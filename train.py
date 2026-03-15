#!/usr/bin/env python3
"""
Complete DRL Training with REAL Server Load Monitoring

This is the CORRECT way to train for server load balancing:
1. Real HTTP servers running on h1, h2, h3
2. Real traffic from clients creating actual CPU load
3. REAL CPU/memory/latency measurement
4. DRL agent learns to balance REAL load
"""

import requests
import json
import time
import yaml
import numpy as np
import os
import logging
import threading
import sys
import argparse

# Setup logger
logger = logging.getLogger('dqn_trainer')
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)5s: %(message)s', '%H:%M:%S'))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)  # Default level

from drl_agent import DQNAgent
# from trainer import get_port_and_flow_stats, get_host_ports, post_flow_entry, detect_dpids
# build_state is now local or imported from controller logic, let's redefine it here to match controller
from traffic_generator import TrafficGenerator, ConstantTraffic, BurstyTraffic, IncrementalTraffic
from real_server_monitor import ServerMonitor, collect_real_server_metrics, calculate_reward_from_real_load
from setup_network import setup_complete_routing

# Import metrics and set up real monitoring
import math
import utils.metrics as metrics_module

# Base URL for Ryu Controller
RYU_BASE_URL = 'http://127.0.0.1:8080'
RYU_URL = f'{RYU_BASE_URL}/sdrlb'  # Keep for backward compatibility with existing code using RYU_URL

def detect_dpids():
    """Query connected switches from Ryu controller"""
    try:
        resp = requests.get(f'{RYU_BASE_URL}/stats/switches', timeout=2.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"[WARN] Failed to detect DPIDs: {e}")
    return []

def is_server_alive(host_ip, port=8000, timeout=1.0, net=None):
    """Check whether a server is alive via a lightweight HTTP probe."""
    if net is not None:
        try:
            client = net.get('h4')
            if client:
                res = client.cmd(f'curl -s -o /dev/null -w "%{{http_code}}" -m {timeout} http://{host_ip}:{port}/ 2>/dev/null')
                return "200" in res
        except Exception:
            pass
    try:
        r = requests.get(f"http://{host_ip}:{port}/", timeout=timeout)
        return r.status_code < 500
    except Exception:
        return True


# Default server IPs used during training (h1, h2, h3 in Mininet)
SERVER_IPS = ['10.0.0.1', '10.0.0.2', '10.0.0.3']
HOST_NAMES = ['h1', 'h2', 'h3']


def build_state(server_metrics, alive=None):
    """
    Build state vector (9 features): conn_share(3) + load_masked(3) + alive(3).

    This MUST match _build_agent_state() in ryu_controller.py so that the
    Q-network sees the same feature layout at training and inference time.

    Args:
        server_metrics: dict keyed by host name ('h1', 'h2', 'h3')
        alive: np.array of shape (3,) with 1.0/0.0 per server, or None (all alive)
    """
    conn_counts = np.array(
        [server_metrics.get(h, {}).get('connections', 0) for h in HOST_NAMES],
        dtype=np.float32,
    )
    load_vals = np.array(
        [server_metrics.get(h, {}).get('load_score', 0.0) for h in HOST_NAMES],
        dtype=np.float32,
    )

    if alive is None:
        alive = np.ones(3, dtype=np.float32)

    # --- Fix 1: safe connection-share normalization ---
    total = conn_counts.sum()
    if total < 1e-8:
        conn_share = np.array([1/3, 1/3, 1/3], dtype=np.float32)
    else:
        conn_share = conn_counts / total

    # --- Fix 3: mask dead-server load scores ---
    load_vals_masked = load_vals * alive

    # --- Fix 4: consistent state vector ---
    state = np.concatenate([conn_share, load_vals_masked, alive])
    assert not np.isnan(state).any(), "state contains NaN"
    return state


def reset_build_state():
    """Reset build_state history between episodes (no-op now, kept for compat)."""
    pass

# Logger is already defined above
# logger = logging.getLogger(__name__)

class RealLoadBalancerTrainer:
    """
    Complete trainer with REAL server load monitoring
    """
    
    def _is_net_running(self) -> bool:
        """Best-effort check whether Mininet network and host shells are available."""
        net = getattr(self, 'net', None)
        if not net:
            return False
        try:
            host = net.hosts[0]
            return bool(getattr(host, 'shell', None))
        except Exception:
            return False

    def safe_host_exec(self, host, cmd: str, timeout: float = 1.0) -> str:
        """
        Safely execute a command on a Mininet host.
        Falls back to popen if host.cmd() would assert (no shell / waiting).
        Never raises AssertionError.
        """
        host_name = getattr(host, 'name', str(host))
        try:
            if not self._is_net_running():
                logger.warning("Network not running — skipping cmd on %s", host_name)
                return ''

            # Try to start shell if missing
            if not getattr(host, 'shell', None):
                try:
                    host.startShell()
                except Exception:
                    logger.debug("startShell failed for %s; will try popen", host_name)

            # Normal blocking cmd
            try:
                return host.cmd(cmd)
            except AssertionError:
                logger.warning("host.cmd assertion for %s — popen fallback", host_name)
                proc = host.popen(cmd, shell=True)
                try:
                    out, _ = proc.communicate(timeout=timeout)
                    return out.decode() if isinstance(out, bytes) else out
                except Exception:
                    return ''
        except Exception as e:
            logger.warning("safe_host_exec error on %s: %s", host_name, e)
            return ''

    def __init__(self, config_path='config.yaml'):
        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.agent = None
        self.net = None
        self.traffic_gen = None
        self.server_monitor = None
        self.training_active = False
        
        # Training statistics
        self.episode_rewards = []
        self.episode_losses = []
        self.episode_metrics = []
        
        # Action logging
        self.action_log_file = 'action_log.csv'
        with open(self.action_log_file, 'w') as f:
            f.write("timestamp,episode,step,state,action,reward,next_state,done\n")
        
        # Weight sync counter (sync every N steps to avoid timeouts)
        self.sync_counter = 0
        self.sync_interval = 10  # Sync every 10 steps instead of every step

        # Ephemeral flow tracking
        self._routing_installed = False
        self.ephemeral_cookies = []
        self._ephemeral_cookie_counter = 0



    def log_action(self, episode, step, state, action, reward, next_state, done):
        """Log action details to CSV"""
        with open(self.action_log_file, 'a') as f:
            state_str = '|'.join([f'{x:.2f}' for x in state])
            next_state_str = '|'.join([f'{x:.2f}' for x in next_state]) if next_state is not None else ''
            f.write(f"{time.time()},{episode},{step},{state_str},{action},{reward:.4f},{next_state_str},{done}\n")
    
    def sync_weights_to_controller(self):
        """Synchronize trained weights to controller's DRL agent"""
        try:
            import torch
            import base64
            import io
            
            # Serialize Q-network weights
            q_net_buffer = io.BytesIO()
            torch.save(self.agent.q_net.state_dict(), q_net_buffer)
            q_net_bytes = q_net_buffer.getvalue()
            q_net_b64 = base64.b64encode(q_net_bytes).decode('utf-8')
            
            # Serialize target network weights
            target_net_buffer = io.BytesIO()
            torch.save(self.agent.target_net.state_dict(), target_net_buffer)
            target_net_bytes = target_net_buffer.getvalue()
            target_net_b64 = base64.b64encode(target_net_bytes).decode('utf-8')
            
            # Send to controller
            resp = requests.post(
                f'{RYU_URL}/update_weights',
                json={
                    'q_net_weights': q_net_b64,
                    'target_net_weights': target_net_b64
                },
                timeout=5.0  # Increased from 2.0 to handle agent initialization
            )
            
            if resp.status_code == 200:
                # Reset failure counter on success
                if not hasattr(self, '_sync_failures'):
                    self._sync_failures = 0
                self._sync_failures = 0
                return True
            else:
                # Only print every 10th failure to reduce spam
                if not hasattr(self, '_sync_failures'):
                    self._sync_failures = 0
                self._sync_failures += 1
                if self._sync_failures == 1 or self._sync_failures % 10 == 0:
                    print(f"⚠️  Weight sync failed: {resp.status_code} (failure #{self._sync_failures})")
                return False
                
        except Exception as e:
            # Only print first error and every 10th error
            if not hasattr(self, '_sync_failures'):
                self._sync_failures = 0
            self._sync_failures += 1
            if self._sync_failures == 1 or self._sync_failures % 10 == 0:
                logger.warning(f"Weight sync error: {e} (failure #{self._sync_failures})")
            return False

    def add_ephemeral_flow(self, sw, flow_spec):
        """Add a flow with a tracking cookie so it can be cleared later."""
        cookie = 0xFACE0000 + self._ephemeral_cookie_counter
        self._ephemeral_cookie_counter += 1
        cmd = f"ovs-ofctl add-flow {sw.name} \"cookie=0x{cookie:08x},{flow_spec}\""
        self.safe_host_exec(sw, cmd)
        self.ephemeral_cookies.append(cookie)

    def clear_ephemeral_flows(self):
        """Clear only flows created by this trainer during episodes."""
        if not self.ephemeral_cookies:
            return
            
        for sw in self.net.switches:
            for cookie in list(self.ephemeral_cookies):
                cmd = f"ovs-ofctl --strict del-flows {sw.name} \"cookie=0x{cookie:08x}/0xffffffff\""
                self.safe_host_exec(sw, cmd)
        self.ephemeral_cookies = []
        logger.debug("Cleared ephemeral flows via cookie.")

    def verify_action_mapping(self, action):
        """
        Verify that the action was correctly applied to the switch.
        Checks if the controller installed the flow for the virtual IP.
        """
        # Mapping: 0 -> h1, 1 -> h2, 2 -> h3
        server_ips = ['10.0.0.1', '10.0.0.2', '10.0.0.3']
        if not (0 <= action < len(server_ips)):
            return False
            
        target_ip = server_ips[action]
        s1 = self.net.get('s1')
        if not s1: 
            return True # persistent/external switch? can't check
            
        # Check flow table for traffic to VIP=10.0.0.100
        # We look for a flow that modifies nw_dst to target_ip
        # This is heuristics-based since exact flow match depends on controller implementation
        try:
            flows = self.safe_host_exec(s1, "ovs-ofctl dump-flows s1")
            
            # Simple check: do we see the target IP in set_field or set_nw_dst?
            # And is it associated with high priority or recent activity?
            # We assume the controller sets a high priority flow for the active selection
            if f"nw_dst={target_ip}" in flows or f"nw_dst:{target_ip}" in flows:
                return True
            
            # If not found directly, maybe we check if the VIP matches are present
            if "nw_dst=10.0.0.100" in flows and f"actions=...{target_ip}..." in flows:
                return True
                
        except Exception:
            pass
            
        return False # Uncertain or failed

    def install_routing_once(self):
        """Idempotent routing installation."""
        if getattr(self, '_routing_installed', False):
            logger.debug("install_routing_once: routing already installed, skipping.")
            return
        
        logger.info("[SETUP] Installing routing flows (idempotent)...")
        # We rely on setup_complete_routing from setup_network.py
        # It has its own print statements, which is fine for the one-time setup.
        if setup_complete_routing():
             logger.info("Routing setup complete!")
             self._routing_installed = True
        else:
             logger.error("Routing setup failed!")

    def setup_network(self):
        """Initialize Mininet network"""
        logger.info("[SETUP] Starting Mininet network...")
        from mininet_topology import start_network
        
        self.net = start_network()
        logger.info("[SETUP] Network started successfully")
        
        # Debug: List Mininet switches
        logger.debug(f"Mininet switches: {[s.name for s in self.net.switches]}")
        
        logger.info("[SETUP] Waiting 15s for switches to connect to controller...")
        time.sleep(15)
        
        # Debug: Check Ryu switches (with retry)
        ryu_switches = []
        for attempt in range(3):
            try:
                resp = requests.get(f'{RYU_BASE_URL}/stats/switches', timeout=3.0)
                if resp.status_code == 200:
                    ryu_switches = resp.json()
                    logger.debug(f"Ryu switches ({len(ryu_switches)}): {sorted(ryu_switches)}")
                    break
                else:
                    logger.debug(f"Switch query attempt {attempt+1} failed: HTTP {resp.status_code}")
            except Exception as e:
                logger.debug(f"Switch query attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(2)
        
        if not ryu_switches:
            logger.warning("Could not query Ryu switches, but continuing anyway...")
        
        # Install routing flows (IDEMPOTENT call)
        self.install_routing_once()
        time.sleep(5)
        
        # Verify connectivity
        logger.info("[SETUP] Verifying connectivity (h1 -> h3)...")
        h1 = self.net.get('h1')
        h3 = self.net.get('h3')
        result = self.safe_host_exec(h1, f'ping -c 3 -W 1 {h3.IP()}')
        if "0% packet loss" in result:
            logger.info("[SETUP] ✅ Connectivity verified!")
        else:
            logger.warning(f"[SETUP] ❌ Connectivity check failed:\n{result}")
            logger.warning("Multi-hop routing might be broken!")
    
    def setup_monitor(self):
        """Start monitoring (Servers are already started by TrafficGenerator)"""
        
        server_hosts = ['h1', 'h2', 'h3']
        
        # Initialize server monitor for REAL metrics
        logger.info("[SETUP] Initializing REAL server monitor...")
        self.server_monitor = ServerMonitor(self.net, server_hosts=server_hosts)
        self.server_monitor.start_monitoring(interval=2.0)
        
        # Set the global monitor in metrics module
        metrics_module.set_server_monitor(self.server_monitor)
        
        logger.info("[SETUP] ✅ Real server monitoring active!")
    
    def setup_traffic_generator(self):
        """Initialize traffic generator"""
        logger.info("[SETUP] Initializing traffic generator...")
        self.traffic_gen = TrafficGenerator(self.net, virtual_ip="10.0.0.100", virtual_port=8000)
        
        # CRITICAL: Start the HTTP servers!
        self.traffic_gen.start_http_servers()
        
        logger.info("[SETUP] Traffic generator ready")
    
    def setup_agent(self):
        """Initialize DRL agent"""
        logger.info("[SETUP] Initializing DRL agent...")
        self.agent = DQNAgent(self.config)
        logger.info(f"[SETUP] Agent created: state_dim={self.config['drl']['state_dim']}, action_dim={self.config['drl']['action_dim']}")
        
        # Enable training mode in controller (disables session persistence)
        try:
            resp = requests.post(
                f'{RYU_URL}/set_training_mode',
                json={'enabled': True},
                timeout=2.0
            )
            if resp.status_code == 200:
                logger.info("[SETUP] ✅ Controller training mode ENABLED (session persistence disabled)")
            else:
                logger.warning(f"[SETUP] ⚠️  Failed to enable training mode: {resp.status_code}")
        except Exception as e:
            logger.warning(f"[SETUP] ⚠️  Could not enable training mode: {e}")
        
        logger.info("[SETUP] ✅ DRL agent ready")
    
    def generate_traffic_thread(self, pattern, duration):
        """Run traffic generation in background"""
        logger.info(f"[TRAFFIC] Starting {pattern.name} pattern for {duration}s")
        
        start_time = time.time()
        request_count = 0
        
        while self.training_active and (time.time() - start_time) < duration:
            elapsed = time.time() - start_time
            current_rate = pattern.get_rate(elapsed)
            
            if current_rate > 0:
                # Calculate batch size for 1 second (or less)
                batch_duration = 1.0 
                batch_size = int(current_rate * batch_duration)
                if batch_size < 1: batch_size = 1
                
                # Send HTTP requests using Apache Bench (ab) for HIGH LOAD
                import random
                client = random.choice(self.traffic_gen.clients)
                
                # Log which client is being used (for debugging)
                if request_count % 500 == 0:  # Log every 500 requests
                    logger.debug(f"[DEBUG] Using client {client.name} ({client.IP()}) for traffic")
                
                success, count = self.traffic_gen.send_batch(
                    client,
                    self.traffic_gen.virtual_ip,
                    self.traffic_gen.virtual_port,
                    count=batch_size,
                    concurrency=min(10, batch_size)
                )
                
                self.traffic_gen.stats['total_requests'] += batch_size
                request_count += batch_size
                
                # Sleep for the batch duration
                time.sleep(batch_duration)
        
        logger.info(f"[TRAFFIC] Pattern completed: {request_count} requests sent")
    
    def reset_episode(self):
        """
        Reset environment state between episodes.
        Lightweight: clears ephemeral flows & controller state, guarantees routing is safe.
        """
        if not self._is_net_running():
            logger.warning("reset_episode called but network not running — skipping.")
            return

        # 1. Clear ephemeral flows (only those we added during episode)
        self.clear_ephemeral_flows()
        
        # 2. Reset controller episode state
        try:
            requests.post(f'{RYU_URL}/reset_episode', timeout=1.0)
        except:
            pass
        
        # 3. Ensure routing is installed (fast check)
        self.install_routing_once()
        
        # 4. Reset build_state history
        reset_build_state()
        
        # 5. Kill lingering connections on servers
        for h_name in ['h1', 'h2', 'h3']:
            host = self.net.get(h_name)
            if host:
                self.safe_host_exec(host, 'ss -K state time-wait dport = :8000 2>/dev/null')
        
        # 6. Reset connection counts in server monitor (FIX 2)
        if self.server_monitor:
            self.server_monitor.reset_connections()
            logger.info("[RESET] Connection counts zeroed for new episode")
        
        # 7. Set controller to external mode
        try:
            requests.post(f'{RYU_URL}/set_algorithm', json={'algorithm': 'external'}, timeout=1.0)
            requests.post(f'{RYU_URL}/set_training_mode', json={'enabled': True}, timeout=1.0)
        except:
            pass
        
        logger.info("[RESET] Episode state cleared")
        
        print("[RESET] ✅ Episode state cleared")
    
    def train_episode(self, episode_num, episode_duration, traffic_pattern):
        """
        Train one episode with REAL traffic and metrics.

        Includes Fix 2 (liveness), Fix 3 (load masking), Fix 4 (state vector),
        and Fix 6 (episode abort on prolonged server death).
        """
        logger.info(f"{'='*30} Episode {episode_num+1} {'='*30}")
        logger.info(f"Pattern: {traffic_pattern.name} | Duration: {episode_duration}s")
        
        # === CRITICAL: Reset environment for independent episodes ===
        self.reset_episode()
        
        # Show initial server status
        logger.debug("[BEFORE] Initial server status:")
        if logger.isEnabledFor(logging.DEBUG):
             self.server_monitor.print_status()
        
        # Start traffic generation
        traffic_thread = threading.Thread(
            target=self.generate_traffic_thread,
            args=(traffic_pattern, episode_duration)
        )
        traffic_thread.daemon = True
        traffic_thread.start()
        
        start_time = time.time()
        total_reward = 0.0
        total_loss = 0.0
        step_count = 0
        action_counts = {}
        
        # Diagnostic: per-feature variance tracking
        diag_conns = {h: [] for h in HOST_NAMES}
        diag_cpus  = {h: [] for h in HOST_NAMES}
        diag_loads = {h: [] for h in HOST_NAMES}
        reward_components = {'imbalance': [], 'reward': []}
        
        # --- Fix 6: Track consecutive dead-server steps for episode abort ---
        consecutive_dead_steps = 0
        DEAD_STEP_THRESHOLD = 3
        episode_aborted = False
        # Remember how many transitions we had before this episode
        memory_start_len = len(self.agent.memory)
        
        # Training loop
        while time.time() - start_time < episode_duration:
            # --- Fix 2: Server liveness check ---
            alive = np.array(
                [float(is_server_alive(ip, net=self.net)) for ip in SERVER_IPS],
                dtype=np.float32,
            )
            
            # --- Fix 6: Track consecutive dead steps ---
            if (alive == 0.0).any():
                consecutive_dead_steps += 1
                dead_hosts = [HOST_NAMES[i] for i in range(3) if alive[i] == 0.0]
                if consecutive_dead_steps >= DEAD_STEP_THRESHOLD:
                    logger.warning(
                        f"Episode {episode_num+1} aborted: server {', '.join(dead_hosts)} "
                        f"unreachable for {consecutive_dead_steps}+ steps"
                    )
                    episode_aborted = True
                    break
            else:
                consecutive_dead_steps = 0
            
            # 1. Observe state s (with liveness and load masking)
            current_metrics = self.server_monitor.get_metrics()
            state = build_state(current_metrics, alive=alive)
            
            # 2. Select action a
            action, _ = self.agent.act(state)
            action_counts[action] = action_counts.get(action, 0) + 1
            
            # 3. Apply action to controller
            try:
                resp = requests.post(f'{RYU_URL}/set_action', json={'action': int(action)}, timeout=0.5)
                # Verify action effect periodically (e.g. every 100 steps or if reward is bad)
                if step_count % 50 == 0:
                     if not self.verify_action_mapping(action):
                         logger.warning(f"Action {action} verification failed! Controller might be unresponsive.")
            except:
                pass
            
            # 4. Wait for traffic to flow through the selected server
            time.sleep(1.0)
            
            # 5. Observe next state s' (re-check liveness for next_state)
            next_alive = np.array(
                [float(is_server_alive(ip, net=self.net)) for ip in SERVER_IPS],
                dtype=np.float32,
            )
            next_metrics = self.server_monitor.get_metrics()
            next_state = build_state(next_metrics, alive=next_alive)
            
            # 6. Compute reward — Per-action reward
            #    +1 = picked least loaded, -1 = picked most loaded
            conn_counts = np.array([
                next_metrics.get('h1', {}).get('connections', 0),
                next_metrics.get('h2', {}).get('connections', 0),
                next_metrics.get('h3', {}).get('connections', 0),
            ], dtype=np.float32)

            # --- Fix 2: Hard penalty for routing to dead server ---
            if alive[action] == 0.0:
                reward = -1.0
                action_reward = -1.0
                imbalance = 0.0
            else:
                chosen_conn = conn_counts[action]
                min_conn = conn_counts.min()
                max_conn = conn_counts.max()
                denom = max_conn - min_conn + 1e-8

                # Per-action component: +1.0 = picked least loaded, -1.0 = picked most loaded
                action_reward = 1.0 - 2.0 * (chosen_conn - min_conn) / denom

                # Secondary: penalise overall imbalance (CV = std / mean)
                mean_conn = conn_counts.mean()
                imbalance = float(np.std(conn_counts) / (mean_conn + 1e-8))

                reward = float(np.clip(action_reward - 0.2 * imbalance, -1.0, 1.0))

            total_reward += reward
            
            # Diagnostic tracking
            for h in HOST_NAMES:
                d = next_metrics.get(h, {})
                diag_conns[h].append(d.get('connections', 0))
                diag_cpus[h].append(d.get('cpu', 0.0))
                diag_loads[h].append(d.get('load_score', 0.0))
            reward_components['imbalance'].append(imbalance)
            reward_components['reward'].append(reward)
            
            # State change assertion (warn, don't crash)
            if step_count > 0 and np.allclose(state, next_state, atol=1e-6):
                logger.debug(f"  ⚠️  WARNING: state ≈ next_state at step {step_count}!")
            
            # 7. Store transition and train
            done = (time.time() - start_time >= episode_duration)
            self.agent.remember(state, action, reward, next_state, done)
            
            loss = self.agent.train()
            if loss is not None:
                total_loss += loss
            
            self.log_action(episode_num, step_count, state, action, reward, next_state, done)
            step_count += 1
            
            # Progress logging (Concise, every 5 steps)
            if step_count % 5 == 0:
                elapsed = time.time() - start_time
                q_val = getattr(self.agent, 'last_q_values', 0)
                g_norm = getattr(self.agent, 'last_grad_norm', 0)
                # Ensure values are float/not none for formatting
                q_v = float(q_val) if q_val is not None else 0.0
                g_n = float(g_norm) if g_norm is not None else 0.0
                
                logger.info(f"[{elapsed:.0f}s] Step {step_count} | "
                      f"R={reward:.4f} (imbal_cv={imbalance:.3f} act_r={float(action_reward):.3f}) | "
                      f"Loss={loss if loss else 0:.4f} Q={q_v:.3f} ∇={g_n:.4f} | "
                      f"Act={action} conns={conn_counts.tolist()} alive={alive.tolist()}")
                
        # Wait for traffic thread
        traffic_thread.join(timeout=2)
        
        # --- Fix 6: Discard this episode's experience if aborted ---
        if episode_aborted:
            # Remove transitions added during this episode from replay buffer
            memory_end_len = len(self.agent.memory)
            transitions_to_remove = memory_end_len - memory_start_len
            if transitions_to_remove > 0:
                for _ in range(transitions_to_remove):
                    self.agent.memory.pop()
                logger.warning(
                    f"Discarded {transitions_to_remove} transitions from aborted episode {episode_num+1}"
                )
            # Still record summary stats even for aborted episodes
            self.episode_rewards.append(0.0)
            self.episode_losses.append(0.0)
            self.episode_metrics.append({
                'total_requests': self.traffic_gen.stats['total_requests'],
                'successful_requests': self.traffic_gen.stats['successful_requests'],
                'load_variance': 0.0,
                'action_distribution': action_counts,
                'aborted': True,
            })
            logger.info(f"Summary Ep {episode_num+1}: ABORTED (server death) | Steps={step_count}")
            logger.info("-" * 40)
            return
        
        # Show final server status
        logger.debug("[AFTER] Final server status:")
        if logger.isEnabledFor(logging.DEBUG):
            self.server_monitor.print_status()
        
        # Update target network
        self.agent.update_target()
        
        # Calculate averages
        avg_reward = total_reward / max(step_count, 1)
        avg_loss = total_loss / max(step_count, 1)
        
        # Get final server metrics
        final_server_metrics = self.server_monitor.get_metrics()
        server_loads = [m['load_score'] for m in final_server_metrics.values()]
        load_variance = float(np.var(server_loads))
        
        # Store statistics
        self.episode_rewards.append(avg_reward)
        self.episode_losses.append(avg_loss)
        self.episode_metrics.append({
            'total_requests': self.traffic_gen.stats['total_requests'],
            'successful_requests': self.traffic_gen.stats['successful_requests'],
            'load_variance': load_variance,
            'action_distribution': action_counts,
            'server_metrics': {k: v.copy() for k, v in final_server_metrics.items()}
        })
        
        # === DIAGNOSTIC: per-feature variance ===
        logger.debug(f"--- FEATURE VARIANCE DIAGNOSTIC (Episode {episode_num+1}) ---")
        for h in HOST_NAMES:
            cv = np.var(diag_conns[h]) if diag_conns[h] else 0
            cpv = np.var(diag_cpus[h]) if diag_cpus[h] else 0
            lv = np.var(diag_loads[h]) if diag_loads[h] else 0
            logger.debug(f"  {h}: var(conns)={cv:.2f}  var(cpu)={cpv:.6f}  var(load)={lv:.6f}")
        imbal_list = reward_components['imbalance']
        rew_list = reward_components['reward']
        logger.debug(f"  Reward: mean={np.mean(rew_list):.6f} std={np.std(rew_list):.6f} mean_imbal={np.mean(imbal_list):.1f}")
        logger.debug("---")
        
        # Print summary
        logger.info(f"Summary Ep {episode_num+1}: Reward={total_reward:.3f} (Avg {avg_reward:.3f}) | Steps={step_count} | LoadVar={load_variance:.6f}")
        logger.info(f"Actions: {action_counts}")
        logger.info("-" * 40)
    
    def evaluate_episode(self, episode_num, duration, pattern):
        """Run a single evaluation episode without exploration"""
        logger.info(f"{'='*10} EVALUATION Episode (Ep {episode_num+1}) {'='*10}")
        self.reset_episode()
        
        # Start traffic
        traffic_thread = threading.Thread(
            target=self.generate_traffic_thread,
            args=(pattern, duration)
        )
        traffic_thread.daemon = True
        traffic_thread.start()
        
        start_time = time.time()
        total_eval_reward = 0.0
        step_count = 0
        
        # Helper to get greedy action
        def get_greedy_action(s):
             action, _ = self.agent.act(s, epsilon=0.0)
             return action
        
        while time.time() - start_time < duration:
            # Check liveness for consistent state vector
            alive = np.array(
                [float(is_server_alive(ip, net=self.net)) for ip in SERVER_IPS],
                dtype=np.float32,
            )
            
            metrics = self.server_monitor.get_metrics()
            state = build_state(metrics, alive=alive)
            action = get_greedy_action(state)
            
            # Apply action
            try:
                requests.post(f'{RYU_URL}/set_action', json={'action': int(action)}, timeout=0.5)
            except:
                pass
            
            time.sleep(1.0)
            
            # Compute reward (consistent with train_episode)
            next_metrics = self.server_monitor.get_metrics()
            conn_counts = np.array([
                next_metrics.get('h1', {}).get('connections', 0),
                next_metrics.get('h2', {}).get('connections', 0),
                next_metrics.get('h3', {}).get('connections', 0),
            ], dtype=np.float32)

            # Dead-server penalty
            if alive[action] == 0.0:
                reward = -1.0
            else:
                chosen_conn = conn_counts[action]
                min_conn, max_conn = conn_counts.min(), conn_counts.max()
                denom = max_conn - min_conn + 1e-8
                action_reward = 1.0 - 2.0 * (chosen_conn - min_conn) / denom
                mean_conn = conn_counts.mean()
                imbalance = float(np.std(conn_counts) / (mean_conn + 1e-8))
                reward = float(np.clip(action_reward - 0.2 * imbalance, -1.0, 1.0))

            total_eval_reward += reward
            step_count += 1
            
        traffic_thread.join(timeout=2)
        
        avg_eval_reward = total_eval_reward / max(1, step_count)
        logger.info(f"EVAL RESULT: Avg Reward = {avg_eval_reward:.4f}")
        return avg_eval_reward

    def train(self):
        """Main training loop"""
        logger.info(f"{'='*30} DRL Training (Real Load) {'='*30}")
        
        try:
            # Setup
            self.setup_network()
            
            # 1. Start Traffic Generator (Starts HTTP Servers)
            self.setup_traffic_generator()
            
            # 2. Start Monitor (After servers are ready)
            self.setup_monitor()
            
            self.setup_agent()
            
            # Training configuration
            num_episodes = self.config['training']['episodes']
            episode_duration = self.config['training']['episode_duration']
            
            # Traffic patterns
            patterns = [
                ConstantTraffic(rate=100, duration=episode_duration),
                BurstyTraffic(base_rate=50, burst_rate=400, duration=episode_duration),
                IncrementalTraffic(start_rate=50, end_rate=300, duration=episode_duration)
            ]
            
            self.training_active = True
            
            # Evaluation config
            eval_every = 20
            baseline_eval = None
            best_eval_reward = -float('inf')
            
            logger.info(f"Starting training for {num_episodes} episodes...")
            logger.info(f"Each episode: {episode_duration}s")
            
            # Train episodes
            for ep in range(num_episodes):
                # Cycle through traffic patterns
                pattern = patterns[ep % len(patterns)]
                
                self.train_episode(ep, episode_duration, pattern)
                
                # Periodic Evaluation
                if (ep + 1) % eval_every == 0:
                    eval_reward = self.evaluate_episode(ep, episode_duration, pattern)
                    
                    if baseline_eval is None:
                        baseline_eval = eval_reward
                        
                    # Calculate improvement
                    perf_impr = eval_reward - baseline_eval
                    perf_pct = 0.0
                    if abs(baseline_eval) > 1e-9:
                         perf_pct = (perf_impr / abs(baseline_eval)) * 100.0
                    
                    logger.info(f"Performance: Eval={eval_reward:.4f} Baseline={baseline_eval:.4f} Impr={perf_impr:.4f} ({perf_pct:.1f}%)")
                    
                    # Update metrics
                    self.episode_metrics[-1]['eval_reward'] = eval_reward
                    self.episode_metrics[-1]['perf_improvement'] = perf_pct
                    
                    # Save best model
                    if eval_reward > best_eval_reward:
                        best_eval_reward = eval_reward
                        self.agent.save_model(f'models/checkpoints/best_model.pth')
                        logger.info("Found new best model! Saved.")
                
                # Save checkpoint every 10 episodes
                if (ep + 1) % 10 == 0:
                    self.save_checkpoint(ep + 1)
                    logger.info(f"✅ Checkpoint saved: episode {ep+1}")
            
        except KeyboardInterrupt:
            logger.warning('Training interrupted by user')
        
        finally:
            self.training_active = False
            self.cleanup()
    
    def save_checkpoint(self, episode):
        """Save model checkpoint"""
        os.makedirs('models/checkpoints', exist_ok=True)
        self.agent.save_model(f'models/checkpoints/dqn_ep{episode}.pth')
    
    def save_final_model(self):
        """Save final trained model and statistics"""
        os.makedirs('models/final', exist_ok=True)
        self.agent.save_model('models/final/dqn_final.pth')
        
        # Save comprehensive training statistics
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_losses': self.episode_losses,
            'episode_metrics': self.episode_metrics,
            'final_epsilon': self.agent.epsilon,
            'total_episodes': len(self.episode_rewards),
            'config': self.config
        }
        
        os.makedirs('logs', exist_ok=True)
        with open('logs/training_with_real_load.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"{'='*30} TRAINING COMPLETED {'='*30}")
        logger.info(f"Final model: models/final/dqn_final.pth")
        logger.info(f"Statistics: logs/training_with_real_load.json")
        logger.info(f"Total episodes: {len(self.episode_rewards)}")
        
        # Show performance improvement
        if len(self.episode_rewards) >= 20:
            first_10_avg = np.mean(self.episode_rewards[:10])
            last_10_avg = np.mean(self.episode_rewards[-10:])
            improvement = last_10_avg - first_10_avg
            logger.info(f"Performance Improvement: {improvement:.3f} ({(improvement/abs(first_10_avg)*100 if first_10_avg != 0 else 0):.1f}%)")
        
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("[CLEANUP] Stopping services...")
        
        if self.server_monitor:
            logger.info("  - Stopping server monitor...")
            self.server_monitor.stop_monitoring()
        
        if self.traffic_gen:
            logger.info("  - Stopping traffic generator...")
            self.traffic_gen.stop()
        
        # Stop HTTP servers
        if self.net and self._is_net_running():
            logger.info("  - Stopping HTTP servers...")
            for host_name in ['h1', 'h2', 'h3']:
                host = self.net.get(host_name)
                if host:
                    self.safe_host_exec(host, 'pkill -f "python3 -m http.server"')
        elif self.net:
            logger.warning("Skipping HTTP server stop — host shells unavailable")
        
        if self.net:
            logger.info("  - Stopping network...")
            self.net.stop()
        
        if self.agent:
            logger.info("  - Saving final model...")
            self.save_final_model()
        
        logger.info("[CLEANUP] Done!")





def main():
    parser = argparse.ArgumentParser(description='DRL Trainer with Real Monitoring')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR'], help='Set logging level')
    args = parser.parse_args()
    
    # Set log level
    logger.setLevel(getattr(logging, args.log_level))
    
    # Run training
    trainer = RealLoadBalancerTrainer(config_path=args.config)
    trainer.train()

if __name__ == '__main__':
    main()