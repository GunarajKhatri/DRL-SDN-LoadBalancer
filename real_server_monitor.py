#!/usr/bin/env python3
"""
Real Server Load Monitoring for Load Balancing Training

This module provides REAL metrics collection from actual servers:
- CPU usage from processes handling requests
- Memory consumption
- Request latency
- Active connections

NOT simulation - actual measurements!
"""

import time
import threading
import requests
import numpy as np
from collections import defaultdict
import os
import re

class ServerMonitor:
    """
    Monitor real server metrics from Mininet hosts
    
    This tracks:
    - CPU usage of HTTP server processes
    - Memory usage
    - Request count and latency
    - Active connections
    """
    
    def __init__(self, net, server_hosts=['h1', 'h2', 'h3']):
        """
        Args:
            net: Mininet network object
            server_hosts: List of host names acting as servers
        """
        self.net = net
        self.server_hosts = server_hosts
        self.metrics = {}
        self.monitoring = False
        self.monitor_thread = None
        
        # Initialize metrics for each server
        for host_name in server_hosts:
            self.metrics[host_name] = {
                'cpu': 0.0,           # CPU usage (0-1)
                'memory': 0.0,        # Memory usage (0-1)
                'rtt': 0.001,         # Latency in seconds
                'connections': 0,     # Active connections
                'requests': 0,        # Total requests handled
                'load_score': 0.0     # Overall load score (0-1)
            }
    
    def start_monitoring(self, interval=2.0):
        """
        Start continuous monitoring of servers
        
        Args:
            interval: Seconds between metric updates
        """
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,)
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"[Monitor] Started monitoring {len(self.server_hosts)} servers")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            self.monitor_thread = None
        print("[Monitor] Stopped monitoring")

    def reset_connections(self):
        """Zero out connection counts for all servers (called between episodes)."""
        for host_name in self.server_hosts:
            if host_name in self.metrics:
                self.metrics[host_name]['connections'] = 0
    
    def _monitor_loop(self, interval):
        """Background monitoring loop"""
        while self.monitoring:
            for host_name in self.server_hosts:
                self._update_server_metrics(host_name)
            time.sleep(interval)
    
    def _update_server_metrics(self, host_name):
        """
        Update metrics for a specific server
        
        Args:
            host_name: Name of host (e.g., 'h1')
        """
        try:
            host = self.net.get(host_name)
            if host is None:
                return
            
            # Get server IP
            server_ip = host.IP()
            
            # 1. Measure CPU usage of HTTP server process
            cpu = self._get_server_cpu(host)
            
            # 2. Measure memory usage
            memory = self._get_server_memory(host)
            
            # 3. Measure response time (RTT)
            rtt = self._measure_response_time(server_ip)
            
            # 4. Count active connections
            connections = self._count_connections(host)
            
            # 5. Calculate overall load score
            load_score = self._calculate_load_score(cpu, memory, connections)
            
            # Update metrics
            self.metrics[host_name].update({
                'cpu': cpu,
                'memory': memory,
                'rtt': rtt,
                'connections': connections,
                'load_score': load_score
            })
            
        except Exception as e:
            print(f"[Monitor] Error updating metrics for {host_name}: {e}")
    
    def _get_server_cpu(self, host):
        """
        Get CPU usage of HTTP server process on host
        
        Returns:
            float: CPU usage 0.0 to 1.0
        """
        try:
            # Find python HTTP server process and get CPU usage
            cmd = "ps aux | grep 'python.*http.server' | grep -v grep | awk '{sum+=$3} END {print sum}'"
            result = host.cmd(cmd)
            
            if result and result.strip():
                cpu_percent = float(result.strip())
                return min(cpu_percent / 100.0, 1.0)  # Normalize to 0-1
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _get_server_memory(self, host):
        """
        Get memory usage of HTTP server process on host
        
        Returns:
            float: Memory usage 0.0 to 1.0
        """
        try:
            # Get memory usage of python processes
            cmd = "ps aux | grep 'python.*http.server' | grep -v grep | awk '{sum+=$4} END {print sum}'"
            result = host.cmd(cmd)
            
            if result and result.strip():
                mem_percent = float(result.strip())
                return min(mem_percent / 100.0, 1.0)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _measure_response_time(self, server_ip, timeout=2, retries=2):
        """
        Measure actual HTTP response time to server with retry logic
        
        Args:
            server_ip: IP address of server
            timeout: Request timeout in seconds
            retries: Number of retry attempts
        
        Returns:
            float: Response time in seconds
        """
        for attempt in range(retries):
            try:
                # Use dedicated client hosts (not servers) for measurement
                # Rotate through clients to avoid bias
                client_hosts = ['h4', 'h5', 'h6']  # Dedicated clients
                
                # Simple round-robin selection
                if not hasattr(self, '_client_index'):
                    self._client_index = 0
                
                client_name = client_hosts[self._client_index % len(client_hosts)]
                self._client_index += 1
                
                client = self.net.get(client_name)
                if client is None:
                    continue
                
                # Use lightweight HTTP HEAD request with curl timing
                cmd = f'curl -s -o /dev/null -w "%{{time_total}}" --head -m {timeout} http://{server_ip}:8000/ 2>/dev/null'
                
                result = client.cmd(cmd)
                
                # Parse the timing output
                if result and result.strip():
                    try:
                        elapsed = float(result.strip())
                        # Sanity check: if curl reports success, use the time
                        if 0 < elapsed < timeout:
                            return elapsed
                    except ValueError:
                        pass
                
                # If first attempt failed, try again with different client
                if attempt < retries - 1:
                    continue
                    
            except Exception as e:
                if attempt < retries - 1:
                    continue
        
        # All retries failed, return timeout
        return timeout
    
    def _count_connections(self, host):
        """
        Count active TCP connections on port 80
        
        Args:
            host: Mininet host object
        
        Returns:
            int: Number of active connections
        """
        try:
            # Count ALL connections on port 8000 (ESTABLISHED, TIME_WAIT, etc.)
            # Exclude the LISTENING socket itself
            cmd = "netstat -an | grep ':8000 ' | grep -v LISTEN | wc -l"
            result = host.cmd(cmd)
            
            if result and result.strip():
                return int(result.strip())
            
            return 0
            
        except Exception:
            return 0
    
    def _calculate_load_score(self, cpu, memory, connections):
        """
        Calculate overall load score for server
        
        Higher score = more loaded
        
        Args:
            cpu: CPU usage 0-1
            memory: Memory usage 0-1
            connections: Number of connections
        
        Returns:
            float: Load score 0.0 (idle) to 1.0 (overloaded)
        """
        # Weighted combination
        # CPU is most important for load balancing
        score = (
            0.6 * cpu +                           # CPU weight: 60%
            0.2 * memory +                        # Memory weight: 20%
            0.2 * min(connections / 1000.0, 1.0)   # Connections weight: 20% (Normalized to 1000 conns)
        )
        
        return min(score, 1.0)
    
    def get_raw_loads(self):
        """Get raw load scores for debugging"""
        return {h: m['load_score'] for h, m in self.metrics.items()}
    
    def get_metrics(self, host_name=None):
        """
        Get current metrics
        
        Args:
            host_name: Specific host name, or None for all
        
        Returns:
            dict or dict of dicts: Current metrics
        """
        if host_name:
            return self.metrics.get(host_name, {})
        return self.metrics.copy()
    
    def get_least_loaded_server(self):
        """
        Find the server with lowest load
        
        Returns:
            tuple: (host_name, load_score)
        """
        if not self.metrics:
            return None, 1.0
        
        min_load = float('inf')
        best_server = None
        
        for host_name, metrics in self.metrics.items():
            load = metrics.get('load_score', 1.0)
            if load < min_load:
                min_load = load
                best_server = host_name
        
        return best_server, min_load
    
    def print_status(self):
        """Print current status of all servers"""
        print("\n" + "="*70)
        print("Server Load Status")
        print("="*70)
        print(f"{'Server':<10} {'CPU':<8} {'Memory':<8} {'RTT(ms)':<10} {'Conns':<8} {'Load':<8}")
        print("-"*70)
        
        for host_name in sorted(self.metrics.keys()):
            m = self.metrics[host_name]
            print(f"{host_name:<10} "
                  f"{m['cpu']*100:>6.1f}% "
                  f"{m['memory']*100:>6.1f}% "
                  f"{m['rtt']*1000:>8.1f} "
                  f"{m['connections']:>6d} "
                  f"{m['load_score']:>6.2f}")
        
        print("="*70 + "\n")


def collect_real_server_metrics(monitor, num_hosts=16):
    """
    Collect metrics in format expected by trainer
    
    Args:
        monitor: ServerMonitor instance
        num_hosts: Total number of hosts in network
    
    Returns:
        list: Metrics for all hosts (servers get real data, others get baseline)
    """
    metrics = []
    server_metrics = monitor.get_metrics()
    
    for i in range(1, num_hosts + 1):
        host_name = f'h{i}'
        
        if host_name in server_metrics:
            # Real metrics from monitored servers
            m = server_metrics[host_name]
            metrics.append({
                'name': host_name,
                'cpu': float(m['cpu']),
                'mem': float(m['memory']),
                'rtt': float(m['rtt'])
            })
        else:
            # Baseline for non-server hosts (clients)
            metrics.append({
                'name': host_name,
                'cpu': 0.1,   # Low baseline
                'mem': 0.2,   # Low baseline
                'rtt': 0.001  # Fast local
            })
    
    return metrics


# Enhanced reward function that considers real load
def calculate_reward_from_real_load(server_monitor, host_metrics, weights):
    """
    Calculate reward based on REAL server load
    
    This rewards:
    - Balanced load across servers
    - Low latency
    - Avoiding overloaded servers
    
    Args:
        server_monitor: ServerMonitor instance
        host_metrics: Host metrics list
        weights: Reward weights dict
    
    Returns:
        float: Reward value
    """
    alpha = weights.get('alpha', 10.0)
    beta = weights.get('beta', 1.0)
    
    # host_metrics argument is deprecated/unused in this new logic
    # if not host_metrics:
    #    return -1.0
    
    # Get server metrics
    server_metrics = server_monitor.get_metrics()
    
    if not server_metrics:
        return -1.0
    
    # Extract loads
    server_loads = [m['load_score'] for m in server_metrics.values()]
    server_rtts = [m['rtt'] for m in server_metrics.values()]
    
    # 1. Load balance metric (lower variance = better)
    load_variance = float(np.var(server_loads))
    
    # 2. Average latency (lower = better)
    avg_rtt = float(np.mean(server_rtts))
    
    # 3. Overload penalty (heavily penalize if any server > 80% load)
    max_load = max(server_loads) if server_loads else 0.0
    overload_penalty = max(0, (max_load - 0.8) * 10.0)
    
    # 4. Underutilization penalty (waste if all servers idle)
    avg_load = np.mean(server_loads) if server_loads else 0.0
    underutil_penalty = max(0, (0.2 - avg_load) * 2.0) if avg_load < 0.2 else 0.0
    
    # Combined reward (higher is better)
    # Goal: Minimize RTT, Minimize Variance, Avoid Overload
    
    # Softer normalization using exponential decay
    # RTT component: reward decreases as RTT increases
    rtt_reward = np.exp(-avg_rtt / 0.02)  # Decay with 20ms characteristic time
    
    # Variance component: reward decreases as variance increases
    var_reward = np.exp(-load_variance / 0.005)  # Decay with 0.005 characteristic variance
    
    # Overload penalty (only kicks in above 80% load)
    overload_penalty = 0.0
    if max_load > 0.8:
        overload_penalty = (max_load - 0.8) * 5.0  # Reduced from 50.0
    
    # Balance penalty (encourage even distribution)
    balance_bonus = 1.0 - min(load_variance / 0.01, 1.0)
    
    # Combined reward (scaled to roughly [-5, +5] range)
    reward = (
        alpha * rtt_reward +           # Reward low latency (0-10 points)
        beta * var_reward +            # Reward low variance (0-1 points)
        balance_bonus -                # Bonus for balance (0-1 points)
        overload_penalty               # Penalty for overload (0-1 points)
    )
    
    # Shift and clip to reasonable range
    reward = reward - 5.0  # Shift to make typical rewards around 0
    reward = np.clip(reward, -10.0, 10.0)  # Prevent explosions
    
    # print(f"DEBUG: Reward={reward:.4f} (RTT={avg_rtt:.4f}, Var={load_variance:.6f})")
    return float(reward)


# Testing function
def test_real_monitoring():
    """Test real monitoring with Mininet"""
    from mininet_topology import start_network
    
    print("\n[TEST] Starting Mininet network...")
    net = start_network()
    
    print("[TEST] Creating server monitor...")
    monitor = ServerMonitor(net, server_hosts=['h1', 'h2', 'h3'])
    
    print("[TEST] Starting HTTP servers...")
    for host_name in ['h1', 'h2', 'h3']:
        host = net.get(host_name)
        ip = host.IP()
        
        # Create a dedicated directory for each server
        host.cmd(f'mkdir -p /tmp/{host_name}')
        
        # Create index file with clear content
        content = f'<html><body><h1>Server: {host_name}</h1><p>IP: {ip}</p></body></html>'
        host.cmd(f'echo "{content}" > /tmp/{host_name}/index.html')
        
        # Verify file was created
        check = host.cmd(f'cat /tmp/{host_name}/index.html')
        if host_name in check:
            print(f"  - {host_name} ({ip}): Index file created ✅")
        else:
            print(f"  - {host_name} ({ip}): Index file FAILED ❌")
        
        # Start HTTP server in that directory
        host.cmd(f'cd /tmp/{host_name} && python3 -m http.server 80 > /tmp/{host_name}.log 2>&1 &')
        print(f"  - {host_name} ({ip}): HTTP server starting...")
    
    print("[TEST] Waiting for servers to initialize...")
    time.sleep(5)  # Wait for servers to fully start
    
    # Extra verification: check if server responds to localhost
    print("[TEST] Verifying servers respond to localhost...")
    for host_name in ['h1', 'h2', 'h3']:
        host = net.get(host_name)
        local_test = host.cmd('curl -s -m 2 http://127.0.0.1/')
        if host_name in local_test:
            print(f"  ✅ {host_name}: Localhost test OK")
        else:
            print(f"  ❌ {host_name}: Localhost test FAILED")
            # Check the log
            log = host.cmd(f'cat /tmp/{host_name}.log')
            print(f"     Log: {log[:100]}")
    
    # Verify servers are responding
    print("[TEST] Verifying HTTP servers...")
    
    # Use h2 to test h1 (both on switch 200)
    # Use h4 to test h3 (both on switch 201)
    test_pairs = [
        ('h2', 'h1'),  # Same switch 200
        ('h4', 'h3'),  # Same switch 201
        ('h2', 'h3'),  # Cross-switch test
    ]
    
    all_working = True
    
    print("\n  Testing same-switch connectivity:")
    for client_name, server_name in test_pairs[:2]:
        client = net.get(client_name)
        server = net.get(server_name)
        ip = server.IP()
        
        # Try with curl (simpler than wget)
        result = client.cmd(f'curl -s -m 2 http://{ip}/ 2>&1')
        
        if server_name in result:
            print(f"  ✅ {client_name} → {server_name} ({ip}): OK - Got: {result[:50]}")
            all_working = all_working and True
        else:
            print(f"  ❌ {client_name} → {server_name} ({ip}): FAILED")
            print(f"     curl output: {result[:150]}")
            all_working = False
    
    print("\n  Testing cross-switch connectivity:")
    client_name, server_name = test_pairs[2]
    client = net.get(client_name)
    server = net.get(server_name)
    ip = server.IP()
    result = client.cmd(f'curl -s -m 2 http://{ip}/ 2>&1')
    
    if server_name in result:
        print(f"  ✅ {client_name} → {server_name} ({ip}): OK - Got: {result[:50]}")
    else:
        print(f"  ❌ {client_name} → {server_name} ({ip}): FAILED")
        print(f"     curl output: {result[:150]}")
        all_working = False
    
    if not all_working:
        print("\n⚠️  Some servers not responding. Debugging...")
        # Check if process is running
        for host_name in ['h1', 'h2', 'h3']:
            host = net.get(host_name)
            ps_result = host.cmd('ps aux | grep "http.server" | grep -v grep')
            if ps_result.strip():
                print(f"  {host_name}: Process running ✅")
            else:
                print(f"  {host_name}: Process NOT running ❌")
            
            # Check if listening on port 80
            port_check = host.cmd('netstat -tuln | grep :80')
            if port_check.strip():
                print(f"  {host_name}: Port 80 listening ✅")
            else:
                print(f"  {host_name}: Port 80 NOT listening ❌")
        
        # Check connectivity from h4
        print("\n  Testing connectivity from h4:")
        h4 = net.get('h4')
        for host_name in ['h1', 'h2', 'h3']:
            host = net.get(host_name)
            ip = host.IP()
            
            # Ping test
            ping_result = h4.cmd(f'ping -c 1 -W 1 {ip}')
            if '1 received' in ping_result:
                print(f"  {host_name} ({ip}): Ping OK ✅")
            else:
                print(f"  {host_name} ({ip}): Ping FAILED ❌")
            
            # Real TCP test with nc (netcat)
            tcp_test = h4.cmd(f'nc -zv -w 2 {ip} 80 2>&1')
            if 'succeeded' in tcp_test or 'open' in tcp_test:
                print(f"  {host_name} ({ip}): TCP port 80 open ✅")
            else:
                print(f"  {host_name} ({ip}): TCP port 80 closed/filtered ❌")
                print(f"     nc output: {tcp_test[:80]}")
            
            # Try telnet as well
            telnet_test = h4.cmd(f'timeout 2 telnet {ip} 80 2>&1 | head -1')
            print(f"  {host_name} ({ip}): Telnet says: {telnet_test[:80]}")
    
    print("\n[TEST] Starting monitoring...")
    monitor.start_monitoring(interval=3)
    
    try:
        print("[TEST] Monitoring for 30 seconds...")
        print("[TEST] Send some requests to see CPU/load changes!\n")
        
        for i in range(10):
            monitor.print_status()
            
            # Show which server is least loaded
            best, load = monitor.get_least_loaded_server()
            print(f"[TEST] Least loaded server: {best} (load={load:.2f})\n")
            
            time.sleep(3)
            
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")
    
    finally:
        monitor.stop_monitoring()
        
        # Stop HTTP servers
        for host_name in ['h1', 'h2', 'h3']:
            host = net.get(host_name)
            host.cmd('pkill -f "python3 -m http.server"')
        
        net.stop()
        print("[TEST] Test completed")


if __name__ == '__main__':
    test_real_monitoring()