#!/usr/bin/env python3
"""
Diagnostic Oracle: Compares DRL agent reward against two baseline policies
in the SAME environment to validate the reward function.

If DRL reward ≈ oracle reward → reward function is broken.
If round-robin beats DRL → policy learning failed.
"""

import requests
import time
import numpy as np
import sys
import os
import json
import threading
import yaml

RYU_URL = 'http://127.0.0.1:8080/sdrlb'
RYU_BASE_URL = 'http://127.0.0.1:8080'


def get_server_metrics(monitor):
    """Get connection and load metrics from monitor."""
    metrics = monitor.get_metrics()
    conns = [metrics.get(h, {}).get('connections', 0) for h in ['h1', 'h2', 'h3']]
    loads = [metrics.get(h, {}).get('load_score', 0) for h in ['h1', 'h2', 'h3']]
    return conns, loads


def compute_reward(conns, loads):
    """Pure connection imbalance reward (matches train.py exactly)."""
    conn_arr = np.array(conns, dtype=np.float64)
    
    conn_mean = conn_arr.mean()
    imbalance = float(np.mean((conn_arr - conn_mean) ** 2))
    
    REWARD_SCALE = 1e-4
    reward = -imbalance * REWARD_SCALE
    
    return reward, imbalance


def reset_episode():
    """Reset controller + flows."""
    from setup_network import setup_complete_routing
    
    dpids = [200, 201, 202, 203, 204, 205, 206, 207]
    for dpid in dpids:
        try:
            requests.post(f'{RYU_BASE_URL}/sdrlb/stats/flowentry/clear',
                          json={'dpid': dpid}, timeout=1.0)
        except:
            pass
    
    try:
        requests.post(f'{RYU_URL}/reset_episode', timeout=1.0)
    except:
        pass
    
    time.sleep(1)
    setup_complete_routing()
    time.sleep(2)


def run_policy(name, action_fn, monitor, traffic_gen, net, duration=30):
    """
    Run a policy for `duration` seconds, collecting rewards.
    
    Args:
        name: Policy name
        action_fn: callable(step, conns, loads) -> action (0, 1, or 2)
        monitor: ServerMonitor instance
        traffic_gen: TrafficGenerator instance
        net: Mininet network
        duration: seconds
    
    Returns:
        dict with summary metrics
    """
    print(f"\n{'='*60}")
    print(f"  ORACLE POLICY: {name}")
    print(f"{'='*60}")
    
    reset_episode()
    
    # Set external mode
    requests.post(f'{RYU_URL}/set_algorithm', json={'algorithm': 'external'}, timeout=1.0)
    requests.post(f'{RYU_URL}/set_training_mode', json={'enabled': True}, timeout=1.0)
    
    # Start traffic in background
    from traffic_generator import BurstyTraffic
    import random
    
    pattern = BurstyTraffic(base_rate=50, burst_rate=200, duration=duration)
    running = [True]
    
    def traffic_thread():
        start = time.time()
        while running[0] and (time.time() - start) < duration:
            rate = pattern.get_rate(time.time() - start)
            if rate > 0:
                batch = max(1, int(rate))
                client = random.choice(traffic_gen.clients)
                traffic_gen.send_batch(client, traffic_gen.virtual_ip,
                                       traffic_gen.virtual_port,
                                       count=batch, concurrency=min(10, batch))
            time.sleep(1.0)
    
    t = threading.Thread(target=traffic_thread, daemon=True)
    t.start()
    
    rewards = []
    imbalances = []
    
    start = time.time()
    step = 0
    
    while time.time() - start < duration:
        conns, loads = get_server_metrics(monitor)
        
        action = action_fn(step, conns, loads)
        requests.post(f'{RYU_URL}/set_action', json={'action': int(action)}, timeout=0.5)
        
        time.sleep(1.0)
        
        # Measure after action
        conns_after, loads_after = get_server_metrics(monitor)
        reward, imbal = compute_reward(conns_after, loads_after)
        
        rewards.append(reward)
        imbalances.append(imbal)
        
        if step % 5 == 0:
            print(f"  [{step}s] Act={action} Conns={conns_after} R={reward:.6f} imbal={imbal:.1f}")
        
        step += 1
    
    running[0] = False
    t.join(timeout=2)
    
    result = {
        'policy': name,
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_imbalance': float(np.mean(imbalances)),
        'final_conns': conns_after,
        'steps': step
    }
    
    print(f"\n  RESULT: mean_R={result['mean_reward']:.6f} ± {result['std_reward']:.6f} | "
          f"mean_imbal={result['mean_imbalance']:.1f}")
    
    return result


def main():
    if os.geteuid() != 0:
        print("❌ Must run as root (for Mininet)")
        sys.exit(1)
    
    # Check controller
    try:
        requests.get(f'{RYU_URL}/stats', timeout=1)
    except:
        print("❌ Ryu controller not running!")
        sys.exit(1)
    
    print("✅ Controller is running")
    
    # Setup network
    from mininet_topology import start_network
    from setup_network import setup_complete_routing
    from traffic_generator import TrafficGenerator
    from real_server_monitor import ServerMonitor
    
    net = start_network()
    time.sleep(15)
    setup_complete_routing()
    time.sleep(5)
    
    traffic_gen = TrafficGenerator(net, virtual_ip="10.0.0.100", virtual_port=8000)
    traffic_gen.start_http_servers()
    
    monitor = ServerMonitor(net, server_hosts=['h1', 'h2', 'h3'])
    monitor.start_monitoring(interval=1.0)
    time.sleep(3)
    
    DURATION = 30
    
    # === Policy 1: Always send to server with MOST connections (worst case) ===
    def always_max_policy(step, conns, loads):
        return int(np.argmax(conns))
    
    # === Policy 2: Round-robin ===
    def round_robin_policy(step, conns, loads):
        return step % 3
    
    # === Policy 3: Always server 0 (collapse policy) ===
    def always_h1_policy(step, conns, loads):
        return 0
    
    # === Policy 4: Least connections (optimal baseline) ===
    def least_conns_policy(step, conns, loads):
        return int(np.argmin(conns))
    
    results = []
    
    for name, fn in [
        ("Always-Max-Connections (worst)", always_max_policy),
        ("Round-Robin", round_robin_policy),
        ("Always-H1 (collapse)", always_h1_policy),
        ("Least-Connections (optimal)", least_conns_policy),
    ]:
        result = run_policy(name, fn, monitor, traffic_gen, net, duration=DURATION)
        results.append(result)
    
    # Summary
    print(f"\n\n{'='*70}")
    print(f"  DIAGNOSTIC ORACLE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Policy':<35} {'Mean R':>12} {'Std R':>12} {'Mean Imbal':>12}")
    print(f"{'-'*75}")
    for r in results:
        print(f"  {r['policy']:<33} {r['mean_reward']:>12.6f} {r['std_reward']:>12.6f} {r['mean_imbalance']:>12.1f}")
    print(f"{'='*75}")
    
    # Check: if all rewards are similar, reward is broken
    mean_rewards = [r['mean_reward'] for r in results]
    reward_range = max(mean_rewards) - min(mean_rewards)
    
    if reward_range < 0.5:
        print("\n⚠️  WARNING: All policies have similar rewards!")
        print("   → Reward function is BROKEN. It does not distinguish good from bad.")
    else:
        print(f"\n✅ Reward function discriminates policies (range={reward_range:.3f})")
        best = results[np.argmax(mean_rewards)]
        worst = results[np.argmin(mean_rewards)]
        print(f"   Best:  {best['policy']} (R={best['mean_reward']:.3f})")
        print(f"   Worst: {worst['policy']} (R={worst['mean_reward']:.3f})")
    
    # Save
    os.makedirs('logs', exist_ok=True)
    with open('logs/diagnostic_oracle.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to logs/diagnostic_oracle.json")
    
    # Cleanup
    monitor.stop_monitoring()
    traffic_gen.stop()
    for h in ['h1', 'h2', 'h3']:
        host = net.get(h)
        if host:
            host.cmd('pkill -f "python3 -m http.server"')
    net.stop()


if __name__ == '__main__':
    main()
