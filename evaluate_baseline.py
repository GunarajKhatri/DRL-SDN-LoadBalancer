#!/usr/bin/env python3
"""
Baseline Algorithm Evaluator
Evaluates Round-Robin, Random, and ECMP/ShortestPath strategies with enhanced metrics.
"""

import requests
import json
import time
import yaml
import numpy as np
import os
import threading
import sys
import argparse
import re
from traffic_generator import TrafficGenerator, ConstantTraffic, BurstyTraffic, IncrementalTraffic
from real_server_monitor import ServerMonitor
from utils.metrics_collector import MetricsCollector

# Base URL for Ryu Controller
RYU_BASE_URL = 'http://127.0.0.1:8080'
RYU_URL = f'{RYU_BASE_URL}/sdrlb'

# ==========================================
# INSTRUMENTED CLASS DEFINITIONS
# ==========================================

class InstrumentedTrafficGenerator(TrafficGenerator):
    """
    Traffic Generator that parses detailed latency stats from ab
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_latency_stats = {
            'mean': 0.0,
            'p95': 0.0,
            'variance': 0.0
        }
        self.latency_samples = []

    def send_batch(self, client, target_ip, port, count, concurrency=10):
        """Override to capture ab output"""
        try:
            # -g generates gnuplot file (tsv), easier to parse? 
            # Or standard output. Let's stick to standard output parsing for now.
            cmd = f"ab -n {count} -c {concurrency} -s 5 http://{target_ip}:{port}/ 2>&1"
            result = client.cmd(cmd)
            
            # DATA PARSING
            stats = {}
            
            # Mean per request (mean)
            # Time per request:       5.123 [ms] (mean)
            mean_match = re.search(r"Time per request:\s+([\d\.]+)\s+\[ms\]\s+\(mean\)", result)
            if mean_match:
                stats['mean'] = float(mean_match.group(1))

            # Connection Times (ms) -> measure variance from "Total" stddev
            #              min  mean[+/-sd] median   max
            # Total:         4    5   1.2      5       8
            # Regex to find Total line
            std_match = re.search(r"Total:\s+\d+\s+\d+\s+([\d\.]+)", result)
            if std_match:
                std = float(std_match.group(1))
                stats['variance'] = std * std
            
            # Percentage served
            #  95%      7
            p95_match = re.search(r"95%\s+(\d+)", result)
            if p95_match:
                stats['p95'] = float(p95_match.group(1))
                
            # Update latest stats if we found valid data
            if 'mean' in stats:
                self.latest_latency_stats = {
                    'mean': stats.get('mean', 0.0),
                    'p95': stats.get('p95', 0.0),
                    'variance': stats.get('variance', 0.0)
                }
                # Keep rolling history? Or just instantaneous?
                # The prompt asks for metrics per time step. We should report what we measured "recently".
            
            # Original logic
            if "Failed requests:        0" in result:
                self.stats["successful_requests"] += count
                self.stats["total_bytes_sent"] += count * 100 
                return True, count
            else:
                fail_match = re.search(r"Failed requests:\s+(\d+)", result)
                if fail_match:
                    failed = int(fail_match.group(1))
                    success = count - failed
                    self.stats["successful_requests"] += success
                    self.stats["failed_requests"] += failed
                    return True, success
                return False, 0
                
        except Exception as e:
            self.stats["failed_requests"] += count
            return False, 0

class BaselineEvaluator:
    def __init__(self, algorithm, duration, config_path='config.yaml'):
        self.algorithm = algorithm
        self.duration = duration
        self.config_path = config_path
        
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.net = None
        self.traffic_gen = None
        self.server_monitor = None
        self.metrics_collector = MetricsCollector(output_dir='logs')
        self.evaluation_active = False

        # Server Map: IP -> (Switch DPID, Port)
        # Based on ryu_controller.py server_pool
        # '10.0.0.1': {'mac': ..., 'port': 3, 'switch': 200},
        # '10.0.0.2': {'mac': ..., 'port': 4, 'switch': 200},
        # '10.0.0.3': {'mac': ..., 'port': 3, 'switch': 201}
        self.server_map = {
            '10.0.0.1': (200, 3),
            '10.0.0.2': (200, 4),
            '10.0.0.3': (201, 3)
        }

    def setup_network(self):
        print("\n[SETUP] Starting Mininet network...")
        from mininet_topology import start_network
        from setup_network import setup_complete_routing
        
        self.net = start_network()
        print("[SETUP] Network started. Waiting for controller...")
        time.sleep(15)
        
        print("[SETUP] Installing routing flows...")
        setup_complete_routing()
        time.sleep(5)
        
        # Verify
        h1 = self.net.get('h1')
        h3 = self.net.get('h3')
        if h1 and h3:
            h1.cmd(f'ping -c 3 -W 1 {h3.IP()}')

    def setup_monitor(self):
        print("\n[SETUP] Initializing REAL server monitor...")
        self.server_monitor = ServerMonitor(self.net, server_hosts=['h1', 'h2', 'h3'])
        self.server_monitor.start_monitoring(interval=1.0)

    def setup_traffic_generator(self):
        print("[SETUP] Initializing traffic generator...")
        # Use INSTRUMENTED class
        self.traffic_gen = InstrumentedTrafficGenerator(self.net, virtual_ip="10.0.0.100", virtual_port=8000)
        self.traffic_gen.start_http_servers()

    def set_controller_algorithm(self):
        print(f"[SETUP] Setting controller algorithm to: {self.algorithm}")
        try:
            requests.post(f'{RYU_URL}/set_algorithm', json={'algorithm': self.algorithm}, timeout=2.0)
            requests.post(f'{RYU_URL}/set_training_mode', json={'enabled': True}, timeout=2.0)
            print(f"[SETUP] ✅ Algorithm {self.algorithm} set (Session persistence DISABLED)")
        except Exception as e:
            print(f"[SETUP] ❌ Error setting algorithm: {e}")
            sys.exit(1)

    def get_port_stats(self):
        """Query Ryu for port stats"""
        stats = {} # {dpid: {port: bytes}}
        for dpid in [200, 201, 202, 203, 204, 205, 206, 207]: # All switches
            try:
                resp = requests.get(f'{RYU_BASE_URL}/stats/port/{dpid}', timeout=0.5)
                if resp.status_code == 200:
                    stats[dpid] = {int(k): v for k, v in resp.json().items()}
            except:
                pass
        return stats

    def generate_traffic_thread(self, pattern, duration):
        print(f"[TRAFFIC] Starting {pattern.name} pattern for {duration}s")
        start_time = time.time()
        
        while self.evaluation_active and (time.time() - start_time) < duration:
            elapsed = time.time() - start_time
            current_rate = pattern.get_rate(elapsed)
            
            if current_rate > 0:
                batch_size = max(1, int(current_rate))
                
                import random
                client = random.choice(self.traffic_gen.clients)
                
                self.traffic_gen.send_batch(
                    client,
                    self.traffic_gen.virtual_ip,
                    self.traffic_gen.virtual_port,
                    count=batch_size,
                    concurrency=min(10, batch_size)
                )
                
                time.sleep(1.0) # 1 sec batches

    def run(self):
        print(f"\n{'='*70}")
        print(f"Evaluator: {self.algorithm.upper()} with ENHANCED METRICS")
        print(f"{'='*70}\n")
        
        try:
            self.setup_network()
            self.setup_traffic_generator()
            self.setup_monitor()
            self.set_controller_algorithm()
            
            self.evaluation_active = True
            
            # Traffic Pattern
            pattern = BurstyTraffic(base_rate=50, burst_rate=200, duration=self.duration)
            
            traffic_thread = threading.Thread(
                target=self.generate_traffic_thread,
                args=(pattern, self.duration)
            )
            traffic_thread.daemon = True
            traffic_thread.start()
            
            start_time = time.time()
            step_count = 0
            
            while time.time() - start_time < self.duration:
                loop_start = time.time()
                
                # 1. Gather all raw data
                port_stats = self.get_port_stats()
                server_metrics = self.server_monitor.get_metrics()
                latency_stats = self.traffic_gen.latest_latency_stats
                
                # 2. Compute derivatives via MetricsCollector
                link_stats = self.metrics_collector.compute_link_metrics(port_stats, self.server_map)
                server_agg_stats = self.metrics_collector.compute_server_metrics(server_metrics)
                
                active_conns = server_agg_stats['connections_total']
                
                # 3. Log
                record = self.metrics_collector.log_step(
                    timestamp=time.time(),
                    throughput_stats=link_stats,
                    latency_stats=latency_stats,
                    server_stats=server_agg_stats,
                    active_connections=active_conns
                )
                
                # Print status
                if step_count % 5 == 0:
                    print(f"[{step_count}s] "
                          f"Tput: {record['throughput_bps']/1e6:.2f} Mbps | "
                          f"Lat(p95): {record['p95_latency']:.1f}ms | "
                          f"Fair(Links): {record['link_fairness']:.3f} | "
                          f"Conns: {active_conns}")
                
                step_count += 1
                
                # Maintain ~1s loop
                process_time = time.time() - loop_start
                sleep_time = max(0.0, 1.0 - process_time)
                time.sleep(sleep_time)
                
            traffic_thread.join(timeout=2)
            
            # Save results
            self.metrics_collector.save_to_csv(f'baseline_{self.algorithm}.csv')
            self.metrics_collector.save_to_json(f'baseline_{self.algorithm}.json')
            
        except KeyboardInterrupt:
            print('\n⚠️  Interrupted')
        finally:
            self.evaluation_active = False
            self.cleanup()

    def cleanup(self):
        if self.server_monitor: self.server_monitor.stop_monitoring()
        if self.traffic_gen: self.traffic_gen.stop()
        if self.net:
            for h in ['h1', 'h2', 'h3']:
                host = self.net.get(h)
                if host: host.cmd('pkill -f "python3 -m http.server"')
            self.net.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, required=True)
    parser.add_argument('--duration', type=int, default=60)
    args = parser.parse_args()
    
    if os.geteuid() != 0:
        print("❌ Must run as root")
        sys.exit(1)
        
    evaluator = BaselineEvaluator(args.algorithm, args.duration)
    evaluator.run()
