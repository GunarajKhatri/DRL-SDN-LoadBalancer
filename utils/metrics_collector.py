import numpy as np
import time
import json
import os

class MetricsCollector:
    """
    Collects and computes detailed metrics for load balancer comparison.
    Matches DRL agent's metric definitions where applicable.
    """
    
    def __init__(self, output_dir='logs'):
        self.output_dir = output_dir
        self.metrics_history = []
        self.start_time = time.time()
        
        # Tracking for throughput calculation
        self.last_bytes = {}
        self.last_check_time = {}
        
    def jains_fairness_index(self, values):
        """
        Compute Jain's Fairness Index.
        J = (sum(x)^2) / (n * sum(x^2))
        """
        if not values or len(values) == 0:
            return 1.0 # Defined as 1 for empty or single user
        
        n = len(values)
        sum_x = sum(values)
        sum_sq = sum(v*v for v in values)
        
        if sum_sq == 0:
            return 1.0 # All zero is "fair"
            
        return (sum_x * sum_x) / (n * sum_sq)

    def compute_link_metrics(self, port_stats, server_ports_map):
        """
        Compute link throughput and fairness.
        
        Args:
            port_stats: dict {dpid: {port: tx_bytes}}
            server_ports_map: dict {server_ip: (dpid, port)}
            
        Returns:
            dict with throughputs, variance, fairness
        """
        # Focus on links connected to servers (downlinks)
        throughputs = []
        metric_map = {}
        
        current_time = time.time()
        
        for server_ip, (dpid, port) in server_ports_map.items():
            key = f"{dpid}:{port}"
            current_bytes = port_stats.get(dpid, {}).get(port, 0)
            
            # Calculate delta
            last_bytes = self.last_bytes.get(key, 0)
            last_time = self.last_check_time.get(key, self.start_time)
            
            # Update history
            self.last_bytes[key] = current_bytes
            self.last_check_time[key] = current_time
            
            # Avoid spike on first reading or div by zero
            dt = current_time - last_time
            if dt > 0 and last_bytes > 0:
                tput_bps = (current_bytes - last_bytes) * 8 / dt
            else:
                tput_bps = 0.0
                
            throughputs.append(tput_bps)
            metric_map[server_ip] = tput_bps
            
        fairness = self.jains_fairness_index(throughputs)
        variance = float(np.var(throughputs)) if throughputs else 0.0
        total_throughput = sum(throughputs)
        
        return {
            'link_throughputs': metric_map,
            'total_throughput_bps': total_throughput,
            'link_fairness': fairness,
            'link_load_variance': variance
        }

    def compute_server_metrics(self, server_metrics):
        """
        Compute aggregate server metrics.
        """
        cpus = [m.get('cpu', 0) for m in server_metrics.values()]
        mems = [m.get('memory', 0) for m in server_metrics.values()]
        conns = [m.get('connections', 0) for m in server_metrics.values()]
        
        return {
            'cpu_mean': float(np.mean(cpus)) if cpus else 0.0,
            'cpu_variance': float(np.var(cpus)) if cpus else 0.0,
            'memory_mean': float(np.mean(mems)) if mems else 0.0,
            'memory_variance': float(np.var(mems)) if mems else 0.0,
            'connections_total': sum(conns),
            'connections_mean': float(np.mean(conns)) if conns else 0.0,
            'connections_peak': max(conns) if conns else 0,
            'server_fairness_cpu': self.jains_fairness_index(cpus),
            'server_fairness_conns': self.jains_fairness_index(conns)
        }

    def log_step(self, timestamp, throughput_stats, latency_stats, server_stats, active_connections):
        """
        Log a single time step.
        """
        record = {
            "t": timestamp,
            # Throughput
            "throughput_bps": throughput_stats.get('total_throughput_bps', 0),
            "link_fairness": throughput_stats.get('link_fairness', 1.0),
            "link_load_variance": throughput_stats.get('link_load_variance', 0.0),
            
            # Latency
            "avg_latency": latency_stats.get('mean', 0.0),
            "p95_latency": latency_stats.get('p95', 0.0),
            "latency_variance": latency_stats.get('variance', 0.0),
            
            # Server Resources
            "cpu_mean": server_stats.get('cpu_mean', 0.0),
            "cpu_variance": server_stats.get('cpu_variance', 0.0),
            "memory_mean": server_stats.get('memory_mean', 0.0),
            "memory_variance": server_stats.get('memory_variance', 0.0),
            "server_fairness_cpu": server_stats.get('server_fairness_cpu', 1.0),
            
            # Connections
            "active_connections": active_connections,
            "server_fairness_connections": server_stats.get('server_fairness_conns', 1.0)
        }
        
        self.metrics_history.append(record)
        return record

    def save_to_csv(self, filename):
        """Save history to CSV"""
        if not self.metrics_history:
            return
            
        import csv
        keys = self.metrics_history[0].keys()
        
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.metrics_history)
            
        print(f"Saved CSV metrics to {path}")

    def save_to_json(self, filename):
        """Save history to JSON"""
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
            
        print(f"Saved JSON metrics to {path}")
