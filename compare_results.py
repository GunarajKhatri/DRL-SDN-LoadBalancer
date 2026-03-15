#!/usr/bin/env python3
"""
Results Comparator
Plots comparison between DRL and Baseline algorithms.
"""

import json
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd

def load_json(filepath):
    with open(filepath) as f:
        return json.load(f)

def load_baseline_logs(log_dir='logs'):
    files = glob.glob(os.path.join(log_dir, 'baseline_*.json'))
    data = {}
    for f in files:
        algo = f.split('_')[-1].replace('.json', '')
        content = load_json(f)
        # Convert list of dicts to DataFrame
        if isinstance(content, list): # Old format
            df = pd.DataFrame(content)
        elif 'metrics' in content: # New format? MetricsCollector uses list of dicts. 
            # evaluate_baseline.py metrics is a list of dicts.
            # But wait, MetricsCollector.save_to_json saves a list of dicts.
            # But evaluate_baseline.py saves `results` dict containing `time_series`.
            # Let's handle both.
            df = pd.DataFrame(content['metrics']) if 'metrics' in content else pd.DataFrame(content)
        
        # If MetricsCollector format (list of dicts)
        if hasattr(content, 'keys') and 'time_series' in content: 
             # evaluate_baseline format
             df = pd.DataFrame(content['time_series'])
        
        data[algo] = df
    return data

def load_drl_logs(log_dir='logs'):
    # DRL logs might be in action_log.csv or training_with_real_load.json
    # action_log.csv is per step.
    csv_path = os.path.join(log_dir, 'action_log.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Filter for the last episode or best episode? 
        # For comparison, we probably want the FINAL behavior.
        # Let's take the last episode.
        last_episode = df['episode'].max()
        df = df[df['episode'] == last_episode]
        # Align time to start at 0
        start_time = df['timestamp'].min()
        df['relative_time'] = df['timestamp'] - start_time
        return {'drl': df}
    return {}

def plot_comparison(baseline_data, drl_data):
    # Combine
    all_data = {**baseline_data, **drl_data}
    
    if not all_data:
        print("No data found!")
        return

    # 1. Throughput
    plt.figure(figsize=(10, 6))
    for algo, df in all_data.items():
        if 'throughput_bps' in df.columns:
            plt.plot(df['relative_time'], df['throughput_bps']/1e6, label=algo.upper())
    
    plt.xlabel('Time (s)')
    plt.ylabel('Throughput (Mbps)')
    plt.title('Network Throughput Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/throughput_comparison.png')
    print("Saved logs/throughput_comparison.png")

    # 2. Latency (p95)
    plt.figure(figsize=(10, 6))
    for algo, df in all_data.items():
        if 'p95_latency' in df.columns:
             plt.plot(df['relative_time'], df['p95_latency'], label=algo.upper())
    
    plt.xlabel('Time (s)')
    plt.ylabel('p95 Latency (ms)')
    plt.title('Latency (p95) Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/latency_comparison.png')
    print("Saved logs/latency_comparison.png")

    # 3. Fairness
    plt.figure(figsize=(10, 6))
    for algo, df in all_data.items():
        if 'link_fairness' in df.columns:
            plt.plot(df['relative_time'], df['link_fairness'], label=algo.upper())
            
    plt.xlabel('Time (s)')
    plt.ylabel('Jain\'s Fairness Index')
    plt.title('Link Fairness Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/fairness_comparison.png')
    print("Saved logs/fairness_comparison.png")

if __name__ == '__main__':
    b_data = load_baseline_logs()
    d_data = load_drl_logs()
    plot_comparison(b_data, d_data)
