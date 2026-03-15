import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Use non-interactive backend
plt.switch_backend('Agg')

def visualize_results(log_path='logs/training_with_real_load.json', output_dir='plots'):
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return

    with open(log_path, 'r') as f:
        data = json.load(f)

    rewards = data.get('episode_rewards', [])
    metrics = data.get('episode_metrics', [])
    
    episodes = np.arange(len(rewards))
    
    # Extract Server Data
    h1_conns, h2_conns, h3_conns = [], [], []
    h1_cpu, h2_cpu, h3_cpu = [], [], []
    
    for m in metrics:
        sm = m.get('server_metrics', {})
        h1_conns.append(sm.get('h1', {}).get('connections', 0))
        h2_conns.append(sm.get('h2', {}).get('connections', 0))
        h3_conns.append(sm.get('h3', {}).get('connections', 0))
        
        h1_cpu.append(sm.get('h1', {}).get('cpu', 0))
        h2_cpu.append(sm.get('h2', {}).get('cpu', 0))
        h3_cpu.append(sm.get('h3', {}).get('cpu', 0))

    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    # 1. Rewards Plot
    ax1.plot(episodes, rewards, color='blue', alpha=0.3, label='Raw Reward')
    # Rolling average
    if len(rewards) > 10:
        rolling_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
        ax1.plot(episodes[9:], rolling_avg, color='navy', linewidth=2, label='10-Ep Moving Avg')
    ax1.set_title('Training Rewards per Episode', fontsize=14)
    ax1.set_ylabel('Avg Reward')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # 2. Connection Distribution Plot
    ax2.plot(episodes, h1_conns, label='h1 (First Host)', color='red', linewidth=1.5)
    ax2.plot(episodes, h2_conns, label='h2', color='green', linewidth=1.5)
    ax2.plot(episodes, h3_conns, label='h3', color='orange', linewidth=1.5)
    ax2.set_title('Server Connection Distribution', fontsize=14)
    ax2.set_ylabel('Active Connections')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    # 3. CPU Usage Plot
    ax3.plot(episodes, h1_cpu, label='h1 CPU', color='red', linestyle='--')
    ax3.plot(episodes, h2_cpu, label='h2 CPU', color='green', linestyle='--')
    ax3.plot(episodes, h3_cpu, label='h3 CPU', color='orange', linestyle='--')
    ax3.set_title('Server CPU Usage (0.0 to 1.0)', fontsize=14)
    ax3.set_ylabel('CPU Load')
    ax3.set_xlabel('Episode')
    ax3.set_ylim(-0.05, 1.1)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()



    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/training_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/training_summary.pdf', bbox_inches='tight')
    
    print(f"✅ Visualizations saved to {output_dir}/training_summary.png")

if __name__ == "__main__":
    visualize_results()
