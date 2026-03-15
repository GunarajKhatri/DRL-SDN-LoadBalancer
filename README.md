# DRL-SDN Load Balancer

A Deep Reinforcement Learning (DRL) based Load Balancer for Software Defined Networks (SDN) using Ryu and Mininet.

## 🚀 Overview
This project implements an intelligent load balancer that uses a Deep Q-Network (DQN) agent to dynamically route traffic across multiple servers in a Fat-Tree topology. The agent learns to minimize latency and server load variance by interacting with the SDN environment in real-time.

## 🛠️ Prerequisites
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.8+
- **Mininet**: Network emulator
- **Ryu**: SDN Controller framework
- **Open vSwitch**: Virtual switch

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GunarajKhatri/drl-sdn-load-balancer.git
   cd drl-sdn-load-balancer
   ```

2. **Create a virtual environment :**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ How to Run

### 1. Start the Controller
Open a terminal and run the Ryu controller:
```bash
# Ensure you are in the project root and venv is activated 
ryu-manager ryu_controller.py
```

### 2. Start Training
Open a second terminal and run the training script:
```bash
sudo ./venv/bin/python3 train.py
```
This will:
- Initialize the Mininet topology (Fat-Tree).
- Connect to the Ryu controller.
- Start the DRL agent training loop.
- Generate traffic using `ab` (ApacheBench).

### 3. Run inference (after training)
1. Start the controller: `ryu-manager ryu_controller.py` (or let it load the model from `config.yaml` at startup).
2. Optionally push weights: `python inference.py` (if the controller did not load the model at startup).
3. Start Mininet: `sudo ./venv/bin/python3 mininet_topology.py`, then in another terminal run `sudo ./venv/bin/python3 setup_network.py` (with Mininet still running).

### 4. See performance and stabilization
- **Quick check** (with Mininet CLI running): From a host inside Mininet start HTTP servers and send traffic to the VIP, then on your host run:
  ```bash
  curl -s http://127.0.0.1:8080/sdrlb/stats
  ```
  This shows `total_requests`, `server_selections` (how many requests went to each server), and `recent_decisions`.
- **Full evaluation** (metrics + stabilization report): Stop the Mininet CLI if it is running, then:
  ```bash
  sudo ./venv/bin/python3 run_inference_eval.py --duration 120
  ```
  This starts Mininet, generates traffic for 120 seconds, records latency and server distribution, and prints a summary plus an estimate of when the load stabilized (fairness ≥ 0.85 for 15s). Results are saved to `logs/inference_eval_<timestamp>.json` and optionally `plots/inference_eval_<timestamp>.png`.

## 📂 Project Structure
- `ryu_controller.py`: The SDN controller logic (ARP, VIP, DRL integration).
- `train.py`: Main training script (Environment loop, Traffic generation).
- `drl_agent.py`: Deep Q-Network agent implementation (PyTorch).
- `mininet_topology.py`: Custom Fat-Tree topology definition.
- `setup_network.py`: Helper script to configure static routes and flows.
- `traffic_generator.py`: Wrapper for generating HTTP traffic.
- `real_server_monitor.py`: Monitors server metrics (CPU, Latency).
- `inference.py`: Pushes trained model weights to the controller for inference.
- `run_inference_eval.py`: Runs traffic, collects metrics, and reports performance and stabilization.

## 📊 Features
- **Dynamic Routing**: DRL agent selects optimal servers per flow.
- **Real-time Monitoring**: Latency and Load tracking.
- **Training Mode**: Disables session persistence for faster learning.
- **VIP Handling**: Virtual IP (10.0.0.100) for transparent load balancing.

## 📝 Configuration
Edit `config.yaml` to adjust:
- Training parameters (Episodes, Batch size, Learning rate).
- Reward function weights.
- Traffic patterns.

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License
[MIT](https://choosealicense.com/licenses/mit/)
