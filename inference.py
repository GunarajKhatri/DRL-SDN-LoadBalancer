#!/usr/bin/env python3
"""
Standalone inference script: load trained DQN model and push weights to the
Ryu controller so it can run with the trained policy without the trainer.
"""

import argparse
import base64
import io
import os
import sys

import requests
import torch
import yaml

# Project root for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drl_agent import DQNAgent


def load_config(config_path: str) -> dict:
    """Load config from config.yaml."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Push trained DQN model weights to the Ryu controller for inference."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to saved model checkpoint (.pth). Default from config inference.model_path.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Controller REST base URL (e.g. http://127.0.0.1:8080/sdrlb). Default from config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    # Resolve paths relative to script directory (project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(script_dir, args.config)
    if not os.path.isfile(config_path):
        print(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)
    config = load_config(config_path)
    model_path = args.model or config.get("inference", {}).get("model_path") or "models/final/dqn_final.pth"
    model_path = model_path if os.path.isabs(model_path) else os.path.join(script_dir, model_path)

    if not os.path.isfile(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)

    controller_url = args.controller
    if controller_url is None:
        # From config: controller_url or inference section
        controller_url = config.get("inference", {}).get("controller_url") or config.get(
            "controller_url", "http://127.0.0.1:8080/sdrlb"
        )
    controller_url = controller_url.rstrip("/")
    update_weights_url = f"{controller_url}/update_weights"

    # Build agent with same dimensions as training
    agent = DQNAgent(config)
    if not agent.load_model(model_path):
        print("[ERROR] Failed to load model.")
        sys.exit(1)

    # Serialize weights (same format as train.py -> update_weights)
    q_net_buffer = io.BytesIO()
    torch.save(agent.q_net.state_dict(), q_net_buffer)
    q_net_b64 = base64.b64encode(q_net_buffer.getvalue()).decode("utf-8")

    target_net_buffer = io.BytesIO()
    torch.save(agent.target_net.state_dict(), target_net_buffer)
    target_net_b64 = base64.b64encode(target_net_buffer.getvalue()).decode("utf-8")

    try:
        resp = requests.post(
            update_weights_url,
            json={"q_net_weights": q_net_b64, "target_net_weights": target_net_b64},
            timeout=10.0,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] Failed to push weights to controller: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"  Response: {e.response.status_code} {e.response.text[:200]}")
        sys.exit(1)

    print(f"[OK] Weights pushed to controller: {update_weights_url}")
    print(f"[OK] Model: {model_path}")
    print("Controller is now running with the trained policy (inference mode).")


if __name__ == "__main__":
    main()
