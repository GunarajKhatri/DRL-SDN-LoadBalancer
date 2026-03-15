#!/usr/bin/env python3
"""
Inference evaluation: run traffic against the DRL load balancer, collect
performance metrics, and report summary + stabilization time.

Only two processes needed: (1) ryu-manager ryu_controller.py, (2) this script.
This script pushes the trained DQN weights to the controller, then starts
Mininet, sets up the network, generates traffic, and records metrics.

Usage:
  sudo python3 run_inference_eval.py --duration 120
  sudo python3 run_inference_eval.py --duration 60 --no-plot
  sudo python3 run_inference_eval.py --model models/final/dqn_final.pth

Output: logs/inference_eval_<timestamp>.json, summary in terminal,
optional plots/inference_eval_<timestamp>.png.
"""

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
from datetime import datetime

import requests
import torch
import yaml

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drl_agent import DQNAgent
from evaluate_baseline import InstrumentedTrafficGenerator
from real_server_monitor import ServerMonitor
from traffic_generator import BurstyTraffic, ConstantTraffic
from utils.metrics_collector import MetricsCollector

RYU_BASE_URL = "http://127.0.0.1:8080"
RYU_URL = f"{RYU_BASE_URL}/sdrlb"

SERVER_MAP = {
    "10.0.0.1": (200, 3),
    "10.0.0.2": (200, 4),
    "10.0.0.3": (201, 3),
}


def push_model_to_controller(script_dir, model_path=None):
    """Load trained DQN from disk and push weights to the controller via POST /sdrlb/update_weights."""
    config_path = os.path.join(script_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    path = model_path or config.get("inference", {}).get("model_path") or "models/final/dqn_final.pth"
    if not os.path.isabs(path):
        path = os.path.join(script_dir, path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    agent = DQNAgent(config)
    if not agent.load_model(path):
        raise RuntimeError("DQNAgent.load_model returned False")
    q_buf = io.BytesIO()
    torch.save(agent.q_net.state_dict(), q_buf)
    q_b64 = base64.b64encode(q_buf.getvalue()).decode("utf-8")
    t_buf = io.BytesIO()
    torch.save(agent.target_net.state_dict(), t_buf)
    t_b64 = base64.b64encode(t_buf.getvalue()).decode("utf-8")
    update_url = f"{RYU_URL}/update_weights"
    r = requests.post(
        update_url,
        json={"q_net_weights": q_b64, "target_net_weights": t_b64},
        timeout=10,
    )
    r.raise_for_status()
    return path


def jains_fairness(values):
    if not values or len(values) == 0:
        return 1.0
    n = len(values)
    s = sum(values)
    s2 = sum(v * v for v in values)
    if s2 == 0:
        return 1.0
    return (s * s) / (n * s2)


def run_inference_eval(duration_sec, plot=True, model_path=None):
    from mininet_topology import start_network
    from setup_network import setup_complete_routing

    script_dir = os.path.dirname(os.path.abspath(__file__))
    net = None
    traffic_gen = None
    server_monitor = None
    metrics_collector = MetricsCollector(output_dir="logs")
    evaluation_active = True
    history = []  # {t, total_requests, server_selections, latency_mean, ...}

    try:
        print("\n[0/7] Pushing trained DQN model to controller...")
        try:
            path = push_model_to_controller(script_dir, model_path=model_path)
            print(f"      Model loaded and weights pushed: {path}")
        except Exception as e:
            print(f"      ERROR: {e}")
            print("      Ensure the controller is running and the model file exists.")
            raise

        print("[1/7] Starting Mininet (stop any existing Mininet CLI first)...")
        net = start_network()
        time.sleep(3)

        print("[2/7] Installing routing flows...")
        setup_complete_routing()
        time.sleep(2)

        print("[3/7] Setting controller to DRL inference (training_mode=false)...")
        try:
            requests.post(f"{RYU_URL}/set_algorithm", json={"algorithm": "drl"}, timeout=5)
            requests.post(f"{RYU_URL}/set_training_mode", json={"enabled": False}, timeout=5)
            print("      Controller: DRL, session persistence ON (inference).")
        except Exception as e:
            print(f"      Warning: could not set controller: {e}")

        print("[4/7] Starting HTTP servers on h1, h2, h3...")
        traffic_gen = InstrumentedTrafficGenerator(
            net, virtual_ip="10.0.0.100", virtual_port=8000
        )
        traffic_gen.start_http_servers()

        print("[5/7] Starting server monitor...")
        server_monitor = ServerMonitor(net, server_hosts=["h1", "h2", "h3"])
        server_monitor.start_monitoring(interval=1.0)
        time.sleep(2)

        def traffic_loop():
            pattern = ConstantTraffic(rate=80, duration=duration_sec + 10)
            start = time.time()
            while evaluation_active and (time.time() - start) < duration_sec:
                elapsed = time.time() - start
                rate = pattern.get_rate(elapsed)
                if rate > 0:
                    import random
                    client = random.choice(traffic_gen.clients)
                    traffic_gen.send_batch(
                        client,
                        traffic_gen.virtual_ip,
                        traffic_gen.virtual_port,
                        count=max(1, int(rate)),
                        concurrency=min(10, max(1, int(rate))),
                    )
                time.sleep(1.0)

        print(f"[6/7] Running traffic for {duration_sec}s...")
        traffic_thread = threading.Thread(target=traffic_loop, daemon=True)
        traffic_thread.start()

        start_time = time.time()
        step = 0
        last_total = 0

        while time.time() - start_time < duration_sec:
            loop_start = time.time()
            t = time.time() - start_time

            # Poll controller
            try:
                r = requests.get(f"{RYU_URL}/stats", timeout=2)
                if r.status_code == 200:
                    vip = r.json()
                    total_requests = vip.get("total_requests", 0)
                    server_selections = vip.get("server_selections", {})
                else:
                    total_requests, server_selections = last_total, {}
            except Exception:
                total_requests, server_selections = last_total, {}
            last_total = total_requests

            # Server metrics
            server_metrics = server_monitor.get_metrics() if server_monitor else {}
            conns = [
                server_metrics.get(h, {}).get("connections", 0)
                for h in ["h1", "h2", "h3"]
            ]

            # Push metrics to controller so the DRL agent sees real load
            host_to_ip = {"h1": "10.0.0.1", "h2": "10.0.0.2", "h3": "10.0.0.3"}
            try:
                ip_metrics = {
                    host_to_ip[h]: dict(m) for h, m in server_metrics.items() if h in host_to_ip
                }
                requests.post(f"{RYU_URL}/update_metrics", json=ip_metrics, timeout=1)
            except Exception:
                pass

            # --- Fix 5: Always include all 3 servers (even those with 0 traffic) ---
            all_sel_counts = [
                server_selections.get('10.0.0.1', 0),
                server_selections.get('10.0.0.2', 0),
                server_selections.get('10.0.0.3', 0),
            ]
            fairness_conns = jains_fairness(all_sel_counts)
            latency = getattr(traffic_gen, "latest_latency_stats", {}) or {}

            record = {
                "t_sec": round(t, 1),
                "total_requests": total_requests,
                "server_selections": dict(server_selections),
                "latency_mean_ms": latency.get("mean", 0),
                "latency_p95_ms": latency.get("p95", 0),
                "fairness_connections": fairness_conns,
                "conns_h1": conns[0] if len(conns) > 0 else 0,
                "conns_h2": conns[1] if len(conns) > 1 else 0,
                "conns_h3": conns[2] if len(conns) > 2 else 0,
            }
            history.append(record)

            if step % 10 == 0:
                print(
                    f"  [{step:3d}s] requests={total_requests:5d} | "
                    f"lat(avg)={record['latency_mean_ms']:.1f}ms | "
                    f"fairness={fairness_conns:.3f} | "
                    f"sel={server_selections}"
                )
            step += 1
            time.sleep(max(0.0, 1.0 - (time.time() - loop_start)))

        evaluation_active = False
        traffic_thread.join(timeout=5)

        # --- Summary & stabilization ---
        if not history:
            print("No data collected.")
            return

        total_requests = history[-1]["total_requests"]
        sel = history[-1].get("server_selections", {})
        # Map to IPs in SERVER_MAP to ensure all are counted (even if 0)
        all_ips = sorted(list(SERVER_MAP.keys()))
        vals = [sel.get(ip, 0) for ip in all_ips]
        fairness_final = jains_fairness(vals)
        latencies = [h["latency_mean_ms"] for h in history if h["latency_mean_ms"] > 0]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        p95_latencies = [h["latency_p95_ms"] for h in history if h["latency_p95_ms"] > 0]
        p95_lat = max(p95_latencies) if p95_latencies else 0

        # Stabilization: first time step after which fairness stays >= 0.85 for 15s
        stable_sec = None
        window = 15
        for i in range(len(history) - window):
            slice_fair = [history[j]["fairness_connections"] for j in range(i, i + window)]
            if all(f >= 0.85 for f in slice_fair):
                stable_sec = history[i]["t_sec"]
                break

        print("\n" + "=" * 60)
        print("INFERENCE EVALUATION SUMMARY")
        print("=" * 60)
        print(f"  Duration:           {duration_sec}s")
        print(f"  Total requests:      {total_requests}")
        print(f"  Server distribution: {sel}")
        print(f"  Fairness (final):    {fairness_final:.3f} (1.0 = perfect)")
        print(f"  Latency (avg):       {avg_lat:.1f} ms")
        print(f"  Latency (p95):       {p95_lat:.1f} ms")
        if stable_sec is not None:
            print(f"  Stabilization:        ~{stable_sec:.0f}s (fairness ≥ 0.85 for {window}s)")
        else:
            print("  Stabilization:        (fairness did not stay ≥ 0.85 for 15s in this run)")
        print("=" * 60)

        # Save log
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"logs/inference_eval_{ts}.json"
        summary = {
            "duration_sec": duration_sec,
            "total_requests": total_requests,
            "server_selections_final": sel,
            "fairness_final": fairness_final,
            "latency_avg_ms": avg_lat,
            "latency_p95_ms": p95_lat,
            "stabilization_sec": stable_sec,
            "history": history,
        }
        with open(log_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nLog saved: {log_path}")

        # Optional plot
        if plot and history:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import numpy as np

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                t_axis = [h["t_sec"] for h in history]
                ax1.plot(t_axis, [h["total_requests"] for h in history], label="Total requests")
                ax1.set_ylabel("Requests")
                ax1.legend()
                ax1.grid(True, alpha=0.7)
                ax1.set_title("Inference evaluation")

                ax2.plot(t_axis, [h["fairness_connections"] for h in history], label="Fairness (conns)")
                lat_vals = [h["latency_mean_ms"] for h in history]
                ax2_twin = ax2.twinx()
                ax2_twin.plot(t_axis, lat_vals, color="orange", alpha=0.8, label="Latency (ms)")
                ax2.set_ylabel("Fairness")
                ax2_twin.set_ylabel("Latency (ms)")
                ax2.set_ylim(0, 1.1)
                ax2.axhline(0.85, color="gray", linestyle="--", alpha=0.7)
                ax2.legend(loc="upper left")
                ax2_twin.legend(loc="upper right")
                ax2.grid(True, alpha=0.7)
                ax2.set_xlabel("Time (s)")

                os.makedirs("plots", exist_ok=True)
                plot_path = f"plots/inference_eval_{ts}.png"
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"Plot saved: {plot_path}")
            except Exception as e:
                print(f"Plot skipped: {e}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        evaluation_active = False
        if server_monitor:
            server_monitor.stop_monitoring()
        if traffic_gen:
            traffic_gen.stop()
        if net:
            for h in ["h1", "h2", "h3"]:
                host = net.get(h)
                if host:
                    host.cmd("pkill -f 'python3 -m http.server'")
            net.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Run inference evaluation: traffic + metrics + stabilization report."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help="Evaluation duration in seconds (default: 120)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model .pth (default: from config inference.model_path)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving PNG plot",
    )
    args = parser.parse_args()

    if os.geteuid() != 0:
        print("Run with sudo (Mininet requires root).")
        sys.exit(1)

    run_inference_eval(args.duration, plot=not args.no_plot, model_path=args.model)


if __name__ == "__main__":
    main()
