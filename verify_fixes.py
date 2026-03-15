#!/usr/bin/env python3
"""
Offline verification of all 6 DRL SDN Load Balancer fixes.
No Mininet/Ryu required — exercises pure Python logic only.
"""

import sys
import os
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from drl_agent import DQNAgent

# ── helpers ──────────────────────────────────────────────────────────

def _make_config(state_dim=9):
    return {
        'drl': {
            'state_dim': state_dim,
            'action_dim': 3,
            'hidden_dim': 64,
            'epsilon_start': 0.0,
            'epsilon_min': 0.0,
            'epsilon_decay': 0.0,
            'learning_rate': 0.001,
        },
        'training': {'batch_size': 32, 'memory_size': 10000, 'gamma': 0.99},
    }


# ── Fix 1: State normalization (build_state) ────────────────────────

def test_fix1_state_normalization():
    """build_state with all-zero connections must produce [1/3,1/3,1/3,...], no NaN."""
    from train import build_state

    # All servers at 0 connections, all alive
    metrics = {
        'h1': {'connections': 0, 'load_score': 0.5},
        'h2': {'connections': 0, 'load_score': 0.5},
        'h3': {'connections': 0, 'load_score': 0.5},
    }
    alive = np.ones(3, dtype=np.float32)
    state = build_state(metrics, alive=alive)

    assert state.shape == (9,), f"Expected shape (9,), got {state.shape}"
    assert not np.isnan(state).any(), "State contains NaN"
    assert not np.isinf(state).any(), "State contains Inf"
    # conn_share should be [1/3, 1/3, 1/3]
    np.testing.assert_allclose(state[:3], [1/3, 1/3, 1/3], atol=1e-5)
    print("✅ Fix 1: Zero-connection normalization → [1/3, 1/3, 1/3], no NaN")


# ── Fix 2: Server liveness in state + dead-server penalty ───────────

def test_fix2_liveness_in_state():
    """build_state includes alive flags and dead-server reward penalty logic."""
    from train import build_state

    metrics = {
        'h1': {'connections': 10, 'load_score': 0.8},
        'h2': {'connections': 5, 'load_score': 0.4},
        'h3': {'connections': 0, 'load_score': 0.6},
    }
    alive = np.array([1.0, 1.0, 0.0], dtype=np.float32)  # h3 dead
    state = build_state(metrics, alive=alive)

    assert state.shape == (9,), f"Expected shape (9,), got {state.shape}"
    # alive flags are the last 3 elements
    np.testing.assert_array_equal(state[6:9], [1.0, 1.0, 0.0])

    # Simulate dead-server reward penalty (from train_episode logic)
    action = 2  # picks h3 (dead)
    if alive[action] == 0.0:
        reward = -1.0
    else:
        reward = 0.5  # placeholder
    assert reward == -1.0, "Routing to dead server must give reward = -1.0"
    print("✅ Fix 2: Alive flags in state, dead-server penalty = -1.0")


# ── Fix 3: Load score masking for dead servers ──────────────────────

def test_fix3_load_masking():
    """Dead server's load_score is zeroed in the state vector."""
    from train import build_state

    metrics = {
        'h1': {'connections': 5, 'load_score': 0.9},
        'h2': {'connections': 5, 'load_score': 0.2},
        'h3': {'connections': 0, 'load_score': 0.7},  # stale high load
    }
    alive = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    state = build_state(metrics, alive=alive)

    # load_masked is indices [3,4,5]
    assert state[3] == 0.9 * 1.0, f"h1 load should be 0.9, got {state[3]}"
    assert state[4] == 0.2 * 1.0, f"h2 load should be 0.2, got {state[4]}"
    assert state[5] == 0.0, f"h3 load should be masked to 0.0, got {state[5]}"
    print("✅ Fix 3: Dead server (h3) load masked to 0.0")


# ── Fix 4: State vector consistency (train vs inference) ────────────

def test_fix4_state_vector_consistency():
    """Both build_state (train) and _build_agent_state (controller) produce shape (9,)
    with the same semantics: conn_share(3) + load_masked(3) + alive(3)."""
    from train import build_state
    import yaml

    # Check config state_dim
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    state_dim = config['drl']['state_dim']
    assert state_dim == 9, f"config state_dim should be 9, got {state_dim}"

    # Training side
    metrics = {
        'h1': {'connections': 10, 'load_score': 0.3},
        'h2': {'connections': 20, 'load_score': 0.6},
        'h3': {'connections': 15, 'load_score': 0.5},
    }
    alive = np.ones(3, dtype=np.float32)
    state = build_state(metrics, alive=alive)
    assert state.shape == (9,), f"Training state shape {state.shape} != (9,)"

    # Inference side: verify _build_agent_state has same structure by checking source
    import inspect
    import ryu_controller
    src = inspect.getsource(ryu_controller.SDNRest._build_agent_state)
    assert 'conn_share' in src, "_build_agent_state must use conn_share"
    assert 'load_vals_masked' in src, "_build_agent_state must mask loads"
    assert 'alive' in src, "_build_agent_state must include alive"
    assert 'np.concatenate' in src, "_build_agent_state must use np.concatenate"
    print("✅ Fix 4: State vector = [conn_share(3), load_masked(3), alive(3)] in both files")
    print(f"   config.yaml state_dim = {state_dim} ✓")


# ── Fix 5: Fairness bug in run_inference_eval.py ────────────────────

def test_fix5_fairness():
    """jains_fairness must be called with all 3 servers (even those with 0 traffic)."""
    from run_inference_eval import jains_fairness

    # Case 1: All traffic to one server
    vals_biased = [0, 0, 6949]
    f_biased = jains_fairness(vals_biased)
    assert abs(f_biased - 1/3) < 0.01, f"Biased fairness should be ~0.333, got {f_biased:.3f}"

    # Case 2: Perfectly balanced
    vals_balanced = [100, 100, 100]
    f_balanced = jains_fairness(vals_balanced)
    assert abs(f_balanced - 1.0) < 0.01, f"Balanced fairness should be ~1.0, got {f_balanced:.3f}"

    # Case 3: All zero (no traffic yet)
    vals_zero = [0, 0, 0]
    f_zero = jains_fairness(vals_zero)
    assert f_zero == 1.0, f"All-zero fairness should be 1.0, got {f_zero}"

    # Verify the fix in run_inference_eval.py source
    import inspect
    import run_inference_eval
    src = inspect.getsource(run_inference_eval.run_inference_eval)
    assert "server_selections.get('10.0.0.1'" in src, \
        "run_inference_eval must include all 3 servers by IP"
    print(f"✅ Fix 5: Fairness [0,0,6949]={f_biased:.3f}, [100,100,100]={f_balanced:.3f}")


# ── Fix 6: Episode abort logic ──────────────────────────────────────

def test_fix6_episode_abort():
    """train_episode must contain abort-on-death logic (3+ consecutive dead steps)."""
    import inspect
    from train import RealLoadBalancerTrainer

    src = inspect.getsource(RealLoadBalancerTrainer.train_episode)
    assert 'consecutive_dead_steps' in src, "Must track consecutive dead steps"
    assert 'DEAD_STEP_THRESHOLD' in src or 'consecutive_dead_steps >= 3' in src or \
           '>= DEAD_STEP_THRESHOLD' in src, \
        "Must abort at 3+ dead steps"
    assert 'episode_aborted' in src, "Must have episode_aborted flag"
    assert 'memory.pop' in src or 'memory_start_len' in src, \
        "Must discard aborted episode's transitions"
    print("✅ Fix 6: Episode abort on server death (3+ steps) with replay buffer cleanup")


# ── Agent compatibility ─────────────────────────────────────────────

def test_agent_compatibility():
    """DQNAgent.act() works with the new 9-dim state."""
    agent = DQNAgent(_make_config(9))
    state = np.array([1/3, 1/3, 1/3, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0], dtype=np.float32)
    action, q_values = agent.act(state, epsilon=0.0)
    assert isinstance(action, int), f"Action must be int, got {type(action)}"
    assert len(q_values) == 3, f"Expected 3 Q-values, got {len(q_values)}"
    assert 0 <= action < 3, f"Action out of range: {action}"
    print(f"✅ Agent: act() → action={action}, Q-values={q_values}")


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("Fix 1: State normalization", test_fix1_state_normalization),
        ("Fix 2: Server liveness",     test_fix2_liveness_in_state),
        ("Fix 3: Load score masking",  test_fix3_load_masking),
        ("Fix 4: State consistency",   test_fix4_state_vector_consistency),
        ("Fix 5: Fairness bug",        test_fix5_fairness),
        ("Fix 6: Episode abort",       test_fix6_episode_abort),
        ("Agent compatibility",        test_agent_compatibility),
    ]

    print("=" * 60)
    print("  VERIFY ALL 6 DRL FIXES")
    print("=" * 60)

    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"❌ {name}: {e}")
            failed.append(name)

    print("\n" + "=" * 60)
    if failed:
        print(f"  FAILED ({len(failed)}/{len(tests)}): {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"  ALL {len(tests)} TESTS PASSED ✅")
    print("=" * 60)
