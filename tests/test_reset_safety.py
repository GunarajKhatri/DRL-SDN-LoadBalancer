#!/usr/bin/env python3
"""
Test: reset_episode() safety against torn-down networks.

Tests:
1. safe_host_exec works when network is running
2. safe_host_exec gracefully skips when network is stopped
3. reset_episode() does not raise when network is stopped

Must run as root (Mininet requirement).
"""

import sys
import os
import time
import logging

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('test_reset_safety')

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_safe_host_exec_while_running():
    """Test that safe_host_exec works normally when net is up."""
    from mininet_topology import start_network
    from setup_network import setup_complete_routing
    from train import RealLoadBalancerTrainer

    logger.info("=== TEST 1: safe_host_exec while network running ===")

    # Create a minimal trainer (we only need .net and the methods)
    trainer = RealLoadBalancerTrainer.__new__(RealLoadBalancerTrainer)
    trainer.net = start_network()
    time.sleep(10)
    setup_complete_routing()
    time.sleep(3)

    # Verify net is running
    assert trainer._is_net_running(), "Network should be running"
    logger.info("✅ _is_net_running() = True")

    # Execute a harmless command on each host
    for h_name in ['h1', 'h2', 'h3']:
        host = trainer.net.get(h_name)
        result = trainer.safe_host_exec(host, 'echo hello')
        assert 'hello' in result, f"Expected 'hello' in output, got: {result!r}"
        logger.info("✅ safe_host_exec on %s returned: %s", h_name, result.strip())

    # Test reset_episode guard (should not raise)
    try:
        trainer.reset_episode()
        logger.info("✅ reset_episode() completed without error")
    except Exception as e:
        logger.error("❌ reset_episode() raised: %s", e)
        raise

    return trainer


def test_safe_host_exec_after_stop(trainer):
    """Test that safe_host_exec gracefully skips after net.stop()."""
    logger.info("\n=== TEST 2: safe_host_exec after network stopped ===")

    # Stop the network
    trainer.net.stop()
    time.sleep(2)

    # _is_net_running should now be False
    assert not trainer._is_net_running(), "Network should NOT be running after stop"
    logger.info("✅ _is_net_running() = False after stop")

    # safe_host_exec should return '' without raising
    host = trainer.net.get('h1')
    if host:
        result = trainer.safe_host_exec(host, 'echo should_not_run')
        assert result == '', f"Expected empty string, got: {result!r}"
        logger.info("✅ safe_host_exec returned '' (skipped)")
    else:
        logger.info("✅ host not found after stop (expected)")

    # reset_episode should skip cleanly
    try:
        trainer.reset_episode()
        logger.info("✅ reset_episode() after stop: skipped cleanly (no error)")
    except Exception as e:
        logger.error("❌ reset_episode() after stop raised: %s", e)
        raise


def main():
    if os.geteuid() != 0:
        print("❌ Must run as root (for Mininet)")
        sys.exit(1)

    print("=" * 60)
    print("  RESET SAFETY TESTS")
    print("=" * 60)

    try:
        trainer = test_safe_host_exec_while_running()
        test_safe_host_exec_after_stop(trainer)
    except Exception as e:
        logger.exception("Test failed: %s", e)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✅")
    print("=" * 60)


if __name__ == '__main__':
    main()
