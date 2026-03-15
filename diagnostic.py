#!/usr/bin/env python3
"""
Diagnostic Script: Verify Action Enforcement
"""
import requests
import time
import sys

RYU_URL = 'http://127.0.0.1:8080/sdrlb'

def check_controller():
    try:
        requests.get(f'{RYU_URL}/stats', timeout=1)
        return True
    except:
        return False

def force_action(action_idx, duration=10):
    print(f"\n[TEST] Forcing Action {action_idx} for {duration}s...")
    
    # Set algorithm to external
    requests.post(f'{RYU_URL}/set_algorithm', json={'algorithm': 'external'})
    
    start_time = time.time()
    while time.time() - start_time < duration:
        # Send action repeatedly (or once if persistent matches, but loop is safer for test)
        requests.post(f'{RYU_URL}/set_action', json={'action': action_idx})
        
        # Get server status to see if connections are shifting
        resp = requests.get(f'{RYU_URL}/server_status')
        status = resp.json().get('servers', [])
        
        # Simplify output
        summary = {s['name']: s['connections'] for s in status}
        print(f"   Action {action_idx} -> Connections: {summary}")
        
        time.sleep(1.0)

def main():
    if not check_controller():
        print("❌ Ryu controller not running!")
        sys.exit(1)
        
    print("✅ Controller is running")
    
    # Force Server 1 (Action 0)
    force_action(0, duration=10)
    
    # Force Server 2 (Action 1)
    force_action(1, duration=10)
    
    # Force Server 3 (Action 2)
    force_action(2, duration=10)
    
    print("\n✅ Diagnostic Complete. Check if connection counts shifted to the target server.")

if __name__ == '__main__':
    main()
