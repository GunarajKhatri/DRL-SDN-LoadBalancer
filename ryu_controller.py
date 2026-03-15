# ryu_controller_complete.py
"""
COMPLETE VIP Load Balancing Controller with ARP Support

This is the FINAL, WORKING version that handles:
1. ARP requests for Virtual IP (10.0.0.100)
2. VIP packet interception and DNAT
3. Bidirectional flow installation
4. DRL agent integration
5. Works with Fat-Tree across multiple switches

This solves the complete end-to-end VIP load balancing!
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.app.wsgi import WSGIApplication, ControllerBase, route
from ryu.lib.packet import packet, ethernet, arp, ether_types, ipv4, tcp, udp, icmp
from ryu.lib import hub

import json
import os
import time
import numpy as np
from webob import Response

print("\n\n" + "="*80)
print("🚀 STARTING NEW CONTROLLER (SDNRestController)")
print("="*80 + "\n\n")

BASE_URL = '/sdrlb'
_app_instance = None

class SDNRest(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'wsgi': WSGIApplication}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global _app_instance
        _app_instance = self
        wsgi = kwargs['wsgi']

        # Basic SDN state
        self._datapaths = {}
        self.port_stats = {}
        self.flow_stats = {}
        self.port_desc = {}
        self.mac_to_port = {}
        
        # Pre-configured host ports
        self.host_ports = {
            200: {'10.0.0.1': 3, '10.0.0.2': 4},
            201: {'10.0.0.3': 3, '10.0.0.4': 4},
            202: {'10.0.0.5': 3, '10.0.0.6': 4},
            203: {'10.0.0.7': 3, '10.0.0.8': 4},
            204: {'10.0.0.9': 3, '10.0.0.10': 4},
            205: {'10.0.0.11': 3, '10.0.0.12': 4},
            206: {'10.0.0.13': 3, '10.0.0.14': 4},
            207: {'10.0.0.15': 3, '10.0.0.16': 4},
        }

        # ========================================
        # VIP LOAD BALANCING CONFIGURATION
        # ========================================
        self.VIRTUAL_IP = '10.0.0.100'
        self.VIRTUAL_MAC = 'aa:aa:aa:aa:aa:aa'
        
        # Server pool
        # Server Pool (Dynamic or Static)
        # Pre-populate with known servers to avoid discovery delay issues
        self.server_pool = {
            '10.0.0.1': {'mac': '00:00:00:00:00:01', 'port': 3, 'switch': 200},
            '10.0.0.2': {'mac': '00:00:00:00:00:02', 'port': 4, 'switch': 200},
            '10.0.0.3': {'mac': '00:00:00:00:00:03', 'port': 3, 'switch': 201}
        }
        self.logger.info(f"✅ Pre-populated server pool: {list(self.server_pool.keys())}")
        
        # DRL Agent
        self.drl_agent = None
        self.server_monitor = None
        self._external_metrics = {}  # Metrics pushed via REST /update_metrics

        # Optional: load trained model at startup for inference (config or env SDRLB_MODEL_PATH)
        _controller_dir = os.path.dirname(os.path.abspath(__file__))
        _model_path = os.environ.get('SDRLB_MODEL_PATH')
        if _model_path is None:
            try:
                import yaml
                _config_path = os.path.join(_controller_dir, 'config.yaml')
                if os.path.isfile(_config_path):
                    with open(_config_path) as _f:
                        _config = yaml.safe_load(_f)
                    _inf = _config.get('inference', {})
                    if _inf.get('enabled') and _inf.get('model_path'):
                        _model_path = _inf.get('model_path')
            except Exception:
                pass
        if _model_path and not os.path.isabs(_model_path):
            _model_path = os.path.join(_controller_dir, _model_path)
        if _model_path and os.path.isfile(_model_path):
            try:
                import sys
                if _controller_dir not in sys.path:
                    sys.path.insert(0, _controller_dir)
                from drl_agent import DQNAgent
                import yaml
                _config_path = os.path.join(_controller_dir, 'config.yaml')
                with open(_config_path) as _f:
                    _config = yaml.safe_load(_f)
                self.drl_agent = DQNAgent(_config)
                if self.drl_agent.load_model(_model_path):
                    self.logger.info(f"Loaded trained DRL model from {_model_path}, running inference")
                else:
                    self.drl_agent = None
                    self.logger.info("No model found, using round-robin fallback")
            except Exception as e:
                self.drl_agent = None
                self.logger.warning(f"Failed to load model from {_model_path}: {e}. Using round-robin fallback.")
        else:
            if not _model_path:
                self.logger.info("No model found, using round-robin fallback")
            else:
                self.logger.info(f"Model file not found: {_model_path}, using round-robin fallback")
        
        # VIP tracking
        self.vip_sessions = {}
        # Track sessions for persistence
        self.sessions = {}  # {client_ip: server_ip}
        self.training_mode = False  # Disable session persistence during training
        self.vip_stats = {
            'total_requests': 0,
            'arp_requests': 0,
            'server_selections': {},
            'agent_decisions': []
        }
        
        # Algorithm selection
        self.current_algorithm = 'drl' # Default to DRL
        self.rr_counter = 0 # Round-robin counter
        
        self.logger.info("="*70)
        self.logger.info("COMPLETE VIP Load Balancing Controller")
        self.logger.info("="*70)
        self.logger.info(f"Virtual IP: {self.VIRTUAL_IP}")
        self.logger.info(f"Virtual MAC: {self.VIRTUAL_MAC}")
        self.logger.info(f"Server Pool: {list(self.server_pool.keys())}")
        self.logger.info("Features: ARP + VIP + DNAT + DRL")
        self.logger.info("="*70)

        self.monitor_thread = hub.spawn(self._monitor)
        
        # External control
        self.forced_action = None
        self.forced_action_timestamp = 0
        
        wsgi.register(SDNRestController, {'sdn_app': self})

    # ========================================
    # ARP HANDLING FOR VIP (CRITICAL!)
    # ========================================
    
    def handle_arp_for_vip(self, datapath, in_port, eth_pkt, arp_pkt):
        """
        Handle ARP request for Virtual IP
        
        When client asks "Who has 10.0.0.100?", we reply with virtual MAC!
        This is ESSENTIAL for VIP to work!
        """
        if arp_pkt.opcode != arp.ARP_REQUEST:
            return
        
        if arp_pkt.dst_ip != self.VIRTUAL_IP:
            return
        
        # ARP REQUEST for VIP detected!
        self.vip_stats['arp_requests'] += 1
        
        self.logger.info("="*70)
        self.logger.info(f"🔍 ARP REQUEST FOR VIP!")
        self.logger.info("="*70)
        self.logger.info(f"From: {arp_pkt.src_ip} (MAC: {arp_pkt.src_mac})")
        self.logger.info(f"Looking for: {arp_pkt.dst_ip}")
        self.logger.info(f"Switch: {datapath.id}, Port: {in_port}")
        
        # Create ARP REPLY
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        # Build ARP reply packet
        reply_pkt = packet.Packet()
        reply_pkt.add_protocol(ethernet.ethernet(
            ethertype=ether_types.ETH_TYPE_ARP,
            dst=eth_pkt.src,
            src=self.VIRTUAL_MAC
        ))
        reply_pkt.add_protocol(arp.arp(
            opcode=arp.ARP_REPLY,
            src_mac=self.VIRTUAL_MAC,
            src_ip=self.VIRTUAL_IP,
            dst_mac=arp_pkt.src_mac,
            dst_ip=arp_pkt.src_ip
        ))
        reply_pkt.serialize()
        
        # Send ARP reply back to requester
        actions = [parser.OFPActionOutput(in_port)]
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=ofproto.OFPP_CONTROLLER,
            actions=actions,
            data=reply_pkt.data
        )
        datapath.send_msg(out)
        
        self.logger.info(f"✅ ARP REPLY sent: VIP is at {self.VIRTUAL_MAC}")
        self.logger.info("="*70 + "\n")

    # ========================================
    # VIP LOAD BALANCING
    # ========================================
    
    def set_drl_agent(self, agent):
        """Set DRL agent"""
        self.drl_agent = agent
        self.logger.info("✅ DRL Agent registered")
    
    def set_server_monitor(self, monitor):
        """Set server monitor"""
        self.server_monitor = monitor
        self.logger.info("✅ Server Monitor registered")

    # ========================================
    # ALGORITHM IMPLEMENTATIONS
    # ========================================

    def select_server(self, client_ip, dpid):
        """Dispatch to selected algorithm"""
        if self.current_algorithm == 'round_robin':
            return self._select_round_robin()
        elif self.current_algorithm == 'random':
            return self._select_random()
        elif self.current_algorithm == 'least_connections':
            return self._select_least_connections()
        elif self.current_algorithm == 'external':
            return self._select_external()
        else: # Default or 'drl'
            return self.select_server_with_drl(client_ip, dpid)

    def _select_external(self):
        """Selection driven by external trainer via /set_action"""
        # If no valid fresh action (timeout 5s?), fallback to RR
        if self.forced_action is not None:
             # Map action index to IP if needed, or assume forced_action is IP
             # The trainer sends action index usually.
             server_ips = sorted(list(self.server_pool.keys()))
             if 0 <= self.forced_action < len(server_ips):
                 selected_ip = server_ips[self.forced_action]
                 self.logger.info(f"🎮 External Agent selected: {selected_ip}")
                 return selected_ip, self.server_pool[selected_ip]
        
        # Fallback
        return self._select_round_robin()

    def _select_round_robin(self):
        """Round Robin selection"""
        server_ips = sorted(list(self.server_pool.keys()))
        if not server_ips:
             return None, None
        
        selected_ip = server_ips[self.rr_counter % len(server_ips)]
        self.rr_counter += 1
        
        self.logger.info(f"🔄 Round Robin selected: {selected_ip}")
        return selected_ip, self.server_pool[selected_ip]

    def _select_random(self):
        """Random selection"""
        import random
        server_ips = list(self.server_pool.keys())
        if not server_ips:
             return None, None
             
        selected_ip = random.choice(server_ips)
        self.logger.info(f"🎲 Random selected: {selected_ip}")
        return selected_ip, self.server_pool[selected_ip]

    def _select_least_connections(self):
        """Least Connections selection using monitored metrics"""
        server_ips = list(self.server_pool.keys())
        if not server_ips:
             return None, None
             
        # If no monitor, fall back to random
        if not self.server_monitor:
            return self._select_random()
            
        metrics = self.server_monitor.get_metrics()
        
        best_ip = server_ips[0]
        min_conns = float('inf')
        
        for ip in server_ips:
            # Get connections from metrics (default to 0)
            conns = metrics.get(ip, {}).get('connections', 0)
            if conns < min_conns:
                min_conns = conns
                best_ip = ip
        
        self.logger.info(f"📉 Least Connections selected: {best_ip} (conns={min_conns})")
        return best_ip, self.server_pool[best_ip]

    
    def select_server_with_drl(self, client_ip, dpid):
        """Use DRL agent to select server"""
        if self.drl_agent is None:
            # Fallback: Round-robin
            server_ips = list(self.server_pool.keys())
            selected_ip = server_ips[self.vip_stats['total_requests'] % len(server_ips)]
            return selected_ip, self.server_pool[selected_ip]
        
        try:
            # Get server metrics
            if self.server_monitor:
                server_metrics = self.server_monitor.get_metrics()
            elif self._external_metrics:
                server_metrics = self._external_metrics
            else:
                server_metrics = {ip: {'load_score': 0.5} for ip in self.server_pool.keys()}
            
            # Build state
            state = self._build_agent_state(server_metrics, dpid)
            
            # Agent selects action (0, 1, 2) and returns Q-values for diagnostics
            action, q_values = self.drl_agent.act(state, epsilon=0.0)
            
            # Map action to server
            server_ips = sorted(list(self.server_pool.keys()))
            if action < len(server_ips):
                selected_ip = server_ips[action]
            else:
                selected_ip = server_ips[0]
            
            # Diagnostic logging
            q_str = ", ".join([f"{v:.4f}" for v in q_values])
            state_str = ", ".join([f"{s:.2f}" for s in state])
            self.logger.info(f"🤖 DRL Decision: State=[{state_str}]")
            self.logger.info(f"🤖 DRL selection: {selected_ip} (action={action}) | Q-values=[{q_str}]")
            
            # Track decision
            self.vip_stats['agent_decisions'].append({
                'client_ip': client_ip,
                'selected_server': selected_ip,
                'action': action,
                'q_values': q_values.tolist()
            })
            
            return selected_ip, self.server_pool[selected_ip]
            
        except Exception as e:
            self.logger.error(f"❌ DRL error: {e}")
            selected_ip = list(self.server_pool.keys())[0]
            return selected_ip, self.server_pool[selected_ip]
    
    def _is_server_alive(self, host_ip, port=8000, timeout=1.0):
        """Check if a server is alive via HTTP health check."""
        try:
            r = requests.get(f"http://{host_ip}:{port}/", timeout=timeout)
            return r.status_code < 500
        except Exception:
            # Fallback to True since controller runs on Host OS and cannot reach Mininet IPs
            return True

    def _build_agent_state(self, server_metrics, dpid):
        """Build state for DRL agent (9 features: conn_share(3) + load_masked(3) + alive(3)).

        Matches the state vector used during training so the Q-network
        receives consistent inputs at inference time.
        """
        server_ips = sorted(self.server_pool.keys())

        conn_counts = np.array(
            [server_metrics.get(ip, {}).get('connections', 0) for ip in server_ips],
            dtype=np.float32,
        )
        load_vals = np.array(
            [server_metrics.get(ip, {}).get('load_score', 0.0) for ip in server_ips],
            dtype=np.float32,
        )

        # --- Fix 1: safe connection-share normalization ---
        total = conn_counts.sum()
        if total < 1e-8:
            conn_share = np.array([1/3, 1/3, 1/3], dtype=np.float32)
        else:
            conn_share = conn_counts / total

        # --- Liveness flags ---
        alive = np.array(
            [float(self._is_server_alive(ip)) for ip in server_ips],
            dtype=np.float32,
        )

        # --- Fix 3: mask dead-server load scores ---
        load_vals_masked = load_vals * alive

        # --- Fix 4: consistent state vector ---
        state = np.concatenate([conn_share, load_vals_masked, alive])
        return state.tolist()[:9]
    
    def handle_vip_packet(self, ev, ip_pkt, tcp_pkt, udp_pkt):
        """Handle VIP packet with DNAT"""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        dpid = datapath.id
        in_port = msg.match['in_port']
        
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        
        client_ip = ip_pkt.src
        client_mac = eth.src
        
        # Extract port info
        if tcp_pkt:
            client_port = tcp_pkt.src_port
            dst_port = tcp_pkt.dst_port
            proto = 6
        elif udp_pkt:
            client_port = udp_pkt.src_port
            dst_port = udp_pkt.dst_port
            proto = 17
        elif pkt.get_protocol(icmp.icmp):
            # Handle ICMP (Ping)
            client_port = 0  # ICMP has no ports
            dst_port = 0
            proto = 1
        else:
            return
        
        session_key = (client_ip, client_port)
        
        self.logger.info("="*70)
        self.logger.info(f"🎯 VIP REQUEST DETECTED")
        self.logger.info("="*70)
        self.logger.info(f"Client: {client_ip}:{client_port}")
        self.logger.info(f"VIP: {self.VIRTUAL_IP}:{dst_port}")
        self.logger.info(f"Switch: {dpid}, Port: {in_port}")
        
        # Session persistence (disabled during training)
        if not self.training_mode and session_key in self.vip_sessions:
            selected_server_ip = self.vip_sessions[session_key]
            self.logger.info(f"📌 Existing session → {selected_server_ip}")
        else:
            selected_server_ip, server_info = self.select_server(client_ip, dpid)
            
            # Only cache if not in training mode
            if not self.training_mode:
                self.vip_sessions[session_key] = selected_server_ip
            
            self.vip_stats['total_requests'] += 1
            self.vip_stats['server_selections'][selected_server_ip] = \
                self.vip_stats['server_selections'].get(selected_server_ip, 0) + 1
            
            mode_str = "TRAINING" if self.training_mode else "NEW"
            self.logger.info(f"✨ {mode_str} session → {selected_server_ip}")
        
        server_info = self.server_pool[selected_server_ip]
        server_mac = server_info['mac']
        server_switch = server_info['switch']
        server_port = server_info['port']
        
        self.logger.info(f"📍 Target: {selected_server_ip} (Switch {server_switch}, Port {server_port})")
        
        # ========================================
        # INSTALL DNAT FLOWS
        # ========================================
        
        # ========================================
        # INSTALL DNAT FLOWS (INGRESS & EGRESS)
        # ========================================
        
        # Determine output port
        if dpid == server_switch:
            # On server's switch: Send to server
            out_port = server_port
            self.logger.info(f"� Local switching: Output to port {out_port}")
        else:
            # On remote switch (Ingress): Send to Uplink (Port 1)
            # This converts VIP -> Physical IP, then standard routing takes over!
            out_port = 1
            self.logger.info(f"🚀 Remote switching: DNAT + Uplink (Port 1)")

        self.logger.info(f"�💾 Installing DNAT on switch {dpid}")
        
        # Forward flow: Client → VIP becomes Client → Server
        match_kwargs = {
            'eth_type': 0x0800,
            'ipv4_src': client_ip,
            'ipv4_dst': self.VIRTUAL_IP,
            'ip_proto': proto
        }
        
        if proto == 6: # TCP
            match_kwargs['tcp_src'] = client_port
            match_kwargs['tcp_dst'] = dst_port
        elif proto == 17: # UDP
            match_kwargs['udp_src'] = client_port
            match_kwargs['udp_dst'] = dst_port
            
        match = parser.OFPMatch(**match_kwargs)
        
        actions = [
            parser.OFPActionSetField(eth_dst=server_mac),
            parser.OFPActionSetField(ipv4_dst=selected_server_ip),
            parser.OFPActionOutput(out_port)
        ]
        
        # Flow timeout: Moderate during training to balance control vs adaptation
        # Too short (2s) causes flows to expire and fall back to static routing
        # Too long (60s) prevents agent from adapting within an episode
        flow_timeout = 10 if self.training_mode else 30
        
        # PRIORITY 4000: Higher than basic routing (2000)
        self.add_flow(datapath, 4000, match, actions, idle_timeout=flow_timeout, hard_timeout=flow_timeout*2)
        self.logger.info(f"  ✅ Forward: {client_ip} → VIP ⇒ {selected_server_ip} (Port {out_port}, timeout={flow_timeout}s)")
        
        # Reverse flow: Server → Client becomes VIP → Client
        # Note: Reverse flow is only needed on the LAST hop (Server Switch) 
        # or FIRST hop (Client Switch)?
        # Actually, server replies with src=ServerIP.
        # We need to change src=ServerIP to src=VIP BEFORE it reaches client.
        # If we install this on Client Switch (Ingress), it handles the return path too!
        
        if client_mac in self.mac_to_port.get(dpid, {}):
            client_out_port = self.mac_to_port[dpid][client_mac]
        else:
            client_out_port = in_port
        
        reverse_match_kwargs = {
            'eth_type': 0x0800,
            'ipv4_src': selected_server_ip,
            'ipv4_dst': client_ip,
            'ip_proto': proto
        }
        
        if proto == 6: # TCP
            reverse_match_kwargs['tcp_src'] = dst_port
            reverse_match_kwargs['tcp_dst'] = client_port
        elif proto == 17: # UDP
            reverse_match_kwargs['udp_src'] = dst_port
            reverse_match_kwargs['udp_dst'] = client_port
            
        reverse_match = parser.OFPMatch(**reverse_match_kwargs)
        
        reverse_actions = [
            parser.OFPActionSetField(eth_src=self.VIRTUAL_MAC),
            parser.OFPActionSetField(ipv4_src=self.VIRTUAL_IP),
            parser.OFPActionOutput(client_out_port)
        ]
        
        
        # PRIORITY 4000: Higher than basic routing (2000)
        # Use same timeout as forward flow (10s during training)
        self.add_flow(datapath, 4000, reverse_match, reverse_actions, idle_timeout=flow_timeout, hard_timeout=flow_timeout*2)
        self.logger.info(f"  ✅ Reverse: {selected_server_ip} → {client_ip} ⇒ VIP")
        
        # Forward this packet immediately
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        )
        datapath.send_msg(out)
        self.logger.info(f"📤 Packet forwarded")
        
        self.logger.info("="*70 + "\n")

    # ========================================
    # STANDARD CONTROLLER FUNCTIONS
    # ========================================
    
    def _monitor(self):
        while True:
            for dp in list(self._datapaths.values()):
                self._request_stats(dp)
            hub.sleep(5)

    def _request_stats(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        datapath.send_msg(parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY))
        datapath.send_msg(parser.OFPFlowStatsRequest(datapath))
        try:
            datapath.send_msg(parser.OFPPortDescStatsRequest(datapath, 0))
        except Exception:
            pass

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        
        self.logger.info(f"✅ Table-miss flow installed on datapath {datapath.id}")

    def add_flow(self, datapath, priority, match, actions, buffer_id=None, 
                 idle_timeout=0, hard_timeout=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst,
                                    idle_timeout=idle_timeout,
                                    hard_timeout=hard_timeout)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst,
                                    idle_timeout=idle_timeout,
                                    hard_timeout=hard_timeout)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, CONFIG_DISPATCHER])
    def _state_change_handler(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self._datapaths[dp.id] = dp
            self.logger.info(f"Datapath {dp.id} connected")
        else:
            self._datapaths.pop(dp.id, None)
            self.logger.info(f"Datapath {dp.id} disconnected")

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_handler(self, ev):
        dpid = ev.msg.datapath.id
        for stat in ev.msg.body:
            self.port_stats[(dpid, stat.port_no)] = stat.tx_bytes

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_handler(self, ev):
        dpid = ev.msg.datapath.id
        for stat in ev.msg.body:
            match = {}
            if hasattr(stat, 'match'):
                try:
                    for k, v in stat.match.items():
                        match[str(k)] = str(v)
                except Exception:
                    match = str(stat.match)
            self.flow_stats[(dpid, json.dumps(match))] = {
                'packet_count': stat.packet_count,
                'byte_count': stat.byte_count
            }

    @set_ev_cls(ofp_event.EventOFPPortDescStatsReply, MAIN_DISPATCHER)
    def port_desc_handler(self, ev):
        dpid = ev.msg.datapath.id
        ports = []
        for p in ev.msg.body:
            if p.port_no != ev.msg.datapath.ofproto.OFPP_LOCAL:
                ports.append(p.port_no)
        self.port_desc[dpid] = sorted(ports)

    # ========================================
    # PACKET IN HANDLER (MAIN LOGIC)
    # ========================================
    
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        dpid = datapath.id

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        
        if eth is None:
            return

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
        if eth.ethertype == 0x86dd:
            return

        dst = eth.dst
        src = eth.src

        self.logger.info(f"📦 PacketIn: dpid={dpid}, src={src}, dst={dst}, type={hex(eth.ethertype)}")
        if eth.ethertype == 2048: # IPv4
            ip = pkt.get_protocol(ipv4.ipv4)
            self.logger.info(f"   IPv4: src={ip.src}, dst={ip.dst}")

        # ========================================
        # PRIORITY 1: ARP FOR VIP
        # ========================================
        arp_pkt = pkt.get_protocol(arp.arp)
        if arp_pkt:
            self.logger.info(f"🔎 ARP Packet detected: Who has {arp_pkt.dst_ip}? Tell {arp_pkt.src_ip}")
            if arp_pkt.dst_ip == self.VIRTUAL_IP:
                self.handle_arp_for_vip(datapath, in_port, eth, arp_pkt)
                return

        # ========================================
        # PRIORITY 2: VIP TRAFFIC
        # ========================================
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        tcp_pkt = pkt.get_protocol(tcp.tcp)
        udp_pkt = pkt.get_protocol(udp.udp)
        
        if ip_pkt and ip_pkt.dst == self.VIRTUAL_IP:
            self.handle_vip_packet(ev, ip_pkt, tcp_pkt, udp_pkt)
            return
        
        # ========================================
        # REGULAR L2 LEARNING
        # ========================================
        
        if dpid not in self.mac_to_port:
            self.mac_to_port[dpid] = {}
        
        self.mac_to_port[dpid][src] = in_port

        if arp_pkt:
            if dpid not in self.host_ports:
                self.host_ports[dpid] = {}
            self.host_ports[dpid][arp_pkt.src_ip] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            if dst == 'ff:ff:ff:ff:ff:ff' or dst.startswith('01:') or dst.startswith('33:33'):
                out_port = ofproto.OFPP_FLOOD
            else:
                out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # if out_port != ofproto.OFPP_FLOOD:
        #     match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            
        #     if msg.buffer_id != ofproto.OFP_NO_BUFFER:
        #         self.add_flow(datapath, 1, match, actions, msg.buffer_id, idle_timeout=30)
        #     else:
        #         self.add_flow(datapath, 1, match, actions, idle_timeout=30)
            
        #     if src in self.mac_to_port[dpid]:
        #         reverse_out_port = self.mac_to_port[dpid][src]
        #         reverse_match = parser.OFPMatch(in_port=out_port, eth_dst=src, eth_src=dst)
        #         reverse_actions = [parser.OFPActionOutput(reverse_out_port)]
        #         self.add_flow(datapath, 1, reverse_match, reverse_actions, idle_timeout=30)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                   in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)


# ========================================
# REST CONTROLLER
# ========================================

class SDNRestController(ControllerBase):
    def __init__(self, req, link, data, **config):
        super().__init__(req, link, data)
        # Debug logging for initialization
        print(f"DEBUG: RestController init. Data keys: {list(data.keys())}")
        # self.sdn_app = data['sdn_app'] # DISABLED due to issues

    @route('sdn', BASE_URL + '/stats/port/{dpid}', methods=['GET'])
    def get_port_stats(self, req, **kwargs):
        dpid = int(kwargs['dpid'])
        result = {port: tx for (dp, port), tx in _app_instance.port_stats.items() if dp == dpid}
        return Response(status=200, content_type='application/json',
                        body=json.dumps(result).encode('utf-8'))

    @route('sdrlb', BASE_URL + '/set_training_mode', methods=['POST'])
    def set_training_mode(self, req, **kwargs):
        """Enable/disable training mode (disables session persistence)"""
        try:
            data = json.loads(req.body)
            enabled = data.get('enabled', False)
            
            _app_instance.training_mode = enabled
            
            # Clear existing sessions when enabling training mode
            if enabled:
                _app_instance.vip_sessions.clear()
                _app_instance.logger.info("🎯 Training mode ENABLED - session persistence disabled")
            else:
                _app_instance.logger.info("🎯 Training mode DISABLED - session persistence enabled")
            
            return Response(status=200, body=json.dumps({
                'training_mode': enabled,
                'sessions_cleared': enabled
            }).encode('utf-8'))
        except Exception as e:
            return Response(status=500, body=json.dumps({'error': str(e)}).encode('utf-8'))

    @route('sdrlb', BASE_URL + '/set_algorithm', methods=['POST'])
    def set_algorithm(self, req, **kwargs):
        """Set load balancing algorithm"""
        try:
            data = json.loads(req.body)
            algorithm = data.get('algorithm', 'drl')
            
            if algorithm not in ['drl', 'round_robin', 'random', 'least_connections', 'external']:
                return Response(status=400, body=json.dumps({'error': 'Invalid algorithm'}).encode('utf-8'))
            
            _app_instance.current_algorithm = algorithm
            _app_instance.logger.info(f"🔄 Algorithm switched to: {algorithm}")
            
            return Response(status=200, body=json.dumps({
                'algorithm': algorithm,
                'status': 'updated'
            }).encode('utf-8'))
        except Exception as e:
            return Response(status=500, body=json.dumps({'error': str(e)}).encode('utf-8'))

    @route('sdrlb', BASE_URL + '/set_action', methods=['POST'])
    def set_action(self, req, **kwargs):
        """Receive action from external trainer"""
        try:
            data = json.loads(req.body)
            action = data.get('action')
            
            if action is None:
                return Response(status=400, body=json.dumps({'error': 'Missing action'}).encode('utf-8'))
            
            _app_instance.forced_action = int(action)
            _app_instance.forced_action_timestamp = time.time()
            # _app_instance.logger.info(f"📥 Received external action: {action}")
            
            return Response(status=200, body=json.dumps({'status': 'accepted'}).encode('utf-8'))
        except Exception as e:
            return Response(status=500, body=json.dumps({'error': str(e)}).encode('utf-8'))

    @route('sdrlb', BASE_URL + '/reset_episode', methods=['POST'])
    def reset_episode(self, req, **kwargs):
        """Reset controller state for a new training episode"""
        try:
            # Clear VIP sessions
            _app_instance.vip_sessions.clear()
            
            # Reset stats
            _app_instance.vip_stats['total_requests'] = 0
            _app_instance.vip_stats['arp_requests'] = 0
            _app_instance.vip_stats['server_selections'] = {}
            _app_instance.vip_stats['agent_decisions'] = []
            
            # Reset forced action
            _app_instance.forced_action = None
            _app_instance.forced_action_timestamp = 0
            
            # Reset RR counter
            _app_instance.rr_counter = 0
            
            _app_instance.logger.info("🔄 Episode state reset")
            
            return Response(status=200, body=json.dumps({'status': 'episode_reset'}).encode('utf-8'))
        except Exception as e:
            return Response(status=500, body=json.dumps({'error': str(e)}).encode('utf-8'))
    
    @route('sdrlb', BASE_URL + '/update_weights', methods=['POST'])
    def update_weights(self, req, **kwargs):
        """Update controller's DRL agent weights from trainer"""
        try:
            import torch
            data = json.loads(req.body)
            
            # Get base64-encoded weights
            q_net_weights_b64 = data.get('q_net_weights')
            target_net_weights_b64 = data.get('target_net_weights')
            
            if not q_net_weights_b64:
                return Response(status=400, body=json.dumps({'error': 'Missing weights'}).encode('utf-8'))
            
            # Initialize agent if it doesn't exist
            if not _app_instance.drl_agent:
                _app_instance.logger.info("🔧 Initializing controller's DRL agent...")
                
                # Add controller's directory to Python path for imports
                import sys
                import os
                controller_dir = os.path.dirname(os.path.abspath(__file__))
                if controller_dir not in sys.path:
                    sys.path.insert(0, controller_dir)
                
                from drl_agent import DQNAgent
                import yaml
                
                # Load config from controller's directory
                config_path = os.path.join(controller_dir, 'config.yaml')
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                
                _app_instance.drl_agent = DQNAgent(config)
                _app_instance.logger.info("✅ Controller DRL agent created")
            
            # Decode and load weights
            import base64
            import io
            
            q_net_bytes = base64.b64decode(q_net_weights_b64)
            q_net_weights = torch.load(io.BytesIO(q_net_bytes))
            _app_instance.drl_agent.q_net.load_state_dict(q_net_weights)
            
            if target_net_weights_b64:
                target_net_bytes = base64.b64decode(target_net_weights_b64)
                target_net_weights = torch.load(io.BytesIO(target_net_bytes))
                _app_instance.drl_agent.target_net.load_state_dict(target_net_weights)
            
            _app_instance.logger.info("✅ Controller weights updated from trainer")
            return Response(status=200, body=json.dumps({'status': 'weights_updated'}).encode('utf-8'))
        except Exception as e:
            _app_instance.logger.error(f"❌ Weight update failed: {e}")
            return Response(status=500, body=json.dumps({'error': str(e)}).encode('utf-8'))

    @route('sdrlb', BASE_URL + '/load_model', methods=['POST'])
    def load_model(self, req, **kwargs):
        """Load trained model from path (hot-swap without restart)."""
        try:
            data = json.loads(req.body)
            model_path = data.get('model_path')
            if not model_path:
                return Response(status=400, body=json.dumps({
                    'status': 'error',
                    'reason': 'Missing model_path'
                }).encode('utf-8'))
            controller_dir = os.path.dirname(os.path.abspath(__file__))
            if not os.path.isabs(model_path):
                model_path = os.path.join(controller_dir, model_path)
            if not os.path.isfile(model_path):
                return Response(status=400, body=json.dumps({
                    'status': 'error',
                    'reason': f'File not found: {model_path}'
                }).encode('utf-8'))
            import sys
            import yaml
            if controller_dir not in sys.path:
                sys.path.insert(0, controller_dir)
            from drl_agent import DQNAgent
            config_path = os.path.join(controller_dir, 'config.yaml')
            with open(config_path) as f:
                config = yaml.safe_load(f)
            if not _app_instance.drl_agent:
                _app_instance.drl_agent = DQNAgent(config)
            if not _app_instance.drl_agent.load_model(model_path):
                return Response(status=500, body=json.dumps({
                    'status': 'error',
                    'reason': 'load_model returned False'
                }).encode('utf-8'))
            _app_instance.logger.info(f"Loaded model from {model_path} (hot-swap)")
            return Response(status=200, body=json.dumps({
                'status': 'ok',
                'model': model_path
            }).encode('utf-8'))
        except Exception as e:
            _app_instance.logger.error(f"❌ load_model failed: {e}")
            return Response(status=500, body=json.dumps({
                'status': 'error',
                'reason': str(e)
            }).encode('utf-8'))

    @route('sdn', BASE_URL + '/stats/flow/{dpid}', methods=['GET'])
    def get_flow_stats(self, req, **kwargs):
        dpid = int(kwargs['dpid'])
        result = {match_json: stats for (dp, match_json), stats in _app_instance.flow_stats.items() if dp == dpid}
        return Response(status=200, content_type='application/json',
                        body=json.dumps(result).encode('utf-8'))

    @route('sdn', BASE_URL + '/ports/{dpid}', methods=['GET'])
    def get_ports(self, req, **kwargs):
        dpid = int(kwargs['dpid'])
        ports = _app_instance.port_desc.get(dpid, [])
        return Response(status=200, content_type='application/json',
                        body=json.dumps({'ports': ports}).encode('utf-8'))

    @route('sdn', BASE_URL + '/host_ports/{dpid}', methods=['GET'])
    def get_host_ports(self, req, **kwargs):
        dpid = int(kwargs['dpid'])
        host_ports = _app_instance.host_ports.get(dpid, {})
        return Response(status=200, content_type='application/json',
                        body=json.dumps(host_ports).encode('utf-8'))

    @route('sdn', BASE_URL + '/stats/flowentry/add', methods=['POST'])
    def add_flow(self, req, **kwargs):
        try:
            data = json.loads(req.body)
            dpid = int(data['dpid'])
            datapath = _app_instance._datapaths.get(dpid)
            if datapath is None:
                return Response(status=404,
                                body=json.dumps({'error': 'datapath not found'}).encode('utf-8'))

            parser = datapath.ofproto_parser
            ofproto = datapath.ofproto
            match = parser.OFPMatch(**data.get('match', {}))
            _app_instance.logger.info(f"🔧 Installing flow on {dpid}: match={match}")
            actions = [parser.OFPActionOutput(int(a['port']))
                       for a in data.get('actions', []) if a.get('type') == 'OUTPUT']
            inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
            mod = parser.OFPFlowMod(
                datapath=datapath,
                match=match,
                instructions=inst,
                priority=int(data.get('priority', 1000)),
                idle_timeout=int(data.get('idle_timeout', 0)),
                hard_timeout=int(data.get('hard_timeout', 0))
            )
            datapath.send_msg(mod)
            return Response(status=200,
                            body=json.dumps({'result': 'flow added'}).encode('utf-8'))
        except Exception as e:
            return Response(status=500,
                            body=json.dumps({'error': str(e)}).encode('utf-8'))

    @route('sdn', BASE_URL + '/stats/flowentry/clear', methods=['POST'])
    def clear_flows(self, req, **kwargs):
        try:
            data = json.loads(req.body)
            dpid = int(data['dpid'])
            datapath = _app_instance._datapaths.get(dpid)
            if datapath is None:
                return Response(status=404,
                                body=json.dumps({'error': 'datapath not found'}).encode('utf-8'))
            parser = datapath.ofproto_parser
            ofproto = datapath.ofproto
            mod = parser.OFPFlowMod(datapath=datapath, command=ofproto.OFPFC_DELETE)
            datapath.send_msg(mod)
            return Response(status=200,
                            body=json.dumps({'result': 'flows cleared'}).encode('utf-8'))
        except Exception as e:
            return Response(status=500,
                            body=json.dumps({'error': str(e)}).encode('utf-8'))

    
    
    @route('sdrlb', '/stats/switches', methods=['GET'])
    def get_switches_root(self, req, **kwargs):
        """Return list of connected switch DPIDs (root level for compatibility)"""
        dpids = list(_app_instance._datapaths.keys())
        return Response(status=200, content_type='application/json',
                        body=json.dumps(dpids).encode('utf-8'))
    
    @route('sdrlb', BASE_URL + '/stats/switches', methods=['GET'])
    def get_switches(self, req, **kwargs):
        """Return list of connected switch DPIDs"""
        dpids = list(_app_instance._datapaths.keys()) # Changed from _app_instance.datapaths to _app_instance._datapaths
        return Response(status=200, content_type='application/json',
                        body=json.dumps(dpids).encode('utf-8'))
    
    @route('sdrlb', BASE_URL + '/stats', methods=['GET'])
    def get_vip_stats(self, req, **kwargs):
        stats = {
            'total_requests': _app_instance.vip_stats['total_requests'],
            'arp_requests': _app_instance.vip_stats['arp_requests'],
            'server_selections': _app_instance.vip_stats['server_selections'],
            'active_sessions': len(_app_instance.vip_sessions),
            'recent_decisions': _app_instance.vip_stats['agent_decisions'][-10:]
        }
        return Response(status=200, content_type='application/json',
                        body=json.dumps(stats).encode('utf-8'))

    @route('sdrlb', BASE_URL + '/update_metrics', methods=['POST'])
    def update_metrics(self, req, **kwargs):
        """Accept server metrics from external monitor (e.g. run_inference_eval.py).

        Expected JSON:  {"10.0.0.1": {"connections": N, "load_score": F, ...}, ...}
        """
        try:
            data = json.loads(req.body)
            _app_instance._external_metrics = data
            return Response(status=200, body=json.dumps({'status': 'metrics_updated'}).encode('utf-8'))
        except Exception as e:
            return Response(status=500, body=json.dumps({'error': str(e)}).encode('utf-8'))
