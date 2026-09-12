import threading
import time
import pandas as pd
from typing import List, Dict
import scapy.all as scapy

from src2.live.live_aggregator import aggregate_packets, _empty_features
from src2.data.feature_engineering import engineer_features, ENGINEERED_FEATURE_COLS
from src2.data.schema import CANONICAL_FEATURES
from src2.models.inference import run_inference

class LivePipeline:
    def __init__(self, model, interface_name=None, pcap_file=None, csv_file=None):
        self.model = model
        self.interface_name = interface_name
        self.pcap_file = pcap_file
        self.csv_file = csv_file
        
        self.history: List[Dict] = []
        self.seq_len = 5
        self.window_duration = 5.0
        
        # Pre-fill history with empty windows so it calculates immediately
        for _ in range(self.seq_len):
            self.history.append(_empty_features())
        
        self.is_running = False
        self._thread = None
        self.latest_result = None
        
        # Stats
        self.total_packets = 0
        self.current_window = 0
        self.status = "INITIALIZED"

    def start(self):
        self.is_running = True
        self.history = [_empty_features() for _ in range(self.seq_len)]
        self.latest_result = None
        self.total_packets = 0
        self.current_window = 0
        self.status = "RUNNING"
        
        if self.csv_file:
            self._thread = threading.Thread(target=self._run_csv_replay, daemon=True)
        elif self.pcap_file:
            self._thread = threading.Thread(target=self._run_pcap_replay, daemon=True)
        else:
            self._thread = threading.Thread(target=self._run_live_capture, daemon=True)
        self._thread.start()

    def stop(self):
        self.is_running = False
        self.status = "STOPPED"
        if self._thread:
            self._thread.join(timeout=2.0)

    def _process_window(self, packets):
        self.current_window += 1
        self.total_packets += len(packets)
        
        # Extract traffic indicators for UI
        from collections import Counter
        src_counter = Counter()
        dst_counter = Counter()
        port_counter = Counter()
        edges_counter = Counter()
        
        for p in packets:
            if p.haslayer("IP"):
                src = p["IP"].src
                dst = p["IP"].dst
                src_counter[src] += 1
                dst_counter[dst] += 1
                edges_counter[(src, dst)] += 1
                if p.haslayer("TCP"):
                    port_counter[p["TCP"].dport] += 1
                elif p.haslayer("UDP"):
                    port_counter[p["UDP"].dport] += 1
                    
        top_src = [ip for ip, _ in src_counter.most_common(3)]
        top_dst = [ip for ip, _ in dst_counter.most_common(3)]
        top_ports = [str(p) for p, _ in port_counter.most_common(3)]
        top_edges = [{"source": s, "target": d, "weight": w} for ((s, d), w) in edges_counter.most_common(10)]
        
        features = aggregate_packets(packets, self.window_duration)
        self.history.append(features)
        
        if len(self.history) > self.seq_len + 1:
            self.history.pop(0)
            
        if len(self.history) == self.seq_len + 1:
            df = pd.DataFrame(self.history)
            df_proc = engineer_features(df)
            
            try:
                result = run_inference(self.model, df_proc, ENGINEERED_FEATURE_COLS)
                result['mode'] = 'LIVE' if not self.pcap_file else 'REPLAY'
                result['window'] = self.current_window
                result['packets'] = self.total_packets
                result['indicators'] = {
                    'src_ip': ", ".join(top_src) if top_src else "Unknown",
                    'dst_ip': ", ".join(top_dst) if top_dst else "Unknown",
                    'ports': ", ".join(top_ports) if top_ports else "Unknown",
                    'edges': top_edges
                }
                self.latest_result = result
                self.status = f"ACTIVE / FORECASTING (Window {self.current_window})"
            except Exception as e:
                import traceback
                print(f"Live inference error: {traceback.format_exc()}")
                self.status = f"ERROR: {e}"
        else:
            self.status = f"COLLECTING HISTORY (Need {self.seq_len + 1}, have {len(self.history)})"
            
    def _run_live_capture(self):
        try:
            while self.is_running:
                # Sniff for 5 seconds
                packets = scapy.sniff(iface=self.interface_name, timeout=self.window_duration)
                if not self.is_running: break
                self._process_window(packets)
        except Exception as e:
            self.status = f"CAPTURE ERROR: {e}"
            self.is_running = False

    def _run_pcap_replay(self):
        try:
            reader = scapy.PcapReader(self.pcap_file)
            current_window_pkts = []
            window_start_time = None
            
            for p in reader:
                if not self.is_running: break
                
                pt = float(p.time)
                if window_start_time is None:
                    window_start_time = pt
                    
                if pt - window_start_time >= self.window_duration:
                    self._process_window(current_window_pkts)
                    current_window_pkts = [p]
                    window_start_time = pt
                    time.sleep(1.0) # Artificial delay to simulate live UI updates
                else:
                    current_window_pkts.append(p)
                    
            if current_window_pkts and self.is_running:
                self._process_window(current_window_pkts)
                
            self.status = "REPLAY COMPLETE"
            self.is_running = False
        except Exception as e:
            self.status = f"REPLAY ERROR: {e}"
            self.is_running = False

    def _run_csv_replay(self):
        try:
            from src2.data.csv_loader import load_and_normalize_csv
            import sys
            import os
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
            from train_pipeline import aggregate_to_windows
            
            df, _ = load_and_normalize_csv(self.csv_file)
            df_proc = engineer_features(df)
            windowed = aggregate_to_windows(df_proc, window_sec=self.window_duration)
            
            # Ensure all required features are present
            for col in ENGINEERED_FEATURE_COLS:
                if col not in windowed.columns:
                    windowed[col] = 0.0
                    
            for idx, row in windowed.iterrows():
                if not self.is_running: break
                
                features = row.to_dict()
                self.history.append(features)
                if len(self.history) > self.seq_len + 1:
                    self.history.pop(0)
                    
                if len(self.history) == self.seq_len + 1:
                    df_hist = pd.DataFrame(self.history)
                    try:
                        result = run_inference(self.model, df_hist, ENGINEERED_FEATURE_COLS)
                        result['mode'] = 'CSV REPLAY'
                        result['window'] = self.current_window
                        result['packets'] = 0
                        self.latest_result = result
                        self.status = f"ACTIVE / FORECASTING (Window {self.current_window})"
                    except Exception as e:
                        import traceback
                        print(f"Live inference error: {traceback.format_exc()}")
                        self.status = f"ERROR: {e}"
                
                self.current_window += 1
                time.sleep(1.0)
                
            self.status = "CSV REPLAY COMPLETE"
            self.is_running = False
        except Exception as e:
            import traceback
            print(f"CSV Replay error: {traceback.format_exc()}")
            self.status = f"CSV REPLAY ERROR: {e}"
            self.is_running = False
