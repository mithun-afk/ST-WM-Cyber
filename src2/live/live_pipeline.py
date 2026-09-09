import threading
import time
import pandas as pd
from typing import List, Dict
import scapy.all as scapy

from src2.live.live_aggregator import aggregate_packets
from src2.data.feature_engineering import engineer_features, ENGINEERED_FEATURE_COLS
from src2.data.schema import CANONICAL_FEATURES
from src2.models.inference import run_inference

class LivePipeline:
    def __init__(self, model, interface_name=None, pcap_file=None):
        self.model = model
        self.interface_name = interface_name
        self.pcap_file = pcap_file
        
        self.history: List[Dict] = []
        self.seq_len = 5
        self.window_duration = 5.0
        
        # Pre-fill history with empty windows so it calculates immediately
        from src2.live.live_aggregator import _empty_features
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
        self.history = []
        self.latest_result = None
        self.total_packets = 0
        self.current_window = 0
        self.status = "RUNNING"
        
        if self.pcap_file:
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
