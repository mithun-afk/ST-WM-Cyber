import pandas as pd

MITRE_INDICATORS = [
    {
        'id': 'T1046',
        'name': 'Network Service Scanning',
        'stage': 'Reconnaissance',
        'condition': lambda r: (r.get('syn_flag_cnt', 0) > 1 and r.get('unique_dst_ports', 0) > 5) or r.get('syn_rate', 0) > 0.6,
        'description': 'High rate of SYN packets or scanning multiple unique destination ports indicates reconnaissance.'
    },
    {
        'id': 'T1110',
        'name': 'Brute Force',
        'stage': 'InitialAccess',
        'condition': lambda r: r.get('flow_pkts_per_sec', 0) > 100 and r.get('rst_flag_cnt', 0) > 0,
        'description': 'High packet rates accompanied by resets often indicate rapid credential guessing.'
    },
    {
        'id': 'T1071',
        'name': 'Application Layer Protocol',
        'stage': 'Command And Control',
        'condition': lambda r: r.get('flow_duration', 0) > 4000000 and r.get('ack_flag_cnt', 0) > 0,
        'description': 'Flow spanning the entire 5-second window with active acknowledgments suggests C2 beaconing.'
    },
    {
        'id': 'T1048',
        'name': 'Exfiltration Over Alternative Protocol',
        'stage': 'Exfiltration',
        'condition': lambda r: r.get('down_up_ratio', 0) > 10 or r.get('fwd_bytes', 0) > 50000,
        'description': 'Massive outbound byte ratio or large forward byte volume indicates data exfiltration.'
    },
    {
        'id': 'T1021',
        'name': 'Remote Services',
        'stage': 'Lateral Movement',
        'condition': lambda r: r.get('pkt_len_mean', 0) > 150 and r.get('ack_flag_cnt', 0) > 5,
        'description': 'Persistent connections with large average packet sizes common in RDP or SSH lateral movement.'
    },
    {
        'id': 'T1486',
        'name': 'Data Encrypted for Impact',
        'stage': 'Impact',
        'condition': lambda r: r.get('flow_bytes_per_sec', 0) > 500000 and r.get('pkt_len_std', 0) > 50,
        'description': 'High bandwidth utilization with varied packet sizes often accompanies ransomware network-share encryption.'
    },
    {
        'id': 'T1498',
        'name': 'Network Denial of Service',
        'stage': 'Impact',
        'condition': lambda r: r.get('syn_flag_cnt', 0) > 10 or r.get('flow_pkts_per_sec', 0) > 500,
        'description': 'Massive flood of SYN packets or extremely high packet rate indicates a Denial of Service attack.'
    }
]

def match_indicators(df_last_timestep: pd.DataFrame) -> list:
    """
    Evaluates rule-based indicators against the last timestep of each temporal window.
    """
    matched = {}
    
    for idx, row in df_last_timestep.iterrows():
        row_dict = row.to_dict()
        for ind in MITRE_INDICATORS:
            try:
                if ind['condition'](row_dict):
                    ind_id = ind['id']
                    if ind_id not in matched:
                        matched[ind_id] = {
                            'id': ind_id,
                            'name': ind['name'],
                            'stage': ind['stage'],
                            'description': ind['description'],
                            'window_indices': [],
                            'count': 0
                        }
                    matched[ind_id]['window_indices'].append(idx)
                    matched[ind_id]['count'] += 1
            except Exception:
                pass
                
    return list(matched.values())
