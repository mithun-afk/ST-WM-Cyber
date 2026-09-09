import pandas as pd

MITRE_INDICATORS = [
    {
        'id': 'T1046',
        'name': 'Network Service Scanning',
        'stage': 'Reconnaissance',
        'condition': lambda r: r.get('syn_rate', 0) > 0.8 and r.get('fwd_pkts', 0) < 5,
        'description': 'High rate of SYN packets with very few forward packets indicates port scanning.'
    },
    {
        'id': 'T1110',
        'name': 'Brute Force',
        'stage': 'InitialAccess',
        'condition': lambda r: r.get('fwd_pkts', 0) > 50 and r.get('rst_rate', 0) > 0.3,
        'description': 'Large number of forward packets paired with a high reset rate indicates brute force authentication.'
    },
    {
        'id': 'T1071',
        'name': 'Application Layer Protocol',
        'stage': 'CommandAndControl',
        'condition': lambda r: r.get('flow_duration', 0) > 60000000 and r.get('ack_flag_cnt', 0) > 100,
        'description': 'Extremely long flow duration with continuous ACK packets suggests C2 beaconing over standard ports.'
    },
    {
        'id': 'T1048',
        'name': 'Exfiltration Over Alternative Protocol',
        'stage': 'Impact',
        'condition': lambda r: r.get('byte_ratio', 0) > 10 and r.get('bwd_pkts', 0) < 5,
        'description': 'Massive outbound byte ratio with almost no return packets indicates data exfiltration.'
    },
    {
        'id': 'T1021',
        'name': 'Remote Services',
        'stage': 'LateralMovement',
        'condition': lambda r: r.get('pkt_len_mean', 0) > 200 and r.get('ack_flag_cnt', 0) > 20 and r.get('syn_flag_cnt', 0) > 0,
        'description': 'Persistent connections with large average packet sizes common in RDP or SSH lateral movement.'
    },
    {
        'id': 'T1486',
        'name': 'Data Encrypted for Impact',
        'stage': 'Impact',
        'condition': lambda r: r.get('flow_bytes_per_sec', 0) > 1000000 and r.get('pkt_len_std', 0) > 100,
        'description': 'Extremely high bandwidth utilization with varied packet sizes often accompanies network-share ransomware encryption.'
    },
    {
        'id': 'T1595',
        'name': 'Active Scanning',
        'stage': 'Reconnaissance',
        'condition': lambda r: r.get('syn_flag_cnt', 0) > 5 and r.get('flow_duration', 0) < 100000,
        'description': 'Multiple SYN flags within a very short flow duration indicates rapid automated vulnerability scanning.'
    },
    {
        'id': 'T1041',
        'name': 'Exfiltration Over C2 Channel',
        'stage': 'Impact',
        'condition': lambda r: r.get('idle_mean', 0) > 1000000 and r.get('byte_ratio', 0) > 5,
        'description': 'High idle times punctuated by large bursts of outbound data indicates exfiltration over an established C2.'
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
