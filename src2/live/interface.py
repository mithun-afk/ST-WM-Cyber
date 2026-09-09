from typing import List, Dict
import scapy.all as scapy

def get_interfaces() -> List[Dict[str, str]]:
    """Returns a list of available network interfaces."""
    try:
        interfaces = scapy.get_working_ifaces()
        iface_list = []
        for iface in interfaces:
            iface_list.append({
                'name': iface.name,
                'description': iface.description,
                'mac': iface.mac,
                'ips': iface.ips
            })
        return iface_list
    except Exception as e:
        # Graceful handling if Npcap/Scapy not properly installed
        print(f"Error enumerating interfaces: {e}")
        return []

def get_interface_by_description(desc: str):
    interfaces = scapy.get_working_ifaces()
    for iface in interfaces:
        if iface.description == desc or iface.name == desc:
            return iface
    return None
