"""
check_iface.py — Network Interface Discovery Utility
=====================================================
Lists all available network interfaces with their details to help
the user choose the correct interface for packet capture.

Usage
-----
    python check_iface.py

Author : AI-Driven Network Anomaly Detection Project
Created: 2026
"""

import sys
import socket

try:
    from scapy.all import get_if_list, get_if_addr, conf
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

import psutil


def get_interfaces_psutil() -> None:
    """List interfaces using psutil (always available)."""
    print("=" * 70)
    print("  Available Network Interfaces (psutil)")
    print("=" * 70)

    interfaces = psutil.net_if_addrs()
    stats = psutil.net_if_stats()

    for idx, (iface_name, addrs) in enumerate(interfaces.items(), 1):
        is_up = stats.get(iface_name, None)
        status = "UP" if is_up and is_up.isup else "DOWN"
        speed = f"{is_up.speed} Mbps" if is_up and is_up.speed > 0 else "N/A"

        print(f"\n  [{idx}] {iface_name}")
        print(f"      Status : {status}")
        print(f"      Speed  : {speed}")

        for addr in addrs:
            if addr.family == socket.AF_INET:
                print(f"      IPv4   : {addr.address}")
                print(f"      Netmask: {addr.netmask}")
            elif addr.family == socket.AF_INET6:
                print(f"      IPv6   : {addr.address}")
            elif addr.family == psutil.AF_LINK:
                print(f"      MAC    : {addr.address}")

    print("\n" + "=" * 70)


def get_interfaces_scapy() -> None:
    """List interfaces using Scapy (if available)."""
    if not SCAPY_AVAILABLE:
        print("\n  [INFO] Scapy not installed — skipping Scapy interface list")
        return

    print("\n" + "=" * 70)
    print("  Available Network Interfaces (Scapy)")
    print("=" * 70)

    try:
        iface_list = get_if_list()
        for idx, iface in enumerate(iface_list, 1):
            try:
                ip_addr = get_if_addr(iface)
            except Exception:
                ip_addr = "N/A"
            print(f"  [{idx}] {iface:40s}  IP: {ip_addr}")

        print(f"\n  Default interface: {conf.iface}")
    except Exception as e:
        print(f"  [ERROR] Could not list Scapy interfaces: {e}")

    print("=" * 70)


def print_usage_hints() -> None:
    """Print helpful usage information."""
    print("\n" + "─" * 70)
    print("  USAGE HINTS")
    print("─" * 70)
    print("""
  To capture packets, use the interface name shown above:

    python capture.py --iface "Wi-Fi"
    python capture.py --iface "Ethernet"
    python capture.py --iface "Local Area Connection"

  For live detection:

    python detect.py --live --iface "Wi-Fi" --count 200

  NOTE: Packet capture requires Administrator privileges.
        Right-click your terminal → "Run as Administrator"
    """)


if __name__ == "__main__":
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║          NETWORK INTERFACE DISCOVERY                      ║")
    print("╚════════════════════════════════════════════════════════════╝")

    get_interfaces_psutil()
    get_interfaces_scapy()
    print_usage_hints()
