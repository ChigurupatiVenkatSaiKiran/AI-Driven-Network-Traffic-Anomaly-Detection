"""
capture.py — Live Network Traffic Capture
==========================================
Captures live network packets using Scapy and extracts features that
align with the UNSW-NB15 feature schema for real-time anomaly detection.

Features Extracted Per Packet
-----------------------------
    - Protocol type (TCP / UDP / ICMP / Other)
    - Packet size (bytes)
    - Source & destination information
    - TCP flags (SYN, ACK, FIN, RST, PSH, URG)
    - TTL values
    - Flow-level statistics (computed over a sliding window)
    - Traffic rate and inter-packet timing

Requirements
------------
    - Administrator / root privileges (for raw socket access)
    - Scapy installed (``pip install scapy``)

Usage
-----
    python capture.py                         # auto-detect interface
    python capture.py --iface "Ethernet"      # specific interface
    python capture.py --count 500             # capture 500 packets

Author : AI-Driven Network Anomaly Detection Project
Created: 2026
"""

import os
import sys
import csv
import time
import signal
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, conf
except ImportError:
    print("[ERROR] Scapy is not installed. Run:  pip install scapy")
    sys.exit(1)

from utils import Config, get_logger

# ── Module Logger ────────────────────────────────────────────────────────
logger = get_logger("capture", "capture.log")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         FLOW TRACKER                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class FlowTracker:
    """
    Maintains per-flow statistics using a sliding window approach.

    A "flow" is defined by the 5-tuple:
        (src_ip, dst_ip, src_port, dst_port, protocol)

    Tracks:
        - Packet counts (source → dest and dest → source)
        - Byte counts
        - Timing information (inter-packet intervals, jitter)
        - Connection state counts
    """

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self.flows: Dict[str, List[Dict]] = defaultdict(list)
        self.connection_counts: Dict[str, int] = defaultdict(int)

    def _flow_key(self, src: str, dst: str, sport: int, dport: int, proto: str) -> str:
        """Generate a unique flow identifier."""
        return f"{src}:{sport}-{dst}:{dport}-{proto}"

    def update(self, packet_info: Dict) -> Dict:
        """
        Add a new packet to its flow and compute flow-level features.

        Returns enriched packet_info with flow statistics.
        """
        key = self._flow_key(
            packet_info["src_ip"], packet_info["dst_ip"],
            packet_info["src_port"], packet_info["dst_port"],
            packet_info["proto"],
        )

        self.flows[key].append(packet_info)
        self.connection_counts[packet_info["dst_ip"]] += 1

        # Keep only the last N packets per flow (sliding window)
        if len(self.flows[key]) > self.window_size:
            self.flows[key] = self.flows[key][-self.window_size:]

        flow_packets = self.flows[key]
        flow_len = len(flow_packets)

        # ── Flow Statistics ──────────────────────────────────────────────
        timestamps = [p["timestamp"] for p in flow_packets]
        sizes = [p["size"] for p in flow_packets]

        # Duration
        duration = timestamps[-1] - timestamps[0] if flow_len > 1 else 0.0

        # Inter-packet timing
        intervals = np.diff(timestamps) if flow_len > 1 else [0.0]
        mean_interval = float(np.mean(intervals)) if len(intervals) > 0 else 0.0
        jitter = float(np.std(intervals)) if len(intervals) > 1 else 0.0

        # Rate
        rate = flow_len / duration if duration > 0 else 0.0

        # Byte statistics
        total_bytes = sum(sizes)
        mean_size = float(np.mean(sizes))

        # Connection count to this destination
        ct_dst = self.connection_counts.get(packet_info["dst_ip"], 0)

        packet_info.update({
            "dur": round(duration, 6),
            "rate": round(rate, 4),
            "spkts": flow_len,
            "sbytes": total_bytes,
            "smean": round(mean_size, 2),
            "sinpkt": round(mean_interval * 1000, 4),     # milliseconds
            "sjit": round(jitter * 1000, 4),
            "ct_dst_ltm": ct_dst,
            "ct_src_ltm": flow_len,
        })

        return packet_info


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                       PACKET FEATURE EXTRACTOR                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class PacketFeatureExtractor:
    """
    Extract features from raw Scapy packets that align with the
    UNSW-NB15 dataset schema.
    """

    # TCP flag mapping
    TCP_FLAGS = {
        "F": "FIN", "S": "SYN", "R": "RST",
        "P": "PSH", "A": "ACK", "U": "URG",
    }

    def __init__(self) -> None:
        self.flow_tracker = FlowTracker()
        self.packet_count = 0

    def extract(self, packet) -> Optional[Dict]:
        """
        Extract features from a single Scapy packet.

        Returns None if the packet has no IP layer.
        """
        if not packet.haslayer(IP):
            return None

        self.packet_count += 1
        ip_layer = packet[IP]
        timestamp = float(packet.time)

        # ── Base features ────────────────────────────────────────────────
        features = {
            "timestamp": timestamp,
            "src_ip": ip_layer.src,
            "dst_ip": ip_layer.dst,
            "size": len(packet),
            "sttl": ip_layer.ttl,
            "dttl": 0,
            "proto": "other",
            "src_port": 0,
            "dst_port": 0,
            "state": "INT",
            "service": "-",
            "tcp_flags": "",
        }

        # ── Protocol-specific features ───────────────────────────────────
        if packet.haslayer(TCP):
            tcp = packet[TCP]
            features["proto"] = "tcp"
            features["src_port"] = tcp.sport
            features["dst_port"] = tcp.dport
            features["tcp_flags"] = str(tcp.flags)

            # Determine connection state from flags
            flags = str(tcp.flags)
            if "S" in flags and "A" not in flags:
                features["state"] = "SYN"
            elif "S" in flags and "A" in flags:
                features["state"] = "SYNACK"
            elif "F" in flags:
                features["state"] = "FIN"
            elif "R" in flags:
                features["state"] = "RST"
            elif "A" in flags:
                features["state"] = "EST"
            else:
                features["state"] = "INT"

            # Common service detection by port
            features["service"] = self._detect_service(tcp.dport)

        elif packet.haslayer(UDP):
            udp = packet[UDP]
            features["proto"] = "udp"
            features["src_port"] = udp.sport
            features["dst_port"] = udp.dport
            features["service"] = self._detect_service(udp.dport)

        elif packet.haslayer(ICMP):
            features["proto"] = "icmp"

        # ── Additional computed fields ───────────────────────────────────
        features["sload"] = features["size"] * 8                  # bits
        features["dload"] = 0
        features["sloss"] = 0
        features["dloss"] = 0
        features["dpkts"] = 0
        features["dbytes"] = 0
        features["dmean"] = 0
        features["dinpkt"] = 0
        features["djit"] = 0
        features["swin"] = 0
        features["dwin"] = 0
        features["stcpb"] = 0
        features["dtcpb"] = 0
        features["tcprtt"] = 0
        features["synack"] = 0
        features["ackdat"] = 0
        features["trans_depth"] = 0
        features["response_body_len"] = 0
        features["ct_srv_src"] = 1
        features["ct_state_ttl"] = 1
        features["ct_src_dport_ltm"] = 1
        features["ct_dst_sport_ltm"] = 1
        features["ct_dst_src_ltm"] = 1
        features["is_ftp_login"] = 0
        features["ct_ftp_cmd"] = 0
        features["ct_flw_http_mthd"] = 0
        features["ct_srv_dst"] = 1
        features["is_sm_ips_ports"] = 1 if features["src_ip"] == features["dst_ip"] else 0

        # ── Flow-level enrichment ────────────────────────────────────────
        features = self.flow_tracker.update(features)

        return features

    @staticmethod
    def _detect_service(port: int) -> str:
        """Map well-known ports to service names."""
        services = {
            20: "ftp-data", 21: "ftp", 22: "ssh", 23: "telnet",
            25: "smtp", 53: "dns", 80: "http", 110: "pop3",
            143: "imap", 443: "https", 993: "imaps", 995: "pop3s",
            3306: "mysql", 3389: "rdp", 5432: "postgres",
            8080: "http-alt", 8443: "https-alt",
        }
        return services.get(port, "-")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         PACKET CAPTURE ENGINE                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class CaptureEngine:
    """
    Live packet capture engine with CSV export.

    Sniffs packets on the given interface, extracts features,
    and writes them to ``outputs/captured.csv``.
    """

    # CSV column order (matching UNSW-NB15 schema)
    CSV_COLUMNS = [
        "timestamp", "dur", "proto", "service", "state",
        "spkts", "dpkts", "sbytes", "dbytes", "rate",
        "sttl", "dttl", "sload", "dload", "sloss", "dloss",
        "sinpkt", "dinpkt", "sjit", "djit",
        "swin", "stcpb", "dtcpb", "dwin",
        "tcprtt", "synack", "ackdat",
        "smean", "dmean",
        "trans_depth", "response_body_len",
        "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
        "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
        "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
        "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
        "src_ip", "dst_ip", "src_port", "dst_port", "size", "tcp_flags",
    ]

    def __init__(self, interface: Optional[str] = None, output_file: str = "captured.csv") -> None:
        self.interface = interface
        self.output_path = Config.OUTPUTS_DIR / output_file
        self.extractor = PacketFeatureExtractor()
        self.captured_count = 0
        self._running = True
        self._csv_writer = None
        self._csv_file = None

    def _packet_callback(self, packet) -> None:
        """Process each captured packet."""
        features = self.extractor.extract(packet)
        if features is None:
            return

        self.captured_count += 1

        # Write to CSV
        row = {col: features.get(col, 0) for col in self.CSV_COLUMNS}
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        # Console output every 10 packets
        if self.captured_count % 10 == 0:
            logger.info(
                "Captured %5d │ %s │ %-5s │ %s:%d → %s:%d │ %d bytes",
                self.captured_count,
                datetime.now().strftime("%H:%M:%S"),
                features["proto"].upper(),
                features["src_ip"], features["src_port"],
                features["dst_ip"], features["dst_port"],
                features["size"],
            )

    def _signal_handler(self, signum, frame) -> None:
        """Handle Ctrl+C gracefully."""
        logger.info("\n⏹  Capture stopped by user (Ctrl+C)")
        self._running = False

    def start(self, packet_count: int = 0, timeout: int = 0) -> str:
        """
        Begin capturing packets.

        Parameters
        ----------
        packet_count : int
            Number of packets to capture (0 = unlimited).
        timeout : int
            Capture duration in seconds (0 = unlimited).

        Returns
        -------
        str  — Path to the saved CSV file.
        """
        signal.signal(signal.SIGINT, self._signal_handler)

        logger.info("╔════════════════════════════════════════════════════════════╗")
        logger.info("║           LIVE NETWORK TRAFFIC CAPTURE                   ║")
        logger.info("╚════════════════════════════════════════════════════════════╝")
        logger.info("  Interface : %s", self.interface or "auto-detect")
        logger.info("  Output    : %s", self.output_path)
        logger.info("  Max pkts  : %s", packet_count or "unlimited")
        logger.info("  Timeout   : %s seconds", timeout or "unlimited")
        logger.info("  Press Ctrl+C to stop\n")

        # Open CSV file
        self._csv_file = open(self.output_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.CSV_COLUMNS)
        self._csv_writer.writeheader()

        try:
            sniff(
                iface=self.interface,
                prn=self._packet_callback,
                count=packet_count if packet_count > 0 else 0,
                timeout=timeout if timeout > 0 else None,
                store=False,
                stop_filter=lambda _: not self._running,
            )
        except PermissionError:
            logger.error("Permission denied — run as Administrator / root")
            logger.error("  Windows: Right-click terminal → 'Run as Administrator'")
            logger.error("  Linux:   sudo python capture.py")
        except Exception as exc:
            logger.error("Capture error: %s", exc)
        finally:
            if self._csv_file:
                self._csv_file.close()

        logger.info("✓ Capture complete — %d packets saved to %s",
                     self.captured_count, self.output_path)
        return str(self.output_path)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                               CLI ENTRY                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Capture live network traffic and extract features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python capture.py                           # auto-detect interface
    python capture.py --iface "Ethernet"        # use specific interface
    python capture.py --count 1000              # capture 1000 packets
    python capture.py --timeout 60              # capture for 60 seconds
    python capture.py --iface "Wi-Fi" -c 500    # Wi-Fi, 500 packets
        """,
    )
    parser.add_argument("--iface", "-i", type=str, default=None,
                        help="Network interface to capture on")
    parser.add_argument("--count", "-c", type=int, default=0,
                        help="Number of packets to capture (0 = unlimited)")
    parser.add_argument("--timeout", "-t", type=int, default=0,
                        help="Capture timeout in seconds (0 = unlimited)")
    parser.add_argument("--output", "-o", type=str, default="captured.csv",
                        help="Output CSV filename (saved in outputs/)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    engine = CaptureEngine(interface=args.iface, output_file=args.output)
    engine.start(packet_count=args.count, timeout=args.timeout)
