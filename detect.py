"""
detect.py — Real-Time Anomaly Detection Engine
================================================
Loads the trained Autoencoder model and performs real-time anomaly
detection on live or captured network traffic.

Workflow
--------
    1. Load trained model, scaler, and threshold
    2. Read packets (from live capture or CSV file)
    3. Transform features using the saved pipeline
    4. Compute reconstruction error per packet
    5. Flag anomalies when error exceeds the dynamic threshold
    6. Generate severity-based alerts
    7. Log all detections to file

Severity Levels
---------------
    - LOW      : error is 1–2× above threshold
    - MEDIUM   : error is 2–3× above threshold
    - HIGH     : error is 3–5× above threshold
    - CRITICAL : error is 5×+ above threshold

Usage
-----
    python detect.py                          # detect from captured.csv
    python detect.py --live                   # live capture + detection
    python detect.py --input custom_data.csv  # custom input file

Author : AI-Driven Network Anomaly Detection Project
Created: 2026
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from utils import Config, FeatureEngineer, get_logger

# ── Module Logger ────────────────────────────────────────────────────────
logger = get_logger("detect", "live_detection.log")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        SEVERITY CLASSIFIER                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class SeverityClassifier:
    """Assign alert severity based on how far the reconstruction error exceeds the threshold."""

    LEVELS = {
        "NORMAL":   {"min": 0.0, "max": 1.0, "color": "\033[92m"},   # green
        "LOW":      {"min": 1.0, "max": 2.0, "color": "\033[93m"},   # yellow
        "MEDIUM":   {"min": 2.0, "max": 3.0, "color": "\033[33m"},   # orange
        "HIGH":     {"min": 3.0, "max": 5.0, "color": "\033[91m"},   # red
        "CRITICAL": {"min": 5.0, "max": float("inf"), "color": "\033[31m"},  # dark red
    }
    RESET = "\033[0m"

    @classmethod
    def classify(cls, error: float, threshold: float) -> str:
        """Return the severity level string."""
        if error <= threshold:
            return "NORMAL"

        ratio = error / threshold
        for level, bounds in cls.LEVELS.items():
            if bounds["min"] <= ratio < bounds["max"]:
                return level

        return "CRITICAL"

    @classmethod
    def colorise(cls, level: str) -> str:
        """Return a terminal-coloured severity string."""
        info = cls.LEVELS.get(level, cls.LEVELS["CRITICAL"])
        return f"{info['color']}{level:8s}{cls.RESET}"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        DETECTION ENGINE                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class DetectionEngine:
    """
    Core anomaly detection engine using the trained Autoencoder.

    Loads the model and preprocessing artefacts, then scores incoming
    network traffic data for anomalies.
    """

    def __init__(self) -> None:
        self.model = None
        self.threshold = 0.0
        self.feature_engineer = FeatureEngineer()
        self.minmax_scaler = None
        self.is_ready = False

    def load_model(self) -> bool:
        """
        Load the trained Autoencoder, scalers, and anomaly threshold.

        Returns True if all artefacts are loaded successfully.
        """
        logger.info("Loading detection model and artefacts …")

        try:
            # Autoencoder model
            from tensorflow import keras
            self.model = keras.models.load_model(str(Config.AUTOENCODER_MODEL))
            logger.info("✓ Autoencoder loaded from %s", Config.AUTOENCODER_MODEL)

            # Feature engineering transformers
            self.feature_engineer.load_transformers()
            logger.info("✓ Feature transformers loaded")

            # MinMax scaler
            minmax_path = Config.MODELS_DIR / "minmax_scaler.pkl"
            self.minmax_scaler = joblib.load(minmax_path)
            logger.info("✓ MinMax scaler loaded")

            # Dynamic threshold
            with open(Config.THRESHOLD_PATH, "r") as f:
                data = json.load(f)
                self.threshold = data["threshold"]
            logger.info("✓ Threshold loaded: %.6f", self.threshold)

            self.is_ready = True
            return True

        except FileNotFoundError as e:
            logger.error("Model artefact not found: %s", e)
            logger.error("Run 'python train.py' first to train the model")
            return False
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            return False

    def detect_single(self, features: np.ndarray) -> Tuple[bool, float, str]:
        """
        Detect anomaly for a single feature vector.

        Returns
        -------
        (is_anomaly, reconstruction_error, severity)
        """
        if not self.is_ready:
            raise RuntimeError("Detection engine not initialised. Call load_model() first.")

        reconstruction = self.model.predict(features.reshape(1, -1), verbose=0)
        error = float(np.mean(np.square(features - reconstruction)))

        is_anomaly = error > self.threshold
        severity = SeverityClassifier.classify(error, self.threshold)

        return is_anomaly, error, severity

    def detect_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Detect anomalies for a batch of feature vectors.

        Returns
        -------
        (is_anomaly_array, error_array, severity_list)
        """
        if not self.is_ready:
            raise RuntimeError("Detection engine not initialised. Call load_model() first.")

        reconstructions = self.model.predict(X, verbose=0)
        errors = np.mean(np.square(X - reconstructions), axis=1)

        is_anomaly = errors > self.threshold
        severities = [SeverityClassifier.classify(e, self.threshold) for e in errors]

        return is_anomaly, errors, severities


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      DETECTION FROM CSV FILE                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def detect_from_csv(input_path: str) -> None:
    """
    Load captured traffic from CSV, run anomaly detection,
    and produce a detailed results file.
    """
    logger.info("╔════════════════════════════════════════════════════════════╗")
    logger.info("║        ANOMALY DETECTION — CSV MODE                      ║")
    logger.info("╚════════════════════════════════════════════════════════════╝")

    # Load engine
    engine = DetectionEngine()
    if not engine.load_model():
        sys.exit(1)

    # Load data
    logger.info("Reading input: %s", input_path)
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logger.error("File not found: %s", input_path)
        sys.exit(1)

    logger.info("Loaded %d records", len(df))

    # Keep metadata columns if they exist
    meta_cols = ["timestamp", "src_ip", "dst_ip", "src_port", "dst_port"]
    meta_df = df[[c for c in meta_cols if c in df.columns]].copy()

    # Transform features
    X_processed, _ = engine.feature_engineer.transform(df)
    X_np = X_processed.values.astype(np.float32)

    # Apply MinMax scaling
    X_mm = engine.minmax_scaler.transform(X_np)

    # Detect
    logger.info("Running anomaly detection on %d samples …", len(X_mm))
    is_anomaly, errors, severities = engine.detect_batch(X_mm)

    # Build results DataFrame
    results = meta_df.copy() if len(meta_df) == len(df) else pd.DataFrame()
    results["reconstruction_error"] = errors
    results["is_anomaly"] = is_anomaly.astype(int)
    results["severity"] = severities
    results["threshold"] = engine.threshold

    # Save results
    output_path = Config.OUTPUTS_DIR / "live_detection_results.csv"
    results.to_csv(output_path, index=False)
    logger.info("Results saved to %s", output_path)

    # Summary
    total = len(results)
    anomalies = int(is_anomaly.sum())
    normal = total - anomalies

    logger.info("\n" + "=" * 50)
    logger.info("  DETECTION SUMMARY")
    logger.info("=" * 50)
    logger.info("  Total packets    : %d", total)
    logger.info("  Normal           : %d  (%.1f%%)", normal, 100 * normal / total)
    logger.info("  Anomalies        : %d  (%.1f%%)", anomalies, 100 * anomalies / total)
    logger.info("")

    # Severity breakdown
    for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        count = severities.count(level)
        if count > 0:
            logger.info("  %-10s : %d", level, count)

    logger.info("=" * 50)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      LIVE CAPTURE + DETECTION                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def detect_live(interface: Optional[str] = None, count: int = 100) -> None:
    """
    Capture packets live and run anomaly detection in real-time.
    """
    logger.info("╔════════════════════════════════════════════════════════════╗")
    logger.info("║        ANOMALY DETECTION — LIVE MODE                     ║")
    logger.info("╚════════════════════════════════════════════════════════════╝")

    # First, capture packets
    from capture import CaptureEngine

    logger.info("Step 1: Capturing %d packets …", count)
    capture = CaptureEngine(interface=interface, output_file="captured.csv")
    csv_path = capture.start(packet_count=count)

    # Then run detection on the captured data
    logger.info("Step 2: Running anomaly detection …")
    detect_from_csv(csv_path)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                               CLI ENTRY                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time network traffic anomaly detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python detect.py                                  # detect from captured.csv
    python detect.py --input outputs/captured.csv     # detect from specific file
    python detect.py --live                           # live capture + detection
    python detect.py --live --iface "Wi-Fi" -c 200   # live, specific interface
        """,
    )
    parser.add_argument("--input", "-f", type=str, default=None,
                        help="Path to input CSV file for detection")
    parser.add_argument("--live", action="store_true",
                        help="Enable live capture + detection mode")
    parser.add_argument("--iface", "-i", type=str, default=None,
                        help="Network interface (for live mode)")
    parser.add_argument("--count", "-c", type=int, default=100,
                        help="Packets to capture in live mode (default: 100)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.live:
        detect_live(interface=args.iface, count=args.count)
    else:
        input_file = args.input or str(Config.OUTPUTS_DIR / "captured.csv")
        detect_from_csv(input_file)
