"""
test_utils.py — Unit Tests for Core Utilities
===============================================
Comprehensive test suite for the utility module covering:
    - Configuration defaults and directory creation
    - Data loading and validation
    - Feature engineering pipeline (encoding, scaling, selection)
    - Metrics calculation
    - Plot generation (file creation, not visual rendering)

Run with:
    python -m pytest test_utils.py -v

Author : AI-Driven Network Anomaly Detection Project
Created: 2026
"""

import os
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils import (
    Config,
    DataLoader,
    FeatureEngineer,
    MetricsCalculator,
    PlotGenerator,
    get_logger,
    save_dataset_statistics,
    save_model_comparison,
)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          TEST FIXTURES                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a small synthetic DataFrame mimicking UNSW-NB15 schema."""
    np.random.seed(42)
    n = 200

    data = {
        "id": range(1, n + 1),
        "dur": np.random.exponential(0.5, n),
        "proto": np.random.choice(["tcp", "udp", "icmp"], n, p=[0.6, 0.3, 0.1]),
        "service": np.random.choice(["http", "dns", "ftp", "-"], n),
        "state": np.random.choice(["FIN", "INT", "CON", "RST"], n),
        "spkts": np.random.randint(1, 50, n),
        "dpkts": np.random.randint(0, 30, n),
        "sbytes": np.random.randint(40, 5000, n),
        "dbytes": np.random.randint(0, 3000, n),
        "rate": np.random.uniform(0, 1000, n),
        "sttl": np.random.randint(30, 255, n),
        "dttl": np.random.randint(0, 255, n),
        "sload": np.random.uniform(0, 1e6, n),
        "dload": np.random.uniform(0, 1e6, n),
        "sloss": np.random.randint(0, 10, n),
        "dloss": np.random.randint(0, 10, n),
        "sinpkt": np.random.uniform(0, 100, n),
        "dinpkt": np.random.uniform(0, 100, n),
        "sjit": np.random.uniform(0, 50, n),
        "djit": np.random.uniform(0, 50, n),
        "swin": np.random.randint(0, 255, n),
        "stcpb": np.random.randint(0, 1e9, n),
        "dtcpb": np.random.randint(0, 1e9, n),
        "dwin": np.random.randint(0, 255, n),
        "tcprtt": np.random.uniform(0, 1, n),
        "synack": np.random.uniform(0, 0.5, n),
        "ackdat": np.random.uniform(0, 0.5, n),
        "smean": np.random.randint(40, 1500, n),
        "dmean": np.random.randint(0, 1000, n),
        "trans_depth": np.random.randint(0, 5, n),
        "response_body_len": np.random.randint(0, 10000, n),
        "ct_srv_src": np.random.randint(1, 10, n),
        "ct_state_ttl": np.random.randint(0, 5, n),
        "ct_dst_ltm": np.random.randint(1, 10, n),
        "ct_src_dport_ltm": np.random.randint(1, 10, n),
        "ct_dst_sport_ltm": np.random.randint(1, 5, n),
        "ct_dst_src_ltm": np.random.randint(1, 10, n),
        "is_ftp_login": np.random.choice([0, 1], n, p=[0.95, 0.05]),
        "ct_ftp_cmd": np.random.randint(0, 3, n),
        "ct_flw_http_mthd": np.random.randint(0, 3, n),
        "ct_src_ltm": np.random.randint(1, 10, n),
        "ct_srv_dst": np.random.randint(1, 10, n),
        "is_sm_ips_ports": np.random.choice([0, 1], n, p=[0.9, 0.1]),
        "attack_cat": np.random.choice(["Normal", "DoS", "Exploits", "Generic"], n,
                                        p=[0.5, 0.2, 0.2, 0.1]),
        "label": np.random.choice([0, 1], n, p=[0.6, 0.4]),
    }
    return pd.DataFrame(data)


@pytest.fixture
def binary_labels():
    """Create sample binary labels and predictions."""
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.15, 0.9, 0.85, 0.4, 0.7, 0.95])
    return y_true, y_pred, y_prob


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          CONFIG TESTS                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TestConfig:
    """Tests for the Config class."""

    def test_project_root_exists(self):
        """Config.PROJECT_ROOT should point to an existing directory."""
        assert Config.PROJECT_ROOT.exists()

    def test_data_directory(self):
        """Config.DATA_DIR should point to the Data folder."""
        assert "Data" in str(Config.DATA_DIR) or "data" in str(Config.DATA_DIR)

    def test_ensure_directories(self):
        """ensure_directories should create all output folders."""
        Config.ensure_directories()
        assert Config.MODELS_DIR.exists()
        assert Config.OUTPUTS_DIR.exists()
        assert Config.LOGS_DIR.exists()

    def test_autoencoder_hyperparameters(self):
        """Verify AE hyper-parameters are sensible."""
        assert len(Config.AE_ENCODING_DIMS) == 3
        assert Config.AE_DROPOUT > 0 and Config.AE_DROPOUT < 1
        assert Config.AE_EPOCHS > 0
        assert Config.AE_BATCH_SIZE > 0

    def test_xgboost_hyperparameters(self):
        """Verify XGBoost search space is defined."""
        assert len(Config.XGB_N_ESTIMATORS) >= 2
        assert len(Config.XGB_MAX_DEPTH) >= 2


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        DATA LOADER TESTS                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TestDataLoader:
    """Tests for the DataLoader class."""

    def test_load_training_data(self):
        """Training CSV should load with correct columns."""
        loader = DataLoader()
        df = loader.load_training_data()
        assert not df.empty
        assert "label" in df.columns
        assert "attack_cat" in df.columns

    def test_load_testing_data(self):
        """Testing CSV should load with correct columns."""
        loader = DataLoader()
        df = loader.load_testing_data()
        assert not df.empty
        assert df.shape[0] > 0

    def test_inspect(self, sample_dataframe):
        """inspect should return a dict with expected keys."""
        loader = DataLoader()
        summary = loader.inspect(sample_dataframe, "Test DF")
        assert "shape" in summary
        assert "missing" in summary
        assert "duplicates" in summary
        assert "label_dist" in summary


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     FEATURE ENGINEER TESTS                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TestFeatureEngineer:
    """Tests for the FeatureEngineer class."""

    def test_fit_transform_produces_output(self, sample_dataframe):
        """fit_transform should return features and labels."""
        fe = FeatureEngineer()
        X, y = fe.fit_transform(sample_dataframe, is_training=True)
        assert X is not None
        assert y is not None
        assert len(X) > 0
        assert len(y) > 0

    def test_id_column_dropped(self, sample_dataframe):
        """The 'id' column should be removed."""
        fe = FeatureEngineer()
        X, _ = fe.fit_transform(sample_dataframe, is_training=True)
        assert "id" not in X.columns

    def test_attack_cat_dropped(self, sample_dataframe):
        """The 'attack_cat' column should not be in features."""
        fe = FeatureEngineer()
        X, _ = fe.fit_transform(sample_dataframe, is_training=True)
        assert "attack_cat" not in X.columns

    def test_label_separated(self, sample_dataframe):
        """Labels should be separated from features."""
        fe = FeatureEngineer()
        X, y = fe.fit_transform(sample_dataframe, is_training=True)
        assert "label" not in X.columns
        assert set(y.unique()).issubset({0, 1})

    def test_no_missing_after_transform(self, sample_dataframe):
        """No missing values should remain after transformation."""
        # Inject some missing values
        df = sample_dataframe.copy()
        df.loc[0, "dur"] = np.nan
        df.loc[5, "proto"] = np.nan

        fe = FeatureEngineer()
        X, _ = fe.fit_transform(df, is_training=True)
        assert X.isnull().sum().sum() == 0

    def test_transform_reuses_fitted_params(self, sample_dataframe):
        """transform() should use parameters fitted during fit_transform()."""
        fe = FeatureEngineer()
        X_train, _ = fe.fit_transform(sample_dataframe, is_training=True)
        X_test, _ = fe.transform(sample_dataframe)

        # Both should have the same columns
        assert list(X_train.columns) == list(X_test.columns)

    def test_feature_columns_stored(self, sample_dataframe):
        """After fit_transform, feature_columns should be populated."""
        fe = FeatureEngineer()
        fe.fit_transform(sample_dataframe, is_training=True)
        assert len(fe.feature_columns) > 0


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      METRICS CALCULATOR TESTS                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TestMetricsCalculator:
    """Tests for the MetricsCalculator class."""

    def test_compute_all_returns_dict(self, binary_labels):
        """compute_all should return a dict with all metric keys."""
        y_true, y_pred, y_prob = binary_labels
        calc = MetricsCalculator()
        metrics = calc.compute_all(y_true, y_pred, y_prob, "Test")

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics

    def test_metrics_in_valid_range(self, binary_labels):
        """All metrics should be between 0 and 1."""
        y_true, y_pred, y_prob = binary_labels
        calc = MetricsCalculator()
        metrics = calc.compute_all(y_true, y_pred, y_prob)

        for key, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{key} = {value} out of range"

    def test_perfect_predictions(self):
        """Perfect predictions should yield metrics of 1.0."""
        y = np.array([0, 0, 1, 1])
        calc = MetricsCalculator()
        metrics = calc.compute_all(y, y, y.astype(float))
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_classification_report_text(self, binary_labels):
        """classification_report_text should return a non-empty string."""
        y_true, y_pred, _ = binary_labels
        report = MetricsCalculator.classification_report_text(y_true, y_pred)
        assert isinstance(report, str)
        assert len(report) > 0
        assert "Normal" in report
        assert "Anomaly" in report


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                       PLOT GENERATOR TESTS                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TestPlotGenerator:
    """Tests for plot generation (verifies file creation, not visual output)."""

    def test_plot_confusion_matrix(self, binary_labels):
        """Confusion matrix plot should be saved to disk."""
        y_true, y_pred, _ = binary_labels
        plotter = PlotGenerator()
        plotter.plot_confusion_matrix(y_true, y_pred, "TestModel", "test_cm.png")
        assert (Config.OUTPUTS_DIR / "test_cm.png").exists()

    def test_plot_label_distribution(self, sample_dataframe):
        """Label distribution plot should be saved."""
        plotter = PlotGenerator()
        plotter.plot_label_distribution(sample_dataframe, "test_label_dist.png")
        assert (Config.OUTPUTS_DIR / "test_label_dist.png").exists()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          LOGGER TESTS                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TestLogger:
    """Tests for the logging setup."""

    def test_logger_creation(self):
        """get_logger should return a configured Logger instance."""
        import logging
        log = get_logger("test_logger")
        assert isinstance(log, logging.Logger)
        assert log.name == "test_logger"

    def test_logger_with_file(self):
        """Logger with file output should create the log file."""
        log = get_logger("test_file_logger", "test_output.log")
        log.info("Test message")
        assert (Config.LOGS_DIR / "test_output.log").exists()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        HELPER FUNCTION TESTS                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TestHelpers:
    """Tests for standalone helper functions."""

    def test_save_model_comparison(self):
        """save_model_comparison should create a CSV file."""
        results = {
            "Model_A": {"accuracy": 0.95, "precision": 0.93, "recall": 0.90},
            "Model_B": {"accuracy": 0.88, "precision": 0.85, "recall": 0.92},
        }
        df = save_model_comparison(results, "test_comparison.csv")
        assert (Config.OUTPUTS_DIR / "test_comparison.csv").exists()
        assert len(df) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
