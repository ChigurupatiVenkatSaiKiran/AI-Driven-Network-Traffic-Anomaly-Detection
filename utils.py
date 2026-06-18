"""
utils.py — Core Utilities for Network Traffic Anomaly Detection
================================================================
Shared utility classes and functions used across the entire pipeline:
    - Configuration management
    - Logging setup
    - Data loading and validation
    - Feature engineering pipeline
    - Evaluation metrics computation
    - Visualization / plot generation

Author : AI-Driven Network Anomaly Detection Project
Created: 2026
"""

import os
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend for servers
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         CONFIGURATION MANAGER                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class Config:
    """Centralised project configuration — paths, hyper-parameters, constants."""

    # ── Project Root (auto-detected) ─────────────────────────────────────
    PROJECT_ROOT = Path(__file__).resolve().parent

    # ── Directory Paths ──────────────────────────────────────────────────
    DATA_DIR      = PROJECT_ROOT / "Data"        # original folder name
    MODELS_DIR    = PROJECT_ROOT / "models"
    OUTPUTS_DIR   = PROJECT_ROOT / "outputs"
    LOGS_DIR      = PROJECT_ROOT / "logs"
    SCREENSHOTS   = PROJECT_ROOT / "screenshots"
    DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

    # ── Dataset Files ────────────────────────────────────────────────────
    TRAIN_CSV = DATA_DIR / "training.csv"
    TEST_CSV  = DATA_DIR / "testing.csv"

    # ── Model Artefact Paths ─────────────────────────────────────────────
    AUTOENCODER_MODEL = MODELS_DIR / "autoencoder.keras"
    XGBOOST_MODEL     = MODELS_DIR / "xgboost_model.pkl"
    IFOREST_MODEL     = MODELS_DIR / "isolation_forest.pkl"
    SCALER_PATH       = MODELS_DIR / "scaler.pkl"
    ENCODERS_PATH     = MODELS_DIR / "label_encoders.pkl"
    THRESHOLD_PATH    = MODELS_DIR / "threshold.json"
    FEATURE_COLS_PATH = MODELS_DIR / "feature_columns.json"

    # ── Autoencoder Hyper-parameters ─────────────────────────────────────
    AE_ENCODING_DIMS = [128, 64, 32]         # encoder layers  (→ bottleneck 32)
    AE_DECODING_DIMS = [64, 128]             # decoder layers (mirror encoder)
    AE_ACTIVATION    = "relu"
    AE_OUTPUT_ACT    = "sigmoid"
    AE_DROPOUT       = 0.1                   # reduced: less regularisation for better recall
    AE_LEARNING_RATE = 5e-4                  # lower LR: more stable convergence
    AE_EPOCHS        = 150                   # more epochs with early stopping guard
    AE_BATCH_SIZE    = 512                   # larger batch: smoother gradients
    AE_PATIENCE      = 15                    # give the model more time to converge
    AE_THRESHOLD_PERCENTILE = 95             # percentile of normal-only errors (replaces mean+N×std)

    # ── XGBoost Hyper-parameters (wider grid for better optimum) ─────────
    XGB_N_ESTIMATORS = [200, 300, 500]
    XGB_MAX_DEPTH    = [6, 8, 10]
    XGB_LEARNING_RATE_LIST = [0.05, 0.1, 0.2]
    XGB_SUBSAMPLE    = 0.8
    XGB_COLSAMPLE    = 0.8
    XGB_MIN_CHILD_W  = 1
    XGB_SCALE_POS    = None                  # set dynamically from class ratio

    # ── Isolation Forest Hyper-parameters ────────────────────────────────
    IF_N_ESTIMATORS    = 300
    IF_CONTAMINATION   = 0.45                # sklearn max is 0.5; set high to match anomaly-heavy dataset
    IF_MAX_SAMPLES     = 0.8

    # ── Feature Engineering ──────────────────────────────────────────────
    DROP_COLUMNS = ["id"]                     # columns to discard
    CATEGORICAL_COLS = ["proto", "service", "state", "attack_cat"]
    TARGET_COL   = "label"
    ATTACK_COL   = "attack_cat"
    VARIANCE_THRESHOLD = 0.001                # lowered: keep more features for models
    CORRELATION_THRESHOLD = 0.98              # raised: only drop near-perfect duplicates

    # ── Visualisation Defaults ───────────────────────────────────────────
    FIGURE_DPI  = 150
    FIGURE_STYLE = "seaborn-v0_8-darkgrid"
    COLOR_NORMAL  = "#2ecc71"
    COLOR_ANOMALY = "#e74c3c"
    COLOR_PALETTE = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12",
                     "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]

    # ── Logging ──────────────────────────────────────────────────────────
    LOG_FORMAT = "%(asctime)s │ %(levelname)-8s │ %(name)-20s │ %(message)s"
    LOG_DATE   = "%Y-%m-%d %H:%M:%S"
    LOG_LEVEL  = logging.INFO

    @classmethod
    def ensure_directories(cls) -> None:
        """Create all required output directories if they don't exist."""
        for directory in [cls.MODELS_DIR, cls.OUTPUTS_DIR, cls.LOGS_DIR,
                          cls.SCREENSHOTS, cls.DASHBOARD_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


# Ensure directories exist on import
Config.ensure_directories()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                            LOGGING SETUP                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Create a configured logger with console and optional file output.

    Parameters
    ----------
    name : str
        Logger name (usually ``__name__`` of the calling module).
    log_file : str, optional
        Filename inside ``logs/`` for persistent log output.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger                        # avoid duplicate handlers

    logger.setLevel(Config.LOG_LEVEL)
    formatter = logging.Formatter(Config.LOG_FORMAT, datefmt=Config.LOG_DATE)

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (optional)
    if log_file:
        file_path = Config.LOGS_DIR / log_file
        fh = logging.FileHandler(file_path, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                              DATA LOADER                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class DataLoader:
    """
    Handles loading and initial inspection of the UNSW-NB15 dataset.

    Attributes
    ----------
    logger : logging.Logger
        Module-level logger instance.
    """

    def __init__(self) -> None:
        self.logger = get_logger("DataLoader", "data_loader.log")

    def load_training_data(self) -> pd.DataFrame:
        """Load the training split from CSV."""
        self.logger.info("Loading training data from %s", Config.TRAIN_CSV)
        df = pd.read_csv(Config.TRAIN_CSV)
        self.logger.info("Training data shape: %s", df.shape)
        return df

    def load_testing_data(self) -> pd.DataFrame:
        """Load the testing split from CSV."""
        self.logger.info("Loading testing data from %s", Config.TEST_CSV)
        df = pd.read_csv(Config.TEST_CSV)
        self.logger.info("Testing data shape: %s", df.shape)
        return df

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both splits and return ``(train_df, test_df)``."""
        return self.load_training_data(), self.load_testing_data()

    def inspect(self, df: pd.DataFrame, name: str = "DataFrame") -> Dict[str, Any]:
        """
        Print and return a diagnostic summary of the DataFrame.

        Returns a dict with keys: shape, dtypes, missing, duplicates, label_dist.
        """
        summary: Dict[str, Any] = {
            "shape": df.shape,
            "dtypes": df.dtypes.value_counts().to_dict(),
            "missing": df.isnull().sum().sum(),
            "missing_per_col": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
            "duplicates": df.duplicated().sum(),
        }

        if Config.TARGET_COL in df.columns:
            summary["label_dist"] = df[Config.TARGET_COL].value_counts().to_dict()

        if Config.ATTACK_COL in df.columns:
            summary["attack_dist"] = df[Config.ATTACK_COL].value_counts().to_dict()

        self.logger.info("─── %s Inspection ───", name)
        self.logger.info("  Shape       : %s", summary["shape"])
        self.logger.info("  Missing     : %d total", summary["missing"])
        self.logger.info("  Duplicates  : %d rows", summary["duplicates"])
        if "label_dist" in summary:
            self.logger.info("  Label dist  : %s", summary["label_dist"])

        return summary


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          FEATURE ENGINEER                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class FeatureEngineer:
    """
    Production-ready feature engineering pipeline for UNSW-NB15 data.

    Stages
    ------
    1. Drop unnecessary columns (e.g. ``id``)
    2. Handle missing values (median / mode imputation)
    3. Remove duplicate records
    4. Encode categorical features (LabelEncoder per column)
    5. Scale numerical features (StandardScaler)
    6. Feature selection (variance + correlation filtering)

    The fitted transformers are saved so that new data (live capture)
    can be transformed identically at inference time.
    """

    def __init__(self) -> None:
        self.logger = get_logger("FeatureEngineer", "feature_eng.log")
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self._is_fitted = False

    # ── public API ───────────────────────────────────────────────────────

    def fit_transform(
        self,
        df: pd.DataFrame,
        is_training: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Run the full pipeline on *training* data and fit all transformers.

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataframe (training split).
        is_training : bool
            If True, fit transformers; if False, reuse already-fitted ones.

        Returns
        -------
        (features_df, labels_series)
            Processed feature matrix and the target column (if present).
        """
        df = df.copy()
        self.logger.info("Starting feature engineering pipeline (training=%s)", is_training)

        # Step 1 — Drop unwanted columns
        df = self._drop_columns(df)

        # Step 2 — Handle missing values
        df = self._handle_missing(df)

        # Step 3 — Remove duplicates
        if is_training:
            df = self._remove_duplicates(df)

        # Separate labels before encoding
        labels = None
        if Config.TARGET_COL in df.columns:
            labels = df[Config.TARGET_COL].copy()
            df = df.drop(columns=[Config.TARGET_COL])

        # Drop the attack_cat column (used only for EDA, not model input)
        if Config.ATTACK_COL in df.columns:
            df = df.drop(columns=[Config.ATTACK_COL])

        # Step 4 — Encode categoricals
        df = self._encode_categoricals(df, fit=is_training)

        # Step 5 — Scale numerics
        df = self._scale_features(df, fit=is_training)

        # Step 6 — Feature selection (only during training)
        if is_training:
            df = self._select_features(df)
            self.feature_columns = df.columns.tolist()
            self._is_fitted = True
        else:
            # Keep only the columns selected during training
            available = [c for c in self.feature_columns if c in df.columns]
            df = df[available]

        self.logger.info("Pipeline complete — output shape: %s", df.shape)
        return df, labels

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Transform new data using previously fitted transformers."""
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer has not been fitted yet. Call fit_transform first.")
        return self.fit_transform(df, is_training=False)

    def save_transformers(self) -> None:
        """Persist fitted scaler, encoders, and selected feature names."""
        import joblib

        joblib.dump(self.scaler, Config.SCALER_PATH)
        joblib.dump(self.label_encoders, Config.ENCODERS_PATH)

        with open(Config.FEATURE_COLS_PATH, "w", encoding="utf-8") as fh:
            json.dump(self.feature_columns, fh, indent=2)

        self.logger.info("Transformers saved to %s", Config.MODELS_DIR)

    def load_transformers(self) -> None:
        """Restore previously saved transformers for inference."""
        import joblib

        self.scaler = joblib.load(Config.SCALER_PATH)
        self.label_encoders = joblib.load(Config.ENCODERS_PATH)

        with open(Config.FEATURE_COLS_PATH, "r", encoding="utf-8") as fh:
            self.feature_columns = json.load(fh)

        self._is_fitted = True
        self.logger.info("Transformers loaded from %s", Config.MODELS_DIR)

    # ── private helpers ──────────────────────────────────────────────────

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns that provide no predictive value."""
        to_drop = [c for c in Config.DROP_COLUMNS if c in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)
            self.logger.info("Dropped columns: %s", to_drop)
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values — median for numerics, mode for categoricals."""
        missing_total = df.isnull().sum().sum()
        if missing_total == 0:
            self.logger.info("No missing values detected")
            return df

        self.logger.info("Imputing %d missing values", missing_total)

        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ("object", "category"):
                    fill_value = df[col].mode()[0] if not df[col].mode().empty else "unknown"
                    df[col] = df[col].fillna(fill_value)
                else:
                    df[col] = df[col].fillna(df[col].median())

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicate rows and log the count."""
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed > 0:
            self.logger.info("Removed %d duplicate rows (%d → %d)", removed, before, len(df))
        else:
            self.logger.info("No duplicate rows found")
        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Label-encode categorical columns."""
        cat_cols = [c for c in Config.CATEGORICAL_COLS
                    if c in df.columns and c not in (Config.TARGET_COL, Config.ATTACK_COL)]

        for col in cat_cols:
            df[col] = df[col].astype(str)
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                self.logger.info("Fitted LabelEncoder for '%s' (%d classes)", col, len(le.classes_))
            else:
                le = self.label_encoders.get(col)
                if le is not None:
                    # Handle unseen labels gracefully
                    known = set(le.classes_)
                    df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
                    df[col] = le.transform(df[col])

        return df

    def _scale_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Standardise all features to zero mean and unit variance."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            self.logger.info("Fitted StandardScaler on %d numeric features", len(numeric_cols))
        else:
            # Handle column mismatch gracefully
            expected = self.scaler.feature_names_in_
            common = [c for c in expected if c in df.columns]
            df[common] = self.scaler.transform(df[common])

        return df

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove low-variance features and one of each pair of
        highly-correlated features.
        """
        # Variance filter
        variances = df.var()
        low_var = variances[variances < Config.VARIANCE_THRESHOLD].index.tolist()
        if low_var:
            df = df.drop(columns=low_var)
            self.logger.info("Removed %d low-variance features: %s", len(low_var), low_var)

        # Correlation filter
        corr_matrix = df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper_tri.columns
                    if any(upper_tri[col] > Config.CORRELATION_THRESHOLD)]
        if to_drop:
            df = df.drop(columns=to_drop)
            self.logger.info("Removed %d highly-correlated features: %s", len(to_drop), to_drop)

        self.logger.info("Feature selection complete — %d features retained", df.shape[1])
        return df


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         METRICS CALCULATOR                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class MetricsCalculator:
    """Compute and format evaluation metrics for binary classification."""

    def __init__(self) -> None:
        self.logger = get_logger("MetricsCalculator")

    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        model_name: str = "Model",
    ) -> Dict[str, float]:
        """
        Calculate accuracy, precision, recall, F1, and ROC-AUC.

        Parameters
        ----------
        y_true  : Ground-truth binary labels.
        y_pred  : Predicted binary labels.
        y_prob  : Predicted probabilities (for ROC-AUC).  Optional.
        model_name : Identifier for logging.

        Returns
        -------
        dict with metric names as keys.
        """
        metrics = {
            "accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
        }

        if y_prob is not None:
            try:
                metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
            except ValueError:
                metrics["roc_auc"] = 0.0
        else:
            metrics["roc_auc"] = 0.0

        self.logger.info("─── %s Metrics ───", model_name)
        for key, value in metrics.items():
            self.logger.info("  %-12s : %.4f", key, value)

        return metrics

    @staticmethod
    def classification_report_text(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None,
    ) -> str:
        """Return sklearn classification report as formatted text."""
        names = target_names or ["Normal", "Anomaly"]
        return classification_report(y_true, y_pred, target_names=names, zero_division=0)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          PLOT GENERATOR                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class PlotGenerator:
    """
    Generate publication-quality visualisations and save to ``outputs/``.

    Every method saves its figure to disk and also returns the figure
    object for optional inline display.
    """

    def __init__(self) -> None:
        self.logger = get_logger("PlotGenerator")
        try:
            plt.style.use(Config.FIGURE_STYLE)
        except OSError:
            plt.style.use("seaborn-v0_8")

    # ── Training Curves ──────────────────────────────────────────────────

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        filename: str = "training_loss.png",
    ) -> plt.Figure:
        """Plot training and validation loss over epochs."""
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = range(1, len(history["loss"]) + 1)
        ax.plot(epochs, history["loss"], label="Training Loss",
                color="#3498db", linewidth=2)
        ax.plot(epochs, history["val_loss"], label="Validation Loss",
                color="#e74c3c", linewidth=2, linestyle="--")

        ax.set_title("Autoencoder Training & Validation Loss", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Mean Squared Error", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        save_path = Config.OUTPUTS_DIR / filename
        fig.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
        self.logger.info("Saved: %s", save_path)
        plt.close(fig)
        return fig

    # ── Reconstruction Error Distribution ────────────────────────────────

    def plot_reconstruction_error(
        self,
        errors_normal: np.ndarray,
        errors_anomaly: np.ndarray,
        threshold: float,
        filename: str = "reconstruction_error.png",
    ) -> plt.Figure:
        """Histogram of reconstruction errors split by true label."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(errors_normal, bins=80, alpha=0.65, label="Normal",
                color=Config.COLOR_NORMAL, density=True)
        ax.hist(errors_anomaly, bins=80, alpha=0.65, label="Anomaly",
                color=Config.COLOR_ANOMALY, density=True)
        ax.axvline(threshold, color="#f39c12", linewidth=2, linestyle="--",
                   label=f"Threshold = {threshold:.4f}")

        ax.set_title("Reconstruction Error Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Reconstruction Error (MSE)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        save_path = Config.OUTPUTS_DIR / filename
        fig.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
        self.logger.info("Saved: %s", save_path)
        plt.close(fig)
        return fig

    # ── Confusion Matrix ─────────────────────────────────────────────────

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        filename: str = "confusion_matrix.png",
    ) -> plt.Figure:
        """Plot a styled confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"],
            ax=ax, linewidths=0.5, linecolor="white",
            annot_kws={"size": 14},
        )
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)

        fig.tight_layout()
        save_path = Config.OUTPUTS_DIR / filename
        fig.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
        self.logger.info("Saved: %s", save_path)
        plt.close(fig)
        return fig

    # ── ROC Curve ────────────────────────────────────────────────────────

    def plot_roc_curve(
        self,
        roc_data: List[Tuple[np.ndarray, np.ndarray, float, str]],
        filename: str = "roc_curve_comparison.png",
    ) -> plt.Figure:
        """
        Plot ROC curves for multiple models on the same axes.

        Parameters
        ----------
        roc_data : list of (fpr, tpr, auc, model_name) tuples.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = Config.COLOR_PALETTE

        for idx, (fpr, tpr, auc_val, name) in enumerate(roc_data):
            color = colors[idx % len(colors)]
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{name}  (AUC = {auc_val:.4f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Classifier")
        ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        save_path = Config.OUTPUTS_DIR / filename
        fig.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
        self.logger.info("Saved: %s", save_path)
        plt.close(fig)
        return fig

    # ── Feature Importance ───────────────────────────────────────────────

    def plot_feature_importance(
        self,
        importances: np.ndarray,
        feature_names: List[str],
        model_name: str = "XGBoost",
        top_n: int = 20,
        filename: str = "feature_importance.png",
    ) -> plt.Figure:
        """Horizontal bar chart of top-N feature importances."""
        indices = np.argsort(importances)[-top_n:]
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.barh(
            range(len(indices)),
            importances[indices],
            color=Config.COLOR_PALETTE[0],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
        ax.set_title(f"Top {top_n} Feature Importances — {model_name}",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.grid(True, alpha=0.3, axis="x")

        fig.tight_layout()
        save_path = Config.OUTPUTS_DIR / filename
        fig.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
        self.logger.info("Saved: %s", save_path)
        plt.close(fig)
        return fig

    # ── Dataset Distribution ─────────────────────────────────────────────

    def plot_label_distribution(
        self,
        df: pd.DataFrame,
        filename: str = "label_distribution.png",
    ) -> plt.Figure:
        """Pie + bar combo showing Normal vs. Anomaly distribution."""
        counts = df[Config.TARGET_COL].value_counts()
        labels = ["Normal", "Anomaly"]
        colors = [Config.COLOR_NORMAL, Config.COLOR_ANOMALY]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Pie chart
        axes[0].pie(counts, labels=labels, colors=colors, autopct="%1.1f%%",
                    startangle=90, textprops={"fontsize": 12})
        axes[0].set_title("Label Distribution (Pie)", fontsize=13, fontweight="bold")

        # Bar chart
        axes[1].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=0.5)
        for i, v in enumerate(counts.values):
            axes[1].text(i, v + 100, f"{v:,}", ha="center", fontweight="bold", fontsize=11)
        axes[1].set_title("Label Distribution (Count)", fontsize=13, fontweight="bold")
        axes[1].set_ylabel("Number of Samples", fontsize=12)
        axes[1].grid(True, alpha=0.3, axis="y")

        fig.suptitle("UNSW-NB15 Dataset — Label Distribution", fontsize=15, fontweight="bold", y=1.02)
        fig.tight_layout()
        save_path = Config.OUTPUTS_DIR / filename
        fig.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
        self.logger.info("Saved: %s", save_path)
        plt.close(fig)
        return fig

    def plot_attack_distribution(
        self,
        df: pd.DataFrame,
        filename: str = "attack_distribution.png",
    ) -> plt.Figure:
        """Bar chart of attack category counts."""
        if Config.ATTACK_COL not in df.columns:
            self.logger.warning("No '%s' column found — skipping", Config.ATTACK_COL)
            return plt.figure()

        counts = df[Config.ATTACK_COL].value_counts()
        fig, ax = plt.subplots(figsize=(12, 6))

        bars = ax.bar(range(len(counts)), counts.values,
                      color=Config.COLOR_PALETTE[:len(counts)],
                      edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=35, ha="right", fontsize=10)

        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    f"{val:,}", ha="center", fontsize=9, fontweight="bold")

        ax.set_title("Attack Category Distribution", fontsize=14, fontweight="bold")
        ax.set_ylabel("Number of Samples", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        save_path = Config.OUTPUTS_DIR / filename
        fig.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
        self.logger.info("Saved: %s", save_path)
        plt.close(fig)
        return fig

    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        filename: str = "correlation_heatmap.png",
    ) -> plt.Figure:
        """Correlation heatmap for numeric features."""
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(16, 14))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                    square=True, linewidths=0.5, ax=ax,
                    cbar_kws={"shrink": 0.8})
        ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")

        fig.tight_layout()
        save_path = Config.OUTPUTS_DIR / filename
        fig.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
        self.logger.info("Saved: %s", save_path)
        plt.close(fig)
        return fig

    def plot_protocol_distribution(
        self,
        df: pd.DataFrame,
        filename: str = "protocol_distribution.png",
    ) -> plt.Figure:
        """Pie chart for protocol distribution."""
        if "proto" not in df.columns:
            return plt.figure()

        counts = df["proto"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 8))

        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=Config.COLOR_PALETTE[:len(counts)],
            startangle=140, textprops={"fontsize": 10},
        )
        ax.set_title("Top 10 Protocol Distribution", fontsize=14, fontweight="bold")

        fig.tight_layout()
        save_path = Config.OUTPUTS_DIR / filename
        fig.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
        self.logger.info("Saved: %s", save_path)
        plt.close(fig)
        return fig

    def plot_numeric_distributions(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        filename: str = "numeric_distributions.png",
    ) -> plt.Figure:
        """Grid of histograms for selected numeric columns."""
        if columns is None:
            columns = ["dur", "spkts", "dpkts", "sbytes", "dbytes",
                        "rate", "sttl", "dttl", "sload", "dload",
                        "smean", "dmean"]
        columns = [c for c in columns if c in df.columns]

        n_cols = 4
        n_rows = (len(columns) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        axes = axes.flatten()

        for idx, col in enumerate(columns):
            ax = axes[idx]
            df[col].hist(bins=50, ax=ax, color=Config.COLOR_PALETTE[0], alpha=0.7, edgecolor="white")
            ax.set_title(col, fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Numeric Feature Distributions", fontsize=15, fontweight="bold", y=1.01)
        fig.tight_layout()
        save_path = Config.OUTPUTS_DIR / filename
        fig.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
        self.logger.info("Saved: %s", save_path)
        plt.close(fig)
        return fig


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                           HELPER FUNCTIONS                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def save_dataset_statistics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    filename: str = "dataset_statistics.txt",
) -> None:
    """Write a comprehensive statistical summary to a text file."""
    path = Config.OUTPUTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  UNSW-NB15 Dataset — Statistical Summary\n")
        f.write("=" * 70 + "\n\n")

        for name, df in [("Training", train_df), ("Testing", test_df)]:
            f.write(f"─── {name} Set ───\n")
            f.write(f"  Shape       : {df.shape}\n")
            f.write(f"  Missing     : {df.isnull().sum().sum()}\n")
            f.write(f"  Duplicates  : {df.duplicated().sum()}\n")
            if Config.TARGET_COL in df.columns:
                dist = df[Config.TARGET_COL].value_counts()
                f.write(f"  Normal      : {dist.get(0, 0):,}\n")
                f.write(f"  Anomaly     : {dist.get(1, 0):,}\n")
            f.write("\n")

        f.write("─── Numeric Feature Statistics (Training) ───\n\n")
        f.write(train_df.describe().to_string())
        f.write("\n")

    logger = get_logger("utils")
    logger.info("Dataset statistics saved to %s", path)


def save_model_comparison(
    results: Dict[str, Dict[str, float]],
    filename: str = "model_comparison.csv",
) -> pd.DataFrame:
    """Save model comparison metrics as a CSV and return the DataFrame."""
    df = pd.DataFrame(results).T
    df.index.name = "Model"
    path = Config.OUTPUTS_DIR / filename
    df.to_csv(path)

    logger = get_logger("utils")
    logger.info("Model comparison saved to %s", path)
    return df
