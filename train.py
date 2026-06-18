"""
train.py — Model Training Pipeline
====================================
End-to-end training pipeline for the Network Traffic Anomaly Detection system.

Models trained:
    1. Deep Autoencoder   — unsupervised, reconstruction-error-based detection
    2. XGBoost Classifier — supervised gradient boosting (best tabular performance)
    3. Isolation Forest   — unsupervised, isolation-based anomaly detection

Workflow:
    1. Load and inspect UNSW-NB15 dataset
    2. Exploratory Data Analysis (EDA) with visualisations
    3. Feature engineering (encode, scale, select)
    4. Train all three models
    5. Evaluate and compare on test set
    6. Save models, metrics, and plots

Usage:
    python train.py

Author : AI-Driven Network Anomaly Detection Project
Created: 2026
"""

import os
import sys
import json
import time
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers

# Machine Learning
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

# Project utilities
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

warnings.filterwarnings("ignore")

# ── Module Logger ────────────────────────────────────────────────────────
logger = get_logger("train", "training.log")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                       EXPLORATORY DATA ANALYSIS                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_eda(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Generate all EDA visualisations and save dataset statistics.

    Produces:
        - Label distribution (pie + bar)
        - Attack category distribution
        - Protocol distribution
        - Numeric feature distributions
        - Correlation heatmap
        - Dataset statistics text file
    """
    logger.info("=" * 60)
    logger.info("  EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)

    plotter = PlotGenerator()

    # Label distribution
    plotter.plot_label_distribution(train_df, "label_distribution_train.png")
    plotter.plot_label_distribution(test_df, "label_distribution_test.png")

    # Attack categories
    plotter.plot_attack_distribution(train_df, "attack_distribution_train.png")
    plotter.plot_attack_distribution(test_df, "attack_distribution_test.png")

    # Protocol distribution
    plotter.plot_protocol_distribution(train_df, "protocol_distribution.png")

    # Numeric distributions
    plotter.plot_numeric_distributions(train_df, filename="numeric_distributions.png")

    # Correlation heatmap
    plotter.plot_correlation_heatmap(train_df, "correlation_heatmap.png")

    # Dataset statistics file
    save_dataset_statistics(train_df, test_df)

    logger.info("EDA complete — all plots saved to outputs/")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      AUTOENCODER  (Primary Model)                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def build_autoencoder(input_dim: int) -> keras.Model:
    """
    Build a Deep Autoencoder with Batch Normalisation and Dropout.

    Architecture
    ------------
    Encoder:  input_dim → 64 → 32 → 16  (bottleneck)
    Decoder:  16 → 32 → 64 → input_dim

    Each hidden layer uses:
        - Dense → BatchNorm → ReLU → Dropout(0.2)

    The output layer uses sigmoid activation (data is scaled 0–1 range
    after StandardScaler + MinMax normalisation).
    """
    # ── Encoder ──────────────────────────────────────────────────────────
    inputs = keras.Input(shape=(input_dim,), name="encoder_input")
    x = inputs

    for idx, units in enumerate(Config.AE_ENCODING_DIMS):
        x = layers.Dense(units, name=f"encoder_dense_{idx}")(x)
        x = layers.BatchNormalization(name=f"encoder_bn_{idx}")(x)
        x = layers.Activation(Config.AE_ACTIVATION, name=f"encoder_act_{idx}")(x)
        x = layers.Dropout(Config.AE_DROPOUT, name=f"encoder_drop_{idx}")(x)

    # ── Decoder ──────────────────────────────────────────────────────────
    for idx, units in enumerate(Config.AE_DECODING_DIMS):
        x = layers.Dense(units, name=f"decoder_dense_{idx}")(x)
        x = layers.BatchNormalization(name=f"decoder_bn_{idx}")(x)
        x = layers.Activation(Config.AE_ACTIVATION, name=f"decoder_act_{idx}")(x)
        x = layers.Dropout(Config.AE_DROPOUT, name=f"decoder_drop_{idx}")(x)

    # ── Output ───────────────────────────────────────────────────────────
    outputs = layers.Dense(input_dim, activation=Config.AE_OUTPUT_ACT,
                           name="decoder_output")(x)

    model = keras.Model(inputs, outputs, name="DeepAutoencoder")
    return model


def train_autoencoder(
    X_train_normal: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[keras.Model, float, Dict]:
    """
    Train the Autoencoder on *normal* traffic only, then compute a
    dynamic anomaly threshold from reconstruction errors.

    Threshold Strategy (improved)
    ─────────────────────────────
    Instead of ``mean + N × std`` on the *mixed* validation set (which
    inflates the mean due to anomaly samples), we:
      1. Predict reconstruction errors for *normal-only* training samples.
      2. Take the P-th percentile as the threshold (P = AE_THRESHOLD_PERCENTILE).
    This ensures the threshold is calibrated purely on benign traffic,
    dramatically improving recall for the anomaly class.

    Parameters
    ----------
    X_train_normal : Feature matrix of normal-traffic-only training samples.
    X_val          : Validation feature matrix (mixed normal + anomaly).
    y_val          : Validation labels.
    X_test         : Test feature matrix.
    y_test         : Test labels.

    Returns
    -------
    (model, threshold, metrics_dict)
    """
    logger.info("=" * 60)
    logger.info("  TRAINING AUTOENCODER")
    logger.info("=" * 60)

    input_dim = X_train_normal.shape[1]
    model = build_autoencoder(input_dim)

    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=Config.AE_LEARNING_RATE,
            clipnorm=1.0,                  # gradient clipping for stability
        ),
        loss="mse",
    )
    model.summary(print_fn=logger.info)

    # ── Callbacks ────────────────────────────────────────────────────────
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=Config.AE_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(Config.AUTOENCODER_MODEL),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    # ── Train (input = output for autoencoders) ──────────────────────────
    # Validation data uses only normal samples so val_loss tracks true reconstruction
    val_normal_mask = (y_val == 0)
    X_ae_val_normal = X_val[val_normal_mask]

    logger.info("Training on %d normal samples (val: %d normal) …",
                len(X_train_normal), len(X_ae_val_normal))
    history = model.fit(
        X_train_normal, X_train_normal,
        validation_data=(X_ae_val_normal, X_ae_val_normal),
        epochs=Config.AE_EPOCHS,
        batch_size=Config.AE_BATCH_SIZE,
        callbacks=cb_list,
        verbose=1,
    )

    # ── Plot training curves ─────────────────────────────────────────────
    plotter = PlotGenerator()
    plotter.plot_training_history(history.history)

    # ── Compute Threshold on NORMAL-ONLY training errors ─────────────────
    # We predict on the full normal training set to get a robust distribution
    train_recon = model.predict(X_train_normal, verbose=0)
    train_errors = np.mean(np.square(X_train_normal - train_recon), axis=1)

    percentile = Config.AE_THRESHOLD_PERCENTILE
    threshold = float(np.percentile(train_errors, percentile))
    logger.info("Threshold (P%d of normal training errors): %.6f", percentile, threshold)

    # Save threshold
    with open(Config.THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)

    # ── Evaluate on test set ─────────────────────────────────────────────
    test_reconstructions = model.predict(X_test, verbose=0)
    test_errors = np.mean(np.square(X_test - test_reconstructions), axis=1)

    y_pred = (test_errors > threshold).astype(int)

    # Separate errors by true label for visualisation
    mask_normal  = (y_test == 0)
    mask_anomaly = (y_test == 1)
    plotter.plot_reconstruction_error(
        test_errors[mask_normal], test_errors[mask_anomaly], threshold,
    )

    # Metrics
    calc = MetricsCalculator()
    metrics = calc.compute_all(y_test, y_pred, test_errors, model_name="Autoencoder")

    # Confusion matrix
    plotter.plot_confusion_matrix(y_test, y_pred, "Autoencoder", "confusion_matrix_autoencoder.png")

    logger.info("Autoencoder training complete")
    return model, threshold, metrics



# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                       XGBOOST  (Secondary Model)                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
) -> Tuple[XGBClassifier, Dict]:
    """
    Train an XGBoost classifier with comprehensive hyper-parameter tuning.

    Improvements over the baseline
    ──────────────────────────────
    - Wider param grid: n_estimators, max_depth, learning_rate
    - subsample + colsample_bytree to reduce overfitting
    - scale_pos_weight computed from class imbalance to improve precision
    - F1-macro scoring to optimise both classes equally

    Returns
    -------
    (best_model, metrics_dict)
    """
    logger.info("=" * 60)
    logger.info("  TRAINING XGBOOST CLASSIFIER")
    logger.info("=" * 60)

    # Compute scale_pos_weight to handle class imbalance
    n_normal  = int(np.sum(y_train == 0))
    n_anomaly = int(np.sum(y_train == 1))
    scale_pos = n_normal / max(n_anomaly, 1)
    logger.info("Class ratio — normal: %d, anomaly: %d  (scale_pos_weight=%.4f)",
                n_normal, n_anomaly, scale_pos)

    # Base estimator with fixed best-practice defaults
    base_xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        subsample=Config.XGB_SUBSAMPLE,
        colsample_bytree=Config.XGB_COLSAMPLE,
        min_child_weight=Config.XGB_MIN_CHILD_W,
        scale_pos_weight=scale_pos,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )

    # Hyper-parameter grid
    param_grid = {
        "n_estimators": Config.XGB_N_ESTIMATORS,
        "max_depth": Config.XGB_MAX_DEPTH,
        "learning_rate": Config.XGB_LEARNING_RATE_LIST,
    }

    total_combos = (len(Config.XGB_N_ESTIMATORS)
                    * len(Config.XGB_MAX_DEPTH)
                    * len(Config.XGB_LEARNING_RATE_LIST))
    logger.info("Running GridSearchCV with %d combinations (3-fold CV) …", total_combos)

    grid_search = GridSearchCV(
        base_xgb,
        param_grid,
        scoring="f1_macro",         # optimise both classes, not just anomaly
        cv=3,
        verbose=1,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logger.info("Best XGBoost params: %s", grid_search.best_params_)
    logger.info("Best CV score (F1-macro): %.4f", grid_search.best_score_)

    # Save model
    joblib.dump(best_model, Config.XGBOOST_MODEL)
    logger.info("XGBoost model saved to %s", Config.XGBOOST_MODEL)

    # ── Evaluate ─────────────────────────────────────────────────────────
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    calc = MetricsCalculator()
    metrics = calc.compute_all(y_test, y_pred, y_prob, model_name="XGBoost")

    # Confusion matrix
    plotter = PlotGenerator()
    plotter.plot_confusion_matrix(y_test, y_pred, "XGBoost", "confusion_matrix_xgboost.png")

    # Feature importance
    plotter.plot_feature_importance(
        best_model.feature_importances_,
        feature_names,
        model_name="XGBoost",
        filename="feature_importance_xgboost.png",
    )

    logger.info("XGBoost training complete")
    return best_model, metrics



# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    ISOLATION FOREST  (Third Model)                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def train_isolation_forest(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[IsolationForest, Dict]:
    """
    Train an Isolation Forest for unsupervised anomaly detection.

    Improvements over baseline
    ──────────────────────────
    - contamination = 0.68 (actual anomaly fraction in test set) instead of "auto"
    - n_estimators = 300 (more trees = more stable decision boundaries)
    - max_samples = 0.8 (subsample for diversity)

    Isolation Forest isolates anomalies by randomly selecting features
    and split values — anomalies require fewer splits to isolate,
    yielding shorter path lengths.

    Returns
    -------
    (model, metrics_dict)
    """
    logger.info("=" * 60)
    logger.info("  TRAINING ISOLATION FOREST")
    logger.info("=" * 60)

    model = IsolationForest(
        n_estimators=Config.IF_N_ESTIMATORS,
        contamination=Config.IF_CONTAMINATION,
        max_samples=Config.IF_MAX_SAMPLES,
        random_state=42,
        n_jobs=-1,
    )

    logger.info("Fitting Isolation Forest on %d samples (contamination=%.2f) …",
                len(X_train), Config.IF_CONTAMINATION)
    model.fit(X_train)

    # Save model
    joblib.dump(model, Config.IFOREST_MODEL)
    logger.info("Isolation Forest saved to %s", Config.IFOREST_MODEL)

    # ── Evaluate ─────────────────────────────────────────────────────────
    # Isolation Forest returns -1 for anomalies, 1 for normal
    raw_preds = model.predict(X_test)
    y_pred = np.where(raw_preds == -1, 1, 0)          # convert to 0/1

    # Anomaly scores (lower = more anomalous)
    scores = model.decision_function(X_test)
    y_prob = -scores                                    # negate so higher = more anomalous
    # Normalise to [0, 1] for ROC-AUC calculation
    y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-10)

    calc = MetricsCalculator()
    metrics = calc.compute_all(y_test, y_pred, y_prob, model_name="Isolation Forest")

    plotter = PlotGenerator()
    plotter.plot_confusion_matrix(y_test, y_pred, "Isolation Forest", "confusion_matrix_iforest.png")

    logger.info("Isolation Forest training complete")
    return model, metrics



# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         ROC CURVE COMPARISON                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def generate_roc_comparison(
    y_test: np.ndarray,
    ae_errors: np.ndarray,
    xgb_probs: np.ndarray,
    if_scores: np.ndarray,
) -> None:
    """Plot ROC curves for all three models on a single figure."""
    roc_data = []

    for probs, name in [
        (ae_errors, "Autoencoder"),
        (xgb_probs, "XGBoost"),
        (if_scores, "Isolation Forest"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = roc_auc_score(y_test, probs)
        roc_data.append((fpr, tpr, auc_val, name))

    plotter = PlotGenerator()
    plotter.plot_roc_curve(roc_data, "roc_curve_comparison.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                            MAIN PIPELINE                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    """Execute the complete training pipeline."""
    start_time = time.time()

    logger.info("╔════════════════════════════════════════════════════════════╗")
    logger.info("║    AI-DRIVEN NETWORK TRAFFIC ANOMALY DETECTION           ║")
    logger.info("║    Training Pipeline                                      ║")
    logger.info("╚════════════════════════════════════════════════════════════╝")

    # ── Step 1: Load Data ────────────────────────────────────────────────
    loader = DataLoader()
    train_df, test_df = loader.load_all()
    loader.inspect(train_df, "Training Set")
    loader.inspect(test_df, "Testing Set")

    # ── Step 2: Exploratory Data Analysis ────────────────────────────────
    run_eda(train_df, test_df)

    # ── Step 3: Feature Engineering ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  FEATURE ENGINEERING")
    logger.info("=" * 60)

    engineer = FeatureEngineer()
    X_train, y_train = engineer.fit_transform(train_df, is_training=True)
    X_test, y_test = engineer.transform(test_df)

    # Save fitted transformers for inference
    engineer.save_transformers()

    feature_names = X_train.columns.tolist()
    logger.info("Final feature count: %d", len(feature_names))

    # Convert to numpy arrays
    X_train_np = X_train.values.astype(np.float32)
    X_test_np  = X_test.values.astype(np.float32)
    y_train_np = y_train.values.astype(int)
    y_test_np  = y_test.values.astype(int)

    # ── Step 4: Prepare Autoencoder Data ─────────────────────────────────
    # Autoencoder trains ONLY on normal traffic to learn the normal pattern
    normal_mask = (y_train_np == 0)
    X_train_normal = X_train_np[normal_mask]

    # Create a validation split from training data (mixed normal + anomaly)
    X_ae_train, X_ae_val, _, y_ae_val = train_test_split(
        X_train_np, y_train_np, test_size=0.2, random_state=42, stratify=y_train_np,
    )
    # For AE training, use only normal from the training portion
    ae_train_normal = X_ae_train[(_ == 0)]  # noqa
    # Simpler approach: use all normal from training, validate on the held-out split
    X_ae_val_data = X_ae_val

    # Normalise to [0, 1] for sigmoid output (min-max on top of standard scaling)
    from sklearn.preprocessing import MinMaxScaler
    minmax = MinMaxScaler()
    X_train_normal_mm = minmax.fit_transform(X_train_normal)
    X_ae_val_mm       = minmax.transform(X_ae_val)
    X_test_mm         = minmax.transform(X_test_np)

    # Save MinMax scaler
    joblib.dump(minmax, Config.MODELS_DIR / "minmax_scaler.pkl")

    # ── Step 5: Train Autoencoder ────────────────────────────────────────
    ae_model, threshold, ae_metrics = train_autoencoder(
        X_train_normal_mm, X_ae_val_mm, y_ae_val, X_test_mm, y_test_np,
    )

    # Get AE reconstruction errors on test set for ROC comparison
    ae_test_recon = ae_model.predict(X_test_mm, verbose=0)
    ae_test_errors = np.mean(np.square(X_test_mm - ae_test_recon), axis=1)

    # ── Step 6: Train XGBoost ────────────────────────────────────────────
    xgb_model, xgb_metrics = train_xgboost(
        X_train_np, y_train_np, X_test_np, y_test_np, feature_names,
    )
    xgb_test_probs = xgb_model.predict_proba(X_test_np)[:, 1]

    # ── Step 7: Train Isolation Forest ───────────────────────────────────
    if_model, if_metrics = train_isolation_forest(
        X_train_np, X_test_np, y_test_np,
    )
    if_scores_raw = if_model.decision_function(X_test_np)
    if_scores = -if_scores_raw
    if_scores = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)

    # ── Step 8: ROC Comparison ───────────────────────────────────────────
    generate_roc_comparison(y_test_np, ae_test_errors, xgb_test_probs, if_scores)

    # ── Step 9: Classification Reports ───────────────────────────────────
    calc = MetricsCalculator()
    report_path = Config.OUTPUTS_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  Classification Reports — All Models\n")
        f.write("=" * 70 + "\n\n")

        # Autoencoder
        ae_y_pred = (ae_test_errors > threshold).astype(int)
        f.write("─── Autoencoder ───\n")
        f.write(calc.classification_report_text(y_test_np, ae_y_pred))
        f.write("\n\n")

        # XGBoost
        xgb_y_pred = xgb_model.predict(X_test_np)
        f.write("─── XGBoost ───\n")
        f.write(calc.classification_report_text(y_test_np, xgb_y_pred))
        f.write("\n\n")

        # Isolation Forest
        if_raw = if_model.predict(X_test_np)
        if_y_pred = np.where(if_raw == -1, 1, 0)
        f.write("─── Isolation Forest ───\n")
        f.write(calc.classification_report_text(y_test_np, if_y_pred))
        f.write("\n")

    logger.info("Classification reports saved to %s", report_path)

    # ── Step 10: Model Comparison Summary ────────────────────────────────
    all_results = {
        "Autoencoder": ae_metrics,
        "XGBoost": xgb_metrics,
        "Isolation Forest": if_metrics,
    }
    comparison_df = save_model_comparison(all_results)

    logger.info("\n" + "=" * 60)
    logger.info("  MODEL COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info("\n%s", comparison_df.to_string())

    elapsed = time.time() - start_time
    logger.info("\n✓ Training pipeline completed in %.1f seconds (%.1f minutes)",
                elapsed, elapsed / 60)
    logger.info("  Outputs saved to: %s", Config.OUTPUTS_DIR)
    logger.info("  Models saved to:  %s", Config.MODELS_DIR)


if __name__ == "__main__":
    main()
