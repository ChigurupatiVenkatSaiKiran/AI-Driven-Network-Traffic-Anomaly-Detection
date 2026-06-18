"""
optimize_models.py — Optimal Threshold Tuning & IF Normal-Only Training
=======================================================================
Strategy to achieve 80-85%+ metrics across ALL models:

1. Autoencoder   — Sweep validation thresholds to find the F1-maximizing
                   cut-point instead of using a fixed P95 heuristic.

2. XGBoost       — Already strong; also tune prediction threshold for
                   better macro-F1 balance.

3. Isolation Forest — Train on NORMAL-ONLY data (like Autoencoder) so
                   it learns the "normal profile". Then sweep decision
                   function thresholds on validation for optimal cut-point.
                   This is the critical fix: IF trained on mixed data has
                   ROC-AUC ~0.52 (random!); on normal-only it jumps to ~0.85+
"""

import json
import warnings
import numpy as np
import joblib
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from utils import (
    Config, DataLoader, FeatureEngineer, MetricsCalculator,
    PlotGenerator, get_logger, save_model_comparison,
)

warnings.filterwarnings("ignore")
logger = get_logger("optimize_models", "optimize_models.log")


# ── Helpers ──────────────────────────────────────────────────────────────────

def find_optimal_threshold(scores: np.ndarray, labels: np.ndarray,
                            name: str = "model") -> float:
    """
    Sweep 200 threshold candidates between the 1st and 99th percentile of
    `scores` and return the threshold that maximises macro-F1 on `labels`.
    """
    lo = np.percentile(scores, 1)
    hi = np.percentile(scores, 99)
    candidates = np.linspace(lo, hi, 200)

    best_f1 = -1.0
    best_t   = candidates[len(candidates) // 2]

    for t in candidates:
        preds = (scores > t).astype(int)
        if len(np.unique(preds)) < 2:          # skip degenerate splits
            continue
        macro = f1_score(labels, preds, average="macro", zero_division=0)
        if macro > best_f1:
            best_f1 = macro
            best_t  = t

    logger.info("[%s] Optimal threshold: %.6f  (val macro-F1=%.4f)", name, best_t, best_f1)
    return float(best_t)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load & engineer features ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  LOADING DATA")
    logger.info("=" * 60)

    loader = DataLoader()
    train_df, test_df = loader.load_all()

    engineer = FeatureEngineer()
    X_train, y_train = engineer.fit_transform(train_df, is_training=True)
    X_test,  y_test  = engineer.transform(test_df)
    engineer.save_transformers()

    X_train_np = X_train.values.astype("float32")
    X_test_np  = X_test.values.astype("float32")
    y_train_np = y_train.values.astype(int)
    y_test_np  = y_test.values.astype(int)

    # Stratified validation split for threshold tuning
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_np, y_train_np,
        test_size=0.20, random_state=42, stratify=y_train_np
    )

    # Normal-only subsets
    X_tr_normal  = X_tr[y_tr == 0]
    X_val_normal = X_val[y_val == 0]

    logger.info("Train: %d  |  Val: %d  |  Test: %d", len(X_tr), len(X_val), len(X_test_np))
    logger.info("Train normal: %d  |  Val normal: %d", len(X_tr_normal), len(X_val_normal))

    # ── 2. MinMax scale for Autoencoder ─────────────────────────────────────
    minmax = MinMaxScaler()
    X_tr_normal_mm = minmax.fit_transform(X_tr_normal)
    X_val_mm       = minmax.transform(X_val)
    X_test_mm      = minmax.transform(X_test_np)
    joblib.dump(minmax, Config.MODELS_DIR / "minmax_scaler.pkl")

    # ── 3. AUTOENCODER — load saved model, tune threshold ───────────────────
    logger.info("=" * 60)
    logger.info("  OPTIMIZING AUTOENCODER THRESHOLD")
    logger.info("=" * 60)

    ae_model = tf.keras.models.load_model(Config.AUTOENCODER_MODEL)

    # Reconstruction errors on VAL set for threshold search
    val_recon  = ae_model.predict(X_val_mm, verbose=0)
    val_errors = np.mean(np.square(X_val_mm - val_recon), axis=1)

    ae_threshold = find_optimal_threshold(val_errors, y_val, name="Autoencoder")

    # Save updated threshold
    with open(Config.THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": ae_threshold}, f, indent=2)
    logger.info("Autoencoder threshold saved: %.6f", ae_threshold)

    # Test-set predictions
    test_recon    = ae_model.predict(X_test_mm, verbose=0)
    ae_test_err   = np.mean(np.square(X_test_mm - test_recon), axis=1)
    ae_y_pred     = (ae_test_err > ae_threshold).astype(int)

    calc = MetricsCalculator()
    ae_metrics = calc.compute_all(y_test_np, ae_y_pred, ae_test_err,
                                   model_name="Autoencoder")

    plotter = PlotGenerator()
    mask_normal  = (y_test_np == 0)
    mask_anomaly = (y_test_np == 1)
    plotter.plot_reconstruction_error(
        ae_test_err[mask_normal], ae_test_err[mask_anomaly], ae_threshold
    )
    plotter.plot_confusion_matrix(y_test_np, ae_y_pred,
                                   "Autoencoder", "confusion_matrix_autoencoder.png")

    # ── 4. XGBOOST — load & also tune prediction threshold ──────────────────
    logger.info("=" * 60)
    logger.info("  OPTIMIZING XGBOOST THRESHOLD")
    logger.info("=" * 60)

    xgb_model   = joblib.load(Config.XGBOOST_MODEL)
    xgb_val_prob = xgb_model.predict_proba(X_val)[:, 1]

    xgb_threshold = find_optimal_threshold(xgb_val_prob, y_val, name="XGBoost")
    logger.info("XGBoost custom threshold: %.4f", xgb_threshold)

    xgb_test_prob = xgb_model.predict_proba(X_test_np)[:, 1]
    xgb_y_pred    = (xgb_test_prob >= xgb_threshold).astype(int)

    xgb_metrics = calc.compute_all(y_test_np, xgb_y_pred, xgb_test_prob,
                                    model_name="XGBoost")
    plotter.plot_confusion_matrix(y_test_np, xgb_y_pred,
                                   "XGBoost", "confusion_matrix_xgboost.png")

    # ── 5. ISOLATION FOREST — train on NORMAL-ONLY + optimal threshold ──────
    logger.info("=" * 60)
    logger.info("  TRAINING ISOLATION FOREST ON NORMAL-ONLY DATA")
    logger.info("=" * 60)
    logger.info("Normal-only training samples: %d", len(X_tr_normal))

    # Train on normal traffic only — contamination=0.05 means we expect
    # only 5% of the normal training pool to look slightly unusual
    if_model = IsolationForest(
        n_estimators=400,
        contamination=0.05,       # low: we trained on clean normal data
        max_samples=0.9,
        max_features=1.0,
        bootstrap=False,
        random_state=42,
        n_jobs=-1,
    )
    if_model.fit(X_tr_normal)
    joblib.dump(if_model, Config.IFOREST_MODEL)
    logger.info("Isolation Forest (normal-only) saved.")

    # Anomaly scores on val set: higher = more anomalous
    val_if_scores  = -if_model.decision_function(X_val)
    if_threshold   = find_optimal_threshold(val_if_scores, y_val, name="IsolationForest")

    # Store threshold for dashboard inference
    with open(Config.MODELS_DIR / "if_threshold.json", "w") as f:
        json.dump({"if_threshold": if_threshold}, f, indent=2)

    # Test-set predictions
    test_if_scores = -if_model.decision_function(X_test_np)
    if_y_pred      = (test_if_scores >= if_threshold).astype(int)

    # Normalise for ROC-AUC
    if_prob = (test_if_scores - test_if_scores.min()) / \
              (test_if_scores.max() - test_if_scores.min() + 1e-10)

    if_metrics = calc.compute_all(y_test_np, if_y_pred, if_prob,
                                   model_name="Isolation Forest")
    plotter.plot_confusion_matrix(y_test_np, if_y_pred,
                                   "Isolation Forest", "confusion_matrix_iforest.png")

    # ── 6. Write Classification Report ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  GENERATING CLASSIFICATION REPORT")
    logger.info("=" * 60)

    report_path = Config.OUTPUTS_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        sep = "=" * 70 + "\n"
        f.write(sep)
        f.write("  Classification Reports -- All Models (Optimized Thresholds)\n")
        f.write(sep + "\n")

        f.write("--- Autoencoder ---\n")
        f.write(calc.classification_report_text(y_test_np, ae_y_pred))
        f.write("\n\n")

        f.write("--- XGBoost ---\n")
        f.write(calc.classification_report_text(y_test_np, xgb_y_pred))
        f.write("\n\n")

        f.write("--- Isolation Forest (trained on normal-only) ---\n")
        f.write(calc.classification_report_text(y_test_np, if_y_pred))
        f.write("\n")

    logger.info("Classification report saved to %s", report_path)

    # ── 7. Model Comparison & ROC Curves ────────────────────────────────────
    all_results = {
        "Autoencoder":      ae_metrics,
        "XGBoost":          xgb_metrics,
        "Isolation Forest": if_metrics,
    }
    save_model_comparison(all_results)

    from sklearn.metrics import roc_curve, roc_auc_score
    roc_data = []
    for probs, name in [
        (ae_test_err,   "Autoencoder"),
        (xgb_test_prob, "XGBoost"),
        (if_prob,       "Isolation Forest"),
    ]:
        fpr, tpr, _ = roc_curve(y_test_np, probs)
        auc_val     = roc_auc_score(y_test_np, probs)
        roc_data.append((fpr, tpr, auc_val, name))
    plotter.plot_roc_curve(roc_data, "roc_curve_comparison.png")
    logger.info("ROC curves saved.")

    # ── 8. Print Summary ─────────────────────────────────────────────────────
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("  FINAL RESULTS SUMMARY")
    logger.info("=" * 60)
    for name, m in all_results.items():
        logger.info(
            "[%s]  acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  auc=%.4f",
            name,
            m.get("accuracy",  0),
            m.get("precision", 0),
            m.get("recall",    0),
            m.get("f1_score",  0),
            m.get("roc_auc",   0),
        )


if __name__ == "__main__":
    main()
