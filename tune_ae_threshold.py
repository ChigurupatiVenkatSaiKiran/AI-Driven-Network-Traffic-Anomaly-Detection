"""
tune_ae_threshold.py — Push AE Classifier above 90% with optimal threshold
===========================================================================
The AE Classifier achieves 91.87% val_accuracy but 86.29% test accuracy due to
the known UNSW-NB15 train/test distribution shift (train: 38% anomaly, test: 68%).

The XGBoost already uses threshold=0.5945 instead of 0.5 to balance precision/recall.
This script does the same for the AE Classifier and regenerates the final report.
"""

import json
import numpy as np
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

from utils import (
    Config, DataLoader, FeatureEngineer, MetricsCalculator,
    PlotGenerator, get_logger, save_model_comparison,
)

import warnings
warnings.filterwarnings("ignore")
logger = get_logger("tune_ae_threshold", "tune_ae.log")


def sweep_threshold(probs, labels, name="model"):
    """Return threshold that maximises macro-F1 on the given set."""
    best_f1, best_t = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 300):
        preds = (probs >= t).astype(int)
        if len(np.unique(preds)) < 2:
            continue
        f = f1_score(labels, preds, average="macro", zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    logger.info("[%s] Optimal threshold: %.4f  (val macro-F1=%.4f)", name, best_t, best_f1)
    return best_t


def log_metrics(name, y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    logger.info("[%s]  acc=%.2f%%  prec=%.2f%%  rec=%.2f%%  f1=%.2f%%  auc=%.4f",
                name, acc*100, prec*100, rec*100, f1*100, auc)
    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1_score": f1, "roc_auc": auc}


def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    loader = DataLoader()
    train_df, test_df = loader.load_all()

    engineer = FeatureEngineer()
    X_train, y_train = engineer.fit_transform(train_df, is_training=True)
    X_test,  y_test  = engineer.transform(test_df)

    X_train_np = X_train.values.astype("float32")
    X_test_np  = X_test.values.astype("float32")
    y_train_np = y_train.values.astype(int)
    y_test_np  = y_test.values.astype(int)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_np, y_train_np,
        test_size=0.20, random_state=42, stratify=y_train_np,
    )

    # MinMax for AE
    minmax = joblib.load(Config.MODELS_DIR / "minmax_scaler.pkl")
    X_val_mm  = minmax.transform(X_val)
    X_test_mm = minmax.transform(X_test_np)

    # ── MODEL 1: AE Classifier with tuned threshold ──────────────────────────
    logger.info("=" * 65)
    logger.info("  AE CLASSIFIER — THRESHOLD TUNING")
    logger.info("=" * 65)

    ae_clf = tf.keras.models.load_model(str(Config.MODELS_DIR / "ae_classifier.keras"))

    # Get probabilities on val set for threshold sweep
    ae_val_prob = ae_clf.predict(X_val_mm, verbose=0).flatten()
    ae_thresh   = sweep_threshold(ae_val_prob, y_val, name="AE_Classifier")

    # Apply tuned threshold to test set
    ae_test_prob = ae_clf.predict(X_test_mm, verbose=0).flatten()
    ae_test_pred = (ae_test_prob >= ae_thresh).astype(int)
    ae_metrics   = log_metrics("Autoencoder", y_test_np, ae_test_pred, ae_test_prob)

    # ── MODEL 2: XGBoost (already tuned in previous script) ──────────────────
    logger.info("=" * 65)
    logger.info("  XGBOOST — THRESHOLD TUNING")
    logger.info("=" * 65)

    xgb_model    = joblib.load(Config.XGBOOST_MODEL)
    xgb_val_prob = xgb_model.predict_proba(X_val)[:, 1]
    xgb_thresh   = sweep_threshold(xgb_val_prob, y_val, name="XGBoost")

    xgb_test_prob = xgb_model.predict_proba(X_test_np)[:, 1]
    xgb_test_pred = (xgb_test_prob >= xgb_thresh).astype(int)
    xgb_metrics   = log_metrics("XGBoost", y_test_np, xgb_test_pred, xgb_test_prob)

    # ── MODEL 3: IF Hybrid Ensemble ───────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("  IF HYBRID ENSEMBLE — THRESHOLD TUNING")
    logger.info("=" * 65)

    if_model    = joblib.load(Config.IFOREST_MODEL)
    ae_full     = tf.keras.models.load_model(Config.AUTOENCODER_MODEL)

    # Rebuild augmented test features
    if_test_score = -if_model.decision_function(X_test_np).reshape(-1, 1)
    ae_test_recon = ae_full.predict(X_test_mm, verbose=0)
    ae_test_err   = np.mean(np.square(X_test_mm - ae_test_recon), axis=1, keepdims=True)
    X_test_aug    = np.hstack([X_test_np, if_test_score, ae_test_err])

    # Rebuild augmented val features (for threshold sweep)
    if_val_score  = -if_model.decision_function(X_val).reshape(-1, 1)
    ae_val_recon  = ae_full.predict(X_val_mm, verbose=0)
    ae_val_err    = np.mean(np.square(X_val_mm - ae_val_recon), axis=1, keepdims=True)
    X_val_aug     = np.hstack([X_val, if_val_score, ae_val_err])

    hybrid_xgb     = joblib.load(Config.MODELS_DIR / "hybrid_if_ensemble.pkl")
    hybrid_val_prob = hybrid_xgb.predict_proba(X_val_aug)[:, 1]
    hybrid_thresh   = sweep_threshold(hybrid_val_prob, y_val, name="IF_Ensemble")

    hybrid_test_prob = hybrid_xgb.predict_proba(X_test_aug)[:, 1]
    hybrid_test_pred = (hybrid_test_prob >= hybrid_thresh).astype(int)
    if_metrics       = log_metrics("Isolation Forest", y_test_np, hybrid_test_pred, hybrid_test_prob)

    # ── Classification Report ─────────────────────────────────────────────────
    calc = MetricsCalculator()
    plotter = PlotGenerator()

    plotter.plot_confusion_matrix(y_test_np, ae_test_pred,
                                  "Autoencoder", "confusion_matrix_autoencoder.png")
    plotter.plot_confusion_matrix(y_test_np, xgb_test_pred,
                                  "XGBoost", "confusion_matrix_xgboost.png")
    plotter.plot_confusion_matrix(y_test_np, hybrid_test_pred,
                                  "Isolation Forest", "confusion_matrix_iforest.png")

    report_path = Config.OUTPUTS_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        sep = "=" * 70 + "\n"
        f.write(sep)
        f.write("  Classification Reports — All Models (Optimized)\n")
        f.write(sep + "\n")

        f.write("--- Deep Autoencoder Classifier (Semi-supervised Fine-tuning) ---\n")
        f.write(calc.classification_report_text(y_test_np, ae_test_pred))
        f.write("\n\n")

        f.write("--- XGBoost Classifier (GridSearch + Threshold Optimized) ---\n")
        f.write(calc.classification_report_text(y_test_np, xgb_test_pred))
        f.write("\n\n")

        f.write("--- Isolation Forest Ensemble (IF + AE + XGBoost Hybrid) ---\n")
        f.write(calc.classification_report_text(y_test_np, hybrid_test_pred))
        f.write("\n")

    logger.info("Classification report saved to %s", report_path)

    # ROC curves
    all_results = {
        "Autoencoder":      ae_metrics,
        "XGBoost":          xgb_metrics,
        "Isolation Forest": if_metrics,
    }
    save_model_comparison(all_results)

    roc_data = []
    for prob, name in [(ae_test_prob, "Autoencoder"),
                       (xgb_test_prob, "XGBoost"),
                       (hybrid_test_prob, "IF Ensemble")]:
        fpr, tpr, _ = roc_curve(y_test_np, prob)
        auc = roc_auc_score(y_test_np, prob)
        roc_data.append((fpr, tpr, auc, name))
    plotter.plot_roc_curve(roc_data, "roc_curve_comparison.png")

    # ── Final Summary ─────────────────────────────────────────────────────────
    logger.info("\n")
    logger.info("=" * 65)
    logger.info("  FINAL RESULTS SUMMARY")
    logger.info("=" * 65)
    for name, m in all_results.items():
        logger.info("[%s]  acc=%.2f%%  prec=%.2f%%  rec=%.2f%%  f1=%.2f%%  auc=%.4f",
                    name,
                    m["accuracy"]  * 100,
                    m["precision"] * 100,
                    m["recall"]    * 100,
                    m["f1_score"]  * 100,
                    m["roc_auc"])

    logger.info("\n--- Full Classification Report ---")
    with open(report_path, encoding="utf-8") as f:
        for line in f:
            logger.info(line.rstrip())


if __name__ == "__main__":
    main()
