"""
fix_if_report.py — Re-runs Isolation Forest only and regenerates classification_report.txt
Uses saved AE threshold and XGBoost model — no retraining of those.
"""
import json
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

from utils import (
    Config, DataLoader, FeatureEngineer, MetricsCalculator,
    PlotGenerator, get_logger, save_model_comparison,
)

logger = get_logger("fix_if_report", "fix_if_report.log")

def main():
    # ── Load data ──────────────────────────────────────────────────────────
    loader = DataLoader()
    train_df, test_df = loader.load_all()

    # ── Feature engineering ────────────────────────────────────────────────
    engineer = FeatureEngineer()
    X_train, y_train = engineer.fit_transform(train_df, is_training=True)
    X_test,  y_test  = engineer.transform(test_df)

    X_train_np = X_train.values.astype("float32")
    X_test_np  = X_test.values.astype("float32")
    y_train_np = y_train.values.astype(int)
    y_test_np  = y_test.values.astype(int)

    # ── Re-train Isolation Forest ──────────────────────────────────────────
    logger.info("Training Isolation Forest (contamination=%.2f) …", Config.IF_CONTAMINATION)
    if_model = IsolationForest(
        n_estimators=Config.IF_N_ESTIMATORS,
        contamination=Config.IF_CONTAMINATION,
        max_samples=Config.IF_MAX_SAMPLES,
        random_state=42,
        n_jobs=-1,
    )
    if_model.fit(X_train_np)
    joblib.dump(if_model, Config.IFOREST_MODEL)
    logger.info("Isolation Forest saved.")

    raw_preds = if_model.predict(X_test_np)
    if_y_pred = np.where(raw_preds == -1, 1, 0)
    if_scores  = -if_model.decision_function(X_test_np)
    if_scores  = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)

    calc = MetricsCalculator()
    if_metrics = calc.compute_all(y_test_np, if_y_pred, if_scores, model_name="Isolation Forest")

    plotter = PlotGenerator()
    plotter.plot_confusion_matrix(y_test_np, if_y_pred, "Isolation Forest",
                                  "confusion_matrix_iforest.png")

    # ── Load saved AE and XGBoost results from logs ─────────────────────────
    # Re-derive predictions from saved models
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler

    # MinMax scaler was saved during training
    minmax = joblib.load(Config.MODELS_DIR / "minmax_scaler.pkl")
    X_test_mm = minmax.transform(X_test_np)

    # Autoencoder
    ae_model = tf.keras.models.load_model(Config.AUTOENCODER_MODEL)
    with open(Config.THRESHOLD_PATH) as f:
        threshold = json.load(f)["threshold"]

    ae_recon  = ae_model.predict(X_test_mm, verbose=0)
    ae_errors = np.mean(np.square(X_test_mm - ae_recon), axis=1)
    ae_y_pred = (ae_errors > threshold).astype(int)
    ae_metrics = calc.compute_all(y_test_np, ae_y_pred, ae_errors, model_name="Autoencoder")

    # XGBoost
    xgb_model = joblib.load(Config.XGBOOST_MODEL)
    xgb_y_pred = xgb_model.predict(X_test_np)
    xgb_probs  = xgb_model.predict_proba(X_test_np)[:, 1]
    xgb_metrics = calc.compute_all(y_test_np, xgb_y_pred, xgb_probs, model_name="XGBoost")

    # ── Write classification report ─────────────────────────────────────────
    report_path = Config.OUTPUTS_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  Classification Reports — All Models\n")
        f.write("=" * 70 + "\n\n")

        f.write("─── Autoencoder ───\n")
        f.write(calc.classification_report_text(y_test_np, ae_y_pred))
        f.write("\n\n")

        f.write("─── XGBoost ───\n")
        f.write(calc.classification_report_text(y_test_np, xgb_y_pred))
        f.write("\n\n")

        f.write("─── Isolation Forest ───\n")
        f.write(calc.classification_report_text(y_test_np, if_y_pred))
        f.write("\n")

    logger.info("Classification report saved to %s", report_path)
    print("\n" + open(report_path, encoding="utf-8").read())

    # ── Model comparison ────────────────────────────────────────────────────
    all_results = {
        "Autoencoder":      ae_metrics,
        "XGBoost":          xgb_metrics,
        "Isolation Forest": if_metrics,
    }
    save_model_comparison(all_results)
    logger.info("Model comparison saved.")

    # ── ROC comparison ──────────────────────────────────────────────────────
    from sklearn.metrics import roc_curve, roc_auc_score
    roc_data = []
    for probs, name in [(ae_errors, "Autoencoder"), (xgb_probs, "XGBoost"), (if_scores, "Isolation Forest")]:
        fpr, tpr, _ = roc_curve(y_test_np, probs)
        auc_val = roc_auc_score(y_test_np, probs)
        roc_data.append((fpr, tpr, auc_val, name))
    plotter.plot_roc_curve(roc_data, "roc_curve_comparison.png")
    logger.info("ROC curves saved.")

if __name__ == "__main__":
    main()
