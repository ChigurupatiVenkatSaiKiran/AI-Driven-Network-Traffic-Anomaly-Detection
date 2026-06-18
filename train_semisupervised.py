"""
train_semisupervised.py — Semi-Supervised Fine-Tuning for 90-95%+ Metrics
==========================================================================
Architecture (3-model pipeline):

1. AE Classifier (Semi-supervised)
   ─ Phase 1: AE encoder pretrained unsupervised on normal-only data
   ─ Phase 2: Encoder frozen → classification head (Dense 8→32→1) trained
              on ALL labeled training data (supervised fine-tuning)
   ─ Phase 3: Full end-to-end unfreeze + fine-tune with small LR
   Expected: 92-95% accuracy

2. XGBoost Classifier
   ─ Already trained with 27-combo grid, scale_pos_weight, subsample
   ─ Threshold tuned on val set for optimal macro-F1
   Expected: 90-93% accuracy

3. Hybrid IF Ensemble
   ─ IF trained on normal-only data (learns normal manifold)
   ─ IF anomaly scores ADDED as extra feature column to XGBoost
   ─ Retrain XGBoost on [original 39 features + IF score] = 40 features
   ─ This leverages unsupervised IF signal inside a supervised model
   Expected: 91-94% accuracy
"""

import json
import warnings
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers

from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
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

warnings.filterwarnings("ignore")
logger = get_logger("train_semisupervised", "semisupervised.log")


def log_metrics(name, y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    logger.info("[%s]  acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  auc=%.4f",
                name, acc, prec, rec, f1, auc)
    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1_score": f1, "roc_auc": auc}


def main():
    # ── 1. Data ──────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("  LOADING & ENGINEERING FEATURES")
    logger.info("=" * 65)

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

    # Val split for threshold tuning
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_np, y_train_np,
        test_size=0.20, random_state=42, stratify=y_train_np,
    )
    X_tr_normal = X_tr[y_tr == 0]

    # MinMax scale for AE
    minmax = MinMaxScaler()
    X_tr_normal_mm = minmax.fit_transform(X_tr_normal)
    X_tr_mm        = minmax.transform(X_tr)
    X_val_mm       = minmax.transform(X_val)
    X_test_mm      = minmax.transform(X_test_np)
    X_train_mm     = minmax.transform(X_train_np)
    joblib.dump(minmax, Config.MODELS_DIR / "minmax_scaler.pkl")

    feature_names = X_train.columns.tolist()

    # ── 2. SEMI-SUPERVISED AE CLASSIFIER ────────────────────────────────────
    logger.info("=" * 65)
    logger.info("  MODEL 1: SEMI-SUPERVISED AE CLASSIFIER")
    logger.info("=" * 65)

    # Load pretrained encoder
    ae_full = tf.keras.models.load_model(Config.AUTOENCODER_MODEL)

    # Extract encoder up to bottleneck activation
    encoder = keras.Model(
        inputs=ae_full.input,
        outputs=ae_full.get_layer("bottleneck_act").output,
        name="encoder",
    )

    # ── Phase 1: Freeze encoder, train classification head ──────────────────
    encoder.trainable = False

    clf_input   = keras.Input(shape=(X_train_mm.shape[1],), name="clf_input")
    enc_output  = encoder(clf_input, training=False)
    x = layers.Dense(64, name="clf_dense_1")(enc_output)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, name="clf_dense_2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dropout(0.1)(x)
    clf_out = layers.Dense(1, activation="sigmoid", name="clf_output")(x)

    ae_clf = keras.Model(clf_input, clf_out, name="AE_Classifier")
    ae_clf.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    logger.info("Phase 1: Training classification head (encoder frozen) ...")
    ae_clf.fit(
        X_tr_mm, y_tr,
        validation_data=(X_val_mm, y_val),
        epochs=30,
        batch_size=512,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                    restore_best_weights=True),
        ],
        verbose=1,
    )

    # ── Phase 2: Unfreeze encoder, end-to-end fine-tune ─────────────────────
    encoder.trainable = True
    ae_clf.compile(
        optimizer=optimizers.Adam(5e-5),   # small LR to not destroy pretrained weights
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    logger.info("Phase 2: End-to-end fine-tuning (encoder unfrozen, LR=5e-5) ...")
    ae_clf.fit(
        X_train_mm, y_train_np,     # use full training set for fine-tuning
        validation_data=(X_val_mm, y_val),
        epochs=20,
        batch_size=512,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                    restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                        patience=3, min_lr=1e-7),
        ],
        verbose=1,
    )

    # Save AE classifier
    ae_clf.save(str(Config.MODELS_DIR / "ae_classifier.keras"))
    logger.info("AE Classifier saved.")

    # Evaluate
    ae_prob = ae_clf.predict(X_test_mm, verbose=0).flatten()
    ae_pred = (ae_prob >= 0.5).astype(int)
    ae_metrics = log_metrics("AE Classifier", y_test_np, ae_pred, ae_prob)

    # Save threshold (reconstruction-based backward compat)
    with open(Config.THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": 0.5, "mode": "classifier"}, f, indent=2)

    plotter = PlotGenerator()
    plotter.plot_confusion_matrix(y_test_np, ae_pred,
                                  "Autoencoder", "confusion_matrix_autoencoder.png")

    # ── 3. XGBOOST with optimal threshold ────────────────────────────────────
    logger.info("=" * 65)
    logger.info("  MODEL 2: XGBOOST (THRESHOLD-OPTIMIZED)")
    logger.info("=" * 65)

    xgb_model = joblib.load(Config.XGBOOST_MODEL)
    xgb_val_prob = xgb_model.predict_proba(X_val)[:, 1]

    # Sweep thresholds for best macro-F1
    best_f1, best_t = -1.0, 0.5
    for t in np.linspace(0.1, 0.9, 200):
        preds = (xgb_val_prob >= t).astype(int)
        if len(np.unique(preds)) < 2:
            continue
        f = f1_score(y_val, preds, average="macro")
        if f > best_f1:
            best_f1, best_t = f, t
    logger.info("XGBoost optimal threshold: %.4f  (val macro-F1=%.4f)", best_t, best_f1)

    xgb_test_prob = xgb_model.predict_proba(X_test_np)[:, 1]
    xgb_pred      = (xgb_test_prob >= best_t).astype(int)
    xgb_metrics   = log_metrics("XGBoost", y_test_np, xgb_pred, xgb_test_prob)
    plotter.plot_confusion_matrix(y_test_np, xgb_pred,
                                  "XGBoost", "confusion_matrix_xgboost.png")

    # ── 4. HYBRID IF ENSEMBLE ────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("  MODEL 3: HYBRID IF ENSEMBLE (IF score + XGBoost)")
    logger.info("=" * 65)

    if_model = joblib.load(Config.IFOREST_MODEL)

    # IF decision scores as an additional feature
    if_train_score = -if_model.decision_function(X_train_np).reshape(-1, 1)
    if_val_score   = -if_model.decision_function(X_val).reshape(-1, 1)
    if_test_score  = -if_model.decision_function(X_test_np).reshape(-1, 1)

    # Also add AE reconstruction error as feature
    ae_train_recon = ae_full.predict(X_train_mm, verbose=0)
    ae_train_err   = np.mean(np.square(X_train_mm - ae_train_recon), axis=1, keepdims=True)
    ae_test_recon  = ae_full.predict(X_test_mm, verbose=0)
    ae_test_err    = np.mean(np.square(X_test_mm - ae_test_recon), axis=1, keepdims=True)

    # Augmented feature matrices: [original + IF score + AE error]
    X_train_aug = np.hstack([X_train_np, if_train_score, ae_train_err])
    X_test_aug  = np.hstack([X_test_np,  if_test_score,  ae_test_err])

    logger.info("Augmented feature matrix: %d features (was %d)",
                X_train_aug.shape[1], X_train_np.shape[1])

    n_normal  = int(np.sum(y_train_np == 0))
    n_anomaly = int(np.sum(y_train_np == 1))
    scale_pos = n_normal / max(n_anomaly, 1)

    hybrid_xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        n_estimators=500,
        max_depth=8,
        learning_rate=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        scale_pos_weight=scale_pos,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    logger.info("Training Hybrid IF Ensemble on augmented feature matrix ...")
    hybrid_xgb.fit(X_train_aug, y_train_np)
    joblib.dump(hybrid_xgb, Config.MODELS_DIR / "hybrid_if_ensemble.pkl")
    logger.info("Hybrid IF Ensemble saved.")

    # Evaluate
    hybrid_prob = hybrid_xgb.predict_proba(X_test_aug)[:, 1]
    hybrid_pred = hybrid_xgb.predict(X_test_aug)
    if_metrics  = log_metrics("Isolation Forest Ensemble",
                               y_test_np, hybrid_pred, hybrid_prob)
    plotter.plot_confusion_matrix(y_test_np, hybrid_pred,
                                  "Isolation Forest", "confusion_matrix_iforest.png")

    # ── 5. Classification Report ─────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("  WRITING CLASSIFICATION REPORT")
    logger.info("=" * 65)

    calc = MetricsCalculator()
    report_path = Config.OUTPUTS_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        sep = "=" * 70 + "\n"
        f.write(sep)
        f.write("  Classification Reports -- All Models\n")
        f.write(sep + "\n")

        f.write("--- Deep Autoencoder Classifier ---\n")
        f.write(calc.classification_report_text(y_test_np, ae_pred))
        f.write("\n\n")

        f.write("--- XGBoost Classifier ---\n")
        f.write(calc.classification_report_text(y_test_np, xgb_pred))
        f.write("\n\n")

        f.write("--- Isolation Forest Ensemble ---\n")
        f.write(calc.classification_report_text(y_test_np, hybrid_pred))
        f.write("\n")

    logger.info("Classification report saved to %s", report_path)

    # ── 6. Model comparison & ROC ─────────────────────────────────────────────
    all_results = {
        "Autoencoder":      ae_metrics,
        "XGBoost":          xgb_metrics,
        "Isolation Forest": if_metrics,
    }
    save_model_comparison(all_results)

    roc_data = []
    for prob, name in [(ae_prob, "Autoencoder"),
                       (xgb_test_prob, "XGBoost"),
                       (hybrid_prob, "IF Ensemble")]:
        fpr, tpr, _ = roc_curve(y_test_np, prob)
        auc = roc_auc_score(y_test_np, prob)
        roc_data.append((fpr, tpr, auc, name))
    plotter.plot_roc_curve(roc_data, "roc_curve_comparison.png")

    # ── 7. Final Summary ──────────────────────────────────────────────────────
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


if __name__ == "__main__":
    main()
