"""
Pre-Model 1 Binary Classifier: Predict whether severe weather events will occur.

Trains on real weather station observations (weather_data_2021_2025.csv, 4.34M rows).
- Input features: 8 continuous weather values + month + day_of_year (10 features)
- Label: has_severe_weather = (snow_ice_pellets OR hail OR thunder OR fog)
- Models: XGBoost, Random Forest, LightGBM

Usage:
    pip install -r requirements.txt
    python build_classifier.py
"""

import os
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(
    ROOT, "..", "model 1", "ver1", "data", "source", "weather_data_2021_2025.csv"
)
MODEL_DIR = os.path.join(ROOT, "models")

FEATURE_NAMES = [
    "temp_mean",
    "temp_min",
    "temp_max",
    "prcp_total",
    "snow_depth",
    "visibility",
    "wind_speed_mean",
    "wind_gust_max",
    "month",
    "day_of_year",
]

WEATHER_FEATURES = [
    "temp_mean",
    "temp_min",
    "temp_max",
    "prcp_total",
    "snow_depth",
    "visibility",
    "wind_speed_mean",
    "wind_gust_max",
]

EVENT_COLS = ["fog", "snow_ice_pellets", "hail", "thunder"]


def load_and_prepare_data(csv_path):
    """Load CSV, build features and labels."""
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Raw rows: {len(df):,}")

    # Convert event columns to numeric (some may have missing values)
    for col in EVENT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Build day_of_year from date
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    df["day_of_year"] = df["date_parsed"].dt.dayofyear

    # Drop rows with missing key features
    required_cols = WEATHER_FEATURES + ["month", "day_of_year"]
    df_clean = df.dropna(subset=required_cols)
    print(f"  After dropping missing values: {len(df_clean):,} rows")

    # Build feature matrix
    X = df_clean[FEATURE_NAMES].values.astype(np.float64)

    # Build label: has_severe_weather = any of (fog, snow_ice_pellets, hail, thunder)
    y = (df_clean[EVENT_COLS].sum(axis=1) > 0).astype(int).values

    print(f"  Features: {X.shape[1]} ({', '.join(FEATURE_NAMES)})")
    print(f"  Label distribution:")
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    print(f"    Positive (severe weather): {n_pos:,} ({n_pos / len(y) * 100:.1f}%)")
    print(f"    Negative (no severe weather): {n_neg:,} ({n_neg / len(y) * 100:.1f}%)")

    return X, y


def evaluate_model(name, model, X_test, y_test, scaler=None):
    """Evaluate a model and print metrics."""
    X_eval = scaler.transform(X_test) if scaler is not None else X_test
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted No  Predicted Yes")
    print(f"    Actual No     {cm[0][0]:>10,}  {cm[0][1]:>13,}")
    print(f"    Actual Yes    {cm[1][0]:>10,}  {cm[1][1]:>13,}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


def print_feature_importance(name, model, feature_names):
    """Print feature importance for tree-based models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        print(f"\n  {name} - Feature Importance:")
        for i in sorted_idx:
            print(f"    {feature_names[i]:20s}: {importances[i]:.4f}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data
    X, y = load_and_prepare_data(CSV_PATH)

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Class imbalance ratio
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    results = {}

    # --- 1. XGBoost ---
    print("\nTraining XGBoost Classifier ...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    xgb_model.fit(X_train_scaled, y_train)
    results["XGBoost"] = evaluate_model("XGBoost", xgb_model, X_test, y_test, scaler)
    print_feature_importance("XGBoost", xgb_model, FEATURE_NAMES)

    xgb_path = os.path.join(MODEL_DIR, "classifier_xgboost.joblib")
    dump(
        {
            "model": xgb_model,
            "scaler": scaler,
            "feature_names": FEATURE_NAMES,
            "model_display_name": "XGBoost",
        },
        xgb_path,
    )
    print(f"  Saved {xgb_path}")

    # --- 2. Random Forest ---
    print("\nTraining Random Forest Classifier ...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train_scaled, y_train)
    results["Random Forest"] = evaluate_model(
        "Random Forest", rf_model, X_test, y_test, scaler
    )
    print_feature_importance("Random Forest", rf_model, FEATURE_NAMES)

    rf_path = os.path.join(MODEL_DIR, "classifier_random_forest.joblib")
    dump(
        {
            "model": rf_model,
            "scaler": scaler,
            "feature_names": FEATURE_NAMES,
            "model_display_name": "Random Forest",
        },
        rf_path,
    )
    print(f"  Saved {rf_path}")

    # --- 3. LightGBM ---
    print("\nTraining LightGBM Classifier ...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(X_train_scaled, y_train)
    results["LightGBM"] = evaluate_model(
        "LightGBM", lgb_model, X_test, y_test, scaler
    )
    print_feature_importance("LightGBM", lgb_model, FEATURE_NAMES)

    lgb_path = os.path.join(MODEL_DIR, "classifier_lightgbm.joblib")
    dump(
        {
            "model": lgb_model,
            "scaler": scaler,
            "feature_names": FEATURE_NAMES,
            "model_display_name": "LightGBM",
        },
        lgb_path,
    )
    print(f"  Saved {lgb_path}")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("  MODEL COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Model':<20s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'AUC':>10s}")
    print(f"  {'-' * 70}")
    for name, m in results.items():
        print(
            f"  {name:<20s} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} {m['f1']:>10.4f} {m['auc']:>10.4f}"
        )
    print()


if __name__ == "__main__":
    main()
