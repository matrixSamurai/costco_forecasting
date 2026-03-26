#!/usr/bin/env python3
"""
Build XGBoost model to predict loss_rate, replacing the sigmoid formula.

1. Generates ~50,000 training samples using the sigmoid formula + noise + interaction effects
2. Trains XGBoost regressor on loss_rate
3. Compares XGBoost vs sigmoid baseline on held-out test set
4. Saves model artifact to loss_model_xgb.joblib

Usage:
    cd pipeline/
    python build_loss_model.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

SHELF_LIFE_CSV = os.path.join(SCRIPT_DIR, "data", "lettuce_shelf_life_by_temp.csv")
SEASONAL_CSV = os.path.join(SCRIPT_DIR, "data", "seasonal_price_indices.csv")
SIGMOID_CSV = os.path.join(SCRIPT_DIR, "data", "sigmoid_parameters.csv")
OUT_MODEL = os.path.join(SCRIPT_DIR, "loss_model_xgb.joblib")
OUT_TRAINING_DATA = os.path.join(SCRIPT_DIR, "training_data.csv")

# ---------------------------------------------------------------------------
# Sigmoid formula (from v2 model — used to generate training labels)
# ---------------------------------------------------------------------------
LETTUCE_TYPES = ["iceberg", "romaine", "butterhead", "leaf", "spring_mix"]
LETTUCE_TYPE_ENCODING = {name: i for i, name in enumerate(LETTUCE_TYPES)}


def load_sigmoid_params(csv_path):
    df = pd.read_csv(csv_path)
    params = dict(zip(df["parameter"], df["value"]))
    return params["k"], params["x0"], params["L_min"], params["beta"]


def sigmoid_loss_rate(consumption_pct, k, x0, L_min=0.07):
    x = np.asarray(consumption_pct, dtype=float)
    rate = L_min + (1.0 - L_min) / (1.0 + np.exp(-k * (x - x0)))
    return np.clip(rate, L_min, 1.0)


def effective_consumption_pct(delay_hours, shelf_life_days, beta=1.8, pre_delay_pct=0.0):
    delay_days = np.asarray(delay_hours, dtype=float) / 24.0
    ratio = delay_days / shelf_life_days
    return np.power(ratio, beta) * 100.0 + pre_delay_pct


# ---------------------------------------------------------------------------
# Shelf life lookup with interpolation for continuous temps
# ---------------------------------------------------------------------------
def load_shelf_life_table(csv_path):
    """Load shelf life CSV and return a dict: lettuce_type -> [(temp_f, shelf_life_days), ...]"""
    df = pd.read_csv(csv_path)
    table = {}
    for lt in LETTUCE_TYPES:
        sub = df[df["lettuce_type"] == lt].sort_values("storage_temp_f")
        table[lt] = list(zip(sub["storage_temp_f"].values, sub["shelf_life_days"].values))
    # Add extrapolated 55°F for varieties missing it
    for lt in ["butterhead", "leaf", "spring_mix"]:
        temps = [t for t, _ in table[lt]]
        if 55 not in temps:
            last_sl = table[lt][-1][1]
            table[lt].append((55, max(1, last_sl // 2)))
            table[lt].sort(key=lambda x: x[0])
    return table


def interpolate_shelf_life(shelf_table, lettuce_type, temp_f):
    """Linear interpolation of shelf life for any continuous temperature."""
    points = shelf_table.get(lettuce_type)
    if points is None:
        return 7.0  # fallback
    temps = [p[0] for p in points]
    lives = [p[1] for p in points]
    if temp_f <= temps[0]:
        return float(lives[0])
    if temp_f >= temps[-1]:
        return float(lives[-1])
    for i in range(len(temps) - 1):
        if temps[i] <= temp_f <= temps[i + 1]:
            frac = (temp_f - temps[i]) / (temps[i + 1] - temps[i])
            return lives[i] + frac * (lives[i + 1] - lives[i])
    return float(lives[-1])


# ---------------------------------------------------------------------------
# Seasonal price index lookup
# ---------------------------------------------------------------------------
def load_seasonal_indices(csv_path):
    """Returns dict: (lettuce_type, month) -> seasonal_index"""
    df = pd.read_csv(csv_path)
    lookup = {}
    for _, row in df.iterrows():
        lookup[(row["lettuce_type"], int(row["month"]))] = float(row["seasonal_index"])
    return lookup


# ---------------------------------------------------------------------------
# Generate training data
# ---------------------------------------------------------------------------
def generate_training_data(shelf_table, seasonal_lookup, k, x0, L_min, beta,
                           n_target=50000, seed=42):
    """
    Generate synthetic training data using the sigmoid formula + noise + interaction effects.

    Dimensions:
      - delay_hours: 0-168 (0-7 days)
      - delay_temp_f: 30-60 (continuous)
      - lettuce_type: 5 varieties
      - organic: 0/1
      - month: 1-12
      - transit_distance_km: 50-7500
    """
    rng = np.random.RandomState(seed)

    # Calculate samples per combo to reach ~n_target
    n_types = len(LETTUCE_TYPES)
    n_organic = 2
    n_months = 12
    samples_per_combo = max(1, n_target // (n_types * n_organic * n_months))

    records = []
    for lt in LETTUCE_TYPES:
        lt_enc = LETTUCE_TYPE_ENCODING[lt]
        for organic in [0, 1]:
            for month in range(1, 13):
                si = seasonal_lookup.get((lt, month), 1.0)
                for _ in range(samples_per_combo):
                    delay_hours = rng.uniform(0, 168)
                    delay_temp_f = rng.uniform(30, 60)
                    transit_distance_km = rng.uniform(50, 7500)
                    pre_delay_pct = rng.uniform(0, 25)

                    sl = interpolate_shelf_life(shelf_table, lt, delay_temp_f)
                    cons = effective_consumption_pct(delay_hours, sl, beta, pre_delay_pct)
                    base_loss = float(sigmoid_loss_rate(cons, k, x0, L_min))

                    # Interaction effect: high temp + long delay → extra loss
                    temp_factor = max(0, (delay_temp_f - 41)) / 19.0  # 0 at 41°F, 1 at 60°F
                    delay_factor = delay_hours / 168.0
                    interaction_boost = 0.08 * temp_factor * delay_factor * delay_factor
                    # Organic has slightly lower loss (better packaging)
                    organic_adj = -0.015 if organic else 0.0
                    # Long distance → slightly more loss from vibration/handling
                    dist_adj = 0.02 * (transit_distance_km / 7500.0)

                    loss_rate = base_loss + interaction_boost + organic_adj + dist_adj
                    # Add Gaussian noise
                    noise = rng.normal(0, 0.02)
                    loss_rate = np.clip(loss_rate + noise, L_min, 1.0)

                    records.append({
                        "delay_hours": round(delay_hours, 2),
                        "delay_temp_f": round(delay_temp_f, 1),
                        "shelf_life_days": round(sl, 2),
                        "consumption_pct": round(float(cons), 2),
                        "lettuce_type_encoded": lt_enc,
                        "organic": organic,
                        "month": month,
                        "seasonal_price_index": round(si, 4),
                        "transit_distance_km": round(transit_distance_km, 1),
                        "pre_delay_consumption_pct": round(pre_delay_pct, 2),
                        "loss_rate": round(float(loss_rate), 6),
                    })

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------------
# Train XGBoost
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "delay_hours",
    "delay_temp_f",
    "shelf_life_days",
    "consumption_pct",
    "lettuce_type_encoded",
    "organic",
    "month",
    "seasonal_price_index",
    "transit_distance_km",
    "pre_delay_consumption_pct",
]


def train_xgboost(df, feature_cols=FEATURE_COLS, target="loss_rate", test_size=0.2, seed=42):
    from xgboost import XGBRegressor

    X = df[feature_cols].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred_xgb = model.predict(X_test)
    y_pred_xgb = np.clip(y_pred_xgb, 0, 1)

    metrics = {
        "xgb_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_xgb))),
        "xgb_mae": float(mean_absolute_error(y_test, y_pred_xgb)),
        "xgb_r2": float(r2_score(y_test, y_pred_xgb)),
    }

    return model, X_test, y_test, y_pred_xgb, metrics


def sigmoid_baseline_metrics(X_test, y_test, feature_cols, k, x0, L_min):
    """Evaluate sigmoid formula as baseline on the same test set."""
    cons_idx = feature_cols.index("consumption_pct")
    consumption = X_test[:, cons_idx]
    y_pred_sig = sigmoid_loss_rate(consumption, k, x0, L_min)
    return {
        "sig_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_sig))),
        "sig_mae": float(mean_absolute_error(y_test, y_pred_sig)),
        "sig_r2": float(r2_score(y_test, y_pred_sig)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Build Loss Rate Model (XGBoost)")
    print("=" * 60)

    # Check inputs
    for path, name in [(SHELF_LIFE_CSV, "shelf_life"), (SEASONAL_CSV, "seasonal"),
                        (SIGMOID_CSV, "sigmoid_params")]:
        if not os.path.exists(path):
            print(f"ERROR: missing {name}: {path}")
            sys.exit(1)

    # Load sigmoid parameters
    k, x0, L_min, beta = load_sigmoid_params(SIGMOID_CSV)
    print(f"\n[1] Sigmoid params: k={k:.4f}, x0={x0:.2f}, L_min={L_min}, beta={beta}")

    # Load shelf life table
    shelf_table = load_shelf_life_table(SHELF_LIFE_CSV)
    total_entries = sum(len(v) for v in shelf_table.values())
    print(f"[2] Shelf life table: {total_entries} entries across {len(shelf_table)} varieties")

    # Load seasonal indices
    seasonal_lookup = load_seasonal_indices(SEASONAL_CSV)
    print(f"[3] Seasonal indices: {len(seasonal_lookup)} entries")

    # Generate training data
    print("\n[4] Generating training data...")
    df = generate_training_data(shelf_table, seasonal_lookup, k, x0, L_min, beta)
    print(f"    Generated {len(df)} samples")
    print(f"    loss_rate stats: mean={df['loss_rate'].mean():.4f}, "
          f"std={df['loss_rate'].std():.4f}, "
          f"min={df['loss_rate'].min():.4f}, max={df['loss_rate'].max():.4f}")
    df.to_csv(OUT_TRAINING_DATA, index=False)
    print(f"    -> {os.path.basename(OUT_TRAINING_DATA)}")

    # Train XGBoost
    print("\n[5] Training XGBoost...")
    model, X_test, y_test, y_pred_xgb, xgb_metrics = train_xgboost(df)

    # Sigmoid baseline
    sig_metrics = sigmoid_baseline_metrics(X_test, y_test, FEATURE_COLS, k, x0, L_min)

    # Print comparison
    print("\n" + "=" * 60)
    print("Model Comparison (test set)")
    print("=" * 60)
    print(f"  {'Metric':<12} {'Sigmoid':>12} {'XGBoost':>12} {'Improvement':>14}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*14}")
    for metric_name, sig_key, xgb_key in [
        ("RMSE", "sig_rmse", "xgb_rmse"),
        ("MAE", "sig_mae", "xgb_mae"),
        ("R²", "sig_r2", "xgb_r2"),
    ]:
        sv = sig_metrics[sig_key]
        xv = xgb_metrics[xgb_key]
        if metric_name == "R²":
            imp = f"+{xv - sv:.4f}"
        else:
            imp = f"-{sv - xv:.4f}" if sv > xv else f"+{xv - sv:.4f}"
        print(f"  {metric_name:<12} {sv:>12.4f} {xv:>12.4f} {imp:>14}")

    # Feature importance
    print("\n  Feature importance:")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx:
        print(f"    {FEATURE_COLS[i]:<30s} {importances[i]:.4f}")

    # Save model artifact
    artifact = {
        "model": model,
        "feature_names": FEATURE_COLS,
        "lettuce_type_encoding": LETTUCE_TYPE_ENCODING,
        "sigmoid_params": {"k": k, "x0": x0, "L_min": L_min, "beta": beta},
        "shelf_life_table": shelf_table,
        "seasonal_lookup": seasonal_lookup,
        "metrics": {**xgb_metrics, **sig_metrics},
    }
    dump(artifact, OUT_MODEL)
    print(f"\n[6] Model saved -> {os.path.basename(OUT_MODEL)}")
    print("Done.")


if __name__ == "__main__":
    main()
