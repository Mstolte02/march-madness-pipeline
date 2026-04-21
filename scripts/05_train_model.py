"""
05 - Model Training & Backtesting
====================================
Trains the prediction model using leave-one-year-out cross-validation.
Compares logistic regression and XGBoost, calibrates probabilities,
and reports feature importances (pillar weights).

Usage:
    python scripts/05_train_model.py
    python scripts/05_train_model.py --model xgb
"""

import argparse
import sys
import os

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    log_loss, brier_score_loss, accuracy_score, classification_report
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROCESSED_DIR, MODELS_DIR, OUTPUT_DIR,
    TOURNEY_YEARS, RANDOM_SEED, ALL_FEATURES
)


# ============================================================
# Data Loading
# ============================================================

def load_training_data() -> pd.DataFrame:
    """Load the matchup-level training data."""
    path = PROCESSED_DIR / "matchup_training_data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found at {path}. "
            "Run 04_feature_engineering.py first."
        )
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} matchup rows across "
          f"{df['season'].nunique()} seasons")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get all diff_* feature columns from the training data."""
    return [c for c in df.columns if c.startswith("diff_")]


# ============================================================
# Leave-One-Year-Out Cross-Validation
# ============================================================

def leave_one_year_out_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_type: str = "logistic"
) -> dict:
    """
    Train on all years except one, predict on the held-out year.
    Rotate through all tournament years.

    Returns dict with predictions, metrics, and feature importances.
    """
    print(f"\n  Running LOYO-CV with {model_type} model...")
    print(f"  Features: {len(feature_cols)}")

    all_preds = []
    all_true = []
    all_proba = []
    yearly_metrics = []
    feature_importances = {col: [] for col in feature_cols}

    available_years = sorted(df["season"].unique())
    test_years = [y for y in available_years if y in TOURNEY_YEARS]

    for test_year in test_years:
        # Split
        train = df[df["season"] != test_year].copy()
        test = df[df["season"] == test_year].copy()

        if len(test) == 0 or len(train) == 0:
            continue

        X_train = train[feature_cols].values
        y_train = train["target"].values
        X_test = test[feature_cols].values
        y_test = test["target"].values

        # Handle any remaining NaN/inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train model
        if model_type == "logistic":
            model = LogisticRegression(
                C=1.0, max_iter=1000, random_state=RANDOM_SEED,
                solver="lbfgs", penalty="l2"
            )
        elif model_type == "xgb":
            try:
                import xgboost as xgb
                model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_SEED,
                    eval_metric="logloss",
                    use_label_encoder=False,
                )
            except ImportError:
                print("  ⚠ XGBoost not installed, falling back to GradientBoosting")
                model = GradientBoostingClassifier(
                    n_estimators=200, max_depth=4,
                    learning_rate=0.05, subsample=0.8,
                    random_state=RANDOM_SEED
                )
        else:
            # Sklearn gradient boosting
            model = GradientBoostingClassifier(
                n_estimators=200, max_depth=4,
                learning_rate=0.05, subsample=0.8,
                random_state=RANDOM_SEED
            )

        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        all_preds.extend(y_pred)
        all_true.extend(y_test)
        all_proba.extend(y_proba)

        # Year-level metrics
        year_ll = log_loss(y_test, y_proba)
        year_brier = brier_score_loss(y_test, y_proba)
        year_acc = accuracy_score(y_test, y_pred)

        yearly_metrics.append({
            "year": test_year,
            "log_loss": year_ll,
            "brier_score": year_brier,
            "accuracy": year_acc,
            "n_games": len(y_test) // 2,  # Div by 2 because we have both perspectives
        })

        # Feature importances
        if hasattr(model, "coef_"):
            for i, col in enumerate(feature_cols):
                feature_importances[col].append(abs(model.coef_[0][i]))
        elif hasattr(model, "feature_importances_"):
            for i, col in enumerate(feature_cols):
                feature_importances[col].append(model.feature_importances_[i])

        print(f"    {test_year}: accuracy={year_acc:.3f}, "
              f"log_loss={year_ll:.4f}, brier={year_brier:.4f}")

    # Aggregate metrics
    all_true = np.array(all_true)
    all_preds = np.array(all_preds)
    all_proba = np.array(all_proba)

    overall = {
        "accuracy": accuracy_score(all_true, all_preds),
        "log_loss": log_loss(all_true, all_proba),
        "brier_score": brier_score_loss(all_true, all_proba),
    }

    # Average feature importances
    avg_importance = {}
    for col, vals in feature_importances.items():
        if vals:
            avg_importance[col] = np.mean(vals)
    avg_importance = dict(sorted(avg_importance.items(), key=lambda x: -x[1]))

    return {
        "overall_metrics": overall,
        "yearly_metrics": yearly_metrics,
        "feature_importances": avg_importance,
        "predictions": all_proba,
        "true_labels": all_true,
    }


# ============================================================
# Baseline Comparison: Seed-Only Model
# ============================================================

def seed_only_baseline(df: pd.DataFrame) -> dict:
    """
    Baseline model that only uses seed differential.
    This is what we're trying to beat.
    """
    print("\n  Computing seed-only baseline...")

    seed_col = None
    for c in df.columns:
        if "seed" in c.lower() and "diff" in c.lower():
            seed_col = c
            break

    if seed_col is None:
        print("    ⚠ No seed differential column found")
        return None

    # Simple logistic regression on seed diff only
    from sklearn.linear_model import LogisticRegression

    all_preds = []
    all_true = []
    all_proba = []

    available_years = sorted(df["season"].unique())
    test_years = [y for y in available_years if y in TOURNEY_YEARS]

    for test_year in test_years:
        train = df[df["season"] != test_year]
        test = df[df["season"] == test_year]
        if len(test) == 0:
            continue

        X_train = train[[seed_col]].values
        y_train = train["target"].values
        X_test = test[[seed_col]].values
        y_test = test["target"].values

        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)

        model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
        model.fit(X_train, y_train)

        all_proba.extend(model.predict_proba(X_test)[:, 1])
        all_preds.extend(model.predict(X_test))
        all_true.extend(y_test)

    if not all_true:
        return None

    return {
        "accuracy": accuracy_score(all_true, all_preds),
        "log_loss": log_loss(all_true, all_proba),
        "brier_score": brier_score_loss(all_true, all_proba),
    }


# ============================================================
# Train Final Model
# ============================================================

def train_final_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_type: str = "logistic"
) -> tuple:
    """
    Train the final model on ALL available data.
    Apply probability calibration.
    Save model + scaler for inference.
    """
    print("\n  Training final model on all data...")

    X = df[feature_cols].values
    y = df["target"].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if model_type == "logistic":
        base_model = LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_SEED,
            solver="lbfgs", penalty="l2"
        )
    else:
        try:
            import xgboost as xgb
            base_model = xgb.XGBClassifier(
                n_estimators=200, max_depth=4,
                learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, random_state=RANDOM_SEED,
                eval_metric="logloss", use_label_encoder=False,
            )
        except ImportError:
            base_model = GradientBoostingClassifier(
                n_estimators=200, max_depth=4,
                learning_rate=0.05, subsample=0.8,
                random_state=RANDOM_SEED
            )

    # Calibrate probabilities using cross-validation
    calibrated = CalibratedClassifierCV(base_model, cv=5, method="isotonic")
    calibrated.fit(X, y)

    print(f"    ✓ Final model trained on {len(X)} matchup rows")

    return calibrated, scaler, feature_cols


# ============================================================
# Reporting
# ============================================================

def print_report(
    cv_results: dict,
    baseline: dict | None,
    model_type: str
):
    """Print a comprehensive evaluation report."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)

    # Overall metrics
    m = cv_results["overall_metrics"]
    print(f"\n  {model_type.upper()} Model (LOYO-CV):")
    print(f"    Accuracy:    {m['accuracy']:.4f}")
    print(f"    Log Loss:    {m['log_loss']:.4f}")
    print(f"    Brier Score: {m['brier_score']:.4f}")

    if baseline:
        print(f"\n  Seed-Only Baseline:")
        print(f"    Accuracy:    {baseline['accuracy']:.4f}")
        print(f"    Log Loss:    {baseline['log_loss']:.4f}")
        print(f"    Brier Score: {baseline['brier_score']:.4f}")

        print(f"\n  Improvement over baseline:")
        print(f"    Accuracy:    {(m['accuracy'] - baseline['accuracy'])*100:+.2f}%")
        print(f"    Log Loss:    {(baseline['log_loss'] - m['log_loss']):+.4f} (lower is better)")
        print(f"    Brier Score: {(baseline['brier_score'] - m['brier_score']):+.4f} (lower is better)")

    # Year-by-year
    print(f"\n  Year-by-Year Performance:")
    print(f"  {'Year':>6} {'Acc':>7} {'LogLoss':>9} {'Brier':>7} {'Games':>6}")
    print(f"  {'─'*6} {'─'*7} {'─'*9} {'─'*7} {'─'*6}")
    for ym in cv_results["yearly_metrics"]:
        print(f"  {ym['year']:>6} {ym['accuracy']:>7.3f} {ym['log_loss']:>9.4f} "
              f"{ym['brier_score']:>7.4f} {ym['n_games']:>6}")

    # Feature importances (PILLAR WEIGHTS!)
    print(f"\n  Feature Importances (Pillar Weights):")
    print(f"  {'─' * 50}")

    # Group by pillar
    from config import KENPOM_FEATURES, ROSTER_FEATURES, KILLSHOT_FEATURES, CONTEXT_FEATURES

    pillar_map = {}
    for f in KENPOM_FEATURES:
        pillar_map[f"diff_{f}"] = "KenPom"
    for f in ROSTER_FEATURES:
        pillar_map[f"diff_{f}"] = "Roster"
    for f in KILLSHOT_FEATURES:
        pillar_map[f"diff_{f}"] = "Killshot"
    for f in CONTEXT_FEATURES:
        pillar_map[f"diff_{f}"] = "Context"

    pillar_totals = {"KenPom": 0, "Roster": 0, "Killshot": 0, "Context": 0, "Other": 0}

    fi = cv_results["feature_importances"]
    total_imp = sum(fi.values()) if fi else 1

    for feat, imp in fi.items():
        pillar = pillar_map.get(feat, "Other")
        pillar_totals[pillar] += imp
        pct = (imp / total_imp) * 100
        print(f"    {feat:<40} {pct:>5.1f}%  [{pillar}]")

    print(f"\n  Pillar-Level Breakdown:")
    print(f"  {'─' * 35}")
    for pillar, total in sorted(pillar_totals.items(), key=lambda x: -x[1]):
        if total > 0:
            pct = (total / total_imp) * 100
            bar = "█" * int(pct / 2)
            print(f"    {pillar:<12} {pct:>5.1f}%  {bar}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train March Madness prediction model")
    parser.add_argument("--model", choices=["logistic", "xgb", "gbm"],
                        default="logistic", help="Model type")
    args = parser.parse_args()

    print("=" * 60)
    print("MARCH MADNESS MODEL TRAINING")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading training data...")
    df = load_training_data()
    feature_cols = get_feature_columns(df)
    print(f"  Features available: {feature_cols}")

    # Cross-validation
    print("\n[2/4] Leave-One-Year-Out Cross-Validation...")
    cv_results = leave_one_year_out_cv(df, feature_cols, model_type=args.model)

    # Baseline comparison
    print("\n[3/4] Seed-only baseline...")
    baseline = seed_only_baseline(df)

    # Print report
    print_report(cv_results, baseline, args.model)

    # Train final model
    print("\n[4/4] Training final model on all data...")
    model, scaler, cols = train_final_model(df, feature_cols, model_type=args.model)

    # Save
    model_path = MODELS_DIR / f"march_madness_{args.model}.joblib"
    scaler_path = MODELS_DIR / f"scaler_{args.model}.joblib"
    cols_path = MODELS_DIR / f"feature_cols_{args.model}.joblib"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(cols, cols_path)

    print(f"\n✓ Model saved: {model_path}")
    print(f"✓ Scaler saved: {scaler_path}")
    print(f"✓ Feature cols saved: {cols_path}")

    # Save report
    report_df = pd.DataFrame(cv_results["yearly_metrics"])
    report_df.to_csv(OUTPUT_DIR / "cv_results.csv", index=False)

    fi_df = pd.DataFrame([
        {"feature": k, "importance": v}
        for k, v in cv_results["feature_importances"].items()
    ])
    fi_df.to_csv(OUTPUT_DIR / "feature_importances.csv", index=False)

    print(f"\n✓ Reports saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
