"""
06 - Predict Matchups
======================
Load the trained model and generate win probabilities for
any pair of teams or for the full tournament bracket.

Usage:
    python scripts/06_predict.py --team1 "Duke" --team2 "Houston"
    python scripts/06_predict.py --bracket bracket_2026.csv
"""

import argparse
import sys
import os

import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, MODELS_DIR, OUTPUT_DIR, CURRENT_SEASON


def load_model(model_type: str = "logistic"):
    """Load trained model, scaler, and feature columns."""
    model = joblib.load(MODELS_DIR / f"march_madness_{model_type}.joblib")
    scaler = joblib.load(MODELS_DIR / f"scaler_{model_type}.joblib")
    feature_cols = joblib.load(MODELS_DIR / f"feature_cols_{model_type}.joblib")
    return model, scaler, feature_cols


def load_team_features(season: int = None) -> pd.DataFrame:
    """Load current season team features."""
    if season is None:
        season = CURRENT_SEASON

    all_features = pd.read_csv(PROCESSED_DIR / "team_features_all.csv")
    current = all_features[all_features["season"] == season].copy()

    if len(current) == 0:
        print(f"  ⚠ No features found for season {season}.")
        print(f"  Available seasons: {sorted(all_features['season'].unique())}")
        return pd.DataFrame()

    return current


def predict_matchup(
    team_a: str,
    team_b: str,
    features_df: pd.DataFrame,
    model,
    scaler,
    feature_cols: list[str],
) -> dict:
    """
    Predict win probability for Team A vs Team B.

    Returns dict with probabilities for both teams.
    """
    # Find teams (fuzzy match)
    a_row = _find_team(team_a, features_df)
    b_row = _find_team(team_b, features_df)

    if a_row is None:
        return {"error": f"Team not found: {team_a}"}
    if b_row is None:
        return {"error": f"Team not found: {team_b}"}

    # Compute feature differential
    diff = {}
    for col in feature_cols:
        base_col = col.replace("diff_", "")
        a_val = float(a_row.get(base_col, 0) or 0)
        b_val = float(b_row.get(base_col, 0) or 0)
        diff[col] = a_val - b_val

    X = np.array([[diff.get(c, 0) for c in feature_cols]])
    X = np.nan_to_num(X, nan=0.0)
    X = scaler.transform(X)

    prob_a = model.predict_proba(X)[0][1]
    prob_b = 1 - prob_a

    return {
        "team_a": a_row["team"],
        "team_b": b_row["team"],
        "prob_a": round(prob_a * 100, 1),
        "prob_b": round(prob_b * 100, 1),
    }


def _find_team(name: str, df: pd.DataFrame) -> dict | None:
    """Find a team by name (case-insensitive, partial match)."""
    name_lower = name.lower().strip()

    # Exact match first
    exact = df[df["team"].str.lower() == name_lower]
    if len(exact) > 0:
        return exact.iloc[0].to_dict()

    # Partial match
    partial = df[df["team"].str.lower().str.contains(name_lower, na=False)]
    if len(partial) > 0:
        return partial.iloc[0].to_dict()

    return None


def predict_bracket(
    bracket_df: pd.DataFrame,
    features_df: pd.DataFrame,
    model,
    scaler,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Predict an entire bracket.

    bracket_df should have columns: team_a, team_b, seed_a, seed_b, region, round
    """
    results = []

    for _, game in bracket_df.iterrows():
        pred = predict_matchup(
            game["team_a"], game["team_b"],
            features_df, model, scaler, feature_cols
        )

        if "error" not in pred:
            results.append({
                "team_a": pred["team_a"],
                "team_b": pred["team_b"],
                "prob_a": pred["prob_a"],
                "prob_b": pred["prob_b"],
                "predicted_winner": pred["team_a"] if pred["prob_a"] > 50 else pred["team_b"],
                "confidence": max(pred["prob_a"], pred["prob_b"]),
                "region": game.get("region", ""),
                "round": game.get("round", ""),
                "seed_a": game.get("seed_a", ""),
                "seed_b": game.get("seed_b", ""),
            })
        else:
            results.append({
                "team_a": game["team_a"],
                "team_b": game["team_b"],
                "error": pred["error"],
            })

    return pd.DataFrame(results)


def interactive_mode(features_df, model, scaler, feature_cols):
    """Interactive mode: enter matchups one at a time."""
    print("\n" + "=" * 50)
    print("INTERACTIVE MATCHUP PREDICTOR")
    print("=" * 50)
    print("Enter matchups as: Team A vs Team B")
    print("Type 'quit' to exit\n")

    while True:
        try:
            user_input = input("Matchup: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ["quit", "exit", "q"]:
            break

        # Parse "Team A vs Team B" or "Team A, Team B"
        for sep in [" vs ", " vs. ", " v ", ", ", " - "]:
            if sep in user_input:
                parts = user_input.split(sep, 1)
                break
        else:
            print("  Format: Team A vs Team B")
            continue

        team_a, team_b = parts[0].strip(), parts[1].strip()
        pred = predict_matchup(team_a, team_b, features_df, model, scaler, feature_cols)

        if "error" in pred:
            print(f"  ⚠ {pred['error']}")
        else:
            winner = pred["team_a"] if pred["prob_a"] > 50 else pred["team_b"]
            conf = max(pred["prob_a"], pred["prob_b"])
            print(f"\n  {pred['team_a']:>25}  {pred['prob_a']:>5.1f}%")
            print(f"  {pred['team_b']:>25}  {pred['prob_b']:>5.1f}%")
            print(f"  {'Predicted winner:':>25}  {winner} ({conf:.1f}% confidence)")
            print()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Predict March Madness matchups")
    parser.add_argument("--team1", type=str, help="First team")
    parser.add_argument("--team2", type=str, help="Second team")
    parser.add_argument("--bracket", type=str, help="Path to bracket CSV")
    parser.add_argument("--model", default="logistic", help="Model type")
    parser.add_argument("--season", type=int, default=CURRENT_SEASON)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("MARCH MADNESS PREDICTION ENGINE")
    print("=" * 60)

    # Load model and data
    print("\n  Loading model and features...")
    model, scaler, feature_cols = load_model(args.model)
    features_df = load_team_features(args.season)

    if len(features_df) == 0:
        return

    print(f"  ✓ {len(features_df)} teams loaded for season {args.season}")

    if args.team1 and args.team2:
        # Single matchup
        pred = predict_matchup(
            args.team1, args.team2,
            features_df, model, scaler, feature_cols
        )
        if "error" in pred:
            print(f"\n  ⚠ {pred['error']}")
        else:
            print(f"\n  {'─' * 45}")
            print(f"  {pred['team_a']:>25}  {pred['prob_a']:>5.1f}%")
            print(f"  {pred['team_b']:>25}  {pred['prob_b']:>5.1f}%")
            print(f"  {'─' * 45}")

    elif args.bracket:
        # Full bracket
        bracket = pd.read_csv(args.bracket)
        results = predict_bracket(
            bracket, features_df, model, scaler, feature_cols
        )
        outpath = OUTPUT_DIR / "bracket_predictions.csv"
        results.to_csv(outpath, index=False)
        print(f"\n✓ Bracket predictions saved to {outpath}")

        # Print summary
        if "predicted_winner" in results.columns:
            print(f"\n  Predicted Winners:")
            for _, row in results.iterrows():
                if "error" not in row:
                    print(f"    {row['predicted_winner']:>25} ({row['confidence']:.1f}%)"
                          f"  [{row.get('region', '')} {row.get('round', '')}]")

    elif args.interactive:
        interactive_mode(features_df, model, scaler, feature_cols)

    else:
        # Default: interactive
        interactive_mode(features_df, model, scaler, feature_cols)


if __name__ == "__main__":
    main()
