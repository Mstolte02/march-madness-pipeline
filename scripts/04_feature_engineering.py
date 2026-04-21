"""
04 - Feature Engineering
=========================
Combines all collected data into the four-pillar feature set,
performs era-adjustment, and creates matchup-level training data.

Run AFTER scripts 01-03 have collected the raw data.

Usage:
    python scripts/04_feature_engineering.py
"""

import sys
import os

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DIR, PROCESSED_DIR, TOURNEY_YEARS,
    KENPOM_FEATURES, ROSTER_FEATURES, KILLSHOT_FEATURES,
    CONTEXT_FEATURES, ALL_FEATURES
)


# ============================================================
# Pillar 1: Era-Adjusted KenPom (Barttorvik)
# ============================================================

def compute_era_adjusted_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score AdjEM, AdjO, AdjD within each season to normalize
    across eras. A +25 AdjEM in 2015 ≠ +25 AdjEM in 2025.

    Also computes luck penalty (regress luck toward 0).
    """
    print("  Computing era-adjusted ratings...")

    numeric_cols = ["adj_o", "adj_d", "adj_em", "adj_t", "barthag"]
    for col in numeric_cols:
        if col in ratings_df.columns:
            ratings_df[col] = pd.to_numeric(ratings_df[col], errors="coerce")

    # Compute AdjEM if missing
    if "adj_em" not in ratings_df.columns and "adj_o" in ratings_df.columns:
        ratings_df["adj_em"] = ratings_df["adj_o"] - ratings_df["adj_d"]

    # Z-score within each season
    z_scored = []
    for season, group in ratings_df.groupby("season"):
        g = group.copy()
        for col, z_col in [("adj_em", "z_adj_em"), ("adj_o", "z_adj_o"),
                            ("adj_d", "z_adj_d"), ("adj_t", "z_adj_t")]:
            if col in g.columns:
                mean = g[col].mean()
                std = g[col].std()
                if std > 0:
                    g[z_col] = (g[col] - mean) / std
                else:
                    g[z_col] = 0.0
            else:
                g[z_col] = 0.0

        # Note: for AdjD, LOWER is better, so we flip the sign
        # so that positive z-score = better defense
        g["z_adj_d"] = -g["z_adj_d"]

        z_scored.append(g)

    result = pd.concat(z_scored, ignore_index=True)

    # Luck penalty: KenPom/Barttorvik luck measures overperformance
    # in close games. For tournament prediction, regress toward 0.
    if "luck" in result.columns:
        result["luck"] = pd.to_numeric(result["luck"], errors="coerce").fillna(0)
        # Penalty = negative of luck (lucky teams get penalized)
        result["luck_penalty"] = -result["luck"] * 0.5  # 50% regression
    else:
        result["luck_penalty"] = 0.0

    # SOS rank (strength of schedule)
    if "sos" in result.columns:
        result["sos_rank"] = pd.to_numeric(result["sos"], errors="coerce")
    elif "sos_rank" not in result.columns:
        result["sos_rank"] = 0.0

    print(f"    ✓ Era-adjusted {len(result)} team-seasons")
    return result


# ============================================================
# Pillar 2: Roster Construction Index
# ============================================================

def load_roster_features(rci_path=None) -> pd.DataFrame:
    """Load pre-computed roster construction index."""
    if rci_path is None:
        rci_path = PROCESSED_DIR / "roster_construction_index.csv"

    if rci_path.exists():
        df = pd.read_csv(rci_path)
        print(f"  Loaded roster construction data: {len(df)} team-seasons")

        # Fill missing values with league averages
        for col in ["blue_chip_ratio", "experience_score",
                     "continuity_index", "nil_consolidation_flag"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median())

        return df
    else:
        print("  ⚠ No roster construction data found")
        return pd.DataFrame()


# ============================================================
# Pillar 3: Killshot Metrics
# ============================================================

def load_killshot_features(ks_path=None) -> pd.DataFrame:
    """Load pre-computed killshot metrics."""
    if ks_path is None:
        ks_path = PROCESSED_DIR / "killshots_all_seasons.csv"

    if ks_path.exists():
        df = pd.read_csv(ks_path)
        print(f"  Loaded killshot data: {len(df)} team-seasons")
        return df
    else:
        print("  ⚠ No killshot data found")
        return pd.DataFrame()


# ============================================================
# Pillar 4: Tournament Context
# ============================================================

# Historical seed win rates (1985-present aggregated)
SEED_WIN_RATES = {
    # (higher_seed, lower_seed): win probability for higher seed
    (1, 16): 0.993, (1, 8): 0.795, (1, 9): 0.857,
    (1, 4): 0.716, (1, 5): 0.667, (1, 12): 0.917,
    (1, 13): 0.958, (2, 15): 0.942, (2, 7): 0.667,
    (2, 10): 0.769, (2, 3): 0.520, (2, 6): 0.625,
    (2, 11): 0.833, (3, 14): 0.849, (3, 6): 0.583,
    (3, 11): 0.690, (3, 7): 0.536, (4, 13): 0.794,
    (4, 5): 0.556, (4, 12): 0.692, (4, 8): 0.571,
    (4, 9): 0.625, (5, 12): 0.648, (5, 4): 0.444,
    (5, 13): 0.750, (6, 11): 0.628, (6, 3): 0.417,
    (6, 14): 0.800, (7, 10): 0.604, (7, 2): 0.333,
    (7, 15): 0.800, (8, 9): 0.514, (8, 1): 0.205,
    (9, 8): 0.486, (9, 1): 0.143, (10, 7): 0.396,
    (10, 2): 0.231, (11, 6): 0.372, (11, 3): 0.310,
    (12, 5): 0.352, (12, 4): 0.308, (13, 4): 0.206,
    (13, 5): 0.250, (14, 3): 0.151, (15, 2): 0.058,
    (16, 1): 0.007,
}


def get_seed_win_rate(seed_a: int, seed_b: int) -> float:
    """Get historical win rate for seed_a vs seed_b."""
    key = (seed_a, seed_b)
    if key in SEED_WIN_RATES:
        return SEED_WIN_RATES[key]
    # Reverse lookup
    reverse_key = (seed_b, seed_a)
    if reverse_key in SEED_WIN_RATES:
        return 1.0 - SEED_WIN_RATES[reverse_key]
    # Default: favor the better (lower) seed slightly
    if seed_a < seed_b:
        return 0.55
    elif seed_a > seed_b:
        return 0.45
    return 0.50


def compute_tournament_context(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute tournament context features for teams with NCAA seeds.

    Features:
    - seed: raw NCAA tournament seed
    - seed_historical_winrate: base rate for that seed reaching R32
    - hot_hand_index: placeholder (needs game-by-game data)
    - close_game_winrate: placeholder (needs game results)
    - neutral_court_adj: placeholder
    - quad1_wins: from Torvik data if available
    """
    print("  Computing tournament context features...")

    if "ncaa_seed" not in ratings_df.columns:
        print("    ⚠ No seed data — will be populated at tournament time")
        return pd.DataFrame()

    tourney = ratings_df[ratings_df["ncaa_seed"].notna()].copy()
    tourney["seed"] = pd.to_numeric(tourney["ncaa_seed"], errors="coerce")
    tourney = tourney.dropna(subset=["seed"])
    tourney["seed"] = tourney["seed"].astype(int)

    # Seed-based historical win rate (for first round as baseline)
    # Use the R64 opponent's likely seed
    seed_opponent_map = {1: 16, 2: 15, 3: 14, 4: 13, 5: 12, 6: 11,
                          7: 10, 8: 9, 9: 8, 10: 7, 11: 6, 12: 5,
                          13: 4, 14: 3, 15: 2, 16: 1}
    tourney["seed_historical_winrate"] = tourney["seed"].map(
        lambda s: get_seed_win_rate(s, seed_opponent_map.get(s, 8))
    )

    # Placeholder features — will be enriched with game-level data
    tourney["hot_hand_index"] = 0.0
    tourney["close_game_winrate"] = 0.5
    tourney["neutral_court_adj"] = 0.0
    tourney["quad1_wins"] = 0.0

    # Try to pull Q1 wins from WAB or other Torvik data
    if "wab" in tourney.columns:
        tourney["quad1_wins"] = pd.to_numeric(tourney["wab"], errors="coerce").fillna(0)

    print(f"    ✓ Tournament context for {len(tourney)} teams")
    return tourney


# ============================================================
# Merge all pillars → matchup-level training data
# ============================================================

def merge_all_features(
    kenpom_df: pd.DataFrame,
    roster_df: pd.DataFrame,
    killshot_df: pd.DataFrame,
    context_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all four pillars into a single team-season feature table.
    Uses team name + season as the join key.
    """
    print("\n  Merging all feature pillars...")

    # Start with KenPom (most complete dataset)
    merged = kenpom_df[["team", "season"] + [c for c in KENPOM_FEATURES if c in kenpom_df.columns]].copy()

    # Merge roster features
    if len(roster_df) > 0:
        roster_cols = ["team", "season"] + [c for c in ROSTER_FEATURES if c in roster_df.columns]
        roster_sub = roster_df[[c for c in roster_cols if c in roster_df.columns]]
        merged = merged.merge(roster_sub, on=["team", "season"], how="left")

    # Merge killshot features
    if len(killshot_df) > 0:
        ks_cols = ["team", "season"] + [c for c in KILLSHOT_FEATURES if c in killshot_df.columns]
        ks_sub = killshot_df[[c for c in ks_cols if c in killshot_df.columns]]
        merged = merged.merge(ks_sub, on=["team", "season"], how="left")

    # Merge tournament context (only for tournament teams)
    if len(context_df) > 0:
        ctx_cols = ["team", "season"] + [c for c in CONTEXT_FEATURES if c in context_df.columns]
        ctx_sub = context_df[[c for c in ctx_cols if c in context_df.columns]]
        merged = merged.merge(ctx_sub, on=["team", "season"], how="left")

    # Fill missing values with reasonable defaults
    for col in merged.columns:
        if col in ["team", "season"]:
            continue
        if merged[col].dtype in [np.float64, np.int64, float, int]:
            merged[col] = merged[col].fillna(merged[col].median())

    print(f"    ✓ Merged dataset: {len(merged)} team-seasons × {len(merged.columns)} features")
    return merged


def create_matchup_training_data(
    team_features: pd.DataFrame,
    tournament_results: pd.DataFrame
) -> pd.DataFrame:
    """
    Create matchup-level training data for the model.

    For each historical tournament game, compute:
    - Feature DIFFERENCE: Team A features - Team B features
    - Target: did Team A win? (1 or 0)

    This is the core modeling insight: we predict based on
    relative strength, not absolute.
    """
    print("\n  Creating matchup-level training data...")

    if len(tournament_results) == 0:
        print("    ⚠ No tournament results — load manually")
        return pd.DataFrame()

    matchups = []
    feature_cols = [c for c in team_features.columns
                    if c not in ["team", "season", "ncaa_seed", "seed", "rank",
                                  "record", "conf", "barthag"]]

    for _, game in tournament_results.iterrows():
        season = game["season"]
        winner = game["winner"]
        loser = game["loser"]

        # Get feature vectors
        w_feats = team_features[
            (team_features["team"] == winner) & (team_features["season"] == season)
        ]
        l_feats = team_features[
            (team_features["team"] == loser) & (team_features["season"] == season)
        ]

        if len(w_feats) == 0 or len(l_feats) == 0:
            continue

        w_row = w_feats.iloc[0]
        l_row = l_feats.iloc[0]

        # Create two rows per game (both perspectives) for balanced training
        # Winner perspective: diff = winner - loser, target = 1
        matchup_w = {"season": season, "team_a": winner, "team_b": loser, "target": 1}
        for col in feature_cols:
            if col in w_row.index and col in l_row.index:
                matchup_w[f"diff_{col}"] = float(w_row[col]) - float(l_row[col])

        # Loser perspective: diff = loser - winner, target = 0
        matchup_l = {"season": season, "team_a": loser, "team_b": winner, "target": 0}
        for col in feature_cols:
            if col in w_row.index and col in l_row.index:
                matchup_l[f"diff_{col}"] = float(l_row[col]) - float(w_row[col])

        matchups.append(matchup_w)
        matchups.append(matchup_l)

    result = pd.DataFrame(matchups)
    print(f"    ✓ Created {len(result)} matchup rows from "
          f"{len(result) // 2} tournament games")
    return result


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    # Load raw data
    print("\n[1/5] Loading Torvik ratings...")
    ratings_path = RAW_DIR / "torvik_ratings_all.csv"
    if ratings_path.exists():
        ratings = pd.read_csv(ratings_path)
    else:
        print("  ⚠ Missing! Run 01_collect_torvik_ratings.py first.")
        ratings = pd.DataFrame()

    # Pillar 1: Era-adjusted KenPom
    print("\n[2/5] Computing era-adjusted efficiency...")
    if len(ratings) > 0:
        kenpom = compute_era_adjusted_ratings(ratings)
    else:
        kenpom = pd.DataFrame()

    # Pillar 2: Roster Construction
    print("\n[3/5] Loading roster construction index...")
    roster = load_roster_features()

    # Pillar 3: Killshots
    print("\n[4/5] Loading killshot metrics...")
    killshots = load_killshot_features()

    # Pillar 4: Tournament Context
    print("\n[5/5] Computing tournament context...")
    context = compute_tournament_context(kenpom) if len(kenpom) > 0 else pd.DataFrame()

    # Merge everything
    if len(kenpom) > 0:
        team_features = merge_all_features(kenpom, roster, killshots, context)
        team_features.to_csv(PROCESSED_DIR / "team_features_all.csv", index=False)
        print(f"\n✓ Team features saved: {PROCESSED_DIR / 'team_features_all.csv'}")

        # Create matchup training data if tournament results exist
        tourney_path = RAW_DIR / "tournament_results.csv"
        if tourney_path.exists():
            tourney_results = pd.read_csv(tourney_path)
            matchup_data = create_matchup_training_data(team_features, tourney_results)
            if len(matchup_data) > 0:
                matchup_data.to_csv(PROCESSED_DIR / "matchup_training_data.csv", index=False)
                print(f"✓ Matchup training data saved: {PROCESSED_DIR / 'matchup_training_data.csv'}")
        else:
            print("\n⚠ No tournament results found at data/raw/tournament_results.csv")
            print("  You'll need a CSV with columns: season, round, winner, loser, winner_seed, loser_seed")
            print("  Download from: https://www.sports-reference.com/cbb/postseason/")
    else:
        print("\n⚠ Cannot build features without Torvik ratings data.")


if __name__ == "__main__":
    main()
