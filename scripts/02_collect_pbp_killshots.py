"""
02 - Collect Play-by-Play Data
================================
Uses CBBpy to pull ESPN play-by-play for all D1 games,
then computes killshot metrics (10-0, 20-0, 30-0 runs).

This is the most data-intensive collection step.
PBP for a full season is ~5,000+ games.

Usage:
    python scripts/02_collect_pbp_killshots.py --season 2025
    python scripts/02_collect_pbp_killshots.py --all

Note: This will take several hours per season due to ESPN rate limits.
      Run overnight or use --resume to pick up where you left off.
"""

import argparse
import time
import sys
import os
import json
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DIR, PROCESSED_DIR, FIRST_SEASON, CURRENT_SEASON,
    ESPN_REQUEST_DELAY,
    KILLSHOT_THRESHOLD, DOUBLE_KILLSHOT_THRESHOLD, TRIPLE_KILLSHOT_THRESHOLD,
    KILLSHOT_WEIGHTS, SECOND_HALF_MULTIPLIER
)


# ============================================================
# STEP 1: Collect game IDs for a season
# ============================================================

def get_season_game_ids(season: int) -> list[str]:
    """
    Get all ESPN game IDs for a season using CBBpy.

    CBBpy season convention: season=2025 means the 2024-25 season.
    Games run roughly from November to April.
    """
    import cbbpy.mens_scraper as ms

    print(f"  Collecting game IDs for {season-1}-{str(season)[2:]} season...")

    all_ids = []
    # NCAA season runs Nov 1 → April 15 (approx)
    start = datetime(season - 1, 11, 1)
    end = datetime(season, 4, 15)

    current = start
    pbar = tqdm(total=(end - start).days, desc="  Scanning dates")
    while current <= end:
        try:
            date_str = current.strftime("%m/%d/%Y")
            ids = ms.get_game_ids(date_str)
            if ids:
                all_ids.extend([str(gid) for gid in ids])
        except Exception:
            pass  # Skip dates with no games (off days, etc.)

        current += timedelta(days=1)
        pbar.update(1)
        time.sleep(0.3)  # Light throttle for date scanning

    pbar.close()
    all_ids = list(set(all_ids))  # Deduplicate
    print(f"  ✓ Found {len(all_ids)} games for season {season}")
    return all_ids


def get_game_pbp(game_id: str) -> pd.DataFrame:
    """
    Pull play-by-play for a single game.

    Returns DataFrame with columns like:
    - game_id, half, time_remaining, play_desc,
      home_score, away_score, scoring_play, etc.
    """
    import cbbpy.mens_scraper as ms

    try:
        pbp = ms.get_game_pbp(game_id)
        if isinstance(pbp, pd.DataFrame) and len(pbp) > 0:
            pbp["game_id"] = game_id
            return pbp
    except Exception as e:
        pass  # Game may not have PBP data available

    return pd.DataFrame()


# ============================================================
# STEP 2: Parse PBP into scoring runs (killshots)
# ============================================================

def detect_killshots(pbp: pd.DataFrame, game_id: str) -> dict:
    """
    Analyze a single game's play-by-play to detect killshot runs.

    A killshot is a scoring run of 10+ unanswered points.
    We track:
    - Who made the run (home or away team)
    - How large the run was
    - When it happened (1st half, 2nd half, OT)

    Returns dict with killshot stats for both teams.
    """
    if pbp is None or len(pbp) == 0:
        return None

    # Standardize column names (CBBpy format varies slightly)
    pbp.columns = [c.lower().strip().replace(" ", "_") for c in pbp.columns]

    # We need score columns — CBBpy typically provides
    # 'home_score' and 'away_score' (cumulative)
    score_cols = _find_score_columns(pbp)
    if score_cols is None:
        return None

    home_col, away_col = score_cols

    # Convert to numeric and forward-fill
    pbp[home_col] = pd.to_numeric(pbp[home_col], errors="coerce").ffill().fillna(0)
    pbp[away_col] = pd.to_numeric(pbp[away_col], errors="coerce").ffill().fillna(0)

    # Detect half/period for pressure weighting
    half_col = _find_half_column(pbp)

    # Walk through the game and track scoring runs
    runs = _find_scoring_runs(pbp, home_col, away_col, half_col)

    # Get team names
    home_team, away_team = _get_team_names(pbp)

    # Aggregate into team-level stats
    result = {
        "game_id": game_id,
        "home_team": home_team,
        "away_team": away_team,
    }

    for team_label, team_name in [("home", home_team), ("away", away_team)]:
        team_runs = [r for r in runs if r["team"] == team_label]
        opp_runs = [r for r in runs if r["team"] != team_label]

        result[f"{team_label}_ks_made"] = sum(1 for r in team_runs if r["points"] >= KILLSHOT_THRESHOLD)
        result[f"{team_label}_ks_allowed"] = sum(1 for r in opp_runs if r["points"] >= KILLSHOT_THRESHOLD)
        result[f"{team_label}_dks_made"] = sum(1 for r in team_runs if r["points"] >= DOUBLE_KILLSHOT_THRESHOLD)
        result[f"{team_label}_dks_allowed"] = sum(1 for r in opp_runs if r["points"] >= DOUBLE_KILLSHOT_THRESHOLD)
        result[f"{team_label}_tks_made"] = sum(1 for r in team_runs if r["points"] >= TRIPLE_KILLSHOT_THRESHOLD)
        result[f"{team_label}_tks_allowed"] = sum(1 for r in opp_runs if r["points"] >= TRIPLE_KILLSHOT_THRESHOLD)

        # Second half killshots
        result[f"{team_label}_ks_made_2h"] = sum(
            1 for r in team_runs
            if r["points"] >= KILLSHOT_THRESHOLD and r.get("is_second_half", False)
        )
        result[f"{team_label}_ks_allowed_2h"] = sum(
            1 for r in opp_runs
            if r["points"] >= KILLSHOT_THRESHOLD and r.get("is_second_half", False)
        )

        # Weighted killshot score
        wks = (
            result[f"{team_label}_ks_made"] * KILLSHOT_WEIGHTS["single"]
            + result[f"{team_label}_dks_made"] * KILLSHOT_WEIGHTS["double"]
            + result[f"{team_label}_tks_made"] * KILLSHOT_WEIGHTS["triple"]
        )
        wks_allowed = (
            result[f"{team_label}_ks_allowed"] * KILLSHOT_WEIGHTS["single"]
            + result[f"{team_label}_dks_allowed"] * KILLSHOT_WEIGHTS["double"]
            + result[f"{team_label}_tks_allowed"] * KILLSHOT_WEIGHTS["triple"]
        )
        result[f"{team_label}_weighted_ks_made"] = wks
        result[f"{team_label}_weighted_ks_allowed"] = wks_allowed
        result[f"{team_label}_weighted_ks_diff"] = wks - wks_allowed

        # Max run length (largest unanswered run in the game)
        if team_runs:
            result[f"{team_label}_max_run"] = max(r["points"] for r in team_runs)
        else:
            result[f"{team_label}_max_run"] = 0

    return result


def _find_score_columns(pbp: pd.DataFrame) -> tuple | None:
    """Find the home and away score columns in the PBP data."""
    possible_home = ["home_score", "home", "score_home", "hscore"]
    possible_away = ["away_score", "away", "score_away", "ascore"]

    home_col = away_col = None
    for col in pbp.columns:
        cl = col.lower().strip()
        if cl in possible_home:
            home_col = col
        elif cl in possible_away:
            away_col = col

    if home_col and away_col:
        return (home_col, away_col)
    return None


def _find_half_column(pbp: pd.DataFrame) -> str | None:
    """Find the half/period column."""
    for col in pbp.columns:
        cl = col.lower().strip()
        if cl in ["half", "period", "half_num", "period_number"]:
            return col
    return None


def _get_team_names(pbp: pd.DataFrame) -> tuple[str, str]:
    """Extract team names from PBP data."""
    for col in pbp.columns:
        cl = col.lower().strip()
        if cl in ["home_team", "home"]:
            home = pbp[col].dropna().iloc[0] if len(pbp[col].dropna()) > 0 else "Home"
            break
    else:
        home = "Home"

    for col in pbp.columns:
        cl = col.lower().strip()
        if cl in ["away_team", "away"]:
            away = pbp[col].dropna().iloc[0] if len(pbp[col].dropna()) > 0 else "Away"
            break
    else:
        away = "Away"

    return str(home), str(away)


def _find_scoring_runs(
    pbp: pd.DataFrame,
    home_col: str,
    away_col: str,
    half_col: str | None
) -> list[dict]:
    """
    Walk through PBP and identify all unanswered scoring runs.

    Logic:
    - Track cumulative score for each team
    - When one team scores and the other doesn't, extend the run
    - When the other team scores, end the current run and start a new one
    - Only record runs of 4+ points (below that is noise)
    """
    runs = []
    current_run_team = None  # "home" or "away"
    current_run_points = 0
    current_run_start_idx = 0

    prev_home = 0
    prev_away = 0

    for idx, row in pbp.iterrows():
        home_score = row[home_col]
        away_score = row[away_col]

        home_scored = home_score - prev_home
        away_scored = away_score - prev_away

        if home_scored > 0 and away_scored == 0:
            # Home team scored
            if current_run_team == "home":
                current_run_points += home_scored
            else:
                # End previous run and start new one
                if current_run_points >= 4 and current_run_team is not None:
                    is_2h = _is_second_half(pbp, current_run_start_idx, half_col)
                    runs.append({
                        "team": current_run_team,
                        "points": current_run_points,
                        "is_second_half": is_2h,
                    })
                current_run_team = "home"
                current_run_points = home_scored
                current_run_start_idx = idx

        elif away_scored > 0 and home_scored == 0:
            # Away team scored
            if current_run_team == "away":
                current_run_points += away_scored
            else:
                if current_run_points >= 4 and current_run_team is not None:
                    is_2h = _is_second_half(pbp, current_run_start_idx, half_col)
                    runs.append({
                        "team": current_run_team,
                        "points": current_run_points,
                        "is_second_half": is_2h,
                    })
                current_run_team = "away"
                current_run_points = away_scored
                current_run_start_idx = idx

        elif home_scored > 0 and away_scored > 0:
            # Both scored on same play row (rare, but handle it)
            # End current run
            if current_run_points >= 4 and current_run_team is not None:
                is_2h = _is_second_half(pbp, current_run_start_idx, half_col)
                runs.append({
                    "team": current_run_team,
                    "points": current_run_points,
                    "is_second_half": is_2h,
                })
            current_run_team = None
            current_run_points = 0

        prev_home = home_score
        prev_away = away_score

    # Don't forget the last run
    if current_run_points >= 4 and current_run_team is not None:
        is_2h = _is_second_half(pbp, current_run_start_idx, half_col)
        runs.append({
            "team": current_run_team,
            "points": current_run_points,
            "is_second_half": is_2h,
        })

    return runs


def _is_second_half(pbp: pd.DataFrame, idx, half_col: str | None) -> bool:
    """Check if a play index is in the second half or OT."""
    if half_col is None:
        # Rough heuristic: second half of the dataframe
        return idx > len(pbp) / 2

    try:
        val = pbp.loc[idx, half_col]
        return int(val) >= 2
    except (ValueError, KeyError):
        return idx > len(pbp) / 2


# ============================================================
# STEP 3: Aggregate killshots to season-level team stats
# ============================================================

def aggregate_killshots_to_teams(game_killshots: list[dict], season: int) -> pd.DataFrame:
    """
    Aggregate game-level killshot data to season-level team stats.

    For each team, compute:
    - Total and per-game killshot metrics
    - Weighted killshot differential
    - Second-half killshot ratio
    """
    team_stats = {}

    for game in game_killshots:
        if game is None:
            continue

        for side in ["home", "away"]:
            team = game[f"{side}_team"]
            if team not in team_stats:
                team_stats[team] = {
                    "games": 0,
                    "ks_made": 0, "ks_allowed": 0,
                    "dks_made": 0, "dks_allowed": 0,
                    "tks_made": 0, "tks_allowed": 0,
                    "ks_made_2h": 0, "ks_allowed_2h": 0,
                    "weighted_ks_made": 0, "weighted_ks_allowed": 0,
                    "max_run": 0,
                }

            stats = team_stats[team]
            stats["games"] += 1
            for key in ["ks_made", "ks_allowed", "dks_made", "dks_allowed",
                         "tks_made", "tks_allowed", "ks_made_2h", "ks_allowed_2h",
                         "weighted_ks_made", "weighted_ks_allowed"]:
                stats[key] += game.get(f"{side}_{key}", 0)
            stats["max_run"] = max(stats["max_run"], game.get(f"{side}_max_run", 0))

    # Convert to DataFrame and compute per-game rates
    rows = []
    for team, stats in team_stats.items():
        g = max(stats["games"], 1)
        row = {
            "team": team,
            "season": season,
            "games": stats["games"],
            "killshot_made_pg": stats["ks_made"] / g,
            "killshot_allowed_pg": stats["ks_allowed"] / g,
            "killshot_diff_pg": (stats["ks_made"] - stats["ks_allowed"]) / g,
            "weighted_ks_made_pg": stats["weighted_ks_made"] / g,
            "weighted_ks_allowed_pg": stats["weighted_ks_allowed"] / g,
            "weighted_ks_diff_pg": (stats["weighted_ks_made"] - stats["weighted_ks_allowed"]) / g,
            "max_run_season": stats["max_run"],
            "ks_made_total": stats["ks_made"],
            "ks_allowed_total": stats["ks_allowed"],
            "dks_made_total": stats["dks_made"],
            "tks_made_total": stats["tks_made"],
        }

        # Second half killshot ratio
        total_ks = stats["ks_made"] + stats["ks_allowed"]
        total_ks_2h = stats["ks_made_2h"] + stats["ks_allowed_2h"]
        if total_ks > 0:
            row["second_half_ks_ratio"] = stats["ks_made_2h"] / max(stats["ks_made"], 1)
        else:
            row["second_half_ks_ratio"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Main execution
# ============================================================

def process_season(season: int, resume: bool = False):
    """Full pipeline for one season: collect PBP → detect killshots → aggregate."""

    # Check for existing progress
    progress_file = RAW_DIR / f"pbp_progress_{season}.json"
    game_ks_file = RAW_DIR / f"killshots_game_level_{season}.csv"

    processed_ids = set()
    game_killshots = []

    if resume and progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
            processed_ids = set(progress.get("processed_ids", []))
            print(f"  Resuming: {len(processed_ids)} games already processed")

    if resume and game_ks_file.exists():
        existing = pd.read_csv(game_ks_file)
        game_killshots = existing.to_dict("records")

    # Step 1: Get game IDs
    ids_file = RAW_DIR / f"game_ids_{season}.json"
    if ids_file.exists():
        with open(ids_file) as f:
            game_ids = json.load(f)
        print(f"  Loaded {len(game_ids)} cached game IDs")
    else:
        game_ids = get_season_game_ids(season)
        with open(ids_file, "w") as f:
            json.dump(game_ids, f)

    remaining = [gid for gid in game_ids if gid not in processed_ids]
    print(f"  {len(remaining)} games to process ({len(processed_ids)} already done)")

    # Step 2: Pull PBP and detect killshots
    for i, game_id in enumerate(tqdm(remaining, desc=f"  Season {season} PBP")):
        try:
            pbp = get_game_pbp(game_id)
            if len(pbp) > 0:
                ks = detect_killshots(pbp, game_id)
                if ks is not None:
                    game_killshots.append(ks)
        except Exception as e:
            pass  # Skip problematic games silently

        processed_ids.add(game_id)

        # Save progress every 100 games
        if (i + 1) % 100 == 0:
            with open(progress_file, "w") as f:
                json.dump({"processed_ids": list(processed_ids)}, f)
            if game_killshots:
                pd.DataFrame(game_killshots).to_csv(game_ks_file, index=False)
            print(f"    Checkpoint: {len(processed_ids)}/{len(game_ids)} games")

        time.sleep(ESPN_REQUEST_DELAY)

    # Save final game-level results
    if game_killshots:
        game_df = pd.DataFrame(game_killshots)
        game_df.to_csv(game_ks_file, index=False)
        print(f"  ✓ Saved {len(game_df)} game-level killshot records")

    # Step 3: Aggregate to team level
    team_df = aggregate_killshots_to_teams(game_killshots, season)
    team_file = PROCESSED_DIR / f"killshots_team_{season}.csv"
    team_df.to_csv(team_file, index=False)
    print(f"  ✓ Saved {len(team_df)} team killshot profiles to {team_file}")

    return team_df


def main():
    parser = argparse.ArgumentParser(description="Collect PBP data and compute killshots")
    parser.add_argument("--season", type=int, default=None,
                        help="Process a single season")
    parser.add_argument("--all", action="store_true",
                        help="Process all seasons in training window")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    print("=" * 60)
    print("PLAY-BY-PLAY COLLECTION & KILLSHOT DETECTION")
    print("=" * 60)
    print()
    print("⚠  This script pulls PBP for every D1 game via ESPN.")
    print("   A full season takes several hours. Use --resume to")
    print("   pick up where you left off if interrupted.")
    print()

    if args.season:
        seasons = [args.season]
    elif args.all:
        seasons = list(range(FIRST_SEASON, CURRENT_SEASON + 1))
    else:
        # Default: just current season
        seasons = [CURRENT_SEASON]

    all_team_dfs = []
    for season in seasons:
        print(f"\n{'─' * 40}")
        print(f"Processing season {season}")
        print(f"{'─' * 40}")
        df = process_season(season, resume=args.resume)
        all_team_dfs.append(df)

    # Combine all seasons
    if all_team_dfs:
        combined = pd.concat(all_team_dfs, ignore_index=True)
        outpath = PROCESSED_DIR / "killshots_all_seasons.csv"
        combined.to_csv(outpath, index=False)
        print(f"\n✓ Combined killshot data: {len(combined)} team-seasons → {outpath}")


if __name__ == "__main__":
    main()
