"""
03 - Collect Recruiting & Roster Data
========================================
Scrapes 247Sports composite rankings and Barttorvik continuity data
to build the Roster Construction Index.

Usage:
    python scripts/03_collect_roster_data.py
    python scripts/03_collect_roster_data.py --season 2025
"""

import argparse
import time
import sys
import os
import re

import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DIR, PROCESSED_DIR, FIRST_SEASON, CURRENT_SEASON,
    REQUEST_DELAY_SECONDS, TORVIK_BASE_URL,
    BLUE_CHIP_CUTOFF, SCHOLARSHIP_PLAYERS,
    NIL_SUPERTEAM_THRESHOLD, NIL_SUPERTEAM_RECRUIT_CUTOFF
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}


# ============================================================
# Part A: Barttorvik Continuity Data
# ============================================================

def scrape_torvik_continuity(season: int) -> pd.DataFrame:
    """
    Barttorvik tracks returning minutes % for each team.
    This is available on the pre-season rankings page.

    Continuity = % of prior season's minutes that return.
    High continuity → team chemistry advantage.
    """
    url = f"{TORVIK_BASE_URL}/trankpre.php?year={season}&conlimit=All"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        table = soup.find("table")
        if table is None:
            return pd.DataFrame()

        rows = table.find_all("tr")
        data = []
        for row in rows[1:]:  # Skip header
            cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if len(cells) >= 3:
                data.append(cells)

        if not data:
            return pd.DataFrame()

        # Headers vary, but typically include team, conf, returning minutes, etc.
        header_cells = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
        max_cols = max(len(header_cells), max(len(r) for r in data))
        while len(header_cells) < max_cols:
            header_cells.append(f"col_{len(header_cells)}")

        data = [r + [""] * (max_cols - len(r)) for r in data]
        df = pd.DataFrame(data, columns=header_cells[:max_cols])
        df["season"] = season

        # Try to identify the continuity/returning minutes column
        for col in df.columns:
            cl = col.lower().strip()
            if any(kw in cl for kw in ["ret", "cont", "return", "min%"]):
                df = df.rename(columns={col: "continuity_pct"})
                break

        return df

    except Exception as e:
        print(f"  ⚠ Could not get continuity for {season}: {e}")
        return pd.DataFrame()


# ============================================================
# Part B: 247Sports Recruiting Rankings
# ============================================================

def scrape_247_team_rankings(season: int) -> pd.DataFrame:
    """
    Scrape 247Sports team recruiting class rankings.

    The recruiting class for the 2024-25 season (season=2025) is
    the high school class of 2024.

    We pull: team, number of 5-star, 4-star, 3-star recruits,
    and the composite ranking of each recruit.
    """
    # 247's team rankings URL pattern
    recruit_year = season - 1  # Class of 2024 plays in 2024-25 season
    url = f"https://247sports.com/Season/{recruit_year}-Basketball/CompositeTeamRankings/"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        teams = []
        # 247 uses specific CSS classes for ranking lists
        ranking_items = soup.find_all("li", class_=re.compile(r"rankings-page__list-item"))

        if not ranking_items:
            # Alternative: look for the ranking table structure
            ranking_items = soup.find_all("div", class_=re.compile(r"team"))

        for item in ranking_items:
            team_data = _parse_247_team_item(item, recruit_year)
            if team_data:
                teams.append(team_data)

        if teams:
            df = pd.DataFrame(teams)
            df["season"] = season
            return df

    except Exception as e:
        print(f"  ⚠ Could not scrape 247 for class of {recruit_year}: {e}")

    return pd.DataFrame()


def _parse_247_team_item(item, recruit_year: int) -> dict | None:
    """Parse a single team's recruiting class from 247Sports HTML."""
    try:
        # Try to find team name
        team_el = item.find(["a", "span"], class_=re.compile(r"team|name|school"))
        if team_el:
            team_name = team_el.get_text(strip=True)
        else:
            return None

        # Try to find star counts
        stars = {"5star": 0, "4star": 0, "3star": 0}
        star_els = item.find_all("span", class_=re.compile(r"star"))
        for el in star_els:
            text = el.get_text(strip=True)
            if "5" in text:
                stars["5star"] += 1
            elif "4" in text:
                stars["4star"] += 1
            elif "3" in text:
                stars["3star"] += 1

        # Try to find total points / rank
        points_el = item.find(["span", "div"], class_=re.compile(r"points|score"))
        points = 0
        if points_el:
            try:
                points = float(points_el.get_text(strip=True).replace(",", ""))
            except ValueError:
                pass

        return {
            "team": team_name,
            "recruit_year": recruit_year,
            "five_star": stars["5star"],
            "four_star": stars["4star"],
            "three_star": stars["3star"],
            "class_points": points,
        }
    except Exception:
        return None


# ============================================================
# Part C: Build Roster Construction Index
# ============================================================

def build_roster_construction_index(
    continuity_df: pd.DataFrame,
    recruiting_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    season: int
) -> pd.DataFrame:
    """
    Combine continuity, recruiting, and experience data into
    the Roster Construction Index.

    For teams where we can't get individual recruit rankings,
    we use the team-level class ranking as a proxy.

    RCI = w1*blue_chip_ratio + w2*experience + w3*continuity + w4*nil_flag
    (weights determined by the model during training)
    """
    teams = set()
    if len(ratings_df) > 0 and "team" in ratings_df.columns:
        teams = set(ratings_df[ratings_df.get("season", season) == season]["team"])
    if not teams and len(continuity_df) > 0 and "team" in continuity_df.columns:
        teams = set(continuity_df["team"])

    rows = []
    for team in teams:
        row = {"team": team, "season": season}

        # Continuity
        if len(continuity_df) > 0 and "team" in continuity_df.columns:
            team_cont = continuity_df[continuity_df["team"].str.contains(team, case=False, na=False)]
            if len(team_cont) > 0 and "continuity_pct" in team_cont.columns:
                try:
                    val = team_cont.iloc[0]["continuity_pct"]
                    row["continuity_index"] = float(str(val).replace("%", "")) / 100
                except (ValueError, TypeError):
                    row["continuity_index"] = None
            else:
                row["continuity_index"] = None
        else:
            row["continuity_index"] = None

        # Recruiting: blue chip ratio
        # We look at the CURRENT roster, which includes recruits from
        # multiple classes. Ideally we'd have per-player ranks, but
        # team-class-level data gives us a good proxy.
        #
        # Approach: Sum up blue chips from the last 4 recruiting classes
        # (4 years of eligibility, 5 with COVID year)
        blue_chips = 0
        total_recruits = 0
        if len(recruiting_df) > 0:
            for offset in range(0, 5):  # Last 5 recruiting classes
                class_year = season - 1 - offset
                class_data = recruiting_df[
                    (recruiting_df.get("recruit_year", -1) == class_year) &
                    (recruiting_df["team"].str.contains(team, case=False, na=False))
                ]
                if len(class_data) > 0:
                    blue_chips += class_data.iloc[0].get("five_star", 0)
                    blue_chips += class_data.iloc[0].get("four_star", 0)
                    total_recruits += (
                        class_data.iloc[0].get("five_star", 0) +
                        class_data.iloc[0].get("four_star", 0) +
                        class_data.iloc[0].get("three_star", 0)
                    )

        row["blue_chip_count"] = blue_chips
        row["blue_chip_ratio"] = blue_chips / SCHOLARSHIP_PLAYERS

        # Experience score proxy: use continuity as a stand-in
        # (higher continuity = more experienced roster)
        row["experience_score"] = row.get("continuity_index", 0.5) or 0.5

        # NIL consolidation flag
        # Proxy: teams with 3+ five-star recruits across recent classes
        five_stars = 0
        if len(recruiting_df) > 0:
            for offset in range(0, 5):
                class_year = season - 1 - offset
                class_data = recruiting_df[
                    (recruiting_df.get("recruit_year", -1) == class_year) &
                    (recruiting_df["team"].str.contains(team, case=False, na=False))
                ]
                if len(class_data) > 0:
                    five_stars += class_data.iloc[0].get("five_star", 0)

        row["nil_consolidation_flag"] = 1 if five_stars >= NIL_SUPERTEAM_THRESHOLD else 0

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Main execution
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Collect roster/recruiting data")
    parser.add_argument("--season", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("ROSTER CONSTRUCTION DATA COLLECTION")
    print("=" * 60)

    seasons = [args.season] if args.season else list(range(FIRST_SEASON, CURRENT_SEASON + 1))

    # Load existing ratings data if available (for team list)
    ratings_path = RAW_DIR / "torvik_ratings_all.csv"
    if ratings_path.exists():
        ratings_df = pd.read_csv(ratings_path)
        print(f"  Loaded {len(ratings_df)} team-seasons from Torvik ratings")
    else:
        ratings_df = pd.DataFrame()
        print("  ⚠ No Torvik ratings found — run 01_collect_torvik_ratings.py first")

    all_continuity = []
    all_recruiting = []
    all_rci = []

    for season in tqdm(seasons, desc="Collecting roster data"):
        print(f"\n  Season {season}:")

        # Continuity
        cont_df = scrape_torvik_continuity(season)
        if len(cont_df) > 0:
            all_continuity.append(cont_df)
            print(f"    ✓ Continuity: {len(cont_df)} teams")
        time.sleep(REQUEST_DELAY_SECONDS)

        # Recruiting
        rec_df = scrape_247_team_rankings(season)
        if len(rec_df) > 0:
            all_recruiting.append(rec_df)
            print(f"    ✓ Recruiting: {len(rec_df)} teams")
        time.sleep(REQUEST_DELAY_SECONDS)

    # Combine
    continuity_combined = pd.concat(all_continuity, ignore_index=True) if all_continuity else pd.DataFrame()
    recruiting_combined = pd.concat(all_recruiting, ignore_index=True) if all_recruiting else pd.DataFrame()

    # Save raw data
    if len(continuity_combined) > 0:
        continuity_combined.to_csv(RAW_DIR / "continuity_all.csv", index=False)
    if len(recruiting_combined) > 0:
        recruiting_combined.to_csv(RAW_DIR / "recruiting_all.csv", index=False)

    # Build RCI for each season
    for season in seasons:
        cont_season = continuity_combined[continuity_combined["season"] == season] if len(continuity_combined) > 0 else pd.DataFrame()
        rci = build_roster_construction_index(cont_season, recruiting_combined, ratings_df, season)
        if len(rci) > 0:
            all_rci.append(rci)

    if all_rci:
        rci_combined = pd.concat(all_rci, ignore_index=True)
        outpath = PROCESSED_DIR / "roster_construction_index.csv"
        rci_combined.to_csv(outpath, index=False)
        print(f"\n✓ Roster Construction Index: {len(rci_combined)} team-seasons → {outpath}")

    print("\nNOTE: For the most accurate blue chip ratios, consider")
    print("  supplementing with manual per-player data from 247sports.com.")
    print("  The team-class-level approach is a solid proxy but misses")
    print("  individual transfer portal additions with their original ranks.")


if __name__ == "__main__":
    main()
