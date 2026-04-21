"""
01 - Collect Barttorvik Team Ratings
=====================================
Scrapes T-Rank data (AdjEM, AdjO, AdjD, AdjT, Luck, SOS, etc.)
for every D1 team across all seasons in our training window.

Usage:
    python scripts/01_collect_torvik_ratings.py
    python scripts/01_collect_torvik_ratings.py --season 2025
"""

import argparse
import time
import sys
import os
import json
import re

import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DIR, FIRST_SEASON, CURRENT_SEASON,
    TORVIK_BASE_URL, REQUEST_DELAY_SECONDS
)


def scrape_torvik_ratings(season: int) -> pd.DataFrame:
    """
    Scrape the T-Rank ratings page for a given season.

    Barttorvik's trank.php returns an HTML table with columns like:
    Rank, Team, Conf, Record, AdjOE, AdjDE, Barthag, EFG%, EFGD%,
    TOR, TORD, ORB, DRB, FTR, FTRD, 2P%, 2PD%, 3P%, 3PD%, AdjT,
    WAB, Seed, etc.

    We also need to grab SOS and Luck from the team detail pages
    or from the extended table view.
    """
    url = f"{TORVIK_BASE_URL}/trank.php?year={season}&conlimit=All"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }

    print(f"  Fetching T-Rank ratings for {season}...")
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    # Barttorvik loads data via JavaScript in some cases.
    # The main ratings are often embedded in a JSON-like structure
    # or rendered in an HTML table. We try both approaches.

    soup = BeautifulSoup(resp.text, "lxml")

    # Approach 1: Look for the main data table
    table = soup.find("table", {"id": "trank-table"})
    if table is None:
        # Try alternative table selectors
        table = soup.find("table")

    if table is not None:
        df = _parse_html_table(table, season)
        if df is not None and len(df) > 0:
            return df

    # Approach 2: Barttorvik often embeds data in JavaScript variables.
    # Look for patterns like: var defined = [{...}, {...}]
    scripts = soup.find_all("script")
    for script in scripts:
        if script.string and "teamData" in script.string:
            df = _parse_js_data(script.string, season)
            if df is not None and len(df) > 0:
                return df

    # Approach 3: Use the direct CSV/JSON export if available
    # Barttorvik has a getteam endpoint
    return _scrape_via_api(season, headers)


def _parse_html_table(table, season: int) -> pd.DataFrame:
    """Parse an HTML table element into a DataFrame."""
    rows = table.find_all("tr")
    if len(rows) < 2:
        return None

    # Extract headers
    header_row = rows[0]
    headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]

    # Extract data
    data = []
    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        if len(cells) >= 5:  # Minimum valid row
            row_data = [cell.get_text(strip=True) for cell in cells]
            data.append(row_data)

    if not data:
        return None

    # Build DataFrame — handle mismatched column counts
    max_cols = max(len(headers), max(len(r) for r in data))
    while len(headers) < max_cols:
        headers.append(f"col_{len(headers)}")
    data = [r + [""] * (max_cols - len(r)) for r in data]

    df = pd.DataFrame(data, columns=headers[:max_cols])
    df["season"] = season
    return df


def _parse_js_data(script_text: str, season: int) -> pd.DataFrame:
    """Extract team data from embedded JavaScript."""
    # Look for JSON array pattern
    match = re.search(r'var\s+\w+\s*=\s*(\[.*?\]);', script_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            df = pd.DataFrame(data)
            df["season"] = season
            return df
        except json.JSONDecodeError:
            pass
    return None


def _scrape_via_api(season: int, headers: dict) -> pd.DataFrame:
    """
    Fallback: Scrape team-by-team from Barttorvik's team detail pages,
    or use the alternative data endpoint.

    Barttorvik has endpoints like:
    https://barttorvik.com/team-tables_json.php?year=2025&type=pointed
    """
    json_urls = [
        f"{TORVIK_BASE_URL}/getteam.php?year={season}",
        f"{TORVIK_BASE_URL}/team-tables_json.php?year={season}&type=pointed",
    ]

    for url in json_urls:
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame(data)
                    df["season"] = season
                    print(f"  ✓ Got {len(df)} teams via API endpoint")
                    return df
        except (requests.RequestException, json.JSONDecodeError, ValueError):
            continue

    print(f"  ⚠ Could not scrape season {season} — you may need to "
          f"download manually from barttorvik.com")
    return pd.DataFrame()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names across different scraping approaches.
    Map common Barttorvik column names to our standard schema.
    """
    col_map = {
        # Common Barttorvik column names → our standard names
        "rk": "rank", "rank": "rank",
        "team": "team", "Team": "team",
        "conf": "conf", "Conf": "conf",
        "rec": "record", "Record": "record",
        "adjoe": "adj_o", "AdjOE": "adj_o", "adj_o": "adj_o",
        "adjde": "adj_d", "AdjDE": "adj_d", "adj_d": "adj_d",
        "barthag": "barthag", "Barthag": "barthag",
        "adjt": "adj_t", "AdjT": "adj_t", "adj_t": "adj_t",
        "wab": "wab", "WAB": "wab",
        "luck": "luck", "Luck": "luck",
        "sos": "sos", "SOS": "sos",
        "efg": "efg_pct", "EFG%": "efg_pct",
        "efgd": "efg_d_pct", "EFGD%": "efg_d_pct",
        "tor": "to_rate", "TORD": "to_rate_d",
        "orb": "orb_rate", "drb": "drb_rate",
        "ftr": "ft_rate", "ftrd": "ft_rate_d",
        "seed": "ncaa_seed", "Seed": "ncaa_seed",
    }

    renamed = {}
    for col in df.columns:
        clean = col.strip().lower().replace(" ", "").replace("%", "")
        if clean in col_map:
            renamed[col] = col_map[clean]
        elif col in col_map:
            renamed[col] = col_map[col]

    df = df.rename(columns=renamed)

    # Compute AdjEM if we have AdjO and AdjD
    if "adj_o" in df.columns and "adj_d" in df.columns:
        try:
            df["adj_o"] = pd.to_numeric(df["adj_o"], errors="coerce")
            df["adj_d"] = pd.to_numeric(df["adj_d"], errors="coerce")
            df["adj_em"] = df["adj_o"] - df["adj_d"]
        except Exception:
            pass

    return df


def collect_all_seasons(seasons: list[int] = None) -> pd.DataFrame:
    """Collect and combine ratings for all seasons."""
    if seasons is None:
        seasons = list(range(FIRST_SEASON, CURRENT_SEASON + 1))

    all_dfs = []
    for season in tqdm(seasons, desc="Collecting Torvik ratings"):
        try:
            df = scrape_torvik_ratings(season)
            if df is not None and len(df) > 0:
                df = standardize_columns(df)
                all_dfs.append(df)
                print(f"  ✓ Season {season}: {len(df)} teams")
            else:
                print(f"  ✗ Season {season}: no data")
        except Exception as e:
            print(f"  ✗ Season {season} error: {e}")

        time.sleep(REQUEST_DELAY_SECONDS)

    if not all_dfs:
        print("\n⚠ No data collected. Check your internet connection and "
              "try again, or download CSVs manually from barttorvik.com")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def main():
    parser = argparse.ArgumentParser(description="Collect Barttorvik T-Rank ratings")
    parser.add_argument("--season", type=int, default=None,
                        help="Collect a single season (e.g., 2025)")
    args = parser.parse_args()

    print("=" * 60)
    print("BARTTORVIK T-RANK DATA COLLECTION")
    print("=" * 60)

    if args.season:
        seasons = [args.season]
    else:
        seasons = list(range(FIRST_SEASON, CURRENT_SEASON + 1))
        print(f"Collecting seasons {FIRST_SEASON}–{CURRENT_SEASON}")

    df = collect_all_seasons(seasons)

    if len(df) > 0:
        outpath = RAW_DIR / "torvik_ratings_all.csv"
        df.to_csv(outpath, index=False)
        print(f"\n✓ Saved {len(df)} team-seasons to {outpath}")
        print(f"  Columns: {list(df.columns)}")
    else:
        print("\n⚠ No data to save.")
        print("\nMANUAL FALLBACK:")
        print("  1. Go to https://barttorvik.com/trank.php")
        print("  2. Set the year dropdown for each season")
        print("  3. Copy/paste the table into a spreadsheet")
        print("  4. Save as data/raw/torvik_ratings_all.csv")
        print("  5. Ensure columns include: team, conf, adj_o, adj_d, "
              "adj_t, barthag, luck, season")


if __name__ == "__main__":
    main()
