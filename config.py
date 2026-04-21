"""
March Madness Prediction Model - Configuration
"""
import os
from pathlib import Path

# === Project Paths ===
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Create directories if they don't exist
for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === Season Range ===
# Training data window — 2015 is the earliest season with reliable
# Barttorvik data and modern tournament format
FIRST_SEASON = 2015
CURRENT_SEASON = 2026  # Update each year

# Tournament years available for backtesting (excludes 2020 - COVID cancellation)
TOURNEY_YEARS = [y for y in range(FIRST_SEASON, CURRENT_SEASON) if y != 2020]

# === Barttorvik Scraping ===
TORVIK_BASE_URL = "https://barttorvik.com"
TORVIK_RATINGS_URL = f"{TORVIK_BASE_URL}/trank.php"

# Request throttling (be respectful to the server)
REQUEST_DELAY_SECONDS = 1.5

# === CBBpy / ESPN ===
# CBBpy pulls from ESPN's public API — no key needed
# Rate limit yourself to avoid getting blocked
ESPN_REQUEST_DELAY = 1.0

# === Killshot Definitions ===
KILLSHOT_THRESHOLD = 10       # 10-0 run = killshot
DOUBLE_KILLSHOT_THRESHOLD = 20  # 20-0 run = double killshot
TRIPLE_KILLSHOT_THRESHOLD = 30  # 30-0 run = triple killshot

# Weighting multipliers for killshot tiers
KILLSHOT_WEIGHTS = {
    "single": 1.0,
    "double": 2.0,
    "triple": 3.0,
}

# Second half / OT killshots get a pressure bonus
SECOND_HALF_MULTIPLIER = 1.5

# === Roster Construction ===
# Blue Chip = top 100 recruit nationally (per 247 Composite)
BLUE_CHIP_CUTOFF = 100
SCHOLARSHIP_PLAYERS = 13

# NIL Consolidation: flag teams with 3+ former top-50 recruits
NIL_SUPERTEAM_THRESHOLD = 3
NIL_SUPERTEAM_RECRUIT_CUTOFF = 50

# === Model Training ===
RANDOM_SEED = 42
TEST_SIZE = 0.2  # Only used if not doing LOOCV

# Feature columns for each pillar
KENPOM_FEATURES = [
    "z_adj_em", "z_adj_o", "z_adj_d", "z_adj_t",
    "luck_penalty", "sos_rank"
]

ROSTER_FEATURES = [
    "blue_chip_ratio", "experience_score", "continuity_index",
    "nil_consolidation_flag"
]

KILLSHOT_FEATURES = [
    "killshot_diff_pg", "killshot_made_pg", "killshot_allowed_pg",
    "weighted_ks_diff_pg", "second_half_ks_ratio"
]

CONTEXT_FEATURES = [
    "seed", "seed_historical_winrate", "hot_hand_index",
    "close_game_winrate", "neutral_court_adj", "quad1_wins"
]

ALL_FEATURES = KENPOM_FEATURES + ROSTER_FEATURES + KILLSHOT_FEATURES + CONTEXT_FEATURES
