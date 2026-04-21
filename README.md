# March Madness Prediction Pipeline

A modular, end-to-end NCAA tournament prediction system built on four analytically distinct feature pillars.

## Overview

Most March Madness models rely almost entirely on efficiency ratings — AdjEM, Barthag, BPI. Those numbers are strong predictors, but they leave signal on the table. This pipeline adds three additional pillars that address things efficiency ratings miss: how a team is *constructed* going into the tournament, how the team *performs under pressure* during the season, and what *historical seed context* tells us about expected outcomes.

The four pillars are:

1. **Era-Adjusted Efficiency (Barttorvik/T-Rank)** — AdjO, AdjD, AdjEM, AdjT, and luck penalty, all z-scored within season to normalize across eras. A +25 AdjEM team in 2015 is not the same as one in 2025.
2. **Roster Construction Index** — Blue-chip recruit ratio (top-100 per 247 Composite), roster continuity (returning minutes % from Barttorvik), and an NIL consolidation flag for rosters with 3+ former top-50 recruits.
3. **Killshot Metrics** — Unanswered scoring run statistics computed from full play-by-play data. A killshot is a 10-0 run; double and triple killshots are weighted more heavily. Second-half killshots get a pressure multiplier. The premise: teams that regularly go on and survive big runs are built differently for single-elimination play.
4. **Tournament Context** — NCAA seed, seed-based historical win rates (1985–present), Quad 1 wins (via WAB), and placeholders for hot-hand and close-game win rate that can be populated at tournament time.

Predictions are made at the matchup level: the model receives the *difference* in each feature between two teams (Team A minus Team B) and outputs a win probability. This framing keeps the model symmetric and generalizes cleanly to any pairing.

Training uses leave-one-year-out cross-validation (2015–2025, excluding 2020) so every metric reflects true out-of-sample performance. The final model applies isotonic probability calibration so outputs are reliable win probabilities, not just rankings.

## Tech Stack

- **requests + BeautifulSoup + lxml** — scraping Barttorvik and 247Sports (HTML table parsing with multiple fallback strategies)
- **CBBpy** — pulls ESPN play-by-play for all D1 games; used to compute killshot metrics
- **pandas / numpy** — data pipeline, era-adjustment, feature merging
- **scikit-learn** — Logistic Regression baseline, GradientBoosting fallback, `CalibratedClassifierCV` for isotonic probability calibration, LOYO-CV evaluation
- **XGBoost** — optional boosted model; gracefully falls back to sklearn GBM if not installed
- **joblib** — model and scaler serialization
- **tqdm** — progress bars for multi-season scraping (PBP collection takes several hours)

## Key Results / Highlights

- **TODO:** Fill in Cross Validation (leaving one year out) accuracy, log loss, and Brier score after running `05_train_model.py`
- **TODO:** Fill in improvement over seed-only baseline (printed by the training script)
- Covers seasons **2015–2026** (excluding 2020 COVID cancellation) — 10 tournament years of backtesting
- Killshot detection processes **5,000+ games per season** of ESPN play-by-play; the `--resume` flag lets you pick up where you left off if interrupted
- Interactive prediction mode: `python scripts/06_predict.py --team1 "Duke" --team2 "Houston"`

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Data you need to provide

One file must be manually sourced (the others are collected by the pipeline):

```
data/raw/tournament_results.csv
```

This should be a CSV with columns: `season, round, winner, loser, winner_seed, loser_seed`  
Download historical results from [Sports Reference CBB](https://www.sports-reference.com/cbb/postseason/).

### Pipeline execution order

```bash
# Step 1: Collect Barttorvik T-Rank ratings (fast, ~2 min)
python scripts/01_collect_torvik_ratings.py

# Step 2: Collect play-by-play and compute killshot metrics
# ⚠ Slow — allow several hours per season. Use --resume if interrupted.
python scripts/02_collect_pbp_killshots.py --all --resume

# Step 3: Collect recruiting and roster continuity data
python scripts/03_collect_roster_data.py

# Step 4: Build the four-pillar feature set and matchup training data
python scripts/04_feature_engineering.py

# Step 5: Train and evaluate the model
python scripts/05_train_model.py --model logistic  # or --model xgb

# Step 6: Predict matchups
python scripts/06_predict.py --team1 "Duke" --team2 "Houston"
python scripts/06_predict.py --bracket bracket_2026.csv
python scripts/06_predict.py --interactive
```

### Configuration

All tunable constants live in `config.py` at the project root:
- `CURRENT_SEASON` — update each year
- `KILLSHOT_THRESHOLD` — default 10 (points for a killshot)
- `BLUE_CHIP_CUTOFF` — default 100 (top-100 national recruit)
- `NIL_SUPERTEAM_THRESHOLD` — default 3 (five-stars to flag as NIL roster)

## Project Structure

```
march-madness-pipeline/
├── config.py                         # All constants, paths, and feature lists
├── scripts/
│   ├── 01_collect_torvik_ratings.py  # Scrapes Barttorvik T-Rank ratings
│   ├── 02_collect_pbp_killshots.py   # ESPN PBP → killshot metrics (slow)
│   ├── 03_collect_roster_data.py     # 247Sports recruiting + Barttorvik continuity
│   ├── 04_feature_engineering.py     # Merges all pillars, builds matchup training data
│   ├── 05_train_model.py             # LOYO-CV, baseline comparison, final model training
│   └── 06_predict.py                 # Matchup and bracket prediction at inference time
├── data/                             # Created automatically — not committed
│   ├── raw/                          # Scraped source data
│   └── processed/                    # Engineered features and training data
├── models/                           # Saved model weights — not committed
├── output/                           # CV results and feature importances
└── requirements.txt
```

## What I Learned / Future Work

The killshot pillar was the most intellectually interesting part to design. The hypothesis — that teams capable of going on big runs and surviving them are built for tournament variance — is intuitive, but encoding it cleanly from raw ESPN play-by-play required a lot of edge case handling (simultaneous scoring rows, CBBpy column name drift, half detection). Whether it actually improves out-of-sample prediction over efficiency alone is what the Cross Validation (leaving one year out) is there to answer honestly.

**Future directions:**
- Add transfer portal data: a team that added three transfers from high-major programs in March is a different team than its Barttorvik rating suggests
- Hot-hand index: currently a placeholder; the PBP data is already collected to compute it (winning streak entering the tournament, margin trend over final 5 games)
- Simulate full brackets rather than predicting individual games, then score by ESPN/Yahoo bracket scoring rules
- Publish a yearly bracket prediction writeup alongside the model output
