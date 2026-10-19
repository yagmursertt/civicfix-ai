# CivicFix AI — AI-Powered Urban Infrastructure Complaint Prioritization

**Group 8** — Tunahan Onur Direk · Yağmur Sert · Sezin Naz Sarı
Capstone: AI for the SDGs · SDG 11 (11.5, 11.7), SDG 9 (9.1), SDG 13 (13.1)

CivicFix AI turns free-text citizen complaints about roads, sidewalks and drainage into a risk-ranked
work queue for municipalities: an LLM classifies each complaint into a 52-type issue taxonomy and a
hazard tier, and a gradient-boosting model adds recurrence risk and rainfall context.

## Assignment 1 — EDA & first experiment
- `CivicFix_EDA.ipynb` — cleaning, descriptive statistics, rainfall link, resolution times, hotspots,
  and the first recurrence-prediction models (executed, with outputs).
- `civicfix_eda.py` — same analysis as a script. `figs/` — all figures and `stats.json`.

## Data
**Dataset:** NYC 311 Service Requests (NYC Open Data, `erm2-nwe9`) — all 2025 complaints of type Street Condition,
Sidewalk Condition, Sewer and Root/Sewer/Sidewalk Condition (132,653 rows). Download with `./fetch_data.sh` (~50 MB).

Weather feature: daily precipitation for Central Park (NOAA GHCN-Daily, USW00094728), joined by date —
`data/nyc_centralpark_2025.csv`.

## Run
```bash
pip install -r requirements.txt
./fetch_data.sh
jupyter notebook CivicFix_EDA.ipynb
```

## First results (train Jan–Aug 2025, test Sep–Nov 2025)
| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| Baseline (majority) | 0.500 | 0.399 | 0.000 |
| Logistic Regression | 0.670 | 0.586 | 0.540 |
| Random Forest | 0.673 | 0.584 | 0.571 |
| Hist. Gradient Boosting | 0.672 | 0.584 | 0.561 |
