# CivicFix AI — AI-Powered Urban Infrastructure Complaint Prioritization

**Group 8** — Tunahan Onur Direk · Yağmur Sert · Sezin Naz Sarı
Capstone: AI for the SDGs · SDG 11 (11.5, 11.7), SDG 9 (9.1), SDG 13 (13.1)

CivicFix AI turns free-text citizen complaints about roads, sidewalks and drainage into a risk-ranked
work queue for municipalities: an LLM classifies each complaint into a 52-type issue taxonomy and a
hazard tier, and a gradient-boosting model adds recurrence risk and rainfall context.

## Assignment 1 — EDA & first experiment
- `CivicFix_EDA.ipynb` — cleaning, descriptive statistics, rainfall link, resolution times, hotspots,
  Istanbul (İSKİ) context and the first recurrence-prediction models (executed, with outputs).
- `civicfix_eda.py` — same analysis as a script. `figs/` — all figures and `stats.json`.

## Data
| Dataset | Source | In repo |
|---|---|---|
| NYC 311 Service Requests 2025 (Street, Sidewalk, Sewer, Root/Sewer/Sidewalk) — 132,653 rows | NYC Open Data `erm2-nwe9` | run `./fetch_data.sh` (~50 MB) |
| Daily precipitation, Central Park (USW00094728), 2025 | NOAA NCEI GHCN-Daily | `data/nyc_centralpark_2025.csv` |
| İSKİ sewer/channel faults by district & neighbourhood, 2023 | İBB Açık Veri | `data/iski_kanal_2023.xlsx` |
| İBB Çözüm Merkezi (153) applications by sector, 2022–2025 | İBB Açık Veri | `data/ibb153_sektorel.xlsx` |

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
