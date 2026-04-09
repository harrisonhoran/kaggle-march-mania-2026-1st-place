# March Machine Learning Mania 2026 — 1st Place Solution

**Final Score:** 0.1097454 (MSE/Brier, 126 games) | **Rank:** 1st / 3,485 teams

# Context
Kaggle Competition: [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/overview)

Forecasting the outcomes of both the men's and women's 2026 collegiate basketball tournaments, by submitting predictions for every possible tournament matchup.

| Prizes & Awards | Participation |
| --- | --- |
| 1st Place - $10,000<br>2nd Place - $8,000<br>3rd Place - $7,000<br>4th-8th Place - $5,000 | 12,252 Entrants<br>3,767 Participants<br>3,462 Teams<br>3,464 Submissions

For the full solution writeup, you can head to the [`kagglewriteup.md`]() file here or follow this [link](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/writeups/march-machine-learning-mania-2026-1st-place-solut) to the Kaggle page.

---

## Repo Structure

```
├── march-mania-2026.py                     # Full pipeline: feature engineering → training → submission
├── kagglewriteup.md                        # Detailed solution writeup
├── requirements.txt                        # Python dependencies
├── output/
│   └── submission_harry_2026.csv           # The exact submission that won
└── data/
    ├── march-machine-learning-mania-2026/  # Not included — download from Kaggle (see Setup)
    └── external/                           # Included in repo
        ├── college-basketball-injury-report_20260318_all.csv
        ├── miya_player_bpr_v2.csv
        └── top_wcbb_players_2025_26.csv
```

---

## Steps to Reproduce

### 1. Clone

```bash
git clone https://github.com/harrisonhoran/kaggle-march-mania-2026-1st-place.git
cd march-mania-2026
```

### 2. Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Download data

Download all files from the [competition data page](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data) into a `data/` directory at the repo root. Required files:

```
data/
├── MTeams.csv
├── MRegularSeasonDetailedResults.csv
├── MNCAATourneyDetailedResults.csv
├── MNCAATourneySeeds.csv
├── MTeamSpellings.csv
├── MSecondaryTourneyTeams.csv
├── MTeamConferences.csv
├── WTeams.csv
├── WRegularSeasonDetailedResults.csv
├── WNCAATourneyDetailedResults.csv
├── WNCAATourneySeeds.csv
├── WTeamSpellings.csv
├── WSecondaryTourneyTeams.csv
├── WTeamConferences.csv
└── SampleSubmissionStage2.csv
```

### 4. Run

```bash
mkdir output
python march-mania-2026.py
```

Outputs `output/submission_harry_2026.csv`, which matches the winning submission.
