# March Machine Learning Mania 2026 — 1st Place Solution

**Final Score:** 0.1097454 (MSE/Brier, 126 games) | **Rank:** 1st / 3,485 teams

# Context
Kaggle Competition: [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/overview)

Forecasting the outcomes of both the men's and women's 2026 collegiate basketball tournaments, by submitting predictions for every possible tournament matchup.

| Prizes & Awards | Participation |
| --- | --- |
| 1st Place - $10,000<br>2nd Place - $8,000<br>3rd Place - $7,000<br>4th-8th Place - $5,000 | 12,252 Entrants<br>3,767 Participants<br>3,462 Teams<br>3,464 Submissions

# Philosophy
- Let the selection committee do the bulk of the work (i.e. trust seeding differential), then consider what they don't account for: continuous ranking of teams rather than strictly ordinal, some injuries (they're vague about how they account for injuries), matchup-specific advantages, etc.
- Treat the men's and women's tournaments as separate entities
- Play the game to win. Basically, work backwards from Brier Score and tweak the model towards a "finish in the money or bust" mentality (see: sharpening edges in post-processing).

# Approach
- Foundation: a blend of [raddar's 2025 kaggle model](https://www.kaggle.com/code/raddar/vilnius-ncaa) and [Rob Mulla's 2023 tutorial](https://youtu.be/cHtAEWkvSMU?si=7YhhIa-TSux3CEQn) as a starting point. For a novice, there's nothing quite like a great tutorial.

## Data Preparation
- Based on raddar's template cited above

### Efficiency Metrics
The following efficiency metrics calculations which feed into my custom rating (harry_Rating). I ended up not using rebound and free throw rate as features.

```    
df = df.assign(
        WPoss = df["WFGA"] - df["WOR"] + df["WTO"] + (df["WFTA"] * .475),
        LPoss = df["LFGA"] - df["LOR"] + df["LTO"] + (df["LFTA"] * .475),
    ).assign(
        Pace = lambda x: (200 / (x["WPoss"] + x["LPoss"])/2),
        WOffEff = lambda x: (x["WScore"] / x["WPoss"]) * 70,
        WDefEff = lambda x: (x["LScore"] / x["LPoss"]) * 70,
        LOffEff = lambda x: (x["LScore"] / x["LPoss"]) * 70,
        LDefEff = lambda x: (x["WScore"] / x["WPoss"]) * 70
    ).assign(
        WNetEff = lambda x: x["WOffEff"] - x["WDefEff"],
        LNetEff = lambda x: x["LOffEff"] - x["LDefEff"]
    ).assign(
        WOffRebRate = df["WOR"] / (df["WOR"] + df["LDR"]),
        LOffRebRate = df["LOR"] / (df["LOR"] + df["WDR"]),
        WFTRate = df["WFTA"] / df["WFGA"],
        LFTRate = df["LFTA"] / df["LFGA"]
    )
```
### Injury Adjustments (Men's Only)
I think there are two main options: a) you can let Vegas do the work, as their lines are injury-adjusted, or b) do it manually, which was my approach. I tried to do the same process for the women but opted out of adjusting for early season injuries (e.g. Ice Brady), so I ended up with no women's adjustments.

Injury Data Source: https://www.rotowire.com/cbasketball/injury-report.php

Player Efficiency Data Source: https://evanmiya.com/?player_ratings (manual data entry)

I just used basic assumptions for how much to prorate injuries based on their rotowire Status:
```
status_weights = {
    "Out For Season": 1.0,
    "Out": 0.75,
    "Game Time Decision": 0.5,
}
```
**Actual Adjustments**
| Team | Player | Status | Adj BPR | Injury Deduction |
|:----:|:------:|:------:|:-------:|:----------------:|
| Alabama | Aden Holloway | Out | 6.00 | 4.50 |
| Arkansas | Karter Knox | Game Time Decision | 2.91 | 1.46 |
| BYU | Richie Saunders | Out For Season | 6.05 | 6.05 |
| Clemson | Carter Welling | Out For Season | 2.98 | 2.98 |
| Duke | Caleb Foster | Out | 3.79 | 2.84 |
| Duke | Patrick Ngongba | Game Time Decision | 7.07 | 3.54 |
| Gonzaga | Braden Huff | Out | 5.36 | 4.02 |
| Gonzaga | Jalen Warley | Game Time Decision | 4.71 | 2.36 |
| Michigan | L.J. Cason | Out For Season | 3.16 | 3.16 |
| North Carolina | Caleb Wilson | Out For Season | 6.27 | 6.27 |
| SMU | B.J. Edwards | Out | 5.10 | 3.82 |
| Texas Tech | JT Toppin | Out For Season | 6.20 | 6.20 |
| Texas Tech | LeJuan Watts | Game Time Decision | 3.11 | 1.55 |
| UCLA | Tyler Bilodeau | Game Time Decision | 4.40 | 2.20 |
| Villanova | Matthew Hodge | Out For Season | 2.65 | 2.65 |
| Wisconsin | Nolan Winter | Game Time Decision | 5.28 | 2.64 |

## Feature Engineering
- The strategy is minimalist. I've done some modeling with NBA data, so I began there and then added a few college-specific features.
- Each feature is a calculated differential between Team 1 and Team 2

### Seeding Differential
```
tourney_data["seed_diff"] = tourney_data["T1_seed"] - tourney_data["T2_seed"]
```
This is self-evident, but it is worth understanding how the committee actually seeds teams.
- [Men's process](https://www.ncaa.org/sports/2026/2/5/ncaa-mens-basketball-tournament-selections.aspx)
- [Women's process](https://www.ncaa.org/sports/2026/2/6/ncaa-womens-basketball-tournament-selections.aspx)

### Custom Rating (harry_Rating) Differential
The goal of creating this was two-fold:
- Account for some of the things seeding doesn't capture
- Have a way to hand-tune rankings based on my opinion of teams

#### Calculation
##### Step 1: 2% Trim Mean Season Average Net Efficiency (calculated during data preparation)

```
# calculate season averages (2% trim)
ss = regular_data.groupby(["Season", "T1_TeamID"])[boxcols].agg(
    lambda x: trim_mean(x, 0.02)
    ).reset_index()
```
##### Step 2: Injury Deductions from Net Rating
All I did was subtract the Injury Deduction from the table above from those teams' Net Ratings.

##### Step 3: Hand-Tuning Custom Ratings with MinMax Scalers
This is where my greenness probably shows most (I'm in my first term of a Data Science Master's program), but this felt like a way I could inject my own opinion into the model.

###### Scalers
###### Strength of Schedule (opp_pts)
I used tournament seeding and secondary tournament participation as a measure of team quality for that season.
```
conditions = [
    regular_data["T2_seed"] <= 4,
    regular_data["T2_seed"] <= 16,
    regular_data["T2_Tourney"].notna(),
]
choices = [6, 4, 2]

regular_data["T1_Opp_Qlty_Pts"] = np.select(conditions, choices, default=0.25)
```
So, 4 tiers of team quality:
| Tier | Criterion | Points |
| --- | --- | --- |
| Tier 1 | 4 seed or better in this year's tournament | 6 |
| Tier 2 | 5 seed or worse in this year's tournament | 4 |
| Tier 3 | Participating in secondary tournament | 2 |
| Tier 4 | No tournament participation | 0.25 |


###### Power Conferences (power_conf)
Forgot the old Pac-Ten.
```
# split men and women in case the values differ
M_conf["Power"] = np.where(
    M_conf["ConfAbbrev"].isin(["big_ten", "acc", "sec", "big_twelve", "big_east", "pac_twelve"]),
    1, 0
)

W_conf["Power"] = np.where(
    W_conf["ConfAbbrev"].isin(["big_ten", "acc", "sec", "big_twelve", "big_east", "pac_twelve"]),
    1, 0
)
```
###### AP Poll Week 6 Top 12 (top12)
Men-only classifier that shies away from recency. "Were they ranked in the AP Poll top 12 in week 6 or not?"

###### Full Calculation
```
scaler_configs = {
    1: {  # men
        "opp_pts": (-0.55, 0.55),
        "power_conf": (1, 1.3),
        "top12": (1, 1.2),  # men only
    },
    0: {  # women
        "opp_pts": (-0.5, 0.5),
        "power_conf": (1, 1.1),
    },
}

for gender, ranges in scaler_configs.items():
    if gender == 1:
        mask = ss["T1_TeamID"] < 3000
    else:
        mask = ss["T1_TeamID"] >= 3000

    scaler_opp = MinMaxScaler(feature_range=ranges["opp_pts"])
    scaler_conf = MinMaxScaler(feature_range=ranges["power_conf"])
    ss.loc[mask, "T1_Opp_Qlty_Pts_MinMax"] = scaler_opp.fit_transform(ss.loc[mask, ["T1_Opp_Qlty_Pts"]])
    ss.loc[mask, "T1_Power_MinMax"] = scaler_conf.fit_transform(ss.loc[mask, ["T1_Power"]])

    if "top12" in ranges:
        scaler_top12 = MinMaxScaler(feature_range=ranges["top12"])
        ss.loc[mask, "T1_Top12_MinMax"] = scaler_top12.fit_transform(ss.loc[mask, ["T1_Top12"]])
    else:
        ss.loc[mask, "T1_Top12_MinMax"] = 1  # neutral multiplier for women

ss["T1_harry_Rating"] = (
    ss["T1_NetEff"]
    * (1 + ss["T1_Opp_Qlty_Pts_MinMax"])
    * ss["T1_Power_MinMax"]
    * ss["T1_Top12_MinMax"]
)
```

### Quality Wins Differential (Opp_Qlty_Pts)
I wanted to reward teams who had accomplished hard things throughout the season. This is an extension of the Opponent Quality Points system (explained the Strength of Schedule scaler section above) and calculated by summing the opp_qlty_pts for wins.

### Feature Selection

#### Men's Tournament
| Feature | Literal Name | Correlation |
| --- | --- | --- |
| Seed Differential | seed_diff | r=-0.5937 | 
| Quality Wins Differential | opp_qlty_pts_won_diff | r=0.4603 |
| harry_Rating Differential | harry_diff | r=0.5971 |

#### Women's Tournament
| Feature | Literal Name | Correlation |
| --- | --- | --- |
| Seed Differential | seed_diff | r=-0.7560 | 
| Quality Wins Differential | opp_qlty_pts_won_diff | r=0.6437 |
| harry_Rating Differential | harry_diff | r=0.7743 |
| Avg. Blocks Differential | avg_blk_diff | r=0.4373 |

## Modeling
### XGBoost Regression on 'Win'
Raw prediction baseline
```
model = xgb.XGBRegressor(
            eval_metric="rmse",
            n_estimators=4_000,
            learning_rate=0.003,
            early_stopping_rounds=100,
            **hparams[gender],
        )

hparams = {
    "men":   dict(max_depth=2, min_child_weight=5, subsample=0.7, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0),
    "women": dict(max_depth=2, min_child_weight=3, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0),
}
```
### Cross-Validation
Leave One Season Out (2003-2025)

### Isotonic Regression to Calibrate
Applied across all games, so not specific to season
```
men_calibrator = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
men_calibrator.fit(men_oof_preds, men_oof_labels)

women_calibrator = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
women_calibrator.fit(women_oof_preds, women_oof_labels)
```
### Impact of Calibration
```
Men's CV Brier (raw):           0.1850
Men's CV Brier (calibrated):    0.1822
Women's CV Brier (raw):         0.1390
Women's CV Brier (calibrated):  0.1357
Total CV Brier (raw):           0.1620
Total CV Brier (calibrated):    0.1590
```
## Post-Processing
The competition page has a [nice dialogue](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/discussion/684446) on this strategy, so you can read up there, but here's my piece of it:

>I didn't hard-code any specific teams, but I did 'sharpen the edges' in post-processing, meaning I rounded predictions >= 97% and <= 3% to the extreme (see chart below). So my gamble was that no outcome with a 1 in 33.33 chance (3/100) would happen. I sharpened 28 first round games and projected to sharpen 35 total games, so that felt about right. Across 35 games P(at least one miss) is 65.6%. This is a negative expected value play but one way to gain an edge on the margins.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F4013222%2F3c96634d5072941bcba7a3aaef2e1284%2Fsharpened_edges_clean.png?generation=1775679624426569&alt=media)

# What Didn't Work Well
- Pretty much every feature other than the ones I used. I tinkered quite a bit and the model just got worse (e.g. rebound and free throw rate)
- Rewarding recent performance

# Mistakes & Opportunities
- Optuna: auto-tuning is sensible, but I was just learning how to use it.
- Impact of coaching: wondering if anyone had luck with this. I didn't take the time to explore it, but my hunch is it matters sometimes (looking at you, Jake Diebler...)
- Incorporating betting/predictive market odds. This makes a ton of sense, but I just didn't take the time to do it.

# Sources
- Competition Data: (https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data)
- AP Poll Data: ([Nishaan Amin](https://www.kaggle.com/datasets/nishaanamin/march-madness-data))
- Player Injuries: ([rotowire](https://www.rotowire.com/cbasketball/injury-report.php))
- Player Ratings: ([EvanMiya.com](https://evanmiya.com/?player_ratings))
- Bracket Visualizer: https://marksmath.org/visualization/data/NCAABrackets/KaggleBrackets/
