# =============================================================================
# harry_model.py
# March Machine Learning Mania 2026 — Men's & Women's submission pipeline
# =============================================================================

# stdlib
import os
import warnings
from pathlib import Path

# third-party
import numpy as np
import pandas as pd
import kagglehub
import xgboost as xgb
from scipy.stats import pointbiserialr, trim_mean
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 999)


# =============================================================================
# Paths
# =============================================================================

NOTEBOOK_DIR = Path(__file__).resolve().parent
DATA_DIR = NOTEBOOK_DIR / "data/march-machine-learning-mania-2026"
DATA_DIR_EXT = NOTEBOOK_DIR / "data/external"
OUTPUT_DIR = NOTEBOOK_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Data Collection — Mania Competition Data (Men's & Women's)
# =============================================================================

# W -> women's basketball, M -> men's basketball
M_teams               = pd.read_csv(DATA_DIR / "MTeams.csv").assign(League="M")
M_regular_results     = pd.read_csv(DATA_DIR / "MRegularSeasonDetailedResults.csv").assign(League="M")
M_tourney_results     = pd.read_csv(DATA_DIR / "MNCAATourneyDetailedResults.csv").assign(League="M")
M_seeds               = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv").assign(League="M")
M_team_spell          = pd.read_csv(DATA_DIR / "MTeamSpellings.csv", encoding="latin1").assign(League="M")
M_sec_tourney_results = pd.read_csv(DATA_DIR / "MSecondaryTourneyTeams.csv").assign(League="M")
M_conf                = pd.read_csv(DATA_DIR / "MTeamConferences.csv").assign(League="M")

W_teams               = pd.read_csv(DATA_DIR / "WTeams.csv").assign(League="W")
W_regular_results     = pd.read_csv(DATA_DIR / "WRegularSeasonDetailedResults.csv").assign(League="W")
W_tourney_results     = pd.read_csv(DATA_DIR / "WNCAATourneyDetailedResults.csv").assign(League="W")
W_seeds               = pd.read_csv(DATA_DIR / "WNCAATourneySeeds.csv").assign(League="W")
W_team_spell          = pd.read_csv(DATA_DIR / "WTeamSpellings.csv", encoding="latin1").assign(League="W")
W_sec_tourney_results = pd.read_csv(DATA_DIR / "WSecondaryTourneyTeams.csv").assign(League="W")
W_conf                = pd.read_csv(DATA_DIR / "WTeamConferences.csv").assign(League="W")

sub2 = pd.read_csv(DATA_DIR / "SampleSubmissionStage2.csv")


# =============================================================================
# Data Collection — Nishaan Amin (Men's Only)
# https://www.kaggle.com/datasets/nishaanamin/march-madness-data
# =============================================================================

dataset_path = kagglehub.dataset_download("nishaanamin/march-madness-data")

dfs = {}
for file in os.listdir(dataset_path):
    if file.endswith(".csv"):
        key = file.replace(".csv", "")
        dfs[key] = pd.read_csv(os.path.join(dataset_path, file))


# =============================================================================
# Feature: AP Poll Week 6 Top-12 (Men's Only)
# =============================================================================

ap_poll    = dfs["AP Poll Data"]
ap_poll_w6 = ap_poll[ap_poll["WEEK"] == 6].copy()

ap_poll_w6["Top12"] = np.where(ap_poll_w6["AP RANK"] <= 12, 1, 0)
ap_poll_w6 = ap_poll_w6.rename(columns={"YEAR": "Season", "TEAM": "TeamName"})
ap_poll_w6["TeamName"] = ap_poll_w6["TeamName"].str.replace(".", "", regex=False)

# Build name → TeamID lookup
name_to_id = {}
for _, r in M_team_spell.iterrows():
    name_to_id[str(r["TeamNameSpelling"]).lower()] = r["TeamID"]
for _, r in M_teams.iterrows():
    name_to_id[r["TeamName"].lower()] = r["TeamID"]

manual_map = {
    "saint mary's":        name_to_id.get("st mary's ca"),
    "saint joseph's":      name_to_id.get("st joseph's pa"),
    "saint louis":         name_to_id.get("st louis"),
    "loyola chicago":      name_to_id.get("loyola-chicago"),
    "college of charleston": name_to_id.get("col charleston"),
    "florida atlantic":    name_to_id.get("fla atlantic"),
    "middle tennessee":    name_to_id.get("mid tennessee"),
    "stephen f austin":    name_to_id.get("sf austin"),
    "george washington":   name_to_id.get("g washington"),
    "little rock":         name_to_id.get("ark little rock"),
    "south dakota st":     name_to_id.get("s dakota st"),
    "north carolina st":   name_to_id.get("nc state"),
    "western kentucky":    name_to_id.get("w kentucky"),
    "western carolina":    name_to_id.get("w carolina"),
    "coastal carolina":    name_to_id.get("coastal car"),
}
name_to_id.update({k: v for k, v in manual_map.items() if v is not None})

ap_poll_w6["TeamID"] = ap_poll_w6["TeamName"].str.lower().map(name_to_id)
ap_poll_w6["Top12"]  = ap_poll_w6["Top12"].fillna(0)


# =============================================================================
# Feature: Power Conferences
# =============================================================================

POWER_CONFS = ["big_ten", "acc", "sec", "big_twelve", "big_east", "pac_twelve"]

M_conf["Power"] = np.where(M_conf["ConfAbbrev"].isin(POWER_CONFS), 1, 0)
W_conf["Power"] = np.where(W_conf["ConfAbbrev"].isin(POWER_CONFS), 1, 0)


# =============================================================================
# Feature: Men's Injury Adjustments (Rotowire + EvanMiya BPR)
# =============================================================================

M_injuries    = pd.read_csv(DATA_DIR_EXT / "college-basketball-injury-report_20260318_all.csv").assign(League="M")

M_player_stats = pd.read_csv(DATA_DIR_EXT / "miya_player_bpr_v2.csv").assign(League="M")

M_player_stats["Adj_BPR"] = (M_player_stats["BPR"] * 0.7).round(3)

status_weights = {
    "Out For Season":    1.0,
    "Out":               0.75,
    "Game Time Decision": 0.5,
}

M_player_inj_adj = M_injuries.merge(M_player_stats, on="Player")
M_player_inj_adj["Weighted_BPR"] = (
    M_player_inj_adj["Adj_BPR"]
    * M_player_inj_adj["Status"].map(status_weights).fillna(0)
).round(3)

M_player_inj_adj = M_player_inj_adj[["Player", "Team_x", "Status", "Adj_BPR", "Weighted_BPR"]].assign(League="M")
M_player_inj_adj = pd.merge(
    M_player_inj_adj,
    M_teams,
    left_on="Team_x",
    right_on="TeamName",
    how="left",
)
M_player_inj_adj = (
    M_player_inj_adj[["TeamName", "TeamID", "Player", "Status", "Adj_BPR", "Weighted_BPR"]]
    .dropna()
    .rename(columns={"Weighted_BPR": "Injury_Deduction_Eff"})
    .sort_values(["TeamName", "Player"])
)

# Aggregate to team level
M_team_inj_adj = (
    M_player_inj_adj
    .groupby("TeamID")
    .agg(Team_Injury_Deduction_Eff=("Injury_Deduction_Eff", "sum"))
)
M_team_inj_adj["Team_Injury_Deduction_Eff"] = M_team_inj_adj["Team_Injury_Deduction_Eff"].round(2)
M_team_inj_adj.index = M_team_inj_adj.index.astype(int)


# =============================================================================
# Consolidate Men's + Women's Data
# =============================================================================

regular_results = pd.concat([M_regular_results, W_regular_results])
tourney_results = pd.concat([M_tourney_results, W_tourney_results])
teams           = pd.concat([M_teams, W_teams])
seeds           = pd.concat([M_seeds, W_seeds])
conf            = pd.concat([M_conf, W_conf])

sec_tourney_results = pd.concat(
    [M_sec_tourney_results, W_sec_tourney_results]
)[["Season", "TeamID", "SecondaryTourney", "League"]]

# 2026 secondary tourney teams (NIT + WBIT)
nit_2026 = pd.DataFrame({
    "Season": 2026,
    "TeamID": [
        1120, 1305, 1206, 1370, 1375, 1293, 1251, 1472,  # Auburn region
        1307, 1143, 1161, 1430, 1358, 1228, 1386, 1205,  # Albuquerque region
        1448, 1173, 1463, 1229, 1298, 1133, 1423, 1244,  # Winston-Salem region
        1409, 1329, 1455, 1414, 1372, 1172, 1461, 1424,  # Tulsa region
    ],
    "SecondaryTourney": "NIT",
    "League": "M",
})

wbit_2026 = pd.DataFrame({
    "Season": 2026,
    "TeamID": [
        3140, 3295, 3401, 3428, 3242, 3243, 3274, 3390,
        3143, 3206, 3349, 3458, 3162, 3217, 3281, 3361,
        3105, 3151, 3270, 3184, 3407, 3210, 3204, 3258,
        3365, 3346, 3256, 3333, 3385, 3371, 3414, 3298,
    ],
    "SecondaryTourney": "WBIT",
    "League": "W",
})

sec_tourney_results = pd.concat(
    [sec_tourney_results, nit_2026, wbit_2026], ignore_index=True
)

# Apply season cutoff
SEASON_CUTOFF = 2003
regular_results     = regular_results.loc[regular_results["Season"] >= SEASON_CUTOFF]
tourney_results     = tourney_results.loc[tourney_results["Season"] >= SEASON_CUTOFF]
sec_tourney_results = sec_tourney_results.loc[sec_tourney_results["Season"] >= SEASON_CUTOFF]
seeds               = seeds.loc[seeds["Season"] >= SEASON_CUTOFF]
conf                = conf.loc[conf["Season"] >= SEASON_CUTOFF]


# =============================================================================
# Data Preparation (raddar-style double mirror)
# =============================================================================

def swap_location(x):
    if x == "H":
        return "A"
    if x == "A":
        return "H"
    return x


def prepare_data(df):
    """Mirror every game row so both teams appear as T1 in the dataset."""
    df = df.rename(columns={"WLoc": "location"})
    df = df[[
        "Season", "DayNum", "LTeamID", "LScore", "WTeamID", "WScore", "NumOT", "location",
        "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF",
        "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF",
    ]]

    df = df.assign(
        WPoss=df["WFGA"] - df["WOR"] + df["WTO"] + (df["WFTA"] * 0.475),
        LPoss=df["LFGA"] - df["LOR"] + df["LTO"] + (df["LFTA"] * 0.475),
    ).assign(
        Pace    = lambda x: (200 / (x["WPoss"] + x["LPoss"]) / 2),
        WOffEff = lambda x: (x["WScore"] / x["WPoss"]) * 70,
        WDefEff = lambda x: (x["LScore"] / x["LPoss"]) * 70,
        LOffEff = lambda x: (x["LScore"] / x["LPoss"]) * 70,
        LDefEff = lambda x: (x["WScore"] / x["WPoss"]) * 70,
    ).assign(
        WNetEff = lambda x: x["WOffEff"] - x["WDefEff"],
        LNetEff = lambda x: x["LOffEff"] - x["LDefEff"],
    ).assign(
        WOffRebRate = df["WOR"] / (df["WOR"] + df["LDR"]),
        LOffRebRate = df["LOR"] / (df["LOR"] + df["WDR"]),
        WFTRate     = df["WFTA"] / df["WFGA"],
        LFTRate     = df["LFTA"] / df["LFGA"],
    )

    # OT adjustment — normalise all counting stats to regulation pace
    adjot = (40 + 5 * df["NumOT"]) / 40
    adjcols = [
        "LScore", "WScore",
        "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF",
        "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF",
        "Pace", "WPoss", "WOffEff", "WDefEff", "WNetEff", "LPoss", "LOffEff", "LDefEff", "LNetEff",
        "WOffRebRate", "LOffRebRate", "WFTRate", "LFTRate",
    ]
    for col in adjcols:
        df[col] = df[col] / adjot

    dfswap = df.copy()
    dfswap["location"] = dfswap["location"].apply(swap_location)

    df.columns     = [x.replace("W", "T1_").replace("L", "T2_") for x in df.columns]
    dfswap.columns = [x.replace("L", "T1_").replace("W", "T2_") for x in dfswap.columns]

    output = pd.concat([df, dfswap]).reset_index(drop=True)
    output["PointDiff"] = output["T1_Score"] - output["T2_Score"]
    output["win"]       = (output["PointDiff"] > 0).astype(int)
    output["men_women"] = output["T1_TeamID"].apply(lambda t: 1 if str(t).startswith("1") else 0)
    return output


regular_data = prepare_data(regular_results)
tourney_data = prepare_data(tourney_results)


# =============================================================================
# Feature Engineering
# =============================================================================

# --- Seeds ---
seeds["seed"] = seeds["Seed"].apply(lambda x: int(x[1:3]))
seeds_T1 = seeds[["Season", "TeamID", "seed"]].copy().rename(columns={"TeamID": "T1_TeamID", "seed": "T1_seed"})
seeds_T2 = seeds[["Season", "TeamID", "seed"]].copy().rename(columns={"TeamID": "T2_TeamID", "seed": "T2_seed"})

regular_data = pd.merge(regular_data, seeds_T1, on=["Season", "T1_TeamID"], how="left")
regular_data = pd.merge(regular_data, seeds_T2, on=["Season", "T2_TeamID"], how="left")

# --- Power Conferences ---
conf_T1 = conf[["Season", "TeamID", "Power"]].copy().rename(columns={"TeamID": "T1_TeamID", "Power": "T1_Power"})
conf_T2 = conf[["Season", "TeamID", "Power"]].copy().rename(columns={"TeamID": "T2_TeamID", "Power": "T2_Power"})

regular_data = pd.merge(regular_data, conf_T1, on=["Season", "T1_TeamID"], how="left")
regular_data = pd.merge(regular_data, conf_T2, on=["Season", "T2_TeamID"], how="left")

# --- AP Poll Week 6 Top-12 ---
ap_poll_w6_T1 = ap_poll_w6[["Season", "TeamID", "Top12"]].copy().rename(columns={"TeamID": "T1_TeamID", "Top12": "T1_Top12"})
ap_poll_w6_T2 = ap_poll_w6[["Season", "TeamID", "Top12"]].copy().rename(columns={"TeamID": "T2_TeamID", "Top12": "T2_Top12"})

regular_data = pd.merge(regular_data, ap_poll_w6_T1, on=["Season", "T1_TeamID"], how="left")
regular_data = pd.merge(regular_data, ap_poll_w6_T2, on=["Season", "T2_TeamID"], how="left")

# --- Secondary Tournaments ---
sec_T1 = sec_tourney_results[["Season", "TeamID", "SecondaryTourney"]].copy().rename(
    columns={"TeamID": "T1_TeamID", "SecondaryTourney": "T1_Tourney"})
sec_T2 = sec_tourney_results[["Season", "TeamID", "SecondaryTourney"]].copy().rename(
    columns={"TeamID": "T2_TeamID", "SecondaryTourney": "T2_Tourney"})

regular_data = pd.merge(regular_data, sec_T1, on=["Season", "T1_TeamID"], how="left")
regular_data = pd.merge(regular_data, sec_T2, on=["Season", "T2_TeamID"], how="left")

# --- Opponent Quality Points ---
conditions = [
    regular_data["T2_seed"] <= 4,
    regular_data["T2_seed"] <= 16,
    regular_data["T2_Tourney"].notna(),
]
choices = [6, 4, 2]

regular_data["T1_Opp_Qlty_Pts"] = np.select(conditions, choices, default=0.25)
regular_data = regular_data.assign(
    T1_Tourney_Quality_Game=np.where(regular_data["T1_Opp_Qlty_Pts"] > 1, 1, 0),
).assign(
    T1_Tourney_Quality_Win=lambda x:  np.where((x["win"] == 1) & (x["T1_Tourney_Quality_Game"] == 1), 1, 0),
    T1_Tourney_Quality_Loss=lambda x: np.where((x["win"] == 0) & (x["T1_Tourney_Quality_Game"] == 1), 1, 0),
)
regular_data["T1_Opp_Qlty_Pts_Won"] = np.where(
    regular_data["win"] == 1, regular_data["T1_Opp_Qlty_Pts"], 0
)


# =============================================================================
# Regular Season Aggregates (2% trimmed mean)
# =============================================================================

win_pct = (
    regular_data
    .groupby(["Season", "T1_TeamID"])
    .agg(win_pct=("win", "mean"))
    .reset_index()
)

boxcols = [
    "T1_Score", "T1_FGM", "T1_FGA", "T1_FGM3", "T1_FGA3", "T1_FTM", "T1_FTA",
    "T1_OR", "T1_DR", "T1_Ast", "T1_TO", "T1_Stl", "T1_Blk", "T1_PF",
    "T2_Score", "T2_FGM", "T2_FGA", "T2_FGM3", "T2_FGA3", "T2_FTM", "T2_FTA",
    "T2_OR", "T2_DR", "T2_Ast", "T2_TO", "T2_Stl", "T2_Blk", "T2_PF",
    "PointDiff", "Pace", "T1_Poss", "T1_OffEff", "T1_DefEff", "T1_NetEff", "T2_Poss",
    "T2_OffEff", "T2_DefEff", "T2_NetEff", "T1_Opp_Qlty_Pts", "T1_Power", "T1_OffRebRate",
    "T2_OffRebRate", "T1_FTRate", "T2_FTRate", "T1_Top12",
]

ss = (
    regular_data
    .groupby(["Season", "T1_TeamID"])[boxcols]
    .agg(lambda x: trim_mean(x, 0.02))
    .reset_index()
)

ss["T1_NetEff_std"]      = regular_data.groupby(["Season", "T1_TeamID"])["T1_NetEff"].std().values
ss["T1_Opp_Qlty_Pts_Won"] = regular_data.groupby(["Season", "T1_TeamID"])["T1_Opp_Qlty_Pts"].sum().values
ss["T1_Top12"]            = ss["T1_Top12"].fillna(0)
ss = ss.merge(win_pct, on=["Season", "T1_TeamID"])


# =============================================================================
# 2026 Injury Adjustment
# =============================================================================

inj_map = M_team_inj_adj["Team_Injury_Deduction_Eff"].to_dict()
mask_2026 = ss["Season"] == 2026
ss.loc[mask_2026, "T1_NetEff"] -= ss.loc[mask_2026, "T1_TeamID"].map(inj_map).fillna(0)


# =============================================================================
# harry_Rating — Custom Power Ranking
# =============================================================================

scaler_configs = {
    1: {"opp_pts": (-0.55, 0.55), "power_conf": (1, 1.3), "top12": (1, 1.2)},  # men
    0: {"opp_pts": (-0.50, 0.50), "power_conf": (1, 1.1)},                      # women
}

for gender, ranges in scaler_configs.items():
    mask = ss["T1_TeamID"] < 3000 if gender == 1 else ss["T1_TeamID"] >= 3000

    scaler_opp  = MinMaxScaler(feature_range=ranges["opp_pts"])
    scaler_conf = MinMaxScaler(feature_range=ranges["power_conf"])
    ss.loc[mask, "T1_Opp_Qlty_Pts_MinMax"] = scaler_opp.fit_transform(ss.loc[mask, ["T1_Opp_Qlty_Pts"]])
    ss.loc[mask, "T1_Power_MinMax"]         = scaler_conf.fit_transform(ss.loc[mask, ["T1_Power"]])

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


# =============================================================================
# Assemble Tourney Training Data
# =============================================================================

ss_T1 = ss.copy()
ss_T1.columns = ["T1_avg_" + c.replace("T1_", "").replace("T2_", "opponent_") for c in ss_T1.columns]
ss_T1 = ss_T1.rename(columns={
    "T1_avg_Season": "Season",
    "T1_avg_TeamID": "T1_TeamID",
    "T1_avg_Opp_Qlty_Pts_Won": "T1_Opp_Qlty_Pts_Won",
})

ss_T2 = ss.copy()
ss_T2.columns = ["T2_avg_" + c.replace("T1_", "").replace("T2_", "opponent_") for c in ss_T2.columns]
ss_T2 = ss_T2.rename(columns={
    "T2_avg_Season": "Season",
    "T2_avg_TeamID": "T2_TeamID",
    "T2_avg_Opp_Qlty_Pts_Won": "T2_Opp_Qlty_Pts_Won",
})

tourney_data = tourney_data[["Season", "T1_TeamID", "T2_TeamID", "PointDiff", "win", "men_women"]]
tourney_data = pd.merge(tourney_data, seeds_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, seeds_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, ss_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, ss_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, conf_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, conf_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, ap_poll_w6_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, ap_poll_w6_T2, on=["Season", "T2_TeamID"], how="left")

tourney_data["seed_diff"]             = tourney_data["T1_seed"] - tourney_data["T2_seed"]
tourney_data["opp_qlty_pts_won_diff"] = tourney_data["T1_Opp_Qlty_Pts_Won"] - tourney_data["T2_Opp_Qlty_Pts_Won"]
tourney_data["avg_blk_diff"]          = tourney_data["T1_avg_Blk"] - tourney_data["T2_avg_Blk"]
tourney_data["harry_diff"]            = tourney_data["T1_avg_harry_Rating"] - tourney_data["T2_avg_harry_Rating"]


# =============================================================================
# Feature Selection
# =============================================================================

m_features = ["seed_diff", "opp_qlty_pts_won_diff", "harry_diff"]
w_features = ["seed_diff", "avg_blk_diff", "opp_qlty_pts_won_diff", "harry_diff"]

print("Men's feature correlations:")
m_tourney_data = tourney_data[tourney_data["men_women"] == 1]
for f in m_features:
    corr, _ = pointbiserialr(m_tourney_data[f], m_tourney_data["PointDiff"])
    print(f"  {f:30s}  r={corr:.4f}")

print("\nWomen's feature correlations:")
w_tourney_data = tourney_data[tourney_data["men_women"] == 0]
for f in w_features:
    corr, _ = pointbiserialr(w_tourney_data[f], w_tourney_data["PointDiff"])
    print(f"  {f:30s}  r={corr:.4f}")

base_cols    = ["Season", "men_women", "T1_TeamID", "T2_TeamID", "PointDiff", "win"]
all_features = list(dict.fromkeys(m_features + w_features))
tourney_data = tourney_data.loc[:, ~tourney_data.columns.duplicated()]
tourney_data = tourney_data[base_cols + all_features]


# =============================================================================
# Modeling — XGBRegressor + Isotonic Calibration (GroupKFold by Season)
# =============================================================================

hparams = {
    "men":   dict(max_depth=2, min_child_weight=5, subsample=0.7, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0),
    "women": dict(max_depth=2, min_child_weight=3, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0),
}

men_models    = []
women_models  = []
men_oof_preds   = []
men_oof_labels  = []
women_oof_preds = []
women_oof_labels = []

for gender, FEATURES in [("men", m_features), ("women", w_features)]:
    data    = tourney_data[tourney_data["men_women"] == (1 if gender == "men" else 0)]
    X_tr    = data[FEATURES]
    y       = data["win"]
    groups  = data["Season"]
    seasons = data["Season"].unique()

    gkf         = GroupKFold(n_splits=data["Season"].nunique())
    cv_briers    = []
    cv_accuracies = []
    fold_models  = []

    for season_idx, (train_index, test_index) in enumerate(gkf.split(X_tr, y, groups)):
        X_train, X_test = X_tr.iloc[train_index], X_tr.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = xgb.XGBRegressor(
            eval_metric="rmse",
            n_estimators=4_000,
            learning_rate=0.003,
            early_stopping_rounds=100,
            **hparams[gender],
        )
        holdout_season = seasons[season_idx]
        print(f"[{gender}] Holdout Season: {holdout_season}")
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=1000)

        y_pred = model.predict(X_test).clip(0.01, 0.99)

        if gender == "men":
            men_oof_preds.extend(y_pred)
            men_oof_labels.extend(y_test.values)
        else:
            women_oof_preds.extend(y_pred)
            women_oof_labels.extend(y_test.values)

        brier    = brier_score_loss(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred > 0.5)
        cv_briers.append(brier)
        cv_accuracies.append(accuracy)
        print(f"[{gender}] Season {holdout_season}: Brier={brier:.4f}, Accuracy={accuracy:.4f}")
        fold_models.append(model)

    print(f"[{gender}] Average CV Brier: {np.mean(cv_briers):.4f}, Accuracy: {np.mean(cv_accuracies):.4f}\n")
    if gender == "men":
        men_models = fold_models
    else:
        women_models = fold_models

# Fit isotonic calibrators on all out-of-fold predictions
men_calibrator = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
men_calibrator.fit(men_oof_preds, men_oof_labels)

women_calibrator = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
women_calibrator.fit(women_oof_preds, women_oof_labels)

# Calibration diagnostics
men_calibrated   = men_calibrator.predict(np.array(men_oof_preds))
women_calibrated = women_calibrator.predict(np.array(women_oof_preds))

print(f"Men's   CV Brier (raw):        {brier_score_loss(men_oof_labels, men_oof_preds):.4f}")
print(f"Men's   CV Brier (calibrated): {brier_score_loss(men_oof_labels, men_calibrated):.4f}")
print(f"Women's CV Brier (raw):        {brier_score_loss(women_oof_labels, women_oof_preds):.4f}")
print(f"Women's CV Brier (calibrated): {brier_score_loss(women_oof_labels, women_calibrated):.4f}")


# =============================================================================
# Generate Predictions for 2026 Submission
# =============================================================================

X = sub2.copy()
X["Season"]    = X["ID"].apply(lambda t: int(t.split("_")[0]))
X["T1_TeamID"] = X["ID"].apply(lambda t: int(t.split("_")[1]))
X["T2_TeamID"] = X["ID"].apply(lambda t: int(t.split("_")[2]))
X["men_women"] = X["T1_TeamID"].apply(lambda t: 1 if str(t)[0] == "1" else 0)

X = pd.merge(X, seeds_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, seeds_T2, on=["Season", "T2_TeamID"], how="left")
X = pd.merge(X, ss_T1,    on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, ss_T2,    on=["Season", "T2_TeamID"], how="left")
X = pd.merge(X, conf_T1,  on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, conf_T2,  on=["Season", "T2_TeamID"], how="left")
X = pd.merge(X, ap_poll_w6_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, ap_poll_w6_T2, on=["Season", "T2_TeamID"], how="left")

X["seed_diff"]             = X["T1_seed"] - X["T2_seed"]
X["opp_qlty_pts_won_diff"] = X["T1_Opp_Qlty_Pts_Won"] - X["T2_Opp_Qlty_Pts_Won"]
X["avg_blk_diff"]          = X["T1_avg_Blk"] - X["T2_avg_Blk"]
X["harry_diff"]            = X["T1_avg_harry_Rating"] - X["T2_avg_harry_Rating"]

men_mask   = X["men_women"] == 1
women_mask = X["men_women"] == 0

for i, model in enumerate(men_models):
    X.loc[men_mask, f"pred_model_m{i}"] = model.predict(X.loc[men_mask, m_features])

for i, model in enumerate(women_models):
    X.loc[women_mask, f"pred_model_w{i}"] = model.predict(X.loc[women_mask, w_features])

men_pred_cols   = [c for c in X.columns if c.startswith("pred_model_m")]
women_pred_cols = [c for c in X.columns if c.startswith("pred_model_w")]

X.loc[men_mask,   "Pred_raw"] = X.loc[men_mask,   men_pred_cols].mean(axis=1)
X.loc[women_mask, "Pred_raw"] = X.loc[women_mask, women_pred_cols].mean(axis=1)

X.loc[men_mask,   "Pred"] = men_calibrator.predict(X.loc[men_mask,   "Pred_raw"].values)
X.loc[women_mask, "Pred"] = women_calibrator.predict(X.loc[women_mask, "Pred_raw"].values)

X["ID"] = (
    X["Season"].astype(str) + "_"
    + X["T1_TeamID"].astype(str) + "_"
    + X["T2_TeamID"].astype(str)
)

submission = X[["ID", "Pred"]].copy()


# =============================================================================
# Post-Processing — Sharpen Edges
# =============================================================================

EDGE = 0.03


def sharpen_edges(prob, temperature=2.5, lower=EDGE, upper=1 - EDGE):
    if prob <= lower or prob >= upper:
        return prob ** temperature / (prob ** temperature + (1 - prob) ** temperature)
    return prob


submission_sharp = submission.copy()
submission_sharp["Pred"] = submission_sharp["Pred"].apply(lambda x: float(x.real))
submission_sharp["Pred"] = submission_sharp["Pred"].clip(0.001, 0.999)
submission_sharp["Pred"] = submission_sharp["Pred"].apply(sharpen_edges)


# =============================================================================
# Export
# =============================================================================

out_path = OUTPUT_DIR / "submission_harry_2026.csv"
submission_sharp.to_csv(out_path, index=False)
print(f"\nSubmission saved to: {out_path}")
print(f"Rows: {len(submission_sharp):,} | Pred range: [{submission_sharp['Pred'].min():.4f}, {submission_sharp['Pred'].max():.4f}]")