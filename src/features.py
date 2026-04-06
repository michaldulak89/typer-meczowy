import pandas as pd
import numpy as np


FEATURE_COLS = [
    "home_scored_home",
    "home_conceded_home",
    "away_scored_away",
    "away_conceded_away",

    "home_form",
    "away_form",
    "home_form_3",
    "away_form_3",

    "home_rating",
    "away_rating",

    "home_points_home",
    "away_points_away",

    "home_clean_sheets_home",
    "away_clean_sheets_away",

    "home_failed_to_score_home",
    "away_failed_to_score_away",

    "avg_home_odds",
    "avg_draw_odds",
    "avg_away_odds",

    "form_diff",
    "form_diff_3",
    "rating_diff",
    "goal_diff",
    "points_diff",
]


def _safe_result(row):
    result = str(row.get("result", "")).strip().upper()
    if result in ["H", "D", "A"]:
        return result

    home_goals = row.get("home_goals")
    away_goals = row.get("away_goals")

    if pd.notna(home_goals) and pd.notna(away_goals):
        if home_goals > away_goals:
            return "H"
        if home_goals < away_goals:
            return "A"
        return "D"

    return None


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date", kind="stable").reset_index(drop=True)

    for col in ["home_goals", "away_goals", "avg_home_odds", "avg_draw_odds", "avg_away_odds"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    # =========================================
    # ŚREDNIE GOLE HOME / AWAY PRZED MECZEM
    # =========================================
    df["home_scored_home"] = (
        df.groupby("home_team", sort=False)["home_goals"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    df["home_conceded_home"] = (
        df.groupby("home_team", sort=False)["away_goals"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    df["away_scored_away"] = (
        df.groupby("away_team", sort=False)["away_goals"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    df["away_conceded_away"] = (
        df.groupby("away_team", sort=False)["home_goals"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # =========================================
    # FORMA 5 I 3 MECZE - ŚREDNIA PUNKTÓW
    # =========================================
    team_points_history = {}
    home_form_5_list = []
    away_form_5_list = []
    home_form_3_list = []
    away_form_3_list = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        team_points_history.setdefault(home, [])
        team_points_history.setdefault(away, [])

        home_hist = team_points_history[home]
        away_hist = team_points_history[away]

        home_last_5 = home_hist[-5:]
        away_last_5 = away_hist[-5:]
        home_last_3 = home_hist[-3:]
        away_last_3 = away_hist[-3:]

        home_form_5_list.append(np.mean(home_last_5) if len(home_last_5) > 0 else 0.0)
        away_form_5_list.append(np.mean(away_last_5) if len(away_last_5) > 0 else 0.0)
        home_form_3_list.append(np.mean(home_last_3) if len(home_last_3) > 0 else 0.0)
        away_form_3_list.append(np.mean(away_last_3) if len(away_last_3) > 0 else 0.0)

        result = _safe_result(row)

        if result == "H":
            team_points_history[home].append(3)
            team_points_history[away].append(0)
        elif result == "D":
            team_points_history[home].append(1)
            team_points_history[away].append(1)
        elif result == "A":
            team_points_history[home].append(0)
            team_points_history[away].append(3)

    df["home_form"] = home_form_5_list
    df["away_form"] = away_form_5_list
    df["home_form_3"] = home_form_3_list
    df["away_form_3"] = away_form_3_list

    # =========================================
    # ELO
    # =========================================
    ratings = {}
    home_rating_list = []
    away_rating_list = []
    K = 20

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        ratings.setdefault(home, 1500.0)
        ratings.setdefault(away, 1500.0)

        r_home = ratings[home]
        r_away = ratings[away]

        home_rating_list.append(r_home)
        away_rating_list.append(r_away)

        result = _safe_result(row)
        if result is None:
            continue

        expected_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))

        if result == "H":
            score_home = 1.0
        elif result == "D":
            score_home = 0.5
        else:
            score_home = 0.0

        ratings[home] = r_home + K * (score_home - expected_home)
        ratings[away] = r_away + K * ((1 - score_home) - (1 - expected_home))

    df["home_rating"] = home_rating_list
    df["away_rating"] = away_rating_list

    # =========================================
    # HOME / AWAY POINTS PER MATCH
    # =========================================
    home_points_raw = []
    away_points_raw = []

    for _, row in df.iterrows():
        result = _safe_result(row)

        if result == "H":
            home_points_raw.append(3)
            away_points_raw.append(0)
        elif result == "D":
            home_points_raw.append(1)
            away_points_raw.append(1)
        elif result == "A":
            home_points_raw.append(0)
            away_points_raw.append(3)
        else:
            home_points_raw.append(np.nan)
            away_points_raw.append(np.nan)

    df["home_points_raw"] = home_points_raw
    df["away_points_raw"] = away_points_raw

    df["home_points_home"] = (
        df.groupby("home_team", sort=False)["home_points_raw"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    df["away_points_away"] = (
        df.groupby("away_team", sort=False)["away_points_raw"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # =========================================
    # CLEAN SHEETS HOME / AWAY
    # =========================================
    df["home_clean_sheet_raw"] = np.where(
        df["away_goals"].notna(),
        (df["away_goals"] == 0).astype(float),
        np.nan
    )

    df["away_clean_sheet_raw"] = np.where(
        df["home_goals"].notna(),
        (df["home_goals"] == 0).astype(float),
        np.nan
    )

    df["home_clean_sheets_home"] = (
        df.groupby("home_team", sort=False)["home_clean_sheet_raw"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    df["away_clean_sheets_away"] = (
        df.groupby("away_team", sort=False)["away_clean_sheet_raw"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # =========================================
    # FAILED TO SCORE HOME / AWAY
    # =========================================
    df["home_failed_to_score_raw"] = np.where(
        df["home_goals"].notna(),
        (df["home_goals"] == 0).astype(float),
        np.nan
    )

    df["away_failed_to_score_raw"] = np.where(
        df["away_goals"].notna(),
        (df["away_goals"] == 0).astype(float),
        np.nan
    )

    df["home_failed_to_score_home"] = (
        df.groupby("home_team", sort=False)["home_failed_to_score_raw"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    df["away_failed_to_score_away"] = (
        df.groupby("away_team", sort=False)["away_failed_to_score_raw"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # =========================================
    # ODDS JAKO FEATURE
    # =========================================
    if df["avg_home_odds"].notna().any():
        df["avg_home_odds"] = df["avg_home_odds"].fillna(df["avg_home_odds"].median())
    else:
        df["avg_home_odds"] = 2.5

    if df["avg_draw_odds"].notna().any():
        df["avg_draw_odds"] = df["avg_draw_odds"].fillna(df["avg_draw_odds"].median())
    else:
        df["avg_draw_odds"] = 3.2

    if df["avg_away_odds"].notna().any():
        df["avg_away_odds"] = df["avg_away_odds"].fillna(df["avg_away_odds"].median())
    else:
        df["avg_away_odds"] = 2.8

    # =========================================
    # TARGETY DODATKOWE
    # =========================================
    df["target_1x"] = np.where(df["result"].isin(["H", "D"]), 1, 0)
    df["target_x2"] = np.where(df["result"].isin(["D", "A"]), 1, 0)

    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["target_over25"] = np.where(df["total_goals"] > 2.5, 1, 0)

    # =========================================
    # RÓŻNICE
    # =========================================
    fill_cols = [
        "home_scored_home",
        "home_conceded_home",
        "away_scored_away",
        "away_conceded_away",
        "home_form",
        "away_form",
        "home_form_3",
        "away_form_3",
        "home_rating",
        "away_rating",
        "home_points_home",
        "away_points_away",
        "home_clean_sheets_home",
        "away_clean_sheets_away",
        "home_failed_to_score_home",
        "away_failed_to_score_away",
    ]

    for col in fill_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["form_diff"] = df["home_form"] - df["away_form"]
    df["form_diff_3"] = df["home_form_3"] - df["away_form_3"]
    df["rating_diff"] = df["home_rating"] - df["away_rating"]
    df["goal_diff"] = df["home_scored_home"] - df["away_scored_away"]
    df["points_diff"] = df["home_points_home"] - df["away_points_away"]

    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def build_match_features(df: pd.DataFrame, home_team: str, away_team: str) -> pd.DataFrame:
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date", kind="stable").reset_index(drop=True)

    df_feat = add_features(df)

    played_df = df_feat[
        df_feat["home_goals"].notna() &
        df_feat["away_goals"].notna()
    ].copy()

    home_home_matches = played_df[played_df["home_team"] == home_team].copy()
    away_away_matches = played_df[played_df["away_team"] == away_team].copy()

    home_matches = played_df[
        (played_df["home_team"] == home_team) | (played_df["away_team"] == home_team)
    ].copy()

    away_matches = played_df[
        (played_df["home_team"] == away_team) | (played_df["away_team"] == away_team)
    ].copy()

    def team_points_avg(matches: pd.DataFrame, team: str, last_n: int = 5) -> float:
        points = []

        if matches.empty:
            return 0.0

        matches = matches.sort_values("date", kind="stable").tail(last_n)

        for _, row in matches.iterrows():
            result = str(row["result"]).strip().upper()

            if row["home_team"] == team:
                if result == "H":
                    points.append(3)
                elif result == "D":
                    points.append(1)
                elif result == "A":
                    points.append(0)
            else:
                if result == "A":
                    points.append(3)
                elif result == "D":
                    points.append(1)
                elif result == "H":
                    points.append(0)

        if len(points) == 0:
            return 0.0

        return float(sum(points) / len(points))

    def latest_rating(df_in: pd.DataFrame, team: str) -> float:
        team_rows = df_in[
            (df_in["home_team"] == team) | (df_in["away_team"] == team)
        ].copy()

        if team_rows.empty:
            return 1500.0

        team_rows = team_rows.sort_values("date", kind="stable")
        last_row = team_rows.iloc[-1]

        if last_row["home_team"] == team:
            return float(last_row["home_rating"])
        return float(last_row["away_rating"])

    def mean_or_zero(series: pd.Series) -> float:
        return float(series.mean()) if not series.empty else 0.0

    avg_home_odds = float(pd.to_numeric(df_feat["avg_home_odds"], errors="coerce").median()) if "avg_home_odds" in df_feat.columns else 2.5
    avg_draw_odds = float(pd.to_numeric(df_feat["avg_draw_odds"], errors="coerce").median()) if "avg_draw_odds" in df_feat.columns else 3.2
    avg_away_odds = float(pd.to_numeric(df_feat["avg_away_odds"], errors="coerce").median()) if "avg_away_odds" in df_feat.columns else 2.8

    if pd.isna(avg_home_odds):
        avg_home_odds = 2.5
    if pd.isna(avg_draw_odds):
        avg_draw_odds = 3.2
    if pd.isna(avg_away_odds):
        avg_away_odds = 2.8

    home_form_5 = team_points_avg(home_matches, home_team, 5)
    away_form_5 = team_points_avg(away_matches, away_team, 5)

    home_form_3 = team_points_avg(home_matches, home_team, 3)
    away_form_3 = team_points_avg(away_matches, away_team, 3)

    home_points_home = mean_or_zero(home_home_matches["home_points_raw"]) if "home_points_raw" in home_home_matches.columns else 0.0
    away_points_away = mean_or_zero(away_away_matches["away_points_raw"]) if "away_points_raw" in away_away_matches.columns else 0.0

    home_clean_sheets_home = mean_or_zero(home_home_matches["home_clean_sheet_raw"]) if "home_clean_sheet_raw" in home_home_matches.columns else 0.0
    away_clean_sheets_away = mean_or_zero(away_away_matches["away_clean_sheet_raw"]) if "away_clean_sheet_raw" in away_away_matches.columns else 0.0

    home_failed_to_score_home = mean_or_zero(home_home_matches["home_failed_to_score_raw"]) if "home_failed_to_score_raw" in home_home_matches.columns else 0.0
    away_failed_to_score_away = mean_or_zero(away_away_matches["away_failed_to_score_raw"]) if "away_failed_to_score_raw" in away_away_matches.columns else 0.0

    result_df = pd.DataFrame([{
        "home_scored_home": mean_or_zero(home_home_matches["home_goals"]),
        "home_conceded_home": mean_or_zero(home_home_matches["away_goals"]),
        "away_scored_away": mean_or_zero(away_away_matches["away_goals"]),
        "away_conceded_away": mean_or_zero(away_away_matches["home_goals"]),

        "home_form": float(home_form_5),
        "away_form": float(away_form_5),
        "home_form_3": float(home_form_3),
        "away_form_3": float(away_form_3),

        "home_rating": latest_rating(df_feat, home_team),
        "away_rating": latest_rating(df_feat, away_team),

        "home_points_home": home_points_home,
        "away_points_away": away_points_away,

        "home_clean_sheets_home": home_clean_sheets_home,
        "away_clean_sheets_away": away_clean_sheets_away,

        "home_failed_to_score_home": home_failed_to_score_home,
        "away_failed_to_score_away": away_failed_to_score_away,

        "avg_home_odds": avg_home_odds,
        "avg_draw_odds": avg_draw_odds,
        "avg_away_odds": avg_away_odds,
    }])

    result_df["form_diff"] = result_df["home_form"] - result_df["away_form"]
    result_df["form_diff_3"] = result_df["home_form_3"] - result_df["away_form_3"]
    result_df["rating_diff"] = result_df["home_rating"] - result_df["away_rating"]
    result_df["goal_diff"] = result_df["home_scored_home"] - result_df["away_scored_away"]
    result_df["points_diff"] = result_df["home_points_home"] - result_df["away_points_away"]

    for col in FEATURE_COLS:
        result_df[col] = pd.to_numeric(result_df[col], errors="coerce").fillna(0.0)

    return result_df