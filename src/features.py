import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    # Średnie gole przed meczem
    team_stats = {}
    home_avg_scored = []
    away_avg_scored = []
    home_avg_conceded = []
    away_avg_conceded = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        if home not in team_stats:
            team_stats[home] = {"scored": [], "conceded": []}
        if away not in team_stats:
            team_stats[away] = {"scored": [], "conceded": []}

        home_scored_hist = team_stats[home]["scored"]
        home_conceded_hist = team_stats[home]["conceded"]
        away_scored_hist = team_stats[away]["scored"]
        away_conceded_hist = team_stats[away]["conceded"]

        home_avg_scored.append(np.mean(home_scored_hist) if home_scored_hist else 0)
        home_avg_conceded.append(np.mean(home_conceded_hist) if home_conceded_hist else 0)
        away_avg_scored.append(np.mean(away_scored_hist) if away_scored_hist else 0)
        away_avg_conceded.append(np.mean(away_conceded_hist) if away_conceded_hist else 0)

        if pd.notna(row["home_goals"]) and pd.notna(row["away_goals"]):
            team_stats[home]["scored"].append(row["home_goals"])
            team_stats[home]["conceded"].append(row["away_goals"])
            team_stats[away]["scored"].append(row["away_goals"])
            team_stats[away]["conceded"].append(row["home_goals"])

    df["home_avg_scored_pre"] = home_avg_scored
    df["home_avg_conceded_pre"] = home_avg_conceded
    df["away_avg_scored_pre"] = away_avg_scored
    df["away_avg_conceded_pre"] = away_avg_conceded

    # Forma ogólna
    form_points_home = []
    form_points_away = []
    team_results = {}

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        result = str(row["result"]).strip().upper()

        if home not in team_results:
            team_results[home] = []
        if away not in team_results:
            team_results[away] = []

        home_form = team_results[home][-5:]
        away_form = team_results[away][-5:]

        form_points_home.append(sum(home_form) if home_form else 0)
        form_points_away.append(sum(away_form) if away_form else 0)

        if result == "H":
            team_results[home].append(3)
            team_results[away].append(0)
        elif result == "D":
            team_results[home].append(1)
            team_results[away].append(1)
        elif result == "A":
            team_results[home].append(0)
            team_results[away].append(3)

    df["home_form"] = form_points_home
    df["away_form"] = form_points_away

    # Średnie dom / wyjazd przed meczem
    home_scored_home = []
    home_conceded_home = []
    away_scored_away = []
    away_conceded_away = []

    home_stats = {}
    away_stats = {}

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        if home not in home_stats:
            home_stats[home] = {"scored": [], "conceded": []}
        if away not in away_stats:
            away_stats[away] = {"scored": [], "conceded": []}

        h_scored = home_stats[home]["scored"]
        h_conceded = home_stats[home]["conceded"]
        a_scored = away_stats[away]["scored"]
        a_conceded = away_stats[away]["conceded"]

        home_scored_home.append(np.mean(h_scored) if h_scored else 0)
        home_conceded_home.append(np.mean(h_conceded) if h_conceded else 0)
        away_scored_away.append(np.mean(a_scored) if a_scored else 0)
        away_conceded_away.append(np.mean(a_conceded) if a_conceded else 0)

        if pd.notna(row["home_goals"]) and pd.notna(row["away_goals"]):
            home_stats[home]["scored"].append(row["home_goals"])
            home_stats[home]["conceded"].append(row["away_goals"])
            away_stats[away]["scored"].append(row["away_goals"])
            away_stats[away]["conceded"].append(row["home_goals"])

    df["home_scored_home"] = home_scored_home
    df["home_conceded_home"] = home_conceded_home
    df["away_scored_away"] = away_scored_away
    df["away_conceded_away"] = away_conceded_away

    # Wypełniamy tylko kolumny liczbowe
    numeric_fill_cols = [
        "home_avg_scored_pre",
        "home_avg_conceded_pre",
        "away_avg_scored_pre",
        "away_avg_conceded_pre",
        "home_form",
        "away_form",
        "home_scored_home",
        "home_conceded_home",
        "away_scored_away",
        "away_conceded_away",
    ]

    for col in numeric_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Elo rating
    ratings = {}
    home_rating = []
    away_rating = []
    K = 20

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        result = str(row["result"]).strip().upper()

        ratings.setdefault(home, 1500)
        ratings.setdefault(away, 1500)

        r_home = ratings[home]
        r_away = ratings[away]

        home_rating.append(r_home)
        away_rating.append(r_away)

        expected = 1 / (1 + 10 ** ((r_away - r_home) / 400))

        if result == "H":
            score = 1
        elif result == "D":
            score = 0.5
        elif result == "A":
            score = 0
        else:
            score = None

        if score is not None:
            ratings[home] += K * (score - expected)
            ratings[away] += K * ((1 - score) - (1 - expected))

    df["home_rating"] = home_rating
    df["away_rating"] = away_rating

    # Różnice
    df["form_diff"] = df["home_form"] - df["away_form"]
    df["rating_diff"] = df["home_rating"] - df["away_rating"]
    df["goal_diff"] = df["home_scored_home"] - df["away_scored_away"]

    return df


def build_match_features(df: pd.DataFrame, home_team: str, away_team: str) -> pd.DataFrame:
    df = add_features(df)

    home_matches = df[(df["home_team"] == home_team) | (df["away_team"] == home_team)].copy()
    away_matches = df[(df["home_team"] == away_team) | (df["away_team"] == away_team)].copy()

    home_home = df[df["home_team"] == home_team].copy()
    away_away = df[df["away_team"] == away_team].copy()

    def team_points(matches: pd.DataFrame, team: str, last_n: int = 5) -> int:
        points = []
        for _, row in matches.tail(last_n).iterrows():
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
        return sum(points)

    home_form = team_points(home_matches, home_team, 5)
    away_form = team_points(away_matches, away_team, 5)

    home_scored_home = home_home["home_goals"].mean() if not home_home.empty else 0
    home_conceded_home = home_home["away_goals"].mean() if not home_home.empty else 0
    away_scored_away = away_away["away_goals"].mean() if not away_away.empty else 0
    away_conceded_away = away_away["home_goals"].mean() if not away_away.empty else 0

    latest_home = home_matches.iloc[-1:] if not home_matches.empty else pd.DataFrame()
    latest_away = away_matches.iloc[-1:] if not away_matches.empty else pd.DataFrame()

    if latest_home.empty:
        home_rating = 1500
    else:
        last_row = latest_home.iloc[0]
        home_rating = last_row["home_rating"] if last_row["home_team"] == home_team else last_row["away_rating"]

    if latest_away.empty:
        away_rating = 1500
    else:
        last_row = latest_away.iloc[0]
        away_rating = last_row["home_rating"] if last_row["home_team"] == away_team else last_row["away_rating"]

    return pd.DataFrame([{
        "home_scored_home": home_scored_home,
        "home_conceded_home": home_conceded_home,
        "away_scored_away": away_scored_away,
        "away_conceded_away": away_conceded_away,
        "home_form": home_form,
        "away_form": away_form,
        "home_rating": home_rating,
        "away_rating": away_rating,
        "form_diff": home_form - away_form,
        "rating_diff": home_rating - away_rating,
        "goal_diff": home_scored_home - away_scored_away,
    }])