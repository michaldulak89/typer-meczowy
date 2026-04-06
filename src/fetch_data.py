import os
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# =========================================
# ŚCIEŻKI
# =========================================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_API_DIR = BASE_DIR / "data" / "raw" / "api_current"
RAW_API_DIR.mkdir(parents=True, exist_ok=True)

# =========================================
# ENV
# =========================================
load_dotenv()
API_KEY = os.getenv("API_FOOTBALL_KEY")

if not API_KEY:
    raise ValueError("Brak API_FOOTBALL_KEY w pliku .env")

BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# =========================================
# KONFIG
# =========================================
CURRENT_SEASON = 2025

LEAGUES = {
    "POL": {"league_id": 106, "name": "Ekstraklasa"},
    "ENG": {"league_id": 39, "name": "Premier League"},
    "FRA": {"league_id": 61, "name": "Ligue 1"},
    "GER": {"league_id": 78, "name": "Bundesliga"},
}

REQUEST_SLEEP = 0.5

# =========================================
# REQUEST
# =========================================
def safe_get(endpoint: str, params: dict) -> dict:
    url = f"{BASE_URL}/{endpoint}"
    response = requests.get(url, headers=HEADERS, params=params, timeout=60)

    try:
        data = response.json()
    except Exception:
        print("❌ Nie udało się odczytać JSON")
        print("URL:", response.url)
        print("STATUS:", response.status_code)
        print(response.text[:1000])
        raise

    print("\n" + "=" * 70)
    print("URL:", response.url)
    print("STATUS:", response.status_code)
    print("RESULTS:", data.get("results"))
    print("ERRORS:", data.get("errors"))

    for k, v in response.headers.items():
        if "ratelimit" in k.lower():
            print(f"{k}: {v}")

    time.sleep(REQUEST_SLEEP)
    return data

# =========================================
# FIXTURES
# =========================================
def normalize_fixture(item: dict, league_code: str) -> dict:
    fixture = item.get("fixture", {})
    league = item.get("league", {})
    teams = item.get("teams", {})
    goals = item.get("goals", {})

    home_goals = goals.get("home")
    away_goals = goals.get("away")

    result = None
    if home_goals is not None and away_goals is not None:
        if home_goals > away_goals:
            result = "H"
        elif home_goals < away_goals:
            result = "A"
        else:
            result = "D"

    return {
        "fixture_id": fixture.get("id"),
        "country": league.get("country"),
        "league": league.get("name"),
        "season": league.get("season"),
        "date": fixture.get("date"),
        "time": None,
        "home_team": teams.get("home", {}).get("name"),
        "away_team": teams.get("away", {}).get("name"),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "result": result,
        "status": fixture.get("status", {}).get("short"),
        "league_code": league_code,
        "avg_home_odds": pd.NA,
        "avg_draw_odds": pd.NA,
        "avg_away_odds": pd.NA,
        "ps_home_odds": pd.NA,
        "ps_draw_odds": pd.NA,
        "ps_away_odds": pd.NA,
        "max_home_odds": pd.NA,
        "max_draw_odds": pd.NA,
        "max_away_odds": pd.NA,
    }

def fetch_fixtures_for_league(league_id: int, league_code: str) -> pd.DataFrame:
    data = safe_get("fixtures", {"league": league_id, "season": CURRENT_SEASON})

    if data.get("errors"):
        print(f"❌ Fixtures error dla {league_code}: {data['errors']}")
        return pd.DataFrame()

    rows = [normalize_fixture(item, league_code) for item in data.get("response", [])]
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)
    return df

# =========================================
# ODDS
# =========================================
def extract_match_winner_odds(bookmaker: dict) -> dict | None:
    for bet in bookmaker.get("bets", []):
        bet_name = str(bet.get("name", "")).strip().lower()

        if bet_name not in ["match winner", "winner", "1x2"]:
            continue

        odds_map = {"1": None, "X": None, "2": None}

        for value in bet.get("values", []):
            label = str(value.get("value", "")).strip().upper()
            odd = pd.to_numeric(value.get("odd"), errors="coerce")

            if label in ["HOME", "1"]:
                odds_map["1"] = odd
            elif label in ["DRAW", "X"]:
                odds_map["X"] = odd
            elif label in ["AWAY", "2"]:
                odds_map["2"] = odd

        return odds_map

    return None

def fetch_odds_for_league(league_id: int) -> pd.DataFrame:
    rows = []
    page = 1

    while True:
        data = safe_get("odds", {
            "league": league_id,
            "season": CURRENT_SEASON,
            "page": page
        })

        if data.get("errors"):
            print(f"❌ Odds error page={page}: {data['errors']}")
            break

        response = data.get("response", [])
        if not response:
            break

        for item in response:
            fixture_id = item.get("fixture", {}).get("id")
            bookmakers = item.get("bookmakers", [])

            for bookmaker in bookmakers:
                odds_map = extract_match_winner_odds(bookmaker)
                if odds_map is None:
                    continue

                rows.append({
                    "fixture_id": fixture_id,
                    "avg_home_odds": odds_map["1"],
                    "avg_draw_odds": odds_map["X"],
                    "avg_away_odds": odds_map["2"],
                    "ps_home_odds": odds_map["1"],
                    "ps_draw_odds": odds_map["X"],
                    "ps_away_odds": odds_map["2"],
                    "max_home_odds": odds_map["1"],
                    "max_draw_odds": odds_map["X"],
                    "max_away_odds": odds_map["2"],
                })

        paging = data.get("paging", {})
        current_page = paging.get("current", page)
        total_pages = paging.get("total", page)

        if current_page >= total_pages:
            break

        page += 1

    if not rows:
        return pd.DataFrame(columns=[
            "fixture_id",
            "avg_home_odds", "avg_draw_odds", "avg_away_odds",
            "ps_home_odds", "ps_draw_odds", "ps_away_odds",
            "max_home_odds", "max_draw_odds", "max_away_odds",
        ])

    odds_df = pd.DataFrame(rows)

    # średnia z bookmakerów dla tego samego meczu
    odds_df = odds_df.groupby("fixture_id", as_index=False).mean(numeric_only=True)
    return odds_df

# =========================================
# MERGE
# =========================================
def merge_fixtures_with_odds(fixtures_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    fixtures_df = fixtures_df.copy()

    odds_cols = [
        "avg_home_odds", "avg_draw_odds", "avg_away_odds",
        "ps_home_odds", "ps_draw_odds", "ps_away_odds",
        "max_home_odds", "max_draw_odds", "max_away_odds",
    ]

    for col in odds_cols:
        if col not in fixtures_df.columns:
            fixtures_df[col] = pd.NA

    if odds_df.empty:
        print("⚠️ Brak odds do scalenia")
        return fixtures_df

    keep_cols = ["fixture_id"] + [c for c in odds_cols if c in odds_df.columns]
    odds_df = odds_df[keep_cols].copy()

    merged = fixtures_df.merge(
        odds_df,
        on="fixture_id",
        how="left",
        suffixes=("", "_from_odds")
    )

    for col in odds_cols:
        from_col = f"{col}_from_odds"
        if from_col in merged.columns:
            merged[col] = merged[col].combine_first(merged[from_col])
            merged.drop(columns=[from_col], inplace=True)

    for col in odds_cols:
        if col not in merged.columns:
            merged[col] = pd.NA

    return merged

# =========================================
# MAIN
# =========================================
def main():
    print("🚀 Start pobierania fixtures + odds z API-Football")

    for league_code, meta in LEAGUES.items():
        print("\n" + "#" * 80)
        print(f"🏆 {meta['name']} ({league_code})")
        print("#" * 80)

        fixtures_df = fetch_fixtures_for_league(meta["league_id"], league_code)
        if fixtures_df.empty:
            print(f"⚠️ Brak fixtures dla {league_code}")
            continue

        odds_df = fetch_odds_for_league(meta["league_id"])
        merged_df = merge_fixtures_with_odds(fixtures_df, odds_df)

        print(f"📊 Fixtures: {len(fixtures_df)}")
        print(f"💰 Odds rows: {len(odds_df)}")

        future_df = merged_df[
            merged_df["home_goals"].isna() | merged_df["away_goals"].isna()
        ].copy()

        if not future_df.empty:
            with_odds = future_df["avg_home_odds"].notna().sum()
            print(f"🔮 Przyszłe mecze: {len(future_df)} | z odds: {with_odds}")
        else:
            print("🔮 Brak przyszłych meczów")

        out_path = RAW_API_DIR / f"{league_code}_api_current.csv"
        merged_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"✅ Zapisano: {out_path}")

if __name__ == "__main__":
    main()