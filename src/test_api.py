import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")
HEADERS = {"x-apisports-key": API_KEY}
BASE_URL = "https://v3.football.api-sports.io"

def test_league(league_id: int, season: int):
    print(f"\n=== TEST LEAGUE={league_id} SEASON={season} ===")

    r = requests.get(
        f"{BASE_URL}/fixtures",
        headers=HEADERS,
        params={"league": league_id, "season": season},
        timeout=30,
    )

    print("STATUS CODE:", r.status_code)

    data = r.json()
    print("ERRORS:", data.get("errors"))
    print("RESULTS:", data.get("results"))

    response = data.get("response", [])
    if response:
        first = response[0]
        print("FIRST MATCH:")
        print({
            "date": first.get("fixture", {}).get("date"),
            "home": first.get("teams", {}).get("home", {}).get("name"),
            "away": first.get("teams", {}).get("away", {}).get("name"),
            "home_goals": first.get("goals", {}).get("home"),
            "away_goals": first.get("goals", {}).get("away"),
            "status": first.get("fixture", {}).get("status", {}).get("short"),
            "league": first.get("league", {}).get("name"),
            "season": first.get("league", {}).get("season"),
        })
    else:
        print("BRAK MECZÓW W RESPONSE")

if __name__ == "__main__":
    test_league(106, 2025)   # Polska
    test_league(39, 2025)    # Anglia
    test_league(61, 2025)    # Francja
    test_league(78, 2025)    # Niemcy