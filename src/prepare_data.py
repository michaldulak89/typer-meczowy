import pandas as pd
from pathlib import Path

# =========================================
# ALIASY KOLUMN -> STANDARD BACKENDU
# =========================================
COLUMN_ALIASES = {
    "country": ["Country", "country", "Kraj"],
    "league": ["League", "league", "Liga", "Div"],
    "season": ["Season", "season", "Sezon"],
    "date": ["Date", "date", "Data", "MatchDate", "GameDate"],
    "time": ["Time", "time", "Godzina"],

    "home_team": ["Home", "HomeTeam", "home_team", "Gospodarze"],
    "away_team": ["Away", "AwayTeam", "away_team", "Goście", "Goscie"],

    "home_goals": ["HG", "FTHG", "home_goals", "GoleDom", "GoalsHome"],
    "away_goals": ["AG", "FTAG", "away_goals", "GoleWyjazd", "GoalsAway"],

    "result": ["Res", "FTR", "result", "Wynik"],

    "ps_home_odds": ["PSCH", "PSH"],
    "ps_draw_odds": ["PSCD", "PSD"],
    "ps_away_odds": ["PSCA", "PSA"],

    "max_home_odds": ["MaxCH", "MaxH"],
    "max_draw_odds": ["MaxCD", "MaxD"],
    "max_away_odds": ["MaxCA", "MaxA"],

    "avg_home_odds": ["AvgCH", "AvgH"],
    "avg_draw_odds": ["AvgCD", "AvgD"],
    "avg_away_odds": ["AvgCA", "AvgA"],
}

# =========================================
# KOLUMNY WYMAGANE
# =========================================
REQUIRED_COLUMNS = [
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
]

# =========================================
# KOLUMNY, KTÓRE ZOSTAWIAMY
# =========================================
KEEP_COLUMNS = [
    "country",
    "league",
    "season",
    "date",
    "time",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    "ps_home_odds",
    "ps_draw_odds",
    "ps_away_odds",
    "max_home_odds",
    "max_draw_odds",
    "max_away_odds",
    "avg_home_odds",
    "avg_draw_odds",
    "avg_away_odds",
]

# =========================================
# POPRAWKI NAZW DRUŻYN
# =========================================
TEAM_NAME_FIXES = {
    "Lech Poznan": "Lech Poznań",
    "Rakow Czestochowa": "Raków Częstochowa",
    "Legia": "Legia Warszawa",
    "Legia W-wa": "Legia Warszawa",
    "Legia Warsaw": "Legia Warszawa",
    "Wisla Krakow": "Wisła Kraków",
    "Wisla": "Wisła Kraków",
    "Cracovia Krakow": "Cracovia",
    "Gornik Zabrze": "Górnik Zabrze",
    "Gornik": "Górnik Zabrze",
    "Zaglebie Lubin": "Zagłębie Lubin",
    "Slask Wroclaw": "Śląsk Wrocław",
    "Gornik Z.": "Górnik Zabrze",
    "Rakow": "Raków Częstochowa",
    "Leczna": "Górnik Łęczna",
    "Lechia Gdansk": "Lechia Gdańsk",
    "Raków": "Raków Częstochowa",
    "Ruch": "Ruch Chorzów",
    "Sandecja Nowy S.": "Sandecja Nowy Sącz",
    "Wisla Plock": "Wisła Płock",
    "Zaglebie Sosnowiec": "Zagłębie Sosnowiec",
    "Zawisza": "Zawisza Bydgoszcz",
    "Zaglebie": "Zagłębie Lubin",
    "Termalica Nieciecza": "Bruk-Bet Termalica",
    "Puszcza": "Puszcza Niepołomice",
    "Jagiellonia": "Jagiellonia Białystok",
    "Jagiellonia Bialystok": "Jagiellonia Białystok",
    "Warta Poznan": "Warta Poznań",
    "Widzew Lodz": "Widzew Łódź",
    "LKS Lodz": "ŁKS Łódź",
    "LKS": "ŁKS Łódź",
    "Pogon Szczecin": "Pogoń Szczecin",
    "Puszcza Niepolomice": "Puszcza Niepołomice",
    "Ruch Chorzow": "Ruch Chorzów",
    "Termalica B-B.": "Bruk-Bet Termalica",
    "Nieciecza": "Bruk-Bet Termalica",
}

# =========================================
# MAPA KODÓW LIG -> POLSKIE NAZWY
# =========================================
LEAGUE_SMART_MAP = {
    # Polska
    "pol": "Polska",

    # Anglia
    "eng": "Anglia",
    "e0": "Anglia",

    # Hiszpania
    "esp": "Hiszpania",
    "spa": "Hiszpania",

    # Niemcy
    "ger": "Niemcy",
    "d1": "Niemcy",

    # Włochy
    "ita": "Włochy",
    "i1": "Włochy",

    # Francja
    "fra": "Francja",
    "f1": "Francja",

    # Portugalia
    "por": "Portugalia",
    "port": "Portugalia",

    # MLS
    "usa": "MLS",
    "mls": "MLS",
}
def auto_league_name(code: str) -> str:
    text = str(code).lower()

    if text == "usa":
        return "MLS"

    for key, value in LEAGUE_SMART_MAP.items():
        if key in text:
            return value

    return text.capitalize()


def fix_text(value):
    if pd.isna(value):
        return value

    text = str(value).strip()

    # typowe objawy złego kodowania
    if any(bad in text for bad in ["Ã", "Å", "Ä", "�"]):
        try:
            text = text.encode("latin1").decode("utf-8")
        except Exception:
            pass

    return text


def read_csv_auto(path):
    separators = [";", ",", "\t"]
    encodings = ["utf-8", "utf-8-sig", "cp1250", "latin1"]

    for sep in separators:
        for enc in encodings:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc)
                if len(df.columns) > 1:
                    print(f"✔ Wykryto separator '{sep}' i encoding '{enc}' dla pliku {Path(path).name}")
                    return df
            except Exception:
                pass

    raise ValueError(f"❌ Nie udało się wczytać pliku: {path}")


def build_rename_map(df: pd.DataFrame) -> dict:
    rename_map = {}
    existing_cols = {str(col).strip().lower(): col for col in df.columns}

    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_key = alias.strip().lower()
            if alias_key in existing_cols:
                rename_map[existing_cols[alias_key]] = standard_name
                break

    return rename_map


def map_result(x):
    x = str(x).strip().upper()
    if x == "H":
        return "1"
    if x == "D":
        return "X"
    if x == "A":
        return "2"
    return None


def clean_team_name(name):
    if pd.isna(name):
        return name
    name = fix_text(name)
    return TEAM_NAME_FIXES.get(name, name)


def normalize_league_code(name: str) -> str:
    text = str(name).strip().upper()

    if text.startswith("POL"):
        return "POL"
    if text.startswith("ESP") or text.startswith("SPA"):
        return "ESP"
    if text.startswith("GER") or text.startswith("D1"):
        return "GER"
    if text.startswith("ENG") or text.startswith("E0"):
        return "ENG"
    if text.startswith("ITA") or text.startswith("I1"):
        return "ITA"
    if text.startswith("FRA") or text.startswith("F1"):
        return "FRA"

    return text


def prepare_data(input_path, output_path):
    print(f"\n🚀 Start przetwarzania: {input_path}")

    df = read_csv_auto(input_path)
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = build_rename_map(df)
    df = df.rename(columns=rename_map)

    # zostaw tylko kolumny standardowe, które istnieją
    df = df[[col for col in KEEP_COLUMNS if col in df.columns]]

    # sprawdź wymagane
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"❌ Brakuje kluczowych kolumn: {missing}")

    # naprawa tekstu
    text_cols = ["country", "league", "season", "time", "home_team", "away_team", "result"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(fix_text)

    # czyszczenie nazw drużyn
    df["home_team"] = df["home_team"].apply(clean_team_name)
    df["away_team"] = df["away_team"].apply(clean_team_name)

    # konwersje typów
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    if "home_goals" in df.columns:
        df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")

    if "away_goals" in df.columns:
        df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")

    odds_cols = [
        "ps_home_odds", "ps_draw_odds", "ps_away_odds",
        "max_home_odds", "max_draw_odds", "max_away_odds",
        "avg_home_odds", "avg_draw_odds", "avg_away_odds"
    ]
    for col in odds_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # dodatkowe kolumny systemowe
    df["result_1x2"] = df["result"].apply(map_result)
    df["is_played"] = df["result"].isin(["H", "D", "A"])
    df["is_future"] = ~df["is_played"]
    df["source_file"] = Path(input_path).name

    raw_code = normalize_league_code(Path(input_path).stem)
    df["league_code"] = raw_code
    df["league_name_pl"] = auto_league_name(raw_code)

    if "date" in df.columns:
        df = df.sort_values("date", na_position="last")

    df = df.reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"✅ Zapisano: {output_path}")
    print(f"📊 Liczba rekordów: {len(df)}")
    print(f"📌 Kolumny końcowe: {list(df.columns)}")


if __name__ == "__main__":
    print("🚀 Start przetwarzania wszystkich lig (foldery)...")

    raw_root = Path("data/raw")
    processed_root = Path("data/processed")

    if not raw_root.exists():
        print("❌ Folder data/raw nie istnieje")
    else:
        league_dirs = [d for d in raw_root.iterdir() if d.is_dir()]

        if not league_dirs:
            print("❌ Brak folderów lig w data/raw/")
        else:
            for league_dir in league_dirs:
                league_code = normalize_league_code(league_dir.name)
                league_name_pl = auto_league_name(league_code)

                print(f"\n🏆 Liga: {league_name_pl} ({league_code})")

                csv_files = list(league_dir.glob("*.csv"))

                if not csv_files:
                    print(f"⚠️ Brak plików CSV w {league_dir}")
                    continue

                dfs = []

                for file in csv_files:
                    try:
                        temp_output = processed_root / f"_temp_{file.stem}.csv"

                        prepare_data(file, temp_output)

                        df_temp = pd.read_csv(temp_output, encoding="utf-8")
                        dfs.append(df_temp)

                        temp_output.unlink()

                    except Exception as e:
                        print(f"❌ Błąd w pliku {file.name}: {e}")

                if not dfs:
                    print(f"❌ Nie udało się przetworzyć ligi {league_code}")
                    continue

                combined = pd.concat(dfs, ignore_index=True)

                # naprawa tekstu jeszcze raz po scaleniu
                for col in ["country", "league", "season", "time", "home_team", "away_team", "result"]:
                    if col in combined.columns:
                        combined[col] = combined[col].apply(fix_text)

                if "date" in combined.columns:
                    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
                    combined = combined.sort_values("date")

                combined["league_code"] = league_code
                combined["league_name_pl"] = auto_league_name(league_code)

                combined = combined.drop_duplicates().reset_index(drop=True)

                output_path = processed_root / f"{league_code}_all_standard.csv"
                combined.to_csv(output_path, index=False, encoding="utf-8")

                print(f"✅ Gotowa liga: {output_path}")
                print(f"📊 Liczba meczów: {len(combined)}")

    print("\n🔥 Wszystkie ligi przetworzone!")