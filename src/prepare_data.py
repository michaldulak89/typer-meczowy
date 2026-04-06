import pandas as pd
from pathlib import Path

# =========================================
# ŚCIEŻKI
# =========================================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
API_CURRENT_DIR = RAW_DIR / "api_current"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# =========================================
# MAPA LIG
# =========================================
LEAGUE_MAP = {
    "POL": "Polska",
    "ENG": "Anglia",
    "ESP": "Hiszpania",
    "GER": "Niemcy",
    "ITA": "Włochy",
    "FRA": "Francja",
    "POR": "Portugalia",
    "USA": "USA",
    "UCL": "Liga Mistrzów",
}

# =========================================
# ALIASY KOLUMN
# =========================================
COLUMN_ALIASES = {
    "date": ["Date", "date", "Data", "MatchDate", "GameDate"],
    "time": ["Time", "time", "Godzina"],
    "home_team": ["HomeTeam", "Home", "home_team", "Gospodarze"],
    "away_team": ["AwayTeam", "Away", "away_team", "Goście", "Goscie"],
    "home_goals": ["FTHG", "HG", "home_goals", "GoalsHome"],
    "away_goals": ["FTAG", "AG", "away_goals", "GoalsAway"],
    "result": ["FTR", "Res", "result", "Wynik"],
    "status": ["status", "Status"],
    "country": ["Country", "country", "Kraj"],
    "league": ["League", "league", "Liga", "Div"],
    "season": ["Season", "season", "Sezon"],
    "league_code": ["league_code", "LeagueCode"],

    "ps_home_odds": ["PSCH", "PSH", "ps_home_odds"],
    "ps_draw_odds": ["PSCD", "PSD", "ps_draw_odds"],
    "ps_away_odds": ["PSCA", "PSA", "ps_away_odds"],

    "max_home_odds": ["MaxCH", "MaxH", "max_home_odds"],
    "max_draw_odds": ["MaxCD", "MaxD", "max_draw_odds"],
    "max_away_odds": ["MaxCA", "MaxA", "max_away_odds"],

    "avg_home_odds": ["AvgCH", "AvgH", "avg_home_odds"],
    "avg_draw_odds": ["AvgCD", "AvgD", "avg_draw_odds"],
    "avg_away_odds": ["AvgCA", "AvgA", "avg_away_odds"],
}

# =========================================
# KOŃCOWE KOLUMNY
# =========================================
FINAL_COLUMNS = [
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
    "status",
    "ps_home_odds",
    "ps_draw_odds",
    "ps_away_odds",
    "max_home_odds",
    "max_draw_odds",
    "max_away_odds",
    "avg_home_odds",
    "avg_draw_odds",
    "avg_away_odds",
    "league_code",
    "league_name_pl",
    "result_1x2",
    "is_played",
    "is_future",
    "source_file",
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
# POMOCNICZE
# =========================================
def normalize_league_code(name: str) -> str:
    text = str(name).strip().upper()

    if "POL" in text or "POLSKA" in text:
        return "POL"
    if "ENG" in text or "ENGLAND" in text or "E0" in text:
        return "ENG"
    if "ESP" in text or "SPA" in text or "SPAIN" in text:
        return "ESP"
    if "GER" in text or "GERMANY" in text or "D1" in text:
        return "GER"
    if "ITA" in text or "ITALY" in text or "I1" in text:
        return "ITA"
    if "FRA" in text or "FRANCE" in text or "F1" in text:
        return "FRA"
    if "POR" in text or "PORTUGAL" in text:
        return "POR"
    if "USA" in text or "MLS" in text:
        return "USA"
    if "UCL" in text or "CHAMPIONS" in text or text == "2":
        return "UCL"

    return "UNK"


def league_name_from_code(code: str) -> str:
    return LEAGUE_MAP.get(code, code)


def fix_text(value):
    if pd.isna(value):
        return value

    text = str(value).strip()

    if any(bad in text for bad in ["Ã", "Å", "Ä", "�"]):
        try:
            text = text.encode("latin1").decode("utf-8")
        except Exception:
            pass

    return text


def fix_team(name):
    if pd.isna(name):
        return name
    name = fix_text(name)
    return TEAM_NAME_FIXES.get(name, name)


def map_result(x):
    x = str(x).strip().upper()
    if x == "H":
        return "1"
    if x == "D":
        return "X"
    if x == "A":
        return "2"
    return None


def derive_result_from_goals(home_goals, away_goals):
    if pd.isna(home_goals) or pd.isna(away_goals):
        return None
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def parse_date_safe(series: pd.Series) -> pd.Series:
    """
    Inteligentne parsowanie dat:
    - wykrywa ISO (API) → YYYY-MM-DD → normalnie
    - wykrywa europejskie → DD.MM / DD/MM → dayfirst=True
    """

    s = series.astype("string").str.strip()

    # 🔍 wykrycie ISO (np. 2025-08-15)
    iso_mask = s.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)

    parsed = pd.Series(pd.NaT, index=s.index)

    # 1. ISO (API)
    if iso_mask.any():
        parsed.loc[iso_mask] = pd.to_datetime(
            s.loc[iso_mask],
            errors="coerce"
        )

    # 2. Europejskie (CSV)
    if (~iso_mask).any():
        parsed.loc[~iso_mask] = pd.to_datetime(
            s.loc[~iso_mask],
            dayfirst=True,
            errors="coerce"
        )

    return parsed


def read_csv_auto(path: Path) -> pd.DataFrame:
    separators = [";", ",", "\t"]
    encodings = ["utf-8", "utf-8-sig", "cp1250", "latin1"]

    for sep in separators:
        for enc in encodings:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc)
                if len(df.columns) > 1:
                    print(f"✔ Wczytano {path.name} | sep='{sep}' | enc='{enc}'")
                    return df
            except Exception:
                pass

    raise ValueError(f"❌ Nie udało się wczytać pliku: {path}")


def build_rename_map(df: pd.DataFrame) -> dict:
    rename_map = {}
    existing = {str(col).strip().lower(): col for col in df.columns}

    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_key = alias.strip().lower()
            if alias_key in existing:
                rename_map[existing[alias_key]] = standard_name
                break

    return rename_map


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def standardize_dataframe(df: pd.DataFrame, league_code: str, source_file: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = build_rename_map(df)
    df = df.rename(columns=rename_map)

    required = ["home_team", "away_team", "date"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Brak kolumny {col}")

    # jeśli nie ma result, dodaj pustą
    if "result" not in df.columns:
        df["result"] = pd.NA

    df = ensure_columns(df)

    # tekst
    text_cols = ["country", "league", "season", "time", "home_team", "away_team", "result", "status"]
    for col in text_cols:
        df[col] = df[col].apply(fix_text)

    # drużyny
    df["home_team"] = df["home_team"].apply(fix_team)
    df["away_team"] = df["away_team"].apply(fix_team)

    # daty - JEDEN FORMAT WEWNĘTRZNY
    df["date"] = parse_date_safe(df["date"])

    nat_count = df["date"].isna().sum()
    if nat_count > 0:
        print(f"⚠ Usuwam rekordy z błędną datą: {nat_count}")
        preview_cols = [c for c in ["home_team", "away_team"] if c in df.columns]
        if preview_cols:
            print(df[df["date"].isna()][preview_cols].head())
        df = df.dropna(subset=["date"]).copy()

    # liczby
    numeric_cols = [
        "home_goals", "away_goals",
        "ps_home_odds", "ps_draw_odds", "ps_away_odds",
        "max_home_odds", "max_draw_odds", "max_away_odds",
        "avg_home_odds", "avg_draw_odds", "avg_away_odds",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # wynik z goli jeśli pusty
    df["result"] = df.apply(
        lambda row: row["result"]
        if pd.notna(row["result"]) and str(row["result"]).strip() != ""
        else derive_result_from_goals(row["home_goals"], row["away_goals"]),
        axis=1
    )

    # systemowe
    df["league_code"] = league_code
    df["league_name_pl"] = league_name_from_code(league_code)
    df["result_1x2"] = df["result"].apply(map_result)
    df["is_played"] = df["result"].isin(["H", "D", "A"])
    df["is_future"] = ~df["is_played"]
    df["source_file"] = source_file

    df = df[FINAL_COLUMNS].copy()
    df = df.sort_values("date", na_position="last").reset_index(drop=True)

    print(f"📊 Rekordy po obróbce: {len(df)}")
    print(f"🧪 NaT po parsowaniu: {df['date'].isna().sum()}")

    return df


def process_history_files() -> dict:
    grouped = {}

    if not RAW_DIR.exists():
        return grouped

    for league_dir in sorted(RAW_DIR.iterdir()):
        if not league_dir.is_dir():
            continue
        if league_dir.name == "api_current":
            continue

        league_code = normalize_league_code(league_dir.name)
        csv_files = sorted(league_dir.glob("*.csv"))

        if not csv_files:
            continue

        print("\n" + "=" * 60)
        print(f"🏆 Historia | {league_dir.name} -> {league_code}")
        print("=" * 60)

        for file in csv_files:
            try:
                raw_df = read_csv_auto(file)
                clean_df = standardize_dataframe(raw_df, league_code, file.name)
                grouped.setdefault(league_code, []).append(clean_df)
                print(f"✅ Historia OK: {file.name}")
            except Exception as e:
                print(f"❌ Błąd historii {file.name}: {e}")

    return grouped


def process_api_current_files(grouped: dict) -> dict:
    if not API_CURRENT_DIR.exists():
        print("ℹ Brak folderu data/raw/api_current")
        return grouped

    api_files = sorted(API_CURRENT_DIR.glob("*_api_current.csv"))
    if not api_files:
        print("ℹ Brak plików *_api_current.csv")
        return grouped

    print("\n" + "=" * 60)
    print("🌐 Obecny sezon z API")
    print("=" * 60)

    for file in api_files:
        try:
            league_code = normalize_league_code(file.stem)
            raw_df = read_csv_auto(file)
            clean_df = standardize_dataframe(raw_df, league_code, file.name)
            grouped.setdefault(league_code, []).append(clean_df)
            print(f"✅ API OK: {file.name} | liga: {league_code}")
        except Exception as e:
            print(f"❌ Błąd API {file.name}: {e}")

    return grouped


def save_processed_files(grouped: dict):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for league_code, dfs in grouped.items():
        if league_code == "UNK":
            print("⚠ Pomijam UNK")
            continue

        if not dfs:
            continue

        combined = pd.concat(dfs, ignore_index=True).copy()

        combined["date"] = parse_date_safe(combined["date"])
        combined = combined.dropna(subset=["date"]).copy()

        for col in ["home_team", "away_team", "league", "country", "season", "result", "status"]:
            if col in combined.columns:
                combined[col] = combined[col].astype("string").str.strip()

        combined = combined.sort_values("date", na_position="last")

        # ten sam mecz - zostaw ostatni rekord
        combined = combined.drop_duplicates(
            subset=["date", "home_team", "away_team"],
            keep="last"
        ).reset_index(drop=True)

        # systemowe jeszcze raz
        combined["league_code"] = league_code
        combined["league_name_pl"] = league_name_from_code(league_code)
        combined["result_1x2"] = combined["result"].apply(map_result)
        combined["is_played"] = combined["result"].isin(["H", "D", "A"])
        combined["is_future"] = ~combined["is_played"]

        # 🔥 JEDEN FORMAT DATY PRZY ZAPISIE
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")

        out_path = PROCESSED_DIR / f"{league_code}_all_standard.csv"
        combined.to_csv(out_path, index=False, encoding="utf-8")

        print(f"\n✅ Zapisano: {out_path}")
        print(f"🏆 Liga: {league_name_from_code(league_code)} ({league_code})")
        print(f"📊 Rekordy: {len(combined)}")


def main():
    print("🚀 Start prepare_data")

    grouped = process_history_files()
    grouped = process_api_current_files(grouped)

    if not grouped:
        print("❌ Brak danych do przetworzenia")
        return

    save_processed_files(grouped)

    print("\n🔥 GOTOWE")


if __name__ == "__main__":
    main()