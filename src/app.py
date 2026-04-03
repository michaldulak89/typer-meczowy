import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from features import add_features, build_match_features

st.set_page_config(page_title="AI Typy Meczów", layout="centered")


FEATURE_COLS = [
    "home_scored_home",
    "home_conceded_home",
    "away_scored_away",
    "away_conceded_away",
    "home_form",
    "away_form",
    "home_rating",
    "away_rating",
    "form_diff",
    "rating_diff",
    "goal_diff",
]


@st.cache_data
def load_all_leagues():
    folder = Path("data/processed")
    files = list(folder.glob("*_all_standard.csv"))

    if not files:
        raise ValueError("Brak plików w folderze data/processed")

    dfs = []
    for file in files:
        df = pd.read_csv(file, encoding="utf-8")

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        numeric_cols = [
            "home_goals", "away_goals",
            "ps_home_odds", "ps_draw_odds", "ps_away_odds",
            "max_home_odds", "max_draw_odds", "max_away_odds",
            "avg_home_odds", "avg_draw_odds", "avg_away_odds",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in ["home_team", "away_team", "league", "country", "season", "league_code", "league_name_pl"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


@st.cache_resource
def train_models():
    df_all = load_all_leagues()
    models = {}
    league_data = {}

    for league_code in sorted(df_all["league_code"].dropna().unique()):
        df_league = df_all[df_all["league_code"] == league_code].copy()

        if len(df_league) < 30:
            continue

        df_league = add_features(df_league)

        df_model = df_league[
            (df_league["home_scored_home"] > 0) &
            (df_league["away_scored_away"] > 0) &
            df_league["result"].isin(["H", "D", "A"])
        ].copy()

        if len(df_model) < 30:
            continue

        split_index = int(len(df_model) * 0.8)
        train_df = df_model.iloc[:split_index]

        X_train = train_df[FEATURE_COLS]
        y_train = train_df["result_1x2"]

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            max_depth=6,
            min_samples_leaf=5
        )
        model.fit(X_train, y_train)

        models[league_code] = model
        league_data[league_code] = df_league

    return models, league_data


models, league_data = train_models()

st.title("⚽ AI Typy Meczów")
st.caption("Wybierz ligę, zobacz ostatnią kolejkę, terminarz i przeanalizuj mecz.")

available_leagues = sorted(models.keys())

if not available_leagues:
    st.error("Brak gotowych lig w data/processed. Najpierw uruchom prepare_data.py.")
    st.stop()

league_options = {}
for code in available_leagues:
    df_tmp = league_data[code]
    if "league_name_pl" in df_tmp.columns and df_tmp["league_name_pl"].notna().any():
        display_name = df_tmp["league_name_pl"].dropna().iloc[0]
    else:
        display_name = code
    league_options[display_name] = code

selected_league_name = st.selectbox("🏆 Wybierz ligę", list(league_options.keys()))
league = league_options[selected_league_name]

df = league_data[league]
model = models[league]

st.header("📅 Ostatnia kolejka")

played_matches = df[
    df["home_goals"].notna() &
    df["away_goals"].notna() &
    df["result"].isin(["H", "D", "A"])
].copy()

if not played_matches.empty:
    if df["date"].notna().any():
        latest_played_date = played_matches["date"].max()
        last_round = played_matches[played_matches["date"] == latest_played_date].copy()
    else:
        last_round = played_matches.tail(10).copy()

    if "date" in last_round.columns and last_round["date"].notna().any():
        last_round["Data"] = last_round["date"].dt.strftime("%d-%m-%Y")
    else:
        last_round["Data"] = "-"

    last_round["Wynik"] = (
        last_round["home_goals"].astype(float).astype(int).astype(str)
        + ":"
        + last_round["away_goals"].astype(float).astype(int).astype(str)
    )

    last_round_display = last_round[["Data", "home_team", "away_team", "Wynik"]].rename(columns={
        "home_team": "Gospodarze",
        "away_team": "Goście"
    })

    st.dataframe(last_round_display, use_container_width=True)
else:
    st.info("Brak rozegranych meczów w tej lidze.")

st.header("🗓️ Najbliższy terminarz")

today = pd.Timestamp.today().normalize()

if "date" in df.columns and df["date"].notna().any():
    fixtures = df[
        (df["date"] >= today) &
        (
            df["home_goals"].isna() |
            df["away_goals"].isna() |
            ~df["result"].isin(["H", "D", "A"])
        )
    ].copy()

    fixtures = fixtures.sort_values("date")

    if not fixtures.empty:
        fixtures["Data"] = fixtures["date"].dt.strftime("%d-%m-%Y")

        fixtures_display = fixtures[["Data", "home_team", "away_team"]].rename(columns={
            "home_team": "Gospodarze",
            "away_team": "Goście"
        })

        st.dataframe(fixtures_display, use_container_width=True)
    else:
        st.info("Brak przyszłych meczów w tej lidze.")
else:
    st.info("Brak poprawnych dat w danych tej ligi.")

st.header("🔎 Analiza meczu")

teams = sorted(
    {
        str(team).strip()
        for team in pd.concat([df["home_team"], df["away_team"]]).dropna()
        if str(team).strip() != ""
    }
)

home_team = st.selectbox("Gospodarz", teams)
away_team = st.selectbox("Gość", teams)

if home_team == away_team:
    st.warning("Wybierz dwie różne drużyny.")
else:
    if st.button("Analizuj"):
        features = build_match_features(df, home_team, away_team)
        features = features[FEATURE_COLS]

        proba = model.predict_proba(features)[0]
        classes = model.classes_
        probs = dict(zip(classes, proba))

        p1 = probs.get("1", 0)
        px = probs.get("X", 0)
        p2 = probs.get("2", 0)

        best_type = max(probs, key=probs.get)
        confidence = probs[best_type]

        typ_map = {"1": "Gospodarze", "X": "Remis", "2": "Goście"}

        st.subheader(f"{home_team} vs {away_team}")
        st.write("### 📊 Prognoza AI")

        st.write(f"🏠 Gospodarze: {p1*100:.1f}%")
        st.progress(float(p1))

        st.write(f"🤝 Remis: {px*100:.1f}%")
        st.progress(float(px))

        st.write(f"✈️ Goście: {p2*100:.1f}%")
        st.progress(float(p2))

        st.success(f"🔥 Typ: {typ_map[best_type]}")

        if confidence > 0.70:
            st.success(f"✅ Wysoka pewność: {confidence*100:.1f}%")
        elif confidence > 0.55:
            st.warning(f"⚠️ Średnia pewność: {confidence*100:.1f}%")
        else:
            st.error(f"❌ Niska pewność: {confidence*100:.1f}%")

        if confidence < 0.55:
            st.warning("Ten mecz nie wygląda na mocny typ według modelu.")

        features_display = features.rename(columns={
            "home_scored_home": "Gole gospodarzy u siebie",
            "home_conceded_home": "Stracone gospodarzy u siebie",
            "away_scored_away": "Gole gości na wyjeździe",
            "away_conceded_away": "Stracone gości na wyjeździe",
            "home_form": "Forma gospodarzy",
            "away_form": "Forma gości",
            "home_rating": "Siła gospodarzy",
            "away_rating": "Siła gości",
            "form_diff": "Różnica formy",
            "rating_diff": "Różnica siły",
            "goal_diff": "Różnica potencjału bramkowego",
        })

        with st.expander("🔍 Szczegóły analizy"):
            st.dataframe(features_display, use_container_width=True)