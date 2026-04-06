import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from features import add_features, build_match_features
import os
from pathlib import Path
import streamlit as st

st.write("CWD:", os.getcwd())
st.write("ROOT EXISTS:", Path(".").resolve())
st.write("SRC EXISTS:", Path("src").exists())
st.write("DATA EXISTS:", Path("data").exists())
st.write("PROCESSED EXISTS:", Path("data/processed").exists())
st.write("PROCESSED FILES:", [f.name for f in Path("data/processed").glob("*_all_standard.csv")])

st.set_page_config(page_title="AI Typy Meczów PRO", layout="centered")

# =========================================
# FEATURE COLUMNS
# =========================================
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

# =========================================
# LIGI
# =========================================
LEAGUE_NAMES = {
    "POL": "Polska",
    "ENG": "Anglia",
    "FRA": "Francja",
    "GER": "Niemcy",
}

ALLOWED_LEAGUES = ["FRA", "GER", "POL", "ENG"]

LEAGUE_SETTINGS = {
    "POL": {
        "min_odds": 2.1,
        "max_odds": 3.0,
        "threshold": 0.05,
        "min_prob": 0.55,
    },
    "GER": {
        "min_odds": 2.0,
        "max_odds": 3.2,
        "threshold": 0.05,
        "min_prob": 0.53,
    },
    "FRA": {
        "min_odds": 2.1,
        "max_odds": 3.0,
        "threshold": 0.05,
        "min_prob": 0.55,
    },
    "ENG": {
        "min_odds": 2.1,
        "max_odds": 3.0,
        "threshold": 0.05,
        "min_prob": 0.55,
    },
}

# =========================================
# MODEL
# =========================================
def make_model():
    return RandomForestClassifier(
        n_estimators=80,
        random_state=42,
        max_depth=6,
        min_samples_leaf=5,
        n_jobs=-1,
    )

# =========================================
# LOAD DATA
# =========================================
@st.cache_data
def load_all_leagues():
    folder = Path("data/processed")
    files = sorted(folder.glob("*_all_standard.csv"))

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

        text_cols = [
            "home_team", "away_team", "league", "country",
            "season", "league_code", "league_name_pl", "result"
        ]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        if "league_code" in df.columns:
            df["league_code"] = df["league_code"].astype(str).str.upper().str.strip()

        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    if "date" in df_all.columns:
        df_all = df_all.sort_values("date").reset_index(drop=True)

    return df_all

# =========================================
# TRAIN MODELS
# =========================================
@st.cache_resource
def train_models():
    df_all = load_all_leagues()
    models = {}
    league_data = {}

    for league_code in sorted(df_all["league_code"].dropna().unique()):
        if league_code not in ALLOWED_LEAGUES:
            continue

        df_league = df_all[df_all["league_code"] == league_code].copy()

        if len(df_league) < 30:
            continue

        df_league = df_league.sort_values("date").reset_index(drop=True)
        df_league = add_features(df_league)

        df_model = df_league[
            (df_league["home_scored_home"] > 0) &
            (df_league["away_scored_away"] > 0) &
            (df_league["result_1x2"].isin(["1", "X", "2"]))
        ].copy()

        # upewnij się, że targety dodatkowe istnieją
        required_targets = ["target_1x", "target_x2", "target_over25"]
        if any(col not in df_model.columns for col in required_targets):
            continue

        if len(df_model) < 30:
            continue

        X_train = df_model[FEATURE_COLS]

        y_train_1x2 = df_model["result_1x2"]
        y_train_1x = df_model["target_1x"]
        y_train_x2 = df_model["target_x2"]
        y_train_over25 = df_model["target_over25"]

        # model główny 1X2
        model_1x2 = RandomForestClassifier(
            n_estimators=80,
            random_state=42,
            max_depth=6,
            min_samples_leaf=5,
            n_jobs=-1,
        )
        model_1x2.fit(X_train, y_train_1x2)

        # model 1X
        model_1x = RandomForestClassifier(
            n_estimators=80,
            random_state=42,
            max_depth=6,
            min_samples_leaf=5,
            n_jobs=-1,
        )
        model_1x.fit(X_train, y_train_1x)

        # model X2
        model_x2 = RandomForestClassifier(
            n_estimators=80,
            random_state=42,
            max_depth=6,
            min_samples_leaf=5,
            n_jobs=-1,
        )
        model_x2.fit(X_train, y_train_x2)

        # model Over 2.5
        model_over25 = RandomForestClassifier(
            n_estimators=80,
            random_state=42,
            max_depth=6,
            min_samples_leaf=5,
            n_jobs=-1,
        )
        model_over25.fit(X_train, y_train_over25)

        models[league_code] = {
            "1x2": model_1x2,
            "1x": model_1x,
            "x2": model_x2,
            "over25": model_over25,
        }

        league_data[league_code] = df_league.copy()

    return models, league_data
# =========================================
# LIVE VALUE BETS
# =========================================
def find_live_value_bets(
    df: pd.DataFrame,
    model,
    min_odds=2.1,
    max_odds=3.0,
    threshold=0.05,
    min_prob=0.55,
):
    df_live = df.copy()
    df_live = df_live.sort_values("date").reset_index(drop=True)

    today = pd.Timestamp.today().normalize()

    future_matches = df_live[
        (df_live["date"].notna()) &
        (df_live["date"] >= today) &
        (
            df_live["home_goals"].isna() |
            df_live["away_goals"].isna() |
            ~df_live["result"].isin(["H", "D", "A"])
        )
    ].copy()

    # bierzemy tylko mecze z kursem
    future_matches = future_matches[
        future_matches["avg_home_odds"].notna() &
        future_matches["avg_draw_odds"].notna() &
        future_matches["avg_away_odds"].notna()
    ].copy()

    if future_matches.empty:
        return pd.DataFrame()

    results = []

    for _, row in future_matches.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        features = build_match_features(df_live.copy(), home_team, away_team)
        features = features[FEATURE_COLS]

        proba = model.predict_proba(features)[0]
        classes = model.classes_
        probs = dict(zip(classes, proba))

        odds_map = {
            "1": row.get("avg_home_odds"),
            "X": row.get("avg_draw_odds"),
            "2": row.get("avg_away_odds"),
        }

        bet_label_map = {
            "1": "1",
            "X": "X",
            "2": "2",
        }

        best_bet = None
        best_prob = 0.0
        best_value = -999.0

        for outcome in ["1", "X", "2"]:
            odd = odds_map.get(outcome)
            prob = probs.get(outcome, 0)

            if pd.isna(odd):
                continue
            if odd < min_odds or odd > max_odds:
                continue
            if prob < min_prob:
                continue

            value = prob * odd - 1

            if value > threshold and value > best_value:
                best_value = value
                best_bet = outcome
                best_prob = prob

        if best_bet:
            results.append({
                "date": row["date"],
                "home_team": home_team,
                "away_team": away_team,
                "bet_code": bet_label_map[best_bet],
                "probability": best_prob,
            })

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["probability", "date"],
        ascending=[False, True]
    ).reset_index(drop=True)

    return results_df

# =========================================
# APP START
# =========================================
models, league_data = train_models()

st.title("⚽ AI Typy Meczów PRO")
st.caption("Tylko najlepsze ligi: Francja, Niemcy, Polska, Anglia.")

available_leagues = [l for l in ALLOWED_LEAGUES if l in models]

if not available_leagues:
    st.error("Brak gotowych lig w data/processed albo brak modeli.")
    st.stop()

league_options = {LEAGUE_NAMES.get(code, code): code for code in available_leagues}

selected_league_name = st.selectbox("🏆 Wybierz ligę", list(league_options.keys()))
league = league_options[selected_league_name]

df = league_data[league].copy()
df = df.sort_values("date").reset_index(drop=True)
model_pack = models[league]
model = model_pack["1x2"]
settings = LEAGUE_SETTINGS[league]
with st.expander("🧠 Ważność cech modelu"):
    importances = pd.DataFrame({
        "Cecha": FEATURE_COLS,
        "Ważność": model.feature_importances_
    }).sort_values("Ważność", ascending=False).reset_index(drop=True)

    st.dataframe(importances, use_container_width=True, hide_index=True)

st.info(
    f"⚙️ Ustawienia ligi: "
    f"min_odds={settings['min_odds']} | "
    f"max_odds={settings['max_odds']} | "
    f"threshold={settings['threshold']} | "
    f"min_prob={settings['min_prob']}"
)

# =========================================
# OSTATNIA KOLEJKA
# =========================================
st.header("📅 Ostatnia kolejka")

played_matches = df[
    df["home_goals"].notna() &
    df["away_goals"].notna() &
    df["result"].isin(["H", "D", "A"])
].copy()

if not played_matches.empty:
    if played_matches["date"].notna().any():
        latest_played_date = played_matches["date"].max()
        last_round = played_matches[played_matches["date"] == latest_played_date].copy()
    else:
        last_round = played_matches.tail(10).copy()

    if last_round["date"].notna().any():
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

    st.dataframe(
        last_round_display,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Brak rozegranych meczów w tej lidze.")

# =========================================
# TERMINARZ
# =========================================
st.header("🗓️ Najbliższy terminarz")

today = pd.Timestamp.today().normalize()

fixtures = df[
    (df["date"].notna()) &
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

    st.dataframe(
        fixtures_display,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Brak przyszłych meczów w tej lidze.")

# =========================================
# ANALIZA MECZU
# =========================================
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
        model_pack = models[league]
        model_1x2 = model_pack["1x2"]

        features = build_match_features(df.copy(), home_team, away_team)
        features = features[FEATURE_COLS]

        # =========================================
        # MODEL 1X2
        # =========================================
        proba = model_1x2.predict_proba(features)[0]
        classes = model_1x2.classes_
        probs = dict(zip(classes, proba))

        p1 = probs.get("1", 0)
        px = probs.get("X", 0)
        p2 = probs.get("2", 0)

        best_type = max(probs, key=probs.get)
        confidence = probs[best_type]

        # =========================================
        # DODATKOWE MODELE
        # =========================================
        proba_1x = model_pack["1x"].predict_proba(features)[0][1]
        proba_x2 = model_pack["x2"].predict_proba(features)[0][1]
        proba_over25 = model_pack["over25"].predict_proba(features)[0][1]

        st.subheader(f"{home_team} vs {away_team}")

        # =========================================
        # 1X2
        # =========================================
        st.write("### 📊 Prognoza 1X2")

        st.write(f"**1**: {p1 * 100:.1f}%")
        st.progress(float(p1))

        st.write(f"**X**: {px * 100:.1f}%")
        st.progress(float(px))

        st.write(f"**2**: {p2 * 100:.1f}%")
        st.progress(float(p2))

        st.success(f"🔥 Główny typ: {best_type}")

        if confidence > 0.70:
            st.success(f"✅ Wysoka pewność: {confidence * 100:.1f}%")
        elif confidence > 0.55:
            st.warning(f"⚠️ Średnia pewność: {confidence * 100:.1f}%")
        else:
            st.error(f"❌ Niska pewność: {confidence * 100:.1f}%")

        # =========================================
        # DODATKOWE RYNKI
        # =========================================
        st.write("### 📈 Dodatkowe rynki")

        st.write(f"**1X**: {proba_1x * 100:.1f}%")
        st.progress(float(proba_1x))

        st.write(f"**X2**: {proba_x2 * 100:.1f}%")
        st.progress(float(proba_x2))

        st.write(f"**Over 2.5**: {proba_over25 * 100:.1f}%")
        st.progress(float(proba_over25))

        # =========================================
        # NAJLEPSZE SYGNAŁY
        # =========================================
        st.write("### 🎯 Najmocniejsze sygnały")

        shown_signal = False

        if proba_1x >= 0.65:
            st.success(f"✅ Mocny sygnał: 1X ({proba_1x * 100:.1f}%)")
            shown_signal = True

        if proba_x2 >= 0.65:
            st.success(f"✅ Mocny sygnał: X2 ({proba_x2 * 100:.1f}%)")
            shown_signal = True

        if proba_over25 >= 0.65:
            st.success(f"✅ Mocny sygnał: Over 2.5 ({proba_over25 * 100:.1f}%)")
            shown_signal = True

        if not shown_signal:
            st.info("Brak bardzo mocnych sygnałów na rynkach dodatkowych.")

        # =========================================
        # SZCZEGÓŁY FEATURES
        # =========================================
        features_display = features.rename(columns={
            "home_scored_home": "Gole gospodarzy u siebie",
            "home_conceded_home": "Stracone gospodarzy u siebie",
            "away_scored_away": "Gole gości na wyjeździe",
            "away_conceded_away": "Stracone gości na wyjeździe",

            "home_form": "Forma gospodarzy (5)",
            "away_form": "Forma gości (5)",
            "home_form_3": "Forma gospodarzy (3)",
            "away_form_3": "Forma gości (3)",

            "home_rating": "Siła gospodarzy",
            "away_rating": "Siła gości",

            "home_points_home": "Pkt gospodarzy u siebie",
            "away_points_away": "Pkt gości na wyjeździe",

            "home_clean_sheets_home": "CS gospodarzy u siebie",
            "away_clean_sheets_away": "CS gości na wyjeździe",

            "home_failed_to_score_home": "Bez gola gospodarzy u siebie",
            "away_failed_to_score_away": "Bez gola gości na wyjeździe",

            "avg_home_odds": "Kurs 1",
            "avg_draw_odds": "Kurs X",
            "avg_away_odds": "Kurs 2",

            "form_diff": "Różnica formy (5)",
            "form_diff_3": "Różnica formy (3)",
            "rating_diff": "Różnica siły",
            "goal_diff": "Różnica potencjału bramkowego",
            "points_diff": "Różnica punktów home/away",
        })

        with st.expander("🔍 Szczegóły analizy"):
            st.dataframe(
                features_display,
                use_container_width=True,
                hide_index=True
            )
# =========================================
# LIVE
# =========================================
st.header("🎯 LIVE – najlepsze typy na najbliższe mecze")

if league not in ALLOWED_LEAGUES:
    st.warning("Ta liga jest wyłączona w trybie LIVE.")
else:
    if st.button("Pokaż najlepsze typy LIVE"):
        with st.spinner("⏳ Szukam najlepszych typów..."):
            live_bets = find_live_value_bets(
                df=df.copy(),
                model=model,
                min_odds=settings["min_odds"],
                max_odds=settings["max_odds"],
                threshold=settings["threshold"],
                min_prob=settings["min_prob"],
            )

        if live_bets.empty:
            st.info("Brak typów spełniających warunki.")
        else:
            display_df = live_bets.copy()
            display_df["Data"] = display_df["date"].dt.strftime("%d-%m-%Y")
            display_df["Mecz"] = display_df["home_team"] + " vs " + display_df["away_team"]
            display_df["Typ"] = display_df["bet_code"]
            display_df["Pewność"] = (display_df["probability"] * 100).round(1).astype(str) + "%"

            st.subheader("📊 Najlepsze typy")
            st.dataframe(
                display_df[["Data", "Mecz", "Typ", "Pewność"]],
                use_container_width=True,
                hide_index=True
            )

            st.subheader("🔥 TOP 5")

            for _, row in live_bets.head(5).iterrows():
                st.markdown(
                    f"""
**{row['home_team']} vs {row['away_team']}**  
📅 {row['date'].strftime('%d-%m-%Y')}  
🎯 Typ: **{row['bet_code']}**  
📈 Pewność: **{row['probability'] * 100:.1f}%**
"""
                )