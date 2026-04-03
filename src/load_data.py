import pandas as pd
from pathlib import Path


def load_all_leagues():
    folder = Path("data/processed")
    files = list(folder.glob("*_standard.csv"))

    if not files:
        raise ValueError("❌ Brak plików w data/processed")

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    return df_all