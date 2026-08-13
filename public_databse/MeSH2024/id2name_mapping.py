# -*- coding: utf-8 -*-
"""Build MeSH UI -> name mapping from descriptor and supplementary CSVs."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

res = []

df = pd.read_csv(ROOT / "d2024.csv")
for mh, entry, ui in zip(df["MH"], df["ENTRY"], df["UI"]):
    if pd.isna(entry):
        entries = []
    else:
        entries = entry.split("||")
    for e in entries:
        e = e.split("|")[0]
        res.append([ui, e])
    res.append([ui, mh])

df = pd.read_csv(ROOT / "c2024.csv")
for mh, entry, ui in zip(df["NM"], df["SY"], df["UI"]):
    if pd.isna(entry):
        entries = []
    else:
        entries = entry.split("||")
    for e in entries:
        e = e.split("|")[0]
        res.append([ui, e])
    res.append([ui, mh])

pd.DataFrame(res, columns=["UI", "name"]).to_csv(ROOT / "id2name.csv", index=False)
