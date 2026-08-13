# -*- coding: utf-8 -*-
"""Convert HPO hp.obo into a flat CSV table."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

lines = open(ROOT / "hp.obo").readlines()

res = []
term_flag = True
term_res = {}
for line in lines:
    if line.startswith("[Term]"):
        res.append(term_res)
        term_res = {}
        term_flag = False
        continue
    if term_flag or line == "\n" or line.startswith("[Typedef]"):
        continue
    k, v = line.split(": ", 1)
    if k in term_res:
        term_res[k] += "||" + v.strip()
    else:
        term_res[k] = v.strip()

df = pd.json_normalize(res)
print(df)
df.to_csv(ROOT / "hp.csv", index=False)
