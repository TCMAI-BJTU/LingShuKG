# -*- coding: utf-8 -*-
"""Collect STITCH chemical IDs and map them to names."""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

chemical_chemical_file = ROOT / "chemical_chemical_700.csv"
df = pd.read_csv(chemical_chemical_file)
chemical_id_set = set()
for chemical1, chemical2 in zip(df["chemical1"], df["chemical2"]):
    chemical_id_set.add(chemical1)
    chemical_id_set.add(chemical2)

chemical_protein_file = ROOT / "stitch_protein_chemical_score700_9606.csv"
df = pd.read_csv(chemical_protein_file)
for chemical_id in df["chemical_id"]:
    chemical_id_set.add(chemical_id)

res = []
chemicals_file = ROOT / "chemicals.v5.0.txt"
for chunk in tqdm(pd.read_csv(chemicals_file, sep="\t", chunksize=100000)):
    for chemical_id, name in zip(chunk["chemical"], chunk["name"]):
        if chemical_id in chemical_id_set:
            res.append((chemical_id, name))

pd.DataFrame(res, columns=["chemical_id", "name"]).to_csv(
    ROOT / "chemicals.v5.0.id_name.csv",
    index=False,
)
