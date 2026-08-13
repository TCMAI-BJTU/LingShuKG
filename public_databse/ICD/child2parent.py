# -*- coding: utf-8 -*-
"""Build ICD-11 child -> parent mapping from indented Title text."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

df = pd.read_excel(ROOT / "ICD-11_en.xlsx")
lines = df["Title"].tolist()

parent_child_dict = {}
parent_stack = []

for line in lines:
    # Indentation depth is encoded by leading dashes.
    level = line.count("-")
    disease = line.strip("- ").strip()
    parent_stack = parent_stack[:level]
    parent = parent_stack[-1] if parent_stack else None
    parent_child_dict[disease] = parent
    parent_stack.append(disease)

print(parent_child_dict)
