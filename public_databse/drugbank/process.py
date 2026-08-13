# -*- coding: utf-8 -*-
"""Parse DrugBank full_database.xml into a drug entity CSV."""

from pathlib import Path

import pandas as pd
from lxml import etree as et

ROOT = Path(__file__).resolve().parent
NS = "{http://www.drugbank.ca}"

file = ROOT / "full_database.xml"
tree = et.parse(str(file))
root = tree.getroot()

res = set()
for child in root:
    synonyms = "||".join(
        i.text
        for i in child.find(f"{NS}synonyms").findall(f"{NS}synonym")
    )
    drugbank_id = child.find(f"{NS}drugbank-id").text
    drugbank_name = child.find(f"{NS}name").text
    description = child.find(f"{NS}description").text
    res.add((drugbank_id, drugbank_name, description, synonyms))

pd.DataFrame(
    res,
    columns=["drug_id", "drug_name", "description", "synonyms"],
).to_csv(ROOT / "drugbank_entity.csv", index=False, encoding="utf_8_sig")
