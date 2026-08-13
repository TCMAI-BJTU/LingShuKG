# -*- coding: utf-8 -*-
# @File    : process.py
# @Software: PyCharm
"""Parse MalaCards disease / gene dumps into cleaned CSV tables."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent

# Official dump columns (tab-separated):
# MalaCards.txt: McId, DiseaseName, DiseaseSlug, Aliases, ExternalIds
# MalaCardsGenesImplications.txt: MCId, DiseaseName, DiseaseSlug, GeneSymbol,
#   isElite, isCancerCensus, GeneDisorderScore, Implications, Publications
DISEASE_TXT = ROOT / "MalaCards.txt"
GENE_TXT = ROOT / "MalaCardsGenesImplications.txt"


def extract_umls_cuis(external_ids: str) -> list[str]:
    """Pull UMLS CUIs from pipe-separated ExternalIds."""
    cuis: list[str] = []
    for item in str(external_ids).split("|"):
        item = item.strip()
        if "UMLS" not in item:
            continue
        # Examples: UMLS:C3149220 or fragments containing :C########
        match = re.search(r":(C\d+)", item)
        if match:
            cuis.append(match.group(1))
            continue
        match = re.search(r"UMLS:(C?\d+)", item)
        if match:
            cui = match.group(1)
            if not cui.startswith("C"):
                cui = "C" + cui
            cuis.append(cui)
    return sorted(set(cuis))


def parse_external_ids(external_ids: str) -> dict[str, list[str]]:
    """Group ExternalIds as {source: [id, ...]}."""
    grouped: dict[str, list[str]] = {}
    if pd.isna(external_ids) or not str(external_ids).strip():
        return grouped
    for item in str(external_ids).split("|"):
        item = item.strip()
        if ":" not in item:
            continue
        source, value = item.split(":", 1)
        grouped.setdefault(source.strip(), []).append(value.strip())
    return grouped


def load_disease_table(path: Path | None = None) -> pd.DataFrame:
    """Load MalaCards.txt disease dump."""
    path = path or DISEASE_TXT
    return pd.read_csv(path, sep="\t", dtype=str)


def process_diseases(input_path: Path | None = None) -> Path:
    """Export cleaned disease table with aliases and UMLS CUIs."""
    df = load_disease_table(input_path)
    rows = []
    for tup in zip(
        df["McId"],
        df["DiseaseName"],
        df["DiseaseSlug"],
        df["Aliases"],
        df["ExternalIds"],
    ):
        mcid, name, slug, aliases, external_ids = tup
        alias_ls = []
        if not pd.isna(aliases) and str(aliases).strip():
            alias_ls = [a.strip() for a in str(aliases).split("|") if a.strip()]
        cui_ls = []
        if not pd.isna(external_ids):
            cui_ls = extract_umls_cuis(external_ids)
        rows.append(
            [
                mcid,
                name,
                slug,
                "|".join(alias_ls),
                "|".join(cui_ls),
                "" if pd.isna(external_ids) else external_ids,
            ]
        )
    out = ROOT / "disease.csv"
    pd.DataFrame(
        rows,
        columns=[
            "McId",
            "DiseaseName",
            "DiseaseSlug",
            "Aliases",
            "UMLS_CUIs",
            "ExternalIds",
        ],
    ).to_csv(out, index=False)
    return out


def process_mcid2cui(input_path: Path | None = None) -> Path:
    """Export McId -> UMLS CUI mapping (one row per pair)."""
    df = load_disease_table(input_path)
    rows = []
    for mcid, external_ids in zip(df["McId"], df["ExternalIds"]):
        if pd.isna(external_ids):
            continue
        for cui in extract_umls_cuis(external_ids):
            rows.append([mcid, cui])
    out = ROOT / "mcid2cui.csv"
    pd.DataFrame(rows, columns=["McId", "CUI"]).drop_duplicates().to_csv(
        out, index=False
    )
    return out


def process_gene_disease(input_path: Path | None = None) -> Path:
    """Export disease-gene associations from the implications dump."""
    path = input_path or GENE_TXT
    if not path.is_file():
        raise FileNotFoundError(f"Gene dump not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str)
    # Normalize occasional header spelling MCId vs McId.
    rename = {}
    if "MCId" in df.columns and "McId" not in df.columns:
        rename["MCId"] = "McId"
    df = df.rename(columns=rename)
    keep = [
        c
        for c in [
            "McId",
            "DiseaseName",
            "DiseaseSlug",
            "GeneSymbol",
            "isElite",
            "isCancerCensus",
            "GeneDisorderScore",
            "Implications",
            "Publications",
        ]
        if c in df.columns
    ]
    out = ROOT / "disease_gene.csv"
    df[keep].to_csv(out, index=False)
    return out


if __name__ == "__main__":
    disease_out = process_diseases()
    cui_out = process_mcid2cui()
    print(f"Wrote {disease_out}")
    print(f"Wrote {cui_out}")
    if GENE_TXT.is_file():
        gene_out = process_gene_disease()
        print(f"Wrote {gene_out}")
    else:
        print(f"Skip gene dump (missing {GENE_TXT.name})")
