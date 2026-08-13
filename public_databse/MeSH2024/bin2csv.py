# -*- coding: utf-8 -*-
"""Convert a MeSH .bin dump into CSV."""

from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def parse_mesh_file(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().split("*NEWRECORD")
        for record_content in content[1:]:
            record = defaultdict(list)
            lines = record_content.strip().split("\n")
            for line in lines:
                if " = " in line:
                    key, value = line.split(" = ", 1)
                    record[key].append(value)
            record = {k: "||".join(v) for k, v in record.items()}
            records.append(record)
    return records


file_path = ROOT / "c2024.bin"
records = parse_mesh_file(file_path)
pd.DataFrame(records).to_csv(ROOT / "c2024.csv", index=False)
