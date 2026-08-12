"""Export the knowledge-graph SQLite database to three CSV tables."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "state" / "kg.sqlite"
DEFAULT_OUTPUT_DIR = BASE_DIR / "csv_output"


EXPORTS = {
    "sources.csv": """
        SELECT source_id, source_name, text
        FROM sources
        ORDER BY source_id
    """,
    "entities.csv": """
        SELECT id, name, entity_type, source_id
        FROM entities
        ORDER BY id
    """,
    "relations.csv": """
        SELECT r.id,
               r.subject_entity_id,
               subject.name AS subject_entity_name,
               r.relation_type,
               r.object_entity_id,
               object.name AS object_entity_name,
               r.source_id
        FROM relations AS r
        JOIN entities AS subject ON subject.id = r.subject_entity_id
        JOIN entities AS object ON object.id = r.object_entity_id
        ORDER BY r.id
    """,
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the knowledge-graph SQLite DB to sources/entities/relations CSV."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"CSV output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser


def open_readonly_database(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"SQLite file does not exist: {path}")
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def export_query(
    connection: sqlite3.Connection,
    query: str,
    output_path: Path,
) -> int:
    cursor = connection.execute(query)
    fieldnames = [column[0] for column in cursor.description]
    rows = cursor.fetchall()
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict(row) for row in rows)
    return len(rows)


def export_database(db_path: Path, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = open_readonly_database(db_path)
    try:
        return {
            filename: export_query(connection, query, output_dir / filename)
            for filename, query in EXPORTS.items()
        }
    finally:
        connection.close()


def main() -> None:
    args = build_argument_parser().parse_args()
    counts = export_database(args.db, args.output_dir)
    for filename, count in counts.items():
        print(f"{filename}: {count}")


if __name__ == "__main__":
    main()
