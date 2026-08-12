"""SQLite persistence for sources, entities, and relations."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from ..core.workspace import ChunkWorkspace
from ..observability import log_slow_operation


def make_source_id(
    source_name: str,
    text: str,
    source_key: str | None = None,
) -> str:
    """Build a stable source ID from the internal source key and chunk text."""
    identity = source_key or source_name
    return hashlib.sha256(f"{identity}\0{text}".encode("utf-8")).hexdigest()


class GraphStore:
    def __init__(
        self,
        path: Path | str,
        check_same_thread: bool = True,
    ) -> None:
        db_path = Path(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = db_path
        with log_slow_operation("database.open", database=str(db_path)):
            self._connection = sqlite3.connect(
                db_path,
                timeout=60,
                check_same_thread=check_same_thread,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            # WAL allows concurrent readers with one writer; busy_timeout waits out short write locks.
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA busy_timeout = 60000")
            self._create_tables()

    def _create_tables(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sources (
                source_id   TEXT PRIMARY KEY,
                source_name TEXT NOT NULL,
                text        TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS entities (
                id          INTEGER PRIMARY KEY,
                name        TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                source_id   TEXT NOT NULL,
                UNIQUE (name, entity_type, source_id),
                FOREIGN KEY (source_id)
                    REFERENCES sources(source_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS relations (
                id                INTEGER PRIMARY KEY,
                subject_entity_id INTEGER NOT NULL,
                relation_type     TEXT NOT NULL,
                object_entity_id  INTEGER NOT NULL,
                source_id         TEXT NOT NULL,
                UNIQUE (subject_entity_id, relation_type, object_entity_id),
                FOREIGN KEY (subject_entity_id)
                    REFERENCES entities(id) ON DELETE CASCADE,
                FOREIGN KEY (object_entity_id)
                    REFERENCES entities(id) ON DELETE CASCADE,
                FOREIGN KEY (source_id)
                    REFERENCES sources(source_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_entities_name
                ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entities_entity_type
                ON entities(entity_type);
            CREATE INDEX IF NOT EXISTS idx_entities_source_id
                ON entities(source_id);
            CREATE INDEX IF NOT EXISTS idx_relations_source_id
                ON relations(source_id);
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def source_exists(self, source_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM sources WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        return row is not None

    def commit_chunk(self, workspace: ChunkWorkspace) -> bool:
        """Atomically commit source, entities, and relations in one transaction."""
        with log_slow_operation(
            "database.commit_chunk",
            entity_count=len(workspace.entities),
            relation_count=len(workspace.relations),
        ):
            # Commit all three tables atomically; any failure rolls the whole transaction back.
            with self._connection:
                # sources PK doubles as an idempotency lock; committed chunks are not rewritten.
                cursor = self._connection.execute(
                    "INSERT OR IGNORE INTO sources VALUES (?, ?, ?)",
                    (workspace.source_id, workspace.source_name, workspace.text),
                )
                if cursor.rowcount == 0:
                    return False
                self._connection.executemany(
                    """
                    INSERT INTO entities (name, entity_type, source_id)
                    VALUES (?, ?, ?)
                    """,
                    [
                        (entity.name, entity.entity_type, workspace.source_id)
                        for entity in workspace.entities.values()
                    ],
                )
                # Relations store DB entity IDs, so map local IDs after entities are inserted.
                rows = self._connection.execute(
                    """
                    SELECT id, name, entity_type
                    FROM entities
                    WHERE source_id = ?
                    """,
                    (workspace.source_id,),
                ).fetchall()
                stored_ids = {
                    (row["name"], row["entity_type"]): int(row["id"])
                    for row in rows
                }
                entity_ids = {
                    local_id: stored_ids[(entity.name, entity.entity_type)]
                    for local_id, entity in workspace.entities.items()
                }
                self._connection.executemany(
                    """
                    INSERT INTO relations (
                        subject_entity_id,
                        relation_type,
                        object_entity_id,
                        source_id
                    ) VALUES (?, ?, ?, ?)
                    """,
                    [
                        (
                            entity_ids[relation.subject_entity_id],
                            relation.predicate,
                            entity_ids[relation.object_entity_id],
                            workspace.source_id,
                        )
                        for relation in workspace.relations.values()
                    ],
                )
            return True

    def list_entities(self) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT id, name, entity_type, source_id
            FROM entities ORDER BY id
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def list_relations(self) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT id,
                   subject_entity_id,
                   relation_type,
                   object_entity_id,
                   source_id
            FROM relations
            ORDER BY id
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def get_source_result(self, source_id: str) -> dict[str, Any] | None:
        source = self._connection.execute(
            "SELECT source_id, source_name FROM sources WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        if source is None:
            return None
        entities = self._connection.execute(
            """
            SELECT id, name, entity_type
            FROM entities
            WHERE source_id = ?
            ORDER BY id
            """,
            (source_id,),
        ).fetchall()
        relations = self._connection.execute(
            """
            SELECT id,
                   subject_entity_id,
                   relation_type,
                   object_entity_id
            FROM relations
            WHERE source_id = ?
            ORDER BY id
            """,
            (source_id,),
        ).fetchall()
        return {
            "source_id": source["source_id"],
            "source_name": source["source_name"],
            "entities": [dict(row) for row in entities],
            "relations": [dict(row) for row in relations],
        }
