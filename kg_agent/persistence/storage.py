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
    """根据内部来源键和片段文本生成稳定来源 ID。"""
    identity = source_key or source_name
    return hashlib.sha256(f"{identity}\0{text}".encode("utf-8")).hexdigest()


class GraphStore:
    def __init__(
        self,
        path: Path | str,
        check_same_thread: bool = True,
    ) -> None:
        """打开 SQLite 图谱库并初始化表结构。"""
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
            # WAL 允许多个读取连接与一个写入连接并行，busy_timeout 负责等待短暂写锁。
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA busy_timeout = 60000")
            self._create_tables()

    def _create_tables(self) -> None:
        """创建来源、实体和关系三张核心表。"""
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
        """关闭 SQLite 连接。"""
        self._connection.close()

    def source_exists(self, source_id: str) -> bool:
        """判断片段是否已经原子提交。"""
        row = self._connection.execute(
            "SELECT 1 FROM sources WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        return row is not None

    def commit_chunk(self, workspace: ChunkWorkspace) -> bool:
        """在一个事务中提交来源、实体和关系。"""
        with log_slow_operation(
            "database.commit_chunk",
            entity_count=len(workspace.entities),
            relation_count=len(workspace.relations),
        ):
            # 上下文管理器保证三张表全部成功才提交，任一步失败都会整体回滚。
            with self._connection:
                # sources 主键同时承担幂等锁；已提交片段不会重复写实体和关系。
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
                # 关系表只保存数据库实体 ID，因此实体落库后再建立本地 ID 映射。
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
        """按写入顺序列出全部实体记录。"""
        rows = self._connection.execute(
            """
            SELECT id, name, entity_type, source_id
            FROM entities ORDER BY id
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def list_relations(self) -> list[dict[str, Any]]:
        """按写入顺序列出全部关系记录。"""
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
        """读取一个已提交片段的来源、实体和关系。"""
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
