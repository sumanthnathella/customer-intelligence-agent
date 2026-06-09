"""
gbrain/store.py — SQLite-backed graph store.

Engine contract:
    upsert_node(type, id, props, embedding?) -> node_id
    add_edge(src_id, type, dst_id, props?)   -> edge_id
    get_node(id) / get_edges(id, type?)
    traverse(start_id, edge_types, depth)    -> subgraph
    vector_search(type, query_embedding, k)  -> nodes
    query(type, filters)                     -> nodes
    snapshot(run_id) -> path
"""
from __future__ import annotations

import json
import logging
import sqlite3
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from shared.config import GBRAIN_DIR

logger = logging.getLogger(__name__)

DB_PATH = GBRAIN_DIR / "brain.db"
SNAPSHOTS_DIR = GBRAIN_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS nodes (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL,
    props       TEXT NOT NULL DEFAULT '{}',
    embedding   BLOB,
    pack_version TEXT,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    id          TEXT PRIMARY KEY,
    src         TEXT NOT NULL,
    type        TEXT NOT NULL,
    dst         TEXT NOT NULL,
    props       TEXT NOT NULL DEFAULT '{}',
    updated_at  TEXT NOT NULL,
    UNIQUE(src, type, dst)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    window      TEXT,
    n           INTEGER,
    pack_version TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nodes_type  ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_edges_src   ON edges(src, type);
CREATE INDEX IF NOT EXISTS idx_edges_dst   ON edges(dst, type);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _pack_embedding(vec: list[float] | np.ndarray | None) -> bytes | None:
    if vec is None:
        return None
    arr = np.array(vec, dtype=np.float32)
    return struct.pack(f"{len(arr)}f", *arr)


def _unpack_embedding(blob: bytes | None) -> list[float] | None:
    if blob is None:
        return None
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


# ---------------------------------------------------------------------------
# GBrainStore
# ---------------------------------------------------------------------------

class GBrainStore:
    """SQLite-backed L5-centric graph store."""

    def __init__(self, db_path: str | Path = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            self._conn.row_factory = sqlite3.Row
            # WAL allows concurrent readers alongside a writer and, with a busy
            # timeout, lets concurrent writers (e.g. parallel build runs) retry
            # instead of failing immediately with "database is locked".
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_DDL)
            self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> GBrainStore:
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def upsert_node(
        self,
        node_type: str,
        node_id: str,
        props: dict[str, Any],
        embedding: list[float] | np.ndarray | None = None,
        pack_version: str | None = None,
    ) -> str:
        """Insert or update a node. Returns node_id."""
        now = _now()
        emb_blob = _pack_embedding(embedding)
        self.conn.execute(
            """
            INSERT INTO nodes (id, type, props, embedding, pack_version, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                props        = excluded.props,
                embedding    = COALESCE(excluded.embedding, nodes.embedding),
                pack_version = COALESCE(excluded.pack_version, nodes.pack_version),
                updated_at   = excluded.updated_at
            """,
            (node_id, node_type, json.dumps(props), emb_blob, pack_version, now),
        )
        self.conn.commit()
        return node_id

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    def query(
        self,
        node_type: str,
        filters: dict[str, Any] | None = None,
        order_by_prop: str | None = None,
        descending: bool = True,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return nodes of a given type, optionally filtered by props."""
        rows = self.conn.execute(
            "SELECT * FROM nodes WHERE type = ?", (node_type,)
        ).fetchall()
        results = [self._row_to_node(r) for r in rows]

        if filters:
            def _match(node: dict[str, Any]) -> bool:
                props = node.get("props", {})
                for k, v in filters.items():
                    if props.get(k) != v:
                        return False
                return True
            results = [n for n in results if _match(n)]

        if order_by_prop:
            results.sort(
                key=lambda n: (n["props"].get(order_by_prop) or 0),
                reverse=descending,
            )
        if limit:
            results = results[:limit]
        return results

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(
        self,
        src_id: str,
        edge_type: str,
        dst_id: str,
        props: dict[str, Any] | None = None,
    ) -> str:
        """Insert or update an edge (upsert on UNIQUE(src, type, dst))."""
        edge_id = f"{src_id}:{edge_type}:{dst_id}"
        now = _now()
        self.conn.execute(
            """
            INSERT INTO edges (id, src, type, dst, props, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(src, type, dst) DO UPDATE SET
                props      = excluded.props,
                updated_at = excluded.updated_at
            """,
            (edge_id, src_id, edge_type, dst_id, json.dumps(props or {}), now),
        )
        self.conn.commit()
        return edge_id

    def get_edges(
        self,
        node_id: str,
        edge_type: str | None = None,
        direction: str = "out",
    ) -> list[dict[str, Any]]:
        """
        Return edges.
        direction="out"  → edges where src == node_id
        direction="in"   → edges where dst == node_id
        direction="both" → either
        """
        if direction == "out":
            clause = "src = ?"
        elif direction == "in":
            clause = "dst = ?"
        else:
            clause = "(src = ? OR dst = ?)"

        if direction == "both":
            params: tuple[Any, ...] = (node_id, node_id)
        else:
            params = (node_id,)

        if edge_type:
            sql = f"SELECT * FROM edges WHERE {clause} AND type = ?"
            params = params + (edge_type,)
        else:
            sql = f"SELECT * FROM edges WHERE {clause}"

        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_edge(r) for r in rows]

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def traverse(
        self,
        start_id: str,
        edge_types: list[str] | None = None,
        depth: int = 2,
        direction: str = "out",
    ) -> dict[str, Any]:
        """BFS traversal. Returns {nodes: [...], edges: [...]}."""
        visited_nodes: dict[str, dict[str, Any]] = {}
        visited_edges: list[dict[str, Any]] = []
        frontier = {start_id}

        for _ in range(depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                if nid in visited_nodes:
                    continue
                node = self.get_node(nid)
                if node:
                    visited_nodes[nid] = node
                edges = self.get_edges(nid, edge_type=None, direction=direction)
                for edge in edges:
                    if edge_types and edge["type"] not in edge_types:
                        continue
                    visited_edges.append(edge)
                    neighbor = edge["dst"] if direction == "out" else edge["src"]
                    if neighbor not in visited_nodes:
                        next_frontier.add(neighbor)
            frontier = next_frontier

        return {"nodes": list(visited_nodes.values()), "edges": visited_edges}

    # ------------------------------------------------------------------
    # Vector search (cosine similarity — in-process)
    # ------------------------------------------------------------------

    def vector_search(
        self,
        node_type: str,
        query_embedding: list[float] | np.ndarray,
        k: int = 10,
    ) -> list[dict[str, Any]]:
        """Return top-k nodes of node_type by cosine similarity."""
        rows = self.conn.execute(
            "SELECT * FROM nodes WHERE type = ? AND embedding IS NOT NULL",
            (node_type,),
        ).fetchall()
        if not rows:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            emb = _unpack_embedding(row["embedding"])
            if emb is None:
                continue
            v = np.array(emb, dtype=np.float32)
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                continue
            sim = float(np.dot(q, v) / (q_norm * v_norm))
            scored.append((sim, self._row_to_node(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:k]]

    # ------------------------------------------------------------------
    # Run registry
    # ------------------------------------------------------------------

    def register_run(
        self,
        run_id: str,
        window: str,
        n: int,
        pack_version: str | None = None,
    ) -> None:
        now = _now()
        self.conn.execute(
            """
            INSERT OR REPLACE INTO runs (run_id, window, n, pack_version, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, window, n, pack_version, now),
        )
        self.conn.commit()

    def list_runs(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self, run_id: str) -> Path:
        """Export current brain state to JSON. Returns snapshot path."""
        nodes = [self._row_to_node(r) for r in self.conn.execute("SELECT * FROM nodes").fetchall()]
        edges = [self._row_to_edge(r) for r in self.conn.execute("SELECT * FROM edges").fetchall()]
        runs = [dict(r) for r in self.conn.execute("SELECT * FROM runs").fetchall()]

        payload = {
            "run_id": run_id,
            "created_at": _now(),
            "nodes": nodes,
            "edges": edges,
            "runs": runs,
        }
        path = SNAPSHOTS_DIR / f"{run_id}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("Snapshot written: %s (%d nodes, %d edges)", path, len(nodes), len(edges))
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["props"] = json.loads(d.get("props") or "{}")
        d.pop("embedding", None)
        return d

    @staticmethod
    def _row_to_edge(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["props"] = json.loads(d.get("props") or "{}")
        return d


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_store: GBrainStore | None = None


def get_store(db_path: str | Path = DB_PATH) -> GBrainStore:
    global _store
    if _store is None:
        _store = GBrainStore(db_path)
        _store.connect()
    return _store
