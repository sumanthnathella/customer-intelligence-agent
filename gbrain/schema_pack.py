"""
gbrain/schema_pack.py — Load, save, and query the frozen taxonomy schema pack.

The schema pack is a versioned JSON file that maps l5_id → full L1–L5 path + definition.
Analytics and the agent read it; tagging/ writes it (one-time, on build).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from shared.config import TAXONOMY_PATH

logger = logging.getLogger(__name__)


class SchemaPack:
    """Holds the frozen L1–L5 taxonomy."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        self._nodes: dict[str, dict[str, Any]] = data.get("nodes", {})

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path = TAXONOMY_PATH) -> SchemaPack:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Taxonomy schema pack not found: {p}\n"
                "Run `uv run python -m tagging.run --build` to generate it first."
            )
        data = json.loads(p.read_text())
        logger.info("Loaded schema pack v%s (%d L5 nodes) from %s", data.get("version"), len(data.get("nodes", {})), p)
        return cls(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SchemaPack:
        return cls(data)

    def save(self, path: str | Path = TAXONOMY_PATH) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._data, indent=2))
        logger.info("Saved schema pack to %s", p)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def version(self) -> str:
        return self._data.get("version", "unknown")

    @property
    def nodes(self) -> dict[str, dict[str, Any]]:
        return self._nodes

    def get_l5(self, l5_id: str) -> dict[str, Any] | None:
        return self._nodes.get(l5_id)

    def all_l5_ids(self) -> list[str]:
        return list(self._nodes.keys())

    def get_path(self, l5_id: str) -> list[str]:
        """Return [L1, L2, L3, L4, L5] labels for a given l5_id."""
        node = self._nodes.get(l5_id)
        if not node:
            return []
        return node.get("path", [])

    def get_definition(self, l5_id: str) -> str:
        node = self._nodes.get(l5_id)
        if not node:
            return ""
        return node.get("definition", "")

    def l5_ids_for_l1(self, l1: str) -> list[str]:
        return [k for k, v in self._nodes.items() if v.get("path", [None])[0] == l1]

    def to_dict(self) -> dict[str, Any]:
        return self._data


def load_pack(path: str | Path = TAXONOMY_PATH) -> SchemaPack:
    return SchemaPack.load(path)
