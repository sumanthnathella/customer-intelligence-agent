"""Unit tests for gbrain store operations."""
import tempfile
from pathlib import Path

import pytest

from gbrain.graph import upsert_affects_edge, upsert_pain_point
from gbrain.store import GBrainStore
from shared.schemas import DriverEdge


class TestGBrainStore:
    @pytest.fixture
    def store(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "test.db"
            s = GBrainStore(db)
            s.connect()
            yield s
            s.close()

    def test_upsert_node(self, store):
        nid = store.upsert_node("pain_point", "A", {"vol": 10})
        assert nid == "A"
        node = store.get_node("A")
        assert node is not None
        assert node["props"]["vol"] == 10

    def test_upsert_update(self, store):
        store.upsert_node("pain_point", "A", {"vol": 10})
        store.upsert_node("pain_point", "A", {"vol": 20})
        node = store.get_node("A")
        assert node["props"]["vol"] == 20

    def test_add_edge_upsert(self, store):
        store.upsert_node("pain_point", "A", {"vol": 1})
        store.upsert_node("dimension", "dim:X", {"dim": "X"})
        eid = store.add_edge("A", "affects", "dim:X", {"lift": 2.0})
        assert eid == "A:affects:dim:X"
        # update
        store.add_edge("A", "affects", "dim:X", {"lift": 3.0})
        edges = store.get_edges("A", "affects", "out")
        assert len(edges) == 1
        assert edges[0]["props"]["lift"] == 3.0

    def test_query(self, store):
        store.upsert_node("pain_point", "A", {"vol": 10})
        store.upsert_node("pain_point", "B", {"vol": 20})
        nodes = store.query("pain_point", order_by_prop="vol", descending=True)
        assert len(nodes) == 2
        assert nodes[0]["props"]["vol"] == 20

    def test_traverse(self, store):
        store.upsert_node("pain_point", "A", {})
        store.upsert_node("period", "A@W01", {})
        store.add_edge("A", "measured_in", "A@W01")
        sub = store.traverse("A", edge_types=["measured_in"], depth=2)
        assert len(sub["nodes"]) == 2
        assert len(sub["edges"]) == 1

    def test_vector_search(self, store):
        store.upsert_node("pain_point", "A", {}, embedding=[1.0, 0.0, 0.0])
        store.upsert_node("pain_point", "B", {}, embedding=[0.0, 1.0, 0.0])
        results = store.vector_search("pain_point", [1.0, 0.0, 0.0], k=2)
        assert len(results) == 2
        assert results[0]["id"] == "A"  # closest

    def test_snapshot(self, store):
        store.upsert_node("pain_point", "A", {"vol": 10})
        path = store.snapshot("test_run")
        assert path.exists()
        data = path.read_text()
        assert "test_run" in data
        assert "A" in data


class TestGraphHelpers:
    @pytest.fixture
    def store(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "test.db"
            s = GBrainStore(db)
            s.connect()
            yield s
            s.close()

    def test_upsert_pain_point(self, store):
        upsert_pain_point(
            store, "A", ["l1", "l2", "l3", "l4", "A"],
            "def", "v1", total_volume=100, severity_avg=3.0,
            latest_zscore=2.0, latest_egregiousness=0.85,
        )
        node = store.get_node("A")
        assert node["props"]["total_volume"] == 100
        assert node["props"]["latest_egregiousness"] == 0.85

    def test_upsert_affects_edge(self, store):
        driver = DriverEdge(
            l5_id="A", dimension="dim", value="X",
            support=50, share=0.5, lift=2.5,
            p_value=0.01, significant=True, excess=30.0, period="W01",
            history=[{"period": "W00", "support": 40, "share": 0.4, "lift": 2.0}]
        )
        upsert_affects_edge(store, driver)
        edges = store.get_edges("A", "affects", "out")
        assert len(edges) == 1
        assert edges[0]["props"]["lift"] == 2.5
        assert edges[0]["props"]["history"][0]["period"] == "W00"
