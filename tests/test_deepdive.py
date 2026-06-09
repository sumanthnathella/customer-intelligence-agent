"""
Tests for the L5 deep-dive: compositional facet breakdown + emergent sub-themes.
"""
from __future__ import annotations

import pandas as pd

from analytics.drivers import facet_breakdown
from analytics.subthemes import compute_subthemes
from shared.schemas import DriverEdge


def _edge(dim, val, share, lift, significant=True, support=60):
    return DriverEdge(
        l5_id="l5_x", dimension=dim, value=val, support=support, share=share,
        lift=lift, p_value=0.001 if significant else 0.5, significant=significant,
        excess=10.0, period="2024-W10",
    )


class TestFacetBreakdown:
    def test_dominant_value_per_dimension(self):
        edges = [
            _edge("vendor", "Globex", 0.45, 2.4),
            _edge("vendor", "Acme", 0.10, 0.8),
            _edge("product_category", "electronics", 0.40, 2.6),
        ]
        out = facet_breakdown(edges, top_values_per_dim=3)
        assert out["vendor"][0]["value"] == "Globex"   # highest share first
        assert out["vendor"][0]["lift"] == 2.4
        assert out["product_category"][0]["value"] == "electronics"

    def test_min_share_filter(self):
        edges = [_edge("carrier", "DHL", 0.02, 1.5)]  # below default 0.05 share
        assert facet_breakdown(edges) == {}


class TestSubthemes:
    def _frame(self):
        rows = []
        # L5 "battery": distinctive word "battery" recurs; background L5 differs.
        for i in range(40):
            rows.append({
                "conversation_id": f"b{i}", "l5_id": "battery_issue",
                "text": "the battery is swollen and overheating badly" if i % 2 == 0
                        else "battery drains and the device shuts down",
                "severity": 4.0,
            })
        for i in range(40):
            rows.append({
                "conversation_id": f"r{i}", "l5_id": "refund_issue",
                "text": "i want a refund for my money back on this charge",
                "severity": 3.0,
            })
        return pd.DataFrame(rows)

    def test_distinctive_theme_surfaces(self):
        out = compute_subthemes(self._frame(), min_l5_docs=10, min_docs_per_theme=3)
        labels = [t["label"] for t in out.get("battery_issue", [])]
        assert any("battery" in lbl for lbl in labels)
        # the refund-specific term should not bleed into the battery L5
        assert not any("refund" in lbl for lbl in labels)

    def test_theme_has_count_and_quote(self):
        out = compute_subthemes(self._frame(), min_l5_docs=10, min_docs_per_theme=3)
        themes = out.get("battery_issue", [])
        assert themes, "expected at least one battery sub-theme"
        t = themes[0]
        assert t["count"] >= 3
        assert 0 < t["share"] <= 1
        assert isinstance(t["quote"], str) and t["quote"]

    def test_small_l5_returns_empty(self):
        df = pd.DataFrame([
            {"conversation_id": "x1", "l5_id": "tiny", "text": "broken thing", "severity": 2.0}
        ])
        out = compute_subthemes(df, min_l5_docs=15)
        assert out.get("tiny") == []
