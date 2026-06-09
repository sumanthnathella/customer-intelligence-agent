"""
Tests for the Harness-1-inspired layer: systemic (bridge/singleton) drivers,
importance tiers + curated-set movement, and BM25 evidence compression.
"""
from __future__ import annotations

from analytics.systemic import (
    compute_systemic_drivers,
    curation_delta,
    importance_tier,
    zscore_trend,
)
from shared.schemas import DriverEdge
from shared.text import compress_to_sentences, keyword_overlap_score, split_sentences


def _edge(l5_id: str, value: str, *, significant: bool, lift: float = 2.0, support: int = 50) -> DriverEdge:
    return DriverEdge(
        l5_id=l5_id,
        dimension="carrier",
        value=value,
        support=support,
        share=0.3,
        lift=lift,
        p_value=0.01 if significant else 0.5,
        significant=significant,
        excess=10.0,
        period="2024-W10",
    )


class TestSystemicDrivers:
    def test_bridge_detection(self):
        # carrier=DroneX is a significant driver across 3 distinct L5s -> bridge
        edges = [
            _edge("l5_a", "DroneX", significant=True),
            _edge("l5_b", "DroneX", significant=True),
            _edge("l5_c", "DroneX", significant=True),
            _edge("l5_a", "UPS", significant=True),  # singleton
        ]
        result = {(s.dimension, s.value): s for s in compute_systemic_drivers(edges, bridge_min_l5=3)}
        drone = result[("carrier", "DroneX")]
        assert drone.is_bridge is True
        assert drone.n_significant == 3
        assert set(drone.affected_l5s) == {"l5_a", "l5_b", "l5_c"}

        ups = result[("carrier", "UPS")]
        assert ups.is_singleton is True
        assert ups.is_bridge is False

    def test_non_significant_excluded(self):
        edges = [
            _edge("l5_a", "DHL", significant=False),
            _edge("l5_b", "DHL", significant=False),
        ]
        # no significant edges -> DHL should not appear at all
        assert compute_systemic_drivers(edges) == []

    def test_systemic_score_normalized(self):
        edges = [
            _edge("l5_a", "DroneX", significant=True, lift=3.0),
            _edge("l5_b", "DroneX", significant=True, lift=3.0),
            _edge("l5_a", "UPS", significant=True, lift=1.5),
        ]
        results = compute_systemic_drivers(edges, bridge_min_l5=2)
        assert results[0].value == "DroneX"  # bridges/higher score first
        assert 0.0 <= results[-1].systemic_score <= 1.0
        assert results[0].systemic_score == 1.0


class TestImportance:
    def test_tiers(self):
        assert importance_tier(0.95) == "very_high"
        assert importance_tier(0.70) == "high"
        assert importance_tier(0.40) == "fair"
        assert importance_tier(0.10) == "low"

    def test_curation_delta(self):
        assert curation_delta(None, "high") == "new"
        assert curation_delta("fair", "high") == "escalated"
        assert curation_delta("very_high", "high") == "de-escalated"
        assert curation_delta("high", "high") == "stable"

    def test_trend(self):
        assert zscore_trend(3.0) == "rising"
        assert zscore_trend(-2.0) == "falling"
        assert zscore_trend(0.5) == "stable"


class TestTextCompression:
    def test_picks_relevant_sentence(self):
        text = (
            "Hi there how are you. | My refund has not arrived after three weeks. | "
            "Thanks for the help."
        )
        out = compress_to_sentences(text, query="refund delayed not arrived", k=1)
        assert "refund" in out.lower()
        assert "how are you" not in out.lower()

    def test_short_text_passthrough(self):
        text = "Only one sentence here."
        assert compress_to_sentences(text, query="anything", k=2) == text

    def test_split_and_overlap(self):
        assert len(split_sentences("a. b | c")) == 3
        assert keyword_overlap_score("refund money charge", "refund charge") == 1.0
        assert keyword_overlap_score("hello world", "refund") == 0.0
