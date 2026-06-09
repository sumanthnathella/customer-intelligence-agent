"""
datagen/driver_model.py — Documented latent driver model.

Plants statistical relationships between L5 pain points and operational dimensions
so that analytics/drivers.py can recover them. This validates the methodology.

Model: P(dim_value | l5_id, signals) = base_rate * lift_factor(l5_id, dim, value, signals)

The planted lifts are recoverable because the tagging happens BEFORE enrichment
(drivers are real correlations, not synthetic heuristics).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Planted driver model — documented, versioned
# ---------------------------------------------------------------------------
# For each L5 family, we plant specific dimension-value lifts.
# Example: late_delivery pain points should over-index on "drone" vehicle_type.

_PLANTED_DRIVERS: dict[str, list[dict[str, Any]]] = {
    # Delivery/fulfillment issues
    "late_delivery": [
        {"dimension": "vehicle_type", "value": "drone", "lift": 3.1},
        {"dimension": "service_level", "value": "1-day", "lift": 2.4},
        {"dimension": "region", "value": "Northeast", "lift": 1.9},
        {"dimension": "season", "value": "holiday", "lift": 2.2},
    ],
    "lost_package": [
        {"dimension": "carrier", "value": "DroneX", "lift": 2.8},
        {"dimension": "fulfillment_type", "value": "standard", "lift": 1.7},
        {"dimension": "customer_segment", "value": "guest", "lift": 2.0},
    ],
    # Damaged goods concentrate in fragile electronics from a specific vendor
    # (also matches the static taxonomy leaf "damaged_item_received").
    "damaged_item": [
        {"dimension": "product_category", "value": "electronics", "lift": 2.6},
        {"dimension": "vendor", "value": "Globex", "lift": 2.4},
        {"dimension": "fulfillment_type", "value": "same_day", "lift": 1.8},
        {"dimension": "vehicle_type", "value": "truck", "lift": 1.5},
    ],
    # Wrong item / fit problems concentrate in apparel.
    "wrong_item": [
        {"dimension": "product_category", "value": "apparel", "lift": 2.3},
        {"dimension": "vendor", "value": "Initech", "lift": 1.9},
        {"dimension": "region", "value": "Canada", "lift": 2.1},
        {"dimension": "payment_method", "value": "invoice", "lift": 1.6},
    ],
    # Billing issues
    "overcharged": [
        {"dimension": "payment_method", "value": "paypal", "lift": 2.3},
        {"dimension": "customer_segment", "value": "premium", "lift": 1.9},
        {"dimension": "device_type", "value": "mobile_app", "lift": 1.5},
    ],
    "refund_slow": [
        {"dimension": "payment_method", "value": "bank_transfer", "lift": 3.5},
        {"dimension": "fulfillment_type", "value": "BOPIS", "lift": 2.0},
    ],
    # Support issues
    "unresponsive_support": [
        {"dimension": "device_type", "value": "phone", "lift": 2.7},
        {"dimension": "customer_segment", "value": "enterprise", "lift": 2.2},
    ],
    "rude_agent": [
        {"dimension": "region", "value": "Southeast", "lift": 1.9},
        {"dimension": "season", "value": "clearance", "lift": 1.6},
    ],
    # ------------------------------------------------------------------
    # Keys matching the static L1–L5 taxonomy leaf ids (see tagging/taxonomy.py).
    # These are what the demo recovers on real tagged data; the generic keys
    # above remain for the synthetic recovery fixtures.
    #
    # Three values are deliberately *systemic* (they recur across many L5s) so
    # analytics surfaces them as "bridges" — single levers worth fixing once:
    #   • carrier=DroneX     → a failing carrier across all delivery issues
    #   • region=West        → regional infrastructure problems
    #   • payment_method=paypal → a payment-processor problem across billing
    # ------------------------------------------------------------------
    "shipment_delayed": [
        {"dimension": "carrier", "value": "DroneX", "lift": 2.6},   # bridge
        {"dimension": "vehicle_type", "value": "drone", "lift": 3.1},
        {"dimension": "service_level", "value": "1-day", "lift": 2.4},
        {"dimension": "region", "value": "West", "lift": 2.0},      # bridge
        {"dimension": "season", "value": "holiday", "lift": 2.2},
        {"dimension": "product_category", "value": "grocery", "lift": 1.8},
    ],
    "package_lost": [
        {"dimension": "carrier", "value": "DroneX", "lift": 2.8},   # bridge
        {"dimension": "customer_segment", "value": "guest", "lift": 2.0},
        {"dimension": "vendor", "value": "Hooli", "lift": 1.9},
    ],
    "missing_item": [
        {"dimension": "carrier", "value": "DroneX", "lift": 2.1},   # bridge
        {"dimension": "fulfillment_type", "value": "same_day", "lift": 2.0},
        {"dimension": "vendor", "value": "Hooli", "lift": 2.2},
    ],
    "service_outage": [
        {"dimension": "region", "value": "West", "lift": 2.3},      # bridge
        {"dimension": "device_type", "value": "mobile_app", "lift": 1.8},
    ],
    "unauthorized_charge": [
        {"dimension": "payment_method", "value": "paypal", "lift": 2.6},  # bridge
        {"dimension": "customer_segment", "value": "premium", "lift": 1.9},
    ],
    "duplicate_charge": [
        {"dimension": "payment_method", "value": "paypal", "lift": 2.2},  # bridge
        {"dimension": "device_type", "value": "mobile_app", "lift": 1.6},
    ],
    "refund_delayed": [
        {"dimension": "payment_method", "value": "paypal", "lift": 2.4},  # bridge
        {"dimension": "fulfillment_type", "value": "BOPIS", "lift": 2.0},
    ],
    "payment_failure": [
        {"dimension": "payment_method", "value": "paypal", "lift": 2.7},  # bridge
        {"dimension": "device_type", "value": "tablet", "lift": 1.7},
    ],
    "long_wait": [
        {"dimension": "region", "value": "West", "lift": 1.9},      # bridge
        {"dimension": "device_type", "value": "phone", "lift": 2.7},
        {"dimension": "customer_segment", "value": "enterprise", "lift": 2.2},
    ],
    "escalation_request": [
        {"dimension": "customer_segment", "value": "enterprise", "lift": 2.1},
        {"dimension": "region", "value": "Southeast", "lift": 1.8},
    ],
}


def _match_l5(l5_id: str) -> list[dict[str, Any]] | None:
    """Find planted drivers for an L5 by keyword matching."""
    for key, drivers in _PLANTED_DRIVERS.items():
        if key in l5_id.lower():
            return drivers
    return None


def plant_dimensions(
    df_a: pd.DataFrame,
    dimension_cols: list[str],
    seed: int = 42,
    base_lift: float = 1.0,
) -> pd.DataFrame:
    """
    Generate synthetic operational dimensions (Table B) for a set of tagged transcripts.

    Uses the planted driver model: each L5 has specific dimension-value combinations
    that are over-represented. The lift values are what analytics/drivers.py should recover.

    Parameters
    ----------
    df_a: Table A DataFrame with l5_id column
    dimension_cols: which columns to generate
    seed: random seed for reproducibility
    base_lift: default baseline lift (no over-indexing)

    Returns DataFrame with: conversation_id + dimension_cols + fact_cols
    """
    rng = np.random.default_rng(seed)

    from datagen.dimensions import DIMENSIONS, FACTS

    records: list[dict[str, Any]] = []
    for _, row in df_a.iterrows():
        l5_id = str(row["l5_id"])
        cid = str(row["conversation_id"])

        planted = _match_l5(l5_id)
        rec: dict[str, Any] = {"conversation_id": cid}

        for dim in dimension_cols:
            if dim not in DIMENSIONS:
                rec[dim] = None
                continue
            values = DIMENSIONS[dim]

            # Default: uniform probability
            probs = np.ones(len(values)) / len(values)

            # Apply planted lifts
            if planted:
                for driver in planted:
                    if driver["dimension"] == dim:
                        try:
                            idx = values.index(driver["value"])
                            probs[idx] *= driver["lift"]
                        except ValueError:
                            pass

            probs = probs / probs.sum()
            rec[dim] = rng.choice(values, p=probs)

        # Facts (numeric)
        for fact, spec in FACTS.items():
            if fact == "order_total":
                # correlate with financial_harm signal
                if row.get("financial_harm", 0) == 1:
                    rec[fact] = round(rng.normal(250, 50), 2)
                else:
                    rec[fact] = round(rng.normal(spec["mean"], 30), 2)
            elif fact == "quantity":
                rec[fact] = int(rng.integers(spec["min"], spec["max"] + 1))
            else:
                rec[fact] = round(float(rng.normal(spec["mean"], spec["mean"] * 0.3)), 1)

        records.append(rec)

    return pd.DataFrame(records)


def get_planted_lifts() -> dict[str, list[dict[str, Any]]]:
    """Return the documented planted driver model for validation."""
    return _PLANTED_DRIVERS
