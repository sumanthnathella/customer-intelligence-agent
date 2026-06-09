"""
datagen/dimensions.py — Define the synthetic operational dimensions for demo.

These are the columns that will appear in Table B (operational dimensions).
The driver model (driver_model.py) plants statistical relationships so that
analytics can recover them — proving the methodology works.
"""
from __future__ import annotations

from typing import Any

# Dimension catalog: name → list of possible values
DIMENSIONS: dict[str, list[str]] = {
    "carrier": ["FedEx", "UPS", "USPS", "DHL", "DroneX"],
    "fulfillment_type": ["standard", "express", "BOPIS", "same_day", "subscription"],
    "vehicle_type": ["truck", "van", "drone", "bicycle", "locker"],
    "region": ["Northeast", "Southeast", "Midwest", "Southwest", "West", "Canada"],
    "service_level": ["1-day", "2-day", "3-5-day", "ground", "economy"],
    "customer_segment": ["enterprise", "premium", "standard", "new", "guest"],
    "season": ["spring", "summer", "fall", "holiday", "clearance"],
    "payment_method": ["credit_card", "paypal", "apple_pay", "bank_transfer", "invoice"],
    "device_type": ["mobile_app", "web", "desktop", "tablet", "phone"],
    "product_category": ["electronics", "apparel", "grocery", "home_kitchen", "toys", "beauty"],
    "vendor": ["Acme", "Globex", "Initech", "Umbrella", "Soylent", "Hooli"],
}

# Facts (numeric columns)
FACTS: dict[str, Any] = {
    "order_total": {"min": 20.0, "max": 500.0, "mean": 120.0},
    "quantity": {"min": 1, "max": 20, "mean": 3},
    "delivery_minutes": {"min": 10, "max": 10080, "mean": 2880},  # 7 days max
}


def get_dimension_cols() -> list[str]:
    """Return list of dimension column names."""
    return list(DIMENSIONS.keys())


def get_fact_cols() -> list[str]:
    """Return list of fact column names."""
    return list(FACTS.keys())
