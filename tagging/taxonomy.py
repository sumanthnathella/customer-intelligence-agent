"""
tagging/taxonomy.py — Induce (or load) the L1–L5 taxonomy schema pack.

Build-time only (called once from tagging/run.py --build).
Uses nomic-embed-text embeddings + HDBSCAN clustering to discover L3–L5 leaves
under fixed L1 anchors (pre_sale / sale / post_sale).

BYO users: skip this and provide your own taxonomy.json.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from shared.config import EMBED_MODEL, PACK_VERSION, TAXONOMY_PATH

logger = logging.getLogger(__name__)

# Fixed L1 anchors (pre-tagged manually)
L1_ANCHORS = ["pre_sale", "sale", "post_sale"]

# Max leaves per the design spec
MAX_L5_LEAVES = 100


def _embed_texts(texts: list[str], model: str = EMBED_MODEL) -> np.ndarray:
    """Embed a list of texts via Ollama. Returns (N, D) float32 array."""
    try:
        import ollama  # type: ignore
    except ImportError as exc:
        raise ImportError("ollama package required for taxonomy induction: pip install ollama") from exc

    embeddings = []
    expected_dim = None
    for text in texts:
        truncated = text[:512]  # nomic-embed-text context window
        resp = ollama.embeddings(model=model, prompt=truncated)
        vec = resp["embedding"]
        if isinstance(vec, list):
            if expected_dim is None:
                expected_dim = len(vec)
            elif len(vec) != expected_dim:
                # Pad or truncate to match expected dimension
                if len(vec) < expected_dim:
                    vec = vec + [0.0] * (expected_dim - len(vec))
                else:
                    vec = vec[:expected_dim]
            embeddings.append(vec)
        else:
            embeddings.append([0.0] * (expected_dim or 768))
    return np.array(embeddings, dtype=np.float32)


def _cluster(embeddings: np.ndarray, min_cluster_size: int = 5) -> np.ndarray:
    """HDBSCAN clustering. Returns label array (-1 = noise)."""
    try:
        import hdbscan  # type: ignore
    except ImportError as exc:
        raise ImportError("hdbscan package required for taxonomy induction.") from exc

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        prediction_data=True,
    )
    return clusterer.fit_predict(embeddings)


def _label_from_texts(texts: list[str], max_words: int = 4) -> str:
    """Heuristic label: most common N-grams (no LLM)."""
    from collections import Counter
    words: list[str] = []
    for t in texts:
        words.extend(re.findall(r"\b[a-z]{3,}\b", t.lower()))
    stop = {"the", "and", "for", "with", "that", "this", "you", "was", "are", "have", "your", "our", "not"}
    filtered = [w for w in words if w not in stop]
    top = Counter(filtered).most_common(max_words)
    return "_".join(w for w, _ in top) if top else "unknown"


def induce_taxonomy(
    conversations: list[str],
    l1_mapping: dict[str, list[str]] | None = None,
    min_cluster_size: int = 5,
    max_leaves: int = MAX_L5_LEAVES,
    pack_version: str = PACK_VERSION,
    output_path: str | Path = TAXONOMY_PATH,
) -> dict[str, Any]:
    """
    Induce a taxonomy schema pack from conversation texts.

    l1_mapping: optional pre-assignment of conversations to L1 buckets.
                If None, all go into a single root (demo / small datasets).

    Returns the schema pack dict and saves to output_path.
    """
    logger.info("Inducing taxonomy from %d conversations...", len(conversations))

    if not conversations:
        raise ValueError("No conversations provided for taxonomy induction.")

    # Embed
    embeddings = _embed_texts(conversations[:min(5000, len(conversations))])
    labels = _cluster(embeddings, min_cluster_size=min_cluster_size)

    # Build L5 nodes
    unique_labels = sorted(set(labels) - {-1})
    if len(unique_labels) > max_leaves:
        unique_labels = unique_labels[:max_leaves]

    nodes: dict[str, dict[str, Any]] = {}
    for label_id in unique_labels:
        mask = labels == label_id
        cluster_texts = [conversations[i] for i in range(len(conversations)) if i < len(labels) and labels[i] == label_id]
        if not cluster_texts:
            continue

        cluster_label = _label_from_texts(cluster_texts[:50])
        l5_id = f"l5_{label_id:03d}_{cluster_label}"

        # Assign to L1 based on most common cue words (heuristic)
        text_blob = " ".join(cluster_texts[:20]).lower()
        if any(w in text_blob for w in ["order", "deliver", "ship", "track", "return", "refund", "charge", "cancel"]):
            l1 = "post_sale"
        elif any(w in text_blob for w in ["purchase", "buy", "checkout", "cart", "payment"]):
            l1 = "sale"
        else:
            l1 = "pre_sale"

        nodes[l5_id] = {
            "l5_id": l5_id,
            "path": [l1, f"{l1}_issues", f"{l1}_{cluster_label}", f"{cluster_label}_detailed", l5_id],
            "definition": f"Cluster {label_id}: {cluster_label} issues (auto-induced)",
            "cluster_size": int(mask.sum()),
        }

    if not nodes:
        # Fallback: single-leaf taxonomy for very small datasets
        logger.warning("HDBSCAN found no clusters; using fallback single-leaf taxonomy.")
        nodes = {
            "l5_000_general_issue": {
                "l5_id": "l5_000_general_issue",
                "path": ["post_sale", "post_sale_issues", "general", "general_detailed", "l5_000_general_issue"],
                "definition": "General customer issue (fallback)",
                "cluster_size": len(conversations),
            }
        }

    pack: dict[str, Any] = {
        "version": pack_version,
        "nodes": nodes,
        "l1_anchors": L1_ANCHORS,
        "n_leaves": len(nodes),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pack, indent=2))
    logger.info("Schema pack v%s saved: %d L5 leaves → %s", pack_version, len(nodes), out)
    return pack


def load_taxonomy(path: str | Path = TAXONOMY_PATH) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Taxonomy not found: {p}")
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Static (hand-curated) taxonomy — no HDBSCAN required
# ---------------------------------------------------------------------------

_STATIC_NODES: list[dict[str, Any]] = [
    # ── PRE-SALE ─────────────────────────────────────────────────────────────
    # L2: Discover
    {
        "l5_id": "pre_sale__discover__product_info__features__product_feature_inquiry",
        "path": ["pre_sale", "discover", "product_info", "features_specs", "product_feature_inquiry"],
        "definition": "Customer asking about product features, specifications, or capabilities before purchasing.",
    },
    {
        "l5_id": "pre_sale__discover__product_info__pricing__pricing_inquiry",
        "path": ["pre_sale", "discover", "product_info", "pricing", "pricing_inquiry"],
        "definition": "Customer asking about price, cost, fees, plans, or promotions before purchasing.",
    },
    {
        "l5_id": "pre_sale__discover__account_setup__onboarding__account_signup",
        "path": ["pre_sale", "discover", "account_setup", "onboarding", "account_signup"],
        "definition": "Customer trying to create, register, or set up a new account.",
    },
    # L2: Search & Compare
    {
        "l5_id": "pre_sale__search_compare__comparison__plans__plan_comparison",
        "path": ["pre_sale", "search_compare", "comparison", "plans_options", "plan_comparison"],
        "definition": "Customer comparing plans, packages, tiers, or product options to decide what to buy.",
    },
    {
        "l5_id": "pre_sale__search_compare__comparison__compatibility__compatibility_check",
        "path": ["pre_sale", "search_compare", "comparison", "compatibility", "compatibility_check"],
        "definition": "Customer asking about compatibility, device requirements, or service eligibility.",
    },
    # ── SALE ─────────────────────────────────────────────────────────────────
    # L2: Build Cart
    {
        "l5_id": "sale__build_cart__promo__coupon__promo_code_issue",
        "path": ["sale", "build_cart", "promo_discount", "coupon_promo", "promo_code_issue"],
        "definition": "Customer having trouble applying a promo code, coupon, or discount at checkout.",
    },
    {
        "l5_id": "sale__build_cart__promo__pricing_error__price_discrepancy",
        "path": ["sale", "build_cart", "promo_discount", "pricing_error", "price_discrepancy"],
        "definition": "Customer reporting an incorrect price or unexpected charge in their cart.",
    },
    # L2: Purchase
    {
        "l5_id": "sale__purchase__payment__failure__payment_failure",
        "path": ["sale", "purchase", "payment", "payment_failure", "payment_failure"],
        "definition": "Customer's payment was declined, failed, or could not be processed.",
    },
    {
        "l5_id": "sale__purchase__payment__billing_error__billing_error",
        "path": ["sale", "purchase", "payment", "billing_error", "billing_error"],
        "definition": "Customer reporting a billing error, overcharge, or incorrect invoice after purchase.",
    },
    {
        "l5_id": "sale__purchase__account_access__login__login_issue",
        "path": ["sale", "purchase", "account_access", "login", "login_issue"],
        "definition": "Customer unable to log in, account locked, or needing a password reset during checkout.",
    },
    # ── POST-SALE ─────────────────────────────────────────────────────────────
    # L2: Tracking & Order Status
    {
        "l5_id": "post_sale__tracking__shipment__delayed__shipment_delayed",
        "path": ["post_sale", "tracking_order_status", "shipment_tracking", "delayed_shipment", "shipment_delayed"],
        "definition": "Customer's package or order is delayed beyond the expected delivery date.",
    },
    {
        "l5_id": "post_sale__tracking__shipment__lost__package_lost",
        "path": ["post_sale", "tracking_order_status", "shipment_tracking", "lost_package", "package_lost"],
        "definition": "Customer reporting a lost, stolen, or never-delivered package.",
    },
    {
        "l5_id": "post_sale__tracking__status__inquiry__order_status_inquiry",
        "path": ["post_sale", "tracking_order_status", "order_status", "status_inquiry", "order_status_inquiry"],
        "definition": "Customer asking for an update or status check on their order.",
    },
    # L2: Modify Order
    {
        "l5_id": "post_sale__modify_order__modification__cancel__order_cancellation",
        "path": ["post_sale", "modify_order", "order_modification", "cancel_order", "order_cancellation"],
        "definition": "Customer requesting to cancel an order, booking, or subscription.",
    },
    {
        "l5_id": "post_sale__modify_order__modification__change__order_change_request",
        "path": ["post_sale", "modify_order", "order_modification", "change_order", "order_change_request"],
        "definition": "Customer requesting to modify, update address, or change details of an existing order.",
    },
    # L2: Receive Order
    {
        "l5_id": "post_sale__receive_order__delivery__wrong_item__wrong_item_received",
        "path": ["post_sale", "receive_order", "delivery_issues", "wrong_item", "wrong_item_received"],
        "definition": "Customer received the wrong item, product model, or service.",
    },
    {
        "l5_id": "post_sale__receive_order__delivery__damaged__damaged_item_received",
        "path": ["post_sale", "receive_order", "delivery_issues", "damaged_item", "damaged_item_received"],
        "definition": "Customer received a damaged, defective, or broken item.",
    },
    {
        "l5_id": "post_sale__receive_order__delivery__missing__missing_item",
        "path": ["post_sale", "receive_order", "delivery_issues", "missing_item", "missing_item"],
        "definition": "Customer's order arrived incomplete or with missing items.",
    },
    {
        "l5_id": "post_sale__receive_order__service_disruption__outage__service_outage",
        "path": ["post_sale", "receive_order", "service_disruption", "outage", "service_outage"],
        "definition": "Customer experiencing a service outage, downtime, disruption, or connectivity issue.",
    },
    # L2: Return & Exchange
    {
        "l5_id": "post_sale__return_exchange__return__initiate__return_request",
        "path": ["post_sale", "return_exchange", "return_process", "return_initiation", "return_request"],
        "definition": "Customer initiating or requesting a return for a purchased product or service.",
    },
    {
        "l5_id": "post_sale__return_exchange__return__exchange__exchange_request",
        "path": ["post_sale", "return_exchange", "return_process", "exchange", "exchange_request"],
        "definition": "Customer requesting to exchange an item for a different size, model, or variant.",
    },
    {
        "l5_id": "post_sale__return_exchange__return_issues__rejected__return_rejected",
        "path": ["post_sale", "return_exchange", "return_issues", "return_rejected", "return_rejected"],
        "definition": "Customer's return request was denied, rejected, or outside the return window.",
    },
    {
        "l5_id": "post_sale__return_exchange__return_issues__label__return_label_issue",
        "path": ["post_sale", "return_exchange", "return_issues", "return_label", "return_label_issue"],
        "definition": "Customer having trouble printing, receiving, or using a return shipping label.",
    },
    # L2: Refunds
    {
        "l5_id": "post_sale__refunds__processing__delayed__refund_delayed",
        "path": ["post_sale", "refunds", "refund_processing", "refund_delayed", "refund_delayed"],
        "definition": "Customer waiting for a refund that has not been processed or received yet.",
    },
    {
        "l5_id": "post_sale__refunds__processing__partial__partial_refund_dispute",
        "path": ["post_sale", "refunds", "refund_processing", "partial_refund", "partial_refund_dispute"],
        "definition": "Customer disputing a partial refund or requesting a full refund.",
    },
    {
        "l5_id": "post_sale__refunds__billing_dispute__unauthorized__unauthorized_charge",
        "path": ["post_sale", "refunds", "billing_disputes", "unauthorized_charge", "unauthorized_charge"],
        "definition": "Customer reporting an unauthorized, fraudulent, or unexpected charge on their account.",
    },
    {
        "l5_id": "post_sale__refunds__billing_dispute__duplicate__duplicate_charge",
        "path": ["post_sale", "refunds", "billing_disputes", "duplicate_charge", "duplicate_charge"],
        "definition": "Customer charged multiple times for the same transaction or order.",
    },
    # L2: Customer Service Experience
    {
        "l5_id": "post_sale__cx_experience__escalation__supervisor__escalation_request",
        "path": ["post_sale", "cx_experience", "agent_escalation", "supervisor", "escalation_request"],
        "definition": "Customer requesting to speak with a supervisor or escalate their unresolved issue.",
    },
    {
        "l5_id": "post_sale__cx_experience__repeat_contact__unresolved__unresolved_repeat_contact",
        "path": ["post_sale", "cx_experience", "repeat_contact", "unresolved_issue", "unresolved_repeat_contact"],
        "definition": "Customer contacting support again for the same unresolved issue (repeat contact).",
    },
    {
        "l5_id": "post_sale__cx_experience__wait_time__long_wait__long_wait_time",
        "path": ["post_sale", "cx_experience", "wait_time", "long_wait", "long_wait_time"],
        "definition": "Customer complaining about long hold times, slow response, or poor service experience.",
    },
    {
        "l5_id": "post_sale__cx_experience__account_mgmt__subscription__subscription_change",
        "path": ["post_sale", "cx_experience", "account_management", "subscription", "subscription_change"],
        "definition": "Customer requesting changes to their subscription, plan, or account settings.",
    },
]


def generate_static_taxonomy(
    pack_version: str = PACK_VERSION,
    output_path: str | Path = TAXONOMY_PATH,
) -> dict[str, Any]:
    """
    Write the hand-curated 30-leaf L1-L5 taxonomy to disk.
    Call this instead of induce_taxonomy() when you don't want HDBSCAN.
    """
    nodes = {n["l5_id"]: n for n in _STATIC_NODES}
    pack: dict[str, Any] = {
        "version": pack_version,
        "nodes": nodes,
        "l1_anchors": L1_ANCHORS,
        "n_leaves": len(nodes),
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pack, indent=2))
    logger.info("Static taxonomy v%s saved: %d L5 leaves → %s", pack_version, len(nodes), out)
    return pack
