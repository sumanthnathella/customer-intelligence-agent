# Data Model (Demo)

> **Scope:** this describes the **demo** dataset — the twcs transcripts (`tagging/`) plus synthetic
> operational dimensions (`datagen/`). It is **not** part of the dataset-agnostic core. A bring-your-own
> user replaces the dimensions here with their real operational metrics per
> [`INPUT_CONTRACT.md`](INPUT_CONTRACT.md).

## Source: Customer Support on Twitter (`twcs/twcs.csv`)

~2.8M tweets, real 2017 timestamps. Bundled locally (no Kaggle download).

| Column | Meaning |
|--------|---------|
| `tweet_id` | Unique tweet id |
| `author_id` | Customer (numeric) or brand handle (e.g. `AppleSupport`) |
| `inbound` | `True` = from customer, `False` = from brand |
| `created_at` | Real timestamp (e.g. `Tue Oct 31 22:10:47 +0000 2017`) |
| `text` | Tweet body |
| `response_tweet_id` | Reply(ies) to this tweet |
| `in_response_to_tweet_id` | Parent tweet |

### Conversation reconstruction
Follow `in_response_to_tweet_id` chains to assemble inbound+outbound turns into one **conversation**:
`conversation_id`, `brand`, `turns[]`, `text` (joined), `start_at`, `customer_text` (inbound only).

### Build / test split
- **Build set**: first ~1M tweets (by conversation, no leakage) → taxonomy + 10k tagged sample.
- **Test reserve**: remaining ~1.8M tweets, consumed later via `--test-batch N` to grow gbrain.

### Time axis
Weekly buckets from real `created_at`. Z-scores compare the latest week vs the trailing baseline.

## Synthetic order-dimension model (`datagen/` — DEMO ONLY)

The twcs data has no order/product data, so `datagen/` **deterministically** attaches a seeded synthetic
order to each **already-tagged** conversation. This module is a stand-in for a real operational-metrics
join and is meant to be **deleted/replaced** by BYO users.

### Documented driver model (why drivers are recoverable)
If dimensions were drawn randomly, every lift ≈ 1.0 and the report's dimensional section would be noise.
Instead `datagen/driver_model.py` defines a **documented latent model** `P(dimension | l5_id, signals)`
that *plants* realistic operational drivers, e.g.:

| Planted driver | Effect |
|----------------|--------|
| `late_delivery` + winter | over-weights `vehicle_type=drone`, `service_level=2-day` |
| `damaged_item` (groceries) | over-weights `vehicle_type=refrigerated`, `package_type=fragile` |
| `unmet_expectations` (electronics) | over-weights `service_level=overnight` |

The analytics engine ([`ANALYTICS.md`](ANALYTICS.md)) then **recovers** these via lift/significance
*without seeing the model* — a faithful simulation of the analysis you'd run on real data, and a built-in
validation target (recovered lift ≈ planted lift). The report labels these dimensions as **simulated**.

Generation is seeded by `conversation_id` so it is reproducible and stable across runs.

> Facts (numeric) are stored both raw **and** bucketed so they can be used as dimensions.

### Core
| Dimension | Example values |
|-----------|----------------|
| `fulfillment_type` | ship-to-home, BOPIS, curbside, dropship, warehouse-to-door |
| `vehicle_type` | truck, van, drone, bicycle, refrigerated, flatbed, container |
| `order_date` / `dow` / `month` / `quarter` / `season` / `promo_period` | derived calendar attrs |
| `order_status` | pending, confirmed, shipped, delivered, returned, cancelled |
| `sales_channel` | website, mobile_app, marketplace, in_store, call_center |

### Product / SKU
| Dimension | Example values |
|-----------|----------------|
| `sku` | `SKU-#####` |
| `sku_description` | product name + variant (color/size), brand |
| `product_category` | electronics, apparel, groceries, home_garden, … |
| `quantity` | units per order (fact + bucket) |
| `weight` / `volume` / `dims` / `hazardous_flag` | physical attrs |

### Customer / Geography
| Dimension | Example values |
|-----------|----------------|
| `customer_segment` | retail, wholesale, VIP, new, returning |
| `shipping_region` | country / state / zip / metro / delivery_zone |
| `distance` | km from fulfillment center (fact + bucket) |

### Financial
| Dimension/Fact | Example values |
|----------------|----------------|
| `order_total` | numeric + bin ($0–25, 25–100, 100–500, 500+) |
| `payment_method` | credit_card, paypal, COD, gift_card |
| `shipping_cost` | free, paid, discounted |
| `discount_applied` | yes/no, promo_code, pct_off |

### Logistics
| Dimension | Example values |
|-----------|----------------|
| `carrier` | UPS, FedEx, USPS, DHL, in_house |
| `service_level` | ground, 2-day, overnight, same-day |
| `package_type` | standard, oversized, fragile, temperature_controlled |

The authoritative value lists live in `datagen/dimensions.py`; the planted correlations live in
`datagen/driver_model.py` (both to be implemented). Neither is imported by the core.
