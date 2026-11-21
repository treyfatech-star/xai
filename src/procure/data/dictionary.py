"""Data dictionary export (docs/data_dictionary.csv)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from procure.config import path

# name: (source, plain-English meaning)
DICTIONARY: dict[str, tuple[str, str]] = {
    "ocid": ("NOCOPO/OCDS", "Open contracting ID of the procurement record."),
    "date": ("NOCOPO/OCDS", "Date the contract was signed."),
    "buyer_id": ("derived", "Anonymised ID of the buying ministry, department or agency."),
    "supplier_id": ("derived", "Anonymised supplier ID built from the normalised name and state."),
    "supplier_name": ("NOCOPO/OCDS", "Registered business name (kept only in the access-controlled supplier table)."),
    "supplier_state": ("NOCOPO/OCDS", "State where the supplier is based."),
    "supplier_lga": ("NOCOPO/OCDS", "Local government area where the supplier is based."),
    "item": ("NOCOPO/OCDS", "Item bought."),
    "item_category": ("NOCOPO/OCDS", "Broad group of the item (medical, food, fuel, ...)."),
    "quantity": ("NOCOPO/OCDS", "Number of units ordered."),
    "unit": ("NOCOPO/OCDS", "Unit of measure."),
    "value": ("NOCOPO/OCDS", "Contract value in naira at the time of signing."),
    "currency": ("derived", "Always NGN after cleaning (USD converted at that month's rate)."),
    "method": ("NOCOPO/OCDS", "Procurement method (open competitive, selective, RFQ, direct)."),
    "n_bidders": ("NOCOPO/OCDS", "Number of suppliers that bid."),
    "delivery_state": ("NOCOPO/OCDS", "State where the goods must be delivered."),
    "delivery_lga": ("NOCOPO/OCDS", "Local government area where the goods must be delivered."),
    "delay_days": ("records / synthetic", "Days delivered after the agreed date."),
    "quality_score": ("records / synthetic", "Inspection score from 1 (very poor) to 5 (excellent)."),
    "short_delivered": ("records / synthetic", "True if fewer units were delivered than ordered."),
    "rejected": ("records / synthetic", "True if the delivery was rejected at inspection."),
    "failed": ("derived", "True if more than 30 days late, short-delivered or rejected."),
    "is_synthetic": ("derived", "True if any outcome on this row was generated, not recorded."),
    "has_real_outcome": ("derived", "True if all outcomes on this row were recorded."),
    "is_imputed": ("derived", "True if any predictor on this row was filled in by median/mode."),
    "imputed_cols": ("derived", "Which predictors were filled in."),
    "cost_outlier": ("derived", "Unit price is unusually high or low for this item (kept, flagged)."),
    "delay_outlier": ("derived", "Delay is unusually long (kept, flagged)."),
    "registered": ("supplier registry", "Registered with the Bureau of Public Procurement."),
    "debarred": ("supplier registry", "Currently debarred from public contracts."),
    "tax_clearance": ("supplier registry", "Holds a valid tax clearance certificate."),
    "cac": ("supplier registry", "Registered with the Corporate Affairs Commission."),
    "in_registry": ("derived", "Supplier was found in the registry file."),
    "year": ("derived", "Year of signing."),
    "month": ("derived", "Month of signing (1-12)."),
    "unit_cost_nominal": ("derived", "Price per unit in naira at the time of signing."),
    "unit_cost_real": ("derived + macro CPI", "Price per unit in 2024 naira (inflation-adjusted)."),
    "log_quantity": ("derived", "Natural log of quantity."),
    "distance_km": ("derived + LGA coordinates", "Straight-line distance from supplier LGA to delivery LGA, km."),
    "same_state": ("derived", "1 if supplier and delivery are in the same state."),
    "rainy_season": ("derived", "1 if signed in the rainy season (May-October)."),
    "conflict_events_90d": ("ACLED", "Security incidents within 50 km of the delivery LGA in the previous 90 days."),
    "conflict_fatalities_90d": ("ACLED", "Deaths from those incidents."),
    "supplier_experience": ("derived", "Number of earlier contracts with this supplier."),
    "past_on_time_rate": ("derived", "Share of the supplier's earlier, completed orders delivered on time (smoothed)."),
    "past_failure_rate": ("derived", "Share of the supplier's earlier, completed orders that failed (smoothed)."),
    "past_quality_mean": ("derived", "Average inspection score of the supplier's earlier orders (smoothed)."),
    "past_price_index": ("derived", "How the supplier's earlier prices compared with the typical price (log ratio)."),
    "log_item_ref_price": ("derived", "Log of the typical 2024-naira price of the item (training years)."),
    "past_outcomes_known": ("derived", "Number of earlier orders whose outcome was known."),
    "source": ("derived", "Raw file the row came from."),
}


def export_dictionary(df: pd.DataFrame, out: Path | None = None) -> Path:
    """Write name, type, source, plain-English meaning for every column of df."""
    out = out or path("docs_dir") / "data_dictionary.csv"
    rows = [{"name": c, "type": str(df[c].dtype), "source": DICTIONARY.get(c, ("derived", ""))[0],
             "meaning": DICTIONARY.get(c, ("", "(undocumented)"))[1]} for c in df.columns]
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
