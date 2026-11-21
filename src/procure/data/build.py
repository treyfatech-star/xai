"""End-to-end data build (make data): raw -> clean -> features -> synthetic -> split -> parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from procure.config import load_settings, path, set_seed
from procure.data import clean, demo_raw, dictionary, features, load, split, synthetic


def build_dataset(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Run the full data pipeline and return all frames (transactions, splits, acled, macro)."""
    set_seed()
    raw = load.load_all(raw_dir)
    df = clean.clean(raw["contracts"], raw["macro"], raw["registry"])
    df = features.add_context_features(df, raw["macro"], raw["acled"])
    df = synthetic.generate_outcomes(df)
    df, ref = features.add_history_features(df)
    parts = split.temporal_split(df)
    ref = ref.rename("ref_price")
    ref.index.name = "item"
    return {"transactions": df, "ref_prices": ref.reset_index(),
            "acled": raw["acled"], "macro": raw["macro"], **parts}


def save(frames: dict[str, pd.DataFrame], out_dir: Path | None = None) -> Path:
    """Write every frame to data/processed/<name>.parquet."""
    out_dir = out_dir or path("processed_dir")
    for name, f in frames.items():
        f.to_parquet(out_dir / f"{name}.parquet", index=False)
    return out_dir


def load_processed(name: str) -> pd.DataFrame:
    """Read data/processed/<name>.parquet."""
    return pd.read_parquet(path("processed_dir") / f"{name}.parquet")


def main() -> None:
    """CLI entry: generate simulated raw data if needed, build, save, print row counts."""
    raw_dir = path("raw_dir")
    if not demo_raw.raw_dir_has_real_data(raw_dir):
        demo_raw.generate(raw_dir)
        print("WARNING: data/raw had no real files - using SIMULATED raw data (see SIMULATED_DATA_README.txt).")
    frames = build_dataset(raw_dir)
    save(frames)
    dictionary.export_dictionary(frames["transactions"])
    states = load_settings()["region"]["states"]
    table = pd.DataFrame({k: frames[k]["delivery_state"].value_counts().reindex(states).fillna(0).astype(int)
                          for k in ["train", "val", "test", "test_synthetic"]})
    table.loc["TOTAL"] = table.sum()
    print("\nRows per state and split:\n" + table.to_string())
    tx = frames["transactions"]
    print(f"\nTransactions: {len(tx)} | synthetic-outcome rows: {int(tx.is_synthetic.sum())} "
          f"| failure rate: {tx.failed.mean():.1%} | suppliers: {tx.supplier_id.nunique()}")


if __name__ == "__main__":
    main()
