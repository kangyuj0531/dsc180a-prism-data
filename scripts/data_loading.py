"""
Module for loading the PRISM data files. 
Automatically loads all dataframes when imported, making them available as:
- consumers
- accounts  
- transactions
- category_mapping

You can also use load_all_data() to get a dictionary or reload with a custom path.
"""
from pathlib import Path
from typing import Dict, Optional
import pandas as pd


def _find_data_path() -> Path:
    """Automatically detect the correct data path based on environment."""
    possible_paths = [
        Path("/uss/hdsi-prismdata"),      # Server absolute path
        Path("../hdsi-prismdata"),         # Relative from scripts/feature_engineering
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # If none exist, return the first one and let it fail with a clear error
    return possible_paths[0]


DEFAULT_BASE = _find_data_path()


def load_all_data(base_path: Optional[str | Path] = None) -> Dict[str, pd.DataFrame]:
    """Load the four PRISM files and convert the date columns.

    Parameters:
        base_path: path to the `hdsi-prismdata` folder. If None, uses
                   the default path.

    Returns:
        dict with keys: `consumers`, `accounts`, `transactions`, `category_mapping`.
    """
    base = base_path if base_path else DEFAULT_BASE

    consumers = pd.read_parquet(base / "q2-ucsd-consDF.pqt")
    accounts = pd.read_parquet(base / "q2-ucsd-acctDF.pqt")
    transactions = pd.read_parquet(base / "q2-ucsd-trxnDF.pqt")
    category_mapping = pd.read_csv(base / "q2-ucsd-cat-map.csv")

    if "balance_date" in accounts.columns:
        accounts["balance_date"] = pd.to_datetime(accounts["balance_date"])
    if "posted_date" in transactions.columns:
        transactions["posted_date"] = pd.to_datetime(transactions["posted_date"])

    # Deduplicate transactions: prefer unique transaction id when available
    try:
        before = len(transactions)
        if "prism_transaction_id" in transactions.columns:
            transactions = (
                transactions.drop_duplicates(subset=["prism_transaction_id"], keep="first")
                            .reset_index(drop=True)
            )
        else:
            transactions = transactions.drop_duplicates(keep="first").reset_index(drop=True)
        removed = before - len(transactions)
        if removed > 0:
            print(f"Removed {removed:,} duplicate transaction rows (from {before:,} to {len(transactions):,}).")
    except Exception:
        # If anything goes wrong during deduplication, fall back silently to the original table
        pass

    return {
        "consumers": consumers,
        "accounts": accounts,
        "transactions": transactions,
        "category_mapping": category_mapping,
    }


# Auto-load dataframes when module is imported
_data = load_all_data()
consumers = _data["consumers"]
accounts = _data["accounts"]
transactions = _data["transactions"]
category_mapping = _data["category_mapping"]


__all__ = ["load_all_data", "consumers", "accounts", "transactions", "category_mapping"]
