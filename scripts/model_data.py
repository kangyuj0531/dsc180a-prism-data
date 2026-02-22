"""
Shared data-loading and preparation utilities for delinquency models.

Typical usage from a notebook in models/:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path('..').resolve()))
    from scripts.model_data import load_and_split, save_sorted_features

    N_FEATURES = 50               # ← change this to adjust feature count

    save_sorted_features()        # writes output/sorted_features.csv once

    # Top-N selected features (no exclusions)
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, features_df = \\
        load_and_split(n_features=N_FEATURES)

    # Top-N selected features WITH scoring exclusions applied
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, features_df = \\
        load_and_split(n_features=N_FEATURES, apply_exclusions=True)

    # All features
    X_train_all, X_val_all, X_test_all, y_train_all, y_val_all, y_test_all, all_cols, _ = \\
        load_and_split(use_all=True)

Scoring exclusion criteria (applied when apply_exclusions=True):
    1. Minimum 30 days of observation history  (n_days__all >= 30)
    2. Minimum 5 transactions total            (n_tx__90d  >= 5  OR  n_tx__180d >= 5)
    3. Not a zero-balance dormant account      (balance__mean__all != 0)
    4. Valid DQ_TARGET label                   (always required)
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Core loaders live in feature_selection; re-export them here for convenience
from scripts.feature_selection import (
    load_features,
    save_sorted_features,
    _resolve_output,
)


# ---------------------------------------------------------------------------
# Scoring exclusions
# ---------------------------------------------------------------------------

# Criteria constants — adjust here to change the exclusion thresholds globally
MIN_OBS_DAYS   = 30   # minimum days of account history
MIN_TX_90D     = 5    # minimum transactions in most-recent 90-day window
MIN_TX_180D    = 5    # fallback: minimum transactions in 180-day window
MEAN_BAL_ZERO_TOL = 1e-6  # absolute tolerance to detect zero-balance dormant accounts


def apply_scoring_exclusions(
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Apply standard scoring exclusions and return the retained subset.

    Exclusion criteria
    ------------------
    1. **Unlabeled** – ``DQ_TARGET`` is NaN.
    2. **Thin file** – fewer than ``MIN_OBS_DAYS`` (default 30) days of history
       (uses column ``n_days__all`` when present).
    3. **Inactive** – fewer than ``MIN_TX_90D`` (default 5) transactions in the
       last 90 days AND fewer than ``MIN_TX_180D`` (default 5) in the last 180
       days (uses columns ``n_tx__90d`` / ``n_tx__180d`` when present).
    4. **Dormant** – mean balance across all history is effectively zero
       (uses column ``balance__mean__all`` when present).

    Criteria that rely on a column not present in ``df`` are silently skipped.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature matrix including ``DQ_TARGET``.
    verbose : bool
        Print a per-criterion exclusion report.

    Returns
    -------
    pd.DataFrame  The filtered dataframe (index preserved).
    """
    n_total = len(df)
    excluded: dict[str, pd.Index] = {}

    # 1. Unlabeled
    mask_unlabeled = df["DQ_TARGET"].isna()
    excluded["unlabeled (no DQ_TARGET)"] = df.index[mask_unlabeled]

    # 2. Thin file
    if "n_days__all" in df.columns:
        mask_thin = (~mask_unlabeled) & (df["n_days__all"] < MIN_OBS_DAYS)
        excluded[f"thin file (<{MIN_OBS_DAYS} obs days)"] = df.index[mask_thin]

    # 3. Inactive
    tx_90  = df["n_tx__90d"]  if "n_tx__90d"  in df.columns else pd.Series(np.nan, index=df.index)
    tx_180 = df["n_tx__180d"] if "n_tx__180d" in df.columns else pd.Series(np.nan, index=df.index)
    if "n_tx__90d" in df.columns or "n_tx__180d" in df.columns:
        base = (~mask_unlabeled)
        if "n_days__all" in df.columns:
            base = base & (df["n_days__all"] >= MIN_OBS_DAYS)
        mask_inactive = base & (tx_90.fillna(0) < MIN_TX_90D) & (tx_180.fillna(0) < MIN_TX_180D)
        excluded[f"inactive (<{MIN_TX_90D} tx in 90d AND 180d)"] = df.index[mask_inactive]

    # 4. Dormant
    if "balance__mean__all" in df.columns:
        base = (~mask_unlabeled)
        if "n_days__all" in df.columns:
            base = base & (df["n_days__all"] >= MIN_OBS_DAYS)
        mask_dormant = base & (df["balance__mean__all"].abs() < MEAN_BAL_ZERO_TOL)
        excluded["dormant (zero mean balance)"] = df.index[mask_dormant]

    # Union of all excluded indices
    all_excluded = pd.Index([])
    for idx in excluded.values():
        all_excluded = all_excluded.union(idx)

    retained = df.drop(index=all_excluded)

    if verbose:
        print("─" * 55)
        print(f"Scoring Exclusion Report")
        print("─" * 55)
        print(f"{'Total consumers':<40} {n_total:>6}")
        for reason, idx in excluded.items():
            print(f"  Excluded – {reason:<28} {len(idx):>6}")
        print(f"{'Total excluded':<40} {len(all_excluded):>6}  ({len(all_excluded)/n_total:.1%})")
        print(f"{'Retained (scoreable)':<40} {len(retained):>6}  ({len(retained)/n_total:.1%})")
        if "DQ_TARGET" in retained.columns:
            print(f"{'DQ rate (retained)':<40} {retained['DQ_TARGET'].mean():>6.2%}")
        print("─" * 55)

    return retained


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------

def prepare_split(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Clean ``features_df``, subset to *feature_cols*, and do a 60/20/20 split.

    Parameters
    ----------
    features_df:
        Full feature matrix including ``DQ_TARGET``.
    feature_cols:
        Ordered list of feature column names to include.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, available_features
    """
    df_clean = features_df[features_df["DQ_TARGET"].notna()].copy()
    y = df_clean["DQ_TARGET"].astype(int).values

    available = [f for f in feature_cols if f in df_clean.columns]
    X = df_clean[available].fillna(0).values

    # 60 / 40 → 20 / 20
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, available


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def load_and_split(
    output_dir: Optional[Union[str, Path]] = None,
    n_features: int = 50,
    use_all: bool = False,
    apply_exclusions: bool = False,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           List[str], pd.DataFrame]:
    """End-to-end pipeline: load → (optionally exclude) → select features → 60/20/20 split.

    Parameters
    ----------
    output_dir:
        Path to the ``output/`` folder.  Defaults to the workspace-level
        ``output/`` directory.
    n_features:
        Number of top-ranked features to use.  Ignored when *use_all* is True.
        **Change this value** in your notebook to experiment with different
        feature set sizes.
    use_all:
        When True, every column except ``DQ_TARGET`` is used and *n_features*
        is ignored.
    apply_exclusions:
        When True, ``apply_scoring_exclusions()`` is called before splitting,
        removing thin files, inactive and dormant accounts.  Set to False
        (default) to match the original full-population behaviour.
    random_state:
        Random seed passed to ``train_test_split``.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray
    y_train, y_val, y_test : np.ndarray
    feature_names : list[str]   — the actual columns used (in order)
    features_df   : pd.DataFrame — the (possibly filtered) data
    """
    features_df, ranked_features = load_features(output_dir)

    if apply_exclusions:
        features_df = apply_scoring_exclusions(features_df, verbose=True)

    if use_all:
        feature_cols = [c for c in features_df.columns if c != "DQ_TARGET"]
        label = "all"
    else:
        feature_cols = ranked_features[:n_features]
        label = f"top-{n_features}"

    X_train, X_val, X_test, y_train, y_val, y_test, available = prepare_split(
        features_df, feature_cols, random_state=random_state
    )

    total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    dq_rate = features_df["DQ_TARGET"].dropna().mean()

    print(f"Features  : {len(available)} ({label})")
    print(f"Samples   : {total}  |  DQ rate: {dq_rate:.2%}")
    print(f"Train     : {X_train.shape[0]}  ({y_train.mean():.2%} positive)")
    print(f"Val       : {X_val.shape[0]}  ({y_val.mean():.2%} positive)")
    print(f"Test      : {X_test.shape[0]}  ({y_test.mean():.2%} positive)")

    return X_train, X_val, X_test, y_train, y_val, y_test, available, features_df
