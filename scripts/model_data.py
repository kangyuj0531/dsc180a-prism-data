"""
Shared data-loading and preparation utilities for delinquency models.

Typical usage from a notebook in models/:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path('..').resolve()))
    from scripts.model_data import load_and_split, save_sorted_features

    N_FEATURES = 50               # ← change this to adjust feature count

    # Top-N features (CSV column order = feature ranking)
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, features_df = \\
        load_and_split(n_features=N_FEATURES)

    # All features
    X_train_all, X_val_all, X_test_all, y_train_all, y_val_all, y_test_all, all_cols, _ = \\
        load_and_split(use_all=True)
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Core loaders live in feature_selection; re-export them here for convenience
from scripts.feature_selection import (
    load_features,
    _resolve_output,
)


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
    random_state: int = 42,
    features_filename: str = "features.csv"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           List[str], pd.DataFrame]:
    """End-to-end pipeline: load → select features → 60/20/20 split.

    Parameters
    ----------
    output_dir:
        Path to the ``output/`` folder.  Defaults to the workspace-level
        ``output/`` directory.
    n_features:
        Number of top features to use (by CSV column order).  Ignored when
        *use_all* is True.  **Change this value** in your notebook to
        experiment with different feature set sizes.
    use_all:
        When True, every column except ``DQ_TARGET`` is used.
    random_state:
        Random seed passed to ``train_test_split``.
    features_filename:
        CSV file in ``output/`` to load.  Column order is treated as the
        feature ranking (pass a ``*_consensus_ordered.csv`` for pre-ranked data).

    Returns
    -------
    X_train, X_val, X_test : np.ndarray
    y_train, y_val, y_test : np.ndarray
    feature_names : list[str]   — the actual columns used (in order)
    features_df   : pd.DataFrame
    """
    features_df, ranked_features = load_features(output_dir, features_filename=features_filename)

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
