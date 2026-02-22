"""
Feature selection utilities for the PRISM delinquency pipeline.

Provides:
  - load_features()        — load features.csv + ranked_features.txt
  - save_sorted_features() — write sorted_features.csv (columns ordered by rank)

These are re-exported by scripts.model_data so notebooks only need:
    from scripts.model_data import load_and_split, save_sorted_features
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_FS_DIR: Path = Path(__file__).resolve().parent
_DEFAULT_OUTPUT: Path = _FS_DIR.parent / "output"


def _resolve_output(output_dir: Optional[Union[str, Path]]) -> Path:
    return Path(output_dir).resolve() if output_dir else _DEFAULT_OUTPUT


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_features(
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load ``features.csv`` and ``ranked_features.txt``.

    Parameters
    ----------
    output_dir:
        Path to the ``output/`` folder.  Defaults to the workspace-level
        ``output/`` directory relative to this file.

    Returns
    -------
    features_df : pd.DataFrame
        Raw feature matrix with ``DQ_TARGET`` column.
    ranked_features : list[str]
        Feature names ordered by importance (best first).
    """
    out = _resolve_output(output_dir)
    features_df = pd.read_csv(out / "features.csv", index_col=0)

    with open(out / "ranked_features.txt", "r") as fh:
        ranked_features = [line.strip() for line in fh if line.strip()]

    return features_df, ranked_features


def save_sorted_features(
    features_df: Optional[pd.DataFrame] = None,
    ranked_features: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Write ``sorted_features.csv`` — ``features.csv`` with columns ordered by rank.

    The CSV places the highest-ranked feature first.  ``DQ_TARGET`` and any
    features absent from ``ranked_features.txt`` are appended at the end.

    Parameters
    ----------
    features_df, ranked_features:
        Pre-loaded objects.  If *None* they are loaded from disk automatically.
    output_dir:
        Path to the ``output/`` folder.

    Returns
    -------
    sorted_df : pd.DataFrame
        The re-ordered dataframe (also saved to disk).
    """
    out = _resolve_output(output_dir)

    if features_df is None or ranked_features is None:
        features_df, ranked_features = load_features(output_dir)

    ranked_present = [f for f in ranked_features if f in features_df.columns]
    remaining = [c for c in features_df.columns if c not in ranked_present]
    sorted_df = features_df[ranked_present + remaining]

    dest = out / "sorted_features.csv"
    sorted_df.to_csv(dest)
    print(f"Saved {dest.name}  ({len(ranked_present)} ranked + {len(remaining)} other columns)")
    return sorted_df
