"""
Feature selection utilities for the PRISM delinquency pipeline.

Provides:
  - load_features() — load a (pre-ordered) features CSV

Re-exported by scripts.model_data so notebooks only need:
    from scripts.model_data import load_and_split
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
    features_filename: str = "features.csv",
) -> Tuple[pd.DataFrame, List[str]]:
    """Load a features CSV.  Column order is used as the feature ranking.

    Parameters
    ----------
    output_dir:
        Path to the ``output/`` folder.  Defaults to the workspace-level
        ``output/`` directory relative to this file.
    features_filename:
        CSV file to load (must contain ``DQ_TARGET``).  Pass a
        pre-ordered file (e.g. ``*_consensus_ordered.csv``) and the
        column order will serve as the feature ranking.

    Returns
    -------
    features_df : pd.DataFrame
        Raw feature matrix with ``DQ_TARGET`` column.
    ranked_features : list[str]
        Feature names in the order they appear in the CSV (excludes ``DQ_TARGET``).
    """
    out = _resolve_output(output_dir)
    features_df = pd.read_csv(out / features_filename, index_col=0)
    ranked_features = [c for c in features_df.columns if c != "DQ_TARGET"]
    return features_df, ranked_features


