from __future__ import annotations

import pandas as pd


def parse_chains(df: pd.DataFrame, column_name: str = "chain_proc_names") -> pd.Series:
    """Parse process chains from an Excel-style string into a list.

    Input column example:
        "svchost.exe ← services.exe ← wininit.exe"

    Output for each row:
        ["wininit.exe", "services.exe", "svchost.exe"]

    Args:
        df: Input dataframe.
        column_name: Column containing process chains.

    Returns:
        Pandas Series where each value is a list of processes from parent -> child.
    """

    def _parse_single_chain(chain_str):
        if not isinstance(chain_str, str):
            return []
        return [p.strip() for p in chain_str.split("←")][::-1]

    return df[column_name].apply(_parse_single_chain)
