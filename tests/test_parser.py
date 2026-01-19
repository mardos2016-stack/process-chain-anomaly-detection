import pandas as pd

from process_chain_model import parse_chains


def test_parse_chains_reverses_order():
    df = pd.DataFrame({"chain_proc_names": ["a.exe ← b.exe ← c.exe"]})
    parsed = parse_chains(df).iloc[0]
    assert parsed == ["c.exe", "b.exe", "a.exe"]
