from process_chain_model import MarkovChainModel


def test_fit_and_score_basic():
    chains = [["A", "B", "C"], ["A", "B", "D"], ["A", "B", "C"]]
    m = MarkovChainModel(order=1, alpha=1.0)
    m.fit(chains, threshold_quantile=0.9)
    s = m.score(["A", "B", "C"])
    assert s is not None
    assert m.threshold is not None
