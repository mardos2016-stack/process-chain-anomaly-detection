from process_chain_model import MarkovChainModel


def test_fit_and_predict_smoke():
    chains = [
        ["a", "b", "c"],
        ["a", "b", "d"],
        ["a", "b", "c"],
    ]
    model = MarkovChainModel(order=1, alpha=1.0)
    model.fit(chains, threshold_quantile=0.95)

    assert model.threshold is not None
    assert model.predict(["a", "b", "c"]) in (1, -1)
