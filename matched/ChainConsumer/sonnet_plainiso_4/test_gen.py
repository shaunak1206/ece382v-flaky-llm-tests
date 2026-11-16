import numpy as np
import pandas as pd
import pytest

from chainconsumer import ChainConsumer


def make_data(n=200, dim=2, seed=0):
    rng = np.random.RandomState(seed)
    return rng.normal(size=(n, dim))


def test_add_chain_returns_self_and_appends():
    c = ChainConsumer()
    result = c.add_chain(make_data(), parameters=["a", "b"])
    assert result is c
    assert len(c.chains) == 1
    assert c.chains[0].parameters == ["a", "b"]


def test_default_chain_names_increment():
    c = ChainConsumer()
    c.add_chain(make_data(), parameters=["a", "b"])
    c.add_chain(make_data(seed=1), parameters=["a", "b"])
    assert c._all_names() == ["Chain 0", "Chain 1"]
    assert c._get_chain_name(1) == "Chain 1"


def test_custom_name_used():
    c = ChainConsumer()
    c.add_chain(make_data(), parameters=["a", "b"], name="mine")
    assert c.chains[0].name == "mine"


def test_add_chain_from_dict():
    c = ChainConsumer()
    d = {"x": np.arange(50, dtype=float), "y": np.arange(50, dtype=float) * 2}
    c.add_chain(d)
    chain = c.chains[0]
    assert chain.parameters == ["x", "y"]
    assert chain.chain.shape == (50, 2)
    np.testing.assert_allclose(chain.chain[:, 1], d["y"])


def test_add_chain_from_dict_with_parameters_fails():
    c = ChainConsumer()
    with pytest.raises(AssertionError):
        c.add_chain({"x": np.arange(10.0)}, parameters=["x"])


def test_add_chain_from_dataframe_extracts_weight_and_posterior():
    c = ChainConsumer()
    n = 100
    rng = np.random.RandomState(3)
    df = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "weight": np.ones(n) * 2.0,
            "posterior": rng.normal(size=n),
        }
    )
    c.add_chain(df)
    chain = c.chains[0]
    assert chain.parameters == ["a", "b"]
    assert chain.chain.shape == (n, 2)
    assert chain.posterior is not None
    assert chain.weights is not None


def test_add_chain_dataframe_with_parameters_fails():
    c = ChainConsumer()
    df = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})
    with pytest.raises(AssertionError):
        c.add_chain(df, parameters=["a", "b"])


def test_add_chain_from_list_of_columns():
    c = ChainConsumer()
    cols = [list(np.arange(30.0)), list(np.arange(30.0) + 1)]
    c.add_chain(cols, parameters=["p", "q"])
    assert c.chains[0].chain.shape == (30, 2)


def test_parameters_default_to_indices():
    c = ChainConsumer()
    c.add_chain(make_data(dim=3))
    assert c.chains[0].parameters == ["0", "1", "2"]


def test_default_parameters_reused_by_later_chains():
    c = ChainConsumer()
    c.add_chain(make_data(), parameters=["a", "b"])
    c.add_chain(make_data(seed=2))
    assert c.chains[1].parameters == ["a", "b"]


def test_default_parameters_dimension_mismatch_fails():
    c = ChainConsumer()
    c.add_chain(make_data(), parameters=["a", "b"])
    with pytest.raises(AssertionError):
        c.add_chain(make_data(dim=3))


def test_too_many_parameter_names_fails():
    c = ChainConsumer()
    with pytest.raises(AssertionError):
        c.add_chain(make_data(dim=2), parameters=["a", "b", "c"])


def test_none_chain_fails():
    c = ChainConsumer()
    with pytest.raises(AssertionError):
        c.add_chain(None)


def test_non_numeric_power_fails():
    c = ChainConsumer()
    with pytest.raises(AssertionError):
        c.add_chain(make_data(), parameters=["a", "b"], power="two")


def test_grid_validation():
    c = ChainConsumer()
    data = make_data()
    with pytest.raises(AssertionError):
        c.add_chain(data, parameters=["a", "b"], grid=True)  # no weights
    with pytest.raises(AssertionError):
        c.add_chain(data, parameters=["a", "b"], grid=True, weights=np.ones(200), walkers=2)


def test_all_parameters_tracked_without_duplicates():
    c = ChainConsumer()
    c.add_chain(make_data(), parameters=["a", "b"])
    c.add_chain(make_data(seed=1), parameters=["b", "c"])
    assert c._all_parameters == ["a", "b", "c"]


def test_add_chain_resets_configured_flag():
    c = ChainConsumer()
    c.add_chain(make_data(), parameters=["a", "b"])
    c._configured = True
    c.add_chain(make_data(seed=1), parameters=["a", "b"])
    assert c._configured is False
    assert c.config == {}


def test_add_marker_creates_non_mcmc_chain():
    c = ChainConsumer()
    c.add_chain(make_data(), parameters=["a", "b"], name="real")
    result = c.add_marker([0.1, 0.2], parameters=["a", "b"], name="marker")
    assert result is c
    assert len(c.chains) == 2
    assert c.chains[-1].mcmc_chain is False
    assert c.chains[-1].name == "marker"
    assert [ch.name for ch in c.get_mcmc_chains()] == ["real"]


def test_add_covariance_creates_non_mcmc_chain(monkeypatch):
    original = np.random.multivariate_normal

    def small(mean, cov, size=None):
        return original(mean, cov, size=500)

    monkeypatch.setattr(np.random, "multivariate_normal", small)
    c = ChainConsumer()
    result = c.add_covariance([0.0, 1.0], [[1.0, 0.1], [0.1, 2.0]], parameters=["a", "b"], name="cov")
    assert result is c
    assert c.chains[0].mcmc_chain is False
    assert c.chains[0].chain.shape == (500, 2)
    assert c.get_mcmc_chains() == []


def test_remove_chain_by_default_index_and_name():
    c = ChainConsumer()
    c.add_chain(make_data(), parameters=["a", "b"], name="one")
    c.add_chain(make_data(seed=1), parameters=["a", "b"], name="two")
    c.add_chain(make_data(seed=2), parameters=["a", "b"], name="three")
    c.remove_chain()
    assert c._all_names() == ["one", "two"]
    c.remove_chain("one")
    assert c._all_names() == ["two"]
    c.remove_chain(0)
    assert c.chains == []
    assert c._all_parameters == []


def test_remove_chain_list_and_updates_parameters():
    c = ChainConsumer()
    c.add_chain(make_data(), parameters=["a", "b"], name="one")
    c.add_chain(make_data(seed=1), parameters=["c", "d"], name="two")
    c.add_chain(make_data(seed=2), parameters=["a", "b"], name="three")
    c.remove_chain([0, "two"])
    assert c._all_names() == ["three"]
    assert c._all_parameters == ["a", "b"]


def test_remove_chain_twice_in_one_call_fails():
    c = ChainConsumer()
    c.add_chain(make_data(), parameters=["a", "b"], name="one")
    with pytest.raises(AssertionError):
        c.remove_chain([0, "one"])


def test_get_chain_errors():
    c = ChainConsumer()
    c.add_chain(make_data(), parameters=["a", "b"], name="one")
    assert c._get_chain("one") == [0]
    assert c._get_chain(0) == [0]
    assert c._get_chain(c.chains[0]) == [0]
    with pytest.raises(AssertionError):
        c._get_chain("missing")
    with pytest.raises(AssertionError):
        c._get_chain(5)
    with pytest.raises(ValueError):
        c._get_chain(1.5)


def test_divide_chain_splits_walkers():
    c = ChainConsumer()
    data = make_data(n=400)
    c.add_chain(data, parameters=["a", "b"], walkers=4, name="main")
    new = c.divide_chain("main")
    assert isinstance(new, ChainConsumer)
    assert new is not c
    assert len(new.chains) == 4
    for sub in new.chains:
        assert sub.chain.shape == (100, 2)
        assert sub.parameters == ["a", "b"]
    np.testing.assert_allclose(new.chains[1].chain, data[100:200])


def test_divide_chain_without_walkers_fails():
    c = ChainConsumer()
    c.add_chain(make_data(), parameters=["a", "b"])
    with pytest.raises(AssertionError):
        c.divide_chain(0)


def test_configure_sets_configured_and_config():
    c = ChainConsumer()
    c.add_chain(make_data(n=1000), parameters=["a", "b"])
    result = c.configure()
    assert result is c
    assert c._configured is True
    assert list(c.config["sigmas"]) == [0, 1, 2]
    assert c.config["max_ticks"] == 5
    assert c.config["summary_area"] if "summary_area" in c.config else True


def test_configure_truth_defaults():
    c = ChainConsumer()
    result = c.configure_truth()
    assert result is c
    assert c._configured_truth is True
    assert c.config_truth["color"] == "#000000"
    assert c.config_truth["zorder"] == 100
    assert c.config_truth["ls"] == "--"
