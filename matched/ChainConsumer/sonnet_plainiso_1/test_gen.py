import numpy as np
import pandas as pd
import pytest

from chainconsumer import ChainConsumer


def make_data(n=100, d=2, seed=0):
    return np.random.RandomState(seed).normal(size=(n, d))


def test_add_chain_returns_self_and_default_names():
    c = ChainConsumer()
    result = c.add_chain(make_data())
    assert result is c
    c.add_chain(make_data(seed=1))
    assert c._all_names() == ["Chain 0", "Chain 1"]
    assert c.chains[0].parameters == ["0", "1"]


def test_add_chain_with_parameters_and_name():
    c = ChainConsumer()
    c.add_chain(make_data(), parameters=["a", "b"], name="mine")
    assert c.chains[0].name == "mine"
    assert c.chains[0].parameters == ["a", "b"]
    assert c._all_parameters == ["a", "b"]
    assert c._get_chain_name(0) == "mine"


def test_add_chain_dict():
    c = ChainConsumer()
    data = {"x": np.arange(50, dtype=float), "y": np.arange(50, dtype=float) * 2}
    c.add_chain(data)
    assert c.chains[0].parameters == ["x", "y"]
    assert c.chains[0].chain.shape == (50, 2)


def test_add_chain_dict_with_parameters_fails():
    c = ChainConsumer()
    with pytest.raises(AssertionError):
        c.add_chain({"x": np.arange(10.0)}, parameters=["x"])


def test_add_chain_dataframe_extracts_weight_and_posterior():
    c = ChainConsumer()
    rng = np.random.RandomState(3)
    df = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60),
                       "weight": np.ones(60), "posterior": rng.normal(size=60)})
    c.add_chain(df)
    chain = c.chains[0]
    assert chain.parameters == ["a", "b"]
    assert chain.chain.shape == (60, 2)
    assert chain.posterior is not None


def test_add_chain_dataframe_with_parameters_fails():
    c = ChainConsumer()
    df = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})
    with pytest.raises(AssertionError):
        c.add_chain(df, parameters=["a", "b"])


def test_add_chain_list_is_transposed():
    c = ChainConsumer()
    cols = [list(np.arange(30.0)), list(np.arange(30.0) * 2)]
    c.add_chain(cols, parameters=["p", "q"])
    assert c.chains[0].chain.shape == (30, 2)


def test_add_chain_1d_becomes_column():
    c = ChainConsumer()
    c.add_chain(np.random.RandomState(0).normal(size=80), parameters=["only"])
    assert c.chains[0].chain.shape == (80, 1)


def test_add_chain_from_csv_and_npy(tmp_path):
    data = make_data(40, 2)
    csv_path = str(tmp_path / "chain.csv")
    pd.DataFrame(data, columns=["u", "v"]).to_csv(csv_path, index=False)
    npy_path = str(tmp_path / "chain.npy")
    np.save(npy_path, data)

    c = ChainConsumer()
    c.add_chain(csv_path)
    c.add_chain(npy_path, parameters=["u", "v"])
    assert c.chains[0].parameters == ["u", "v"]
    assert c.chains[0].chain.shape == (40, 2)
    assert c.chains[1].chain.shape == (40, 2)


def test_add_chain_invalid_inputs():
    c = ChainConsumer()
    with pytest.raises(AssertionError):
        c.add_chain(None)
    with pytest.raises(AssertionError):
        c.add_chain(make_data(d=2), parameters=["a", "b", "c"])
    with pytest.raises(AssertionError):
        c.add_chain(make_data(d=2), power="high")


def test_default_parameters_reused_and_checked():
    c = ChainConsumer()
    c.add_chain(make_data(d=2), parameters=["a", "b"])
    c.add_chain(make_data(d=2, seed=5))
    assert c.chains[1].parameters == ["a", "b"]
    with pytest.raises(AssertionError):
        c.add_chain(make_data(d=3))


def test_grid_with_meshgrid_weights():
    c = ChainConsumer()
    x = np.linspace(0, 1, 4)
    y = np.linspace(0, 2, 4)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    weights = np.exp(-((xx - 0.5) ** 2 + (yy - 1) ** 2))
    c.add_chain([x, y], parameters=["x", "y"], weights=weights, grid=True)
    assert c.chains[0].chain.shape == (16, 2)


def test_grid_requires_weights_and_no_walkers():
    c = ChainConsumer()
    with pytest.raises(AssertionError):
        c.add_chain(make_data(), grid=True)
    with pytest.raises(AssertionError):
        c.add_chain(make_data(), grid=True, weights=np.ones(100), walkers=2)


def test_add_marker_is_not_mcmc_chain():
    c = ChainConsumer()
    c.add_chain(make_data(), parameters=["a", "b"])
    result = c.add_marker([0.1, 0.2], parameters=["a", "b"], name="marker")
    assert result is c
    assert c.chains[-1].name == "marker"
    assert c.chains[-1].mcmc_chain is False
    assert c.chains[-1].chain.shape == (2, 2)
    assert len(c.get_mcmc_chains()) == 1


def test_add_covariance_marks_chain_not_mcmc(monkeypatch):
    orig = np.random.multivariate_normal
    monkeypatch.setattr(np.random, "multivariate_normal",
                        lambda m, cov, size=None: orig(m, cov, size=2000))
    c = ChainConsumer()
    result = c.add_covariance([1.0, 2.0], [[1.0, 0.1], [0.1, 2.0]],
                              parameters=["a", "b"], name="fisher")
    assert result is c
    assert c.chains[0].mcmc_chain is False
    assert c.chains[0].name == "fisher"
    assert c.chains[0].chain.shape == (2000, 2)
    assert c.get_mcmc_chains() == []


def test_remove_chain_default_name_index_and_list():
    c = ChainConsumer()
    c.add_chain(make_data(), parameters=["a", "b"], name="one")
    c.add_chain(make_data(seed=1), parameters=["a", "c"], name="two")
    c.add_chain(make_data(seed=2), parameters=["a", "d"], name="three")
    c.add_chain(make_data(seed=3), parameters=["a", "e"], name="four")

    assert c.remove_chain() is c
    assert c._all_names() == ["one", "two", "three"]
    c.remove_chain("two")
    assert c._all_names() == ["one", "three"]
    assert c._all_parameters == ["a", "b", "d"]
    c.remove_chain([0, "three"])
    assert c.chains == []
    assert c._all_parameters == []


def test_remove_chain_errors():
    c = ChainConsumer()
    c.add_chain(make_data(), name="one")
    with pytest.raises(AssertionError):
        c.remove_chain([0, "one"])
    with pytest.raises(AssertionError):
        c.remove_chain("missing")
    with pytest.raises(AssertionError):
        c.remove_chain(5)
    with pytest.raises(ValueError):
        c.remove_chain([1.5])


def test_divide_chain_splits_walkers():
    c = ChainConsumer()
    data = make_data(100, 2)
    c.add_chain(data, parameters=["a", "b"], walkers=4)
    new = c.divide_chain(0)
    assert isinstance(new, ChainConsumer)
    assert new is not c
    assert len(new.chains) == 4
    for sub in new.chains:
        assert sub.chain.shape == (25, 2)
        assert sub.parameters == ["a", "b"]
    assert new.chains[0].name == "Chain 0"
    np.testing.assert_allclose(new.chains[1].chain, data[25:50])


def test_divide_chain_without_walkers_fails():
    c = ChainConsumer()
    c.add_chain(make_data())
    with pytest.raises(AssertionError):
        c.divide_chain(0)


def test_configure_truth_defaults_and_overrides():
    c = ChainConsumer()
    assert c.configure_truth() is c
    assert c.config_truth["ls"] == "--"
    assert c.config_truth["linewidth"] == 1
    assert c.config_truth["color"] == "#000000"
    assert c.config_truth["zorder"] == 100
    assert c._configured_truth is True

    c.configure_truth(linestyle=":", color="#ff0000", lw=3)
    assert "ls" not in c.config_truth
    assert c.config_truth["color"] == "#ff0000"
    assert "linewidth" not in c.config_truth


def test_configure_and_reset_on_add_chain():
    c = ChainConsumer()
    c.add_chain(make_data(), parameters=["a", "b"])
    assert c._configured is False
    assert c.configure() is c
    assert c._configured is True
    assert list(c.config["sigmas"]) == [0, 1, 2]
    assert c.config["max_ticks"] == 5
    c.add_chain(make_data(seed=4), parameters=["a", "b"])
    assert c._configured is False
    assert c.config == {}
