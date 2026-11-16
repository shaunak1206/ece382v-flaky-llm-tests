import numpy as np
import pandas as pd
import pytest

from chainconsumer import ChainConsumer


@pytest.fixture
def data():
    rng = np.random.RandomState(42)
    return rng.normal(size=(200, 3))


def make_consumer(data, names=("a", "b", "c"), **kwargs):
    c = ChainConsumer()
    c.add_chain(data, parameters=list(names), **kwargs)
    return c


def test_init_empty_state():
    c = ChainConsumer()
    assert c.chains == []
    assert c._all_parameters == []
    assert c._configured is False
    assert c.config == {}
    assert isinstance(ChainConsumer.__version__, str)


def test_add_chain_ndarray_returns_self_and_defaults(data):
    c = ChainConsumer()
    result = c.add_chain(data, parameters=["a", "b", "c"])
    assert result is c
    assert len(c.chains) == 1
    assert c.chains[0].name == "Chain 0"
    assert c.chains[0].parameters == ["a", "b", "c"]
    assert c.chains[0].chain.shape == data.shape
    assert c._all_parameters == ["a", "b", "c"]


def test_add_chain_custom_name_and_chaining(data):
    c = ChainConsumer()
    c.add_chain(data, parameters=["a", "b", "c"], name="first").add_chain(data, name="second")
    assert c._all_names() == ["first", "second"]
    assert c._get_chain_name(1) == "second"


def test_add_chain_without_parameters_generates_names(data):
    c = ChainConsumer()
    c.add_chain(data)
    assert c.chains[0].parameters == ["0", "1", "2"]


def test_add_chain_dict_uses_keys():
    rng = np.random.RandomState(1)
    d = {"x": rng.normal(size=100), "y": rng.normal(size=100)}
    c = ChainConsumer()
    c.add_chain(d)
    assert c.chains[0].parameters == ["x", "y"]
    assert c.chains[0].chain.shape == (100, 2)
    assert np.allclose(c.chains[0].chain[:, 0], d["x"])


def test_add_chain_dict_with_parameters_fails():
    d = {"x": np.arange(10.0), "y": np.arange(10.0)}
    c = ChainConsumer()
    with pytest.raises(AssertionError):
        c.add_chain(d, parameters=["x", "y"])


def test_add_chain_dataframe_extracts_weight_and_posterior():
    rng = np.random.RandomState(2)
    df = pd.DataFrame(
        {
            "x": rng.normal(size=50),
            "y": rng.normal(size=50),
            "weight": np.full(50, 2.0),
            "posterior": rng.normal(size=50),
        }
    )
    c = ChainConsumer()
    c.add_chain(df)
    chain = c.chains[0]
    assert chain.parameters == ["x", "y"]
    assert chain.chain.shape == (50, 2)
    assert chain.posterior is not None
    assert np.allclose(chain.weights, 2.0)


def test_add_chain_dataframe_with_parameters_fails():
    df = pd.DataFrame({"x": np.arange(10.0), "y": np.arange(10.0)})
    c = ChainConsumer()
    with pytest.raises(AssertionError):
        c.add_chain(df, parameters=["x", "y"])


def test_add_chain_none_fails():
    c = ChainConsumer()
    with pytest.raises(AssertionError):
        c.add_chain(None)


def test_add_chain_list_is_transposed():
    c = ChainConsumer()
    c.add_chain([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], parameters=["a", "b"])
    assert c.chains[0].chain.shape == (4, 2)
    assert np.allclose(c.chains[0].chain[:, 0], [1, 2, 3, 4])


def test_add_chain_1d_array_becomes_column():
    rng = np.random.RandomState(3)
    c = ChainConsumer()
    c.add_chain(rng.normal(size=100), parameters=["x"])
    assert c.chains[0].chain.shape == (100, 1)


def test_default_parameters_reused(data):
    c = ChainConsumer()
    c.add_chain(data, parameters=["a", "b", "c"])
    c.add_chain(data)
    assert c.chains[1].parameters == ["a", "b", "c"]


def test_default_parameters_dimension_mismatch(data):
    c = ChainConsumer()
    c.add_chain(data, parameters=["a", "b", "c"])
    with pytest.raises(AssertionError):
        c.add_chain(data[:, :2])


def test_too_many_parameter_names_fails(data):
    c = ChainConsumer()
    with pytest.raises(AssertionError):
        c.add_chain(data, parameters=["a", "b", "c", "d"])


def test_non_numeric_power_fails(data):
    c = ChainConsumer()
    with pytest.raises(AssertionError):
        c.add_chain(data, parameters=["a", "b", "c"], power="2")


def test_add_chain_from_csv_and_npy(tmp_path, data):
    csv_path = tmp_path / "chain.csv"
    pd.DataFrame(data, columns=["a", "b", "c"]).to_csv(str(csv_path), index=False)
    npy_path = tmp_path / "chain.npy"
    np.save(str(npy_path), data)

    c = ChainConsumer()
    c.add_chain(str(csv_path))
    c.add_chain(str(npy_path))
    assert c.chains[0].parameters == ["a", "b", "c"]
    assert c.chains[0].chain.shape == data.shape
    assert c.chains[1].chain.shape == data.shape


def test_add_marker_not_mcmc(data):
    c = make_consumer(data)
    c.add_marker([0.0, 0.0, 0.0], parameters=["a", "b", "c"], name="marker")
    assert len(c.chains) == 2
    assert c.chains[-1].mcmc_chain is False
    assert c.chains[-1].name == "marker"
    assert len(c.get_mcmc_chains()) == 1
    assert c.get_mcmc_chains()[0].name == "Chain 0"


def test_add_covariance_samples():
    mean = [1.0, -2.0]
    cov = [[1.0, 0.3], [0.3, 2.0]]
    c = ChainConsumer()
    result = c.add_covariance(mean, cov, parameters=["x", "y"], name="cov")
    assert result is c
    chain = c.chains[0]
    assert chain.mcmc_chain is False
    assert chain.name == "cov"
    assert chain.parameters == ["x", "y"]
    assert np.allclose(chain.chain.mean(axis=0), mean, atol=0.05)
    assert c.get_mcmc_chains() == []


def test_remove_chain_default_and_by_name_and_index(data):
    c = ChainConsumer()
    c.add_chain(data, parameters=["a", "b", "c"], name="one")
    c.add_chain(data[:, :2], parameters=["a", "d"], name="two")
    c.add_chain(data, parameters=["a", "b", "c"], name="three")

    c.remove_chain()
    assert c._all_names() == ["one", "two"]
    assert c._all_parameters == ["a", "b", "c", "d"]

    c.remove_chain("one")
    assert c._all_names() == ["two"]
    assert c._all_parameters == ["a", "d"]

    c.remove_chain(0)
    assert c.chains == []
    assert c._all_parameters == []


def test_remove_chain_multiple_and_errors(data):
    c = ChainConsumer()
    for n in ["x", "y", "z"]:
        c.add_chain(data, parameters=["a", "b", "c"], name=n)
    c.remove_chain([0, "z"])
    assert c._all_names() == ["y"]
    with pytest.raises(AssertionError):
        c.remove_chain("missing")
    with pytest.raises(AssertionError):
        c.remove_chain(5)
    with pytest.raises(AssertionError):
        c.remove_chain([0, "y"])


def test_get_chain_invalid_type(data):
    c = make_consumer(data)
    with pytest.raises(ValueError):
        c._get_chain(1.5)
    assert c._get_chain(c.chains[0]) == [0]
    assert c._get_chain("Chain 0") == [0]


def test_divide_chain_splits_walkers():
    rng = np.random.RandomState(5)
    data = rng.normal(size=(100, 2))
    c = ChainConsumer()
    c.add_chain(data, parameters=["a", "b"], walkers=4)
    divided = c.divide_chain(0)
    assert isinstance(divided, ChainConsumer)
    assert len(divided.chains) == 4
    for sub in divided.chains:
        assert sub.chain.shape == (25, 2)
        assert sub.parameters == ["a", "b"]
    assert np.allclose(divided.chains[1].chain, data[25:50])


def test_divide_chain_without_walkers_fails(data):
    c = make_consumer(data)
    with pytest.raises(AssertionError):
        c.divide_chain(0)


def test_grid_with_multidimensional_weights():
    x = np.linspace(0, 1, 3)
    y = np.linspace(0, 1, 3)
    weights = np.random.RandomState(7).uniform(size=(3, 3))
    c = ChainConsumer()
    c.add_chain([x, y], parameters=["x", "y"], weights=weights, grid=True)
    assert c.chains[0].chain.shape == (9, 2)
    assert c.chains[0].weights.size == 9


def test_grid_with_walkers_fails():
    x = np.linspace(0, 1, 5)
    c = ChainConsumer()
    with pytest.raises(AssertionError):
        c.add_chain(x[:, None], parameters=["x"], weights=np.ones(5), grid=True, walkers=2)


def test_grid_without_weights_fails():
    x = np.linspace(0, 1, 5)
    c = ChainConsumer()
    with pytest.raises(AssertionError):
        c.add_chain(x[:, None], parameters=["x"], grid=True)


def test_configure_sets_state(data):
    c = make_consumer(data)
    result = c.configure(usetex=False)
    assert result is c
    assert c._configured is True
    assert list(c.config["sigmas"]) == [0, 1, 2]
    assert c.config["usetex"] is False
    assert c.config["legend_kwargs"]["loc"] == "upper right"


def test_configure_invalid_statistics_fails(data):
    c = make_consumer(data)
    with pytest.raises(AssertionError):
        c.configure(statistics="not_a_statistic")


def test_adding_chain_resets_configuration(data):
    c = make_consumer(data)
    c.configure(usetex=False)
    assert c._configured is True
    c.add_chain(data, name="other")
    assert c._configured is False
    assert c.config == {}


def test_configure_truth_defaults():
    c = ChainConsumer()
    result = c.configure_truth()
    assert result is c
    assert c.config_truth["color"] == "#000000"
    assert c.config_truth["ls"] == "--"
    assert c.config_truth["zorder"] == 100
    c.configure_truth(color="#ff0000", linestyle=":")
    assert c.config_truth["color"] == "#ff0000"
    assert "ls" not in c.config_truth
