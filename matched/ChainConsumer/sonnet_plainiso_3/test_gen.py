import numpy as np
import pandas as pd
import pytest

from chainconsumer import ChainConsumer


def make_data(n=200, dim=2, seed=0):
    rng = np.random.RandomState(seed)
    return rng.normal(size=(n, dim))


def test_init_state():
    c = ChainConsumer()
    assert c.chains == []
    assert c.config == {}
    assert c.config_truth == {}
    assert c._configured is False
    assert c._all_parameters == []
    assert ChainConsumer.__version__ == "0.31.1"


def test_add_chain_returns_self_and_default_parameter_names():
    c = ChainConsumer()
    result = c.add_chain(make_data())
    assert result is c
    assert len(c.chains) == 1
    assert c.chains[0].parameters == ["0", "1"]
    assert c.chains[0].name == "Chain 0"


def test_add_chain_with_parameters_and_name():
    c = ChainConsumer()
    c.add_chain(make_data(), parameters=["a", "b"], name="mine")
    assert c.chains[0].name == "mine"
    assert c.chains[0].parameters == ["a", "b"]
    assert c._all_parameters == ["a", "b"]


def test_default_names_increment():
    c = ChainConsumer()
    c.add_chain(make_data(seed=1)).add_chain(make_data(seed=2))
    assert c._all_names() == ["Chain 0", "Chain 1"]


def test_add_chain_none_asserts():
    c = ChainConsumer()
    with pytest.raises(AssertionError):
        c.add_chain(None)


def test_add_chain_dict():
    c = ChainConsumer()
    rng = np.random.RandomState(3)
    c.add_chain({"x": rng.normal(size=100), "y": rng.normal(size=100)})
    assert c.chains[0].parameters == ["x", "y"]
    assert c.chains[0].chain.shape == (100, 2)


def test_add_chain_dict_with_parameters_asserts():
    c = ChainConsumer()
    with pytest.raises(AssertionError):
        c.add_chain({"x": np.arange(10.0)}, parameters=["x"])


def test_add_chain_list_is_transposed():
    c = ChainConsumer()
    rng = np.random.RandomState(4)
    cols = [rng.normal(size=50).tolist(), rng.normal(size=50).tolist()]
    c.add_chain(cols, parameters=["a", "b"])
    assert c.chains[0].chain.shape == (50, 2)
    np.testing.assert_allclose(c.chains[0].chain[:, 0], cols[0])


def test_add_chain_dataframe_extracts_weight_and_posterior():
    c = ChainConsumer()
    rng = np.random.RandomState(5)
    df = pd.DataFrame(
        {
            "a": rng.normal(size=60),
            "b": rng.normal(size=60),
            "weight": np.ones(60) * 2.0,
            "posterior": rng.normal(size=60),
        }
    )
    c.add_chain(df)
    chain = c.chains[0]
    assert chain.parameters == ["a", "b"]
    assert chain.chain.shape == (60, 2)
    assert chain.posterior is not None


def test_add_chain_dataframe_with_parameters_asserts():
    c = ChainConsumer()
    df = pd.DataFrame({"a": np.arange(5.0), "b": np.arange(5.0)})
    with pytest.raises(AssertionError):
        c.add_chain(df, parameters=["a", "b"])


def test_one_dimensional_chain_becomes_column():
    c = ChainConsumer()
    c.add_chain(np.random.RandomState(6).normal(size=100), parameters=["x"])
    assert c.chains[0].chain.shape == (100, 1)


def test_too_many_parameter_names_asserts():
    c = ChainConsumer()
    with pytest.raises(AssertionError):
        c.add_chain(make_data(dim=2), parameters=["a", "b", "c"])


def test_default_parameters_reused_and_dimension_checked():
    c = ChainConsumer()
    c.add_chain(make_data(seed=1), parameters=["a", "b"])
    c.add_chain(make_data(seed=2))
    assert c.chains[1].parameters == ["a", "b"]
    with pytest.raises(AssertionError):
        c.add_chain(make_data(dim=3))


def test_non_numeric_power_asserts():
    c = ChainConsumer()
    with pytest.raises(AssertionError):
        c.add_chain(make_data(), power="2")


def test_grid_with_multidimensional_weights_builds_meshgrid():
    c = ChainConsumer()
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([10.0, 11.0, 12.0])
    weights = np.random.RandomState(7).uniform(size=(3, 3))
    c.add_chain([x, y], parameters=["x", "y"], weights=weights, grid=True)
    chain = c.chains[0]
    assert chain.chain.shape == (9, 2)
    assert chain.weights.size == 9
    assert chain.grid


def test_grid_requires_weights_and_no_walkers():
    c = ChainConsumer()
    with pytest.raises(AssertionError):
        c.add_chain(make_data(), grid=True)
    with pytest.raises(AssertionError):
        c.add_chain(make_data(), grid=True, weights=np.ones(200), walkers=2)


def test_add_covariance_is_not_mcmc_chain():
    c = ChainConsumer()
    c.add_chain(make_data(), parameters=["a", "b"], name="real")
    result = c.add_covariance([1.0, 2.0], [[1.0, 0.0], [0.0, 4.0]], parameters=["a", "b"], name="cov")
    assert result is c
    assert len(c.chains) == 2
    assert c.chains[-1].mcmc_chain is False
    mcmc = c.get_mcmc_chains()
    assert len(mcmc) == 1
    assert mcmc[0].name == "real"


def test_add_marker():
    c = ChainConsumer()
    result = c.add_marker([1.0, 2.0], parameters=["a", "b"], name="marker")
    assert result is c
    chain = c.chains[0]
    assert chain.mcmc_chain is False
    assert chain.chain.shape == (2, 2)
    assert chain.name == "marker"
    assert c.get_mcmc_chains() == []


def test_remove_chain_default_name_and_index():
    c = ChainConsumer()
    c.add_chain(make_data(seed=1), parameters=["a", "b"], name="one")
    c.add_chain(make_data(seed=2), parameters=["a", "b"], name="two")
    c.add_chain(make_data(seed=3), parameters=["c", "d"], name="three")
    assert c.remove_chain() is c
    assert c._all_names() == ["one", "two"]
    assert c._all_parameters == ["a", "b"]
    c.remove_chain("one")
    assert c._all_names() == ["two"]
    c.remove_chain(0)
    assert c.chains == []
    assert c._all_parameters == []


def test_remove_chain_list_and_duplicates():
    c = ChainConsumer()
    for i in range(3):
        c.add_chain(make_data(seed=i), name="n%d" % i)
    with pytest.raises(AssertionError):
        c.remove_chain([0, "n0"])
    c.remove_chain([0, "n2"])
    assert c._all_names() == ["n1"]


def test_get_chain_errors():
    c = ChainConsumer()
    c.add_chain(make_data(), name="only")
    assert c._get_chain("only") == [0]
    assert c._get_chain(0) == [0]
    assert c._get_chain(c.chains[0]) == [0]
    with pytest.raises(AssertionError):
        c._get_chain("missing")
    with pytest.raises(AssertionError):
        c._get_chain(5)
    with pytest.raises(ValueError):
        c._get_chain(1.5)
    assert c._get_chain_name(0) == "only"


def test_divide_chain():
    c = ChainConsumer()
    c.add_chain(make_data(n=100), parameters=["a", "b"], walkers=2)
    con = c.divide_chain(0)
    assert isinstance(con, ChainConsumer)
    assert len(con.chains) == 2
    for chain in con.chains:
        assert chain.chain.shape == (50, 2)
        assert chain.parameters == ["a", "b"]


def test_divide_chain_without_walkers_asserts():
    c = ChainConsumer()
    c.add_chain(make_data())
    with pytest.raises(AssertionError):
        c.divide_chain(0)


def test_configure_basic_sets_config():
    c = ChainConsumer()
    c.add_chain(make_data(), parameters=["a", "b"])
    result = c.configure()
    assert result is c
    assert c._configured is True
    np.testing.assert_array_equal(c.config["sigmas"], [0, 1, 2])
    assert c.config["flip"] is True
    assert c.config["max_ticks"] == 5


def test_configure_invalid_inputs():
    c = ChainConsumer()
    c.add_chain(make_data())
    with pytest.raises(AssertionError):
        c.configure(statistics="not_a_stat")
    with pytest.raises(AssertionError):
        c.configure(summary_area=1.5)


def test_configure_truth_defaults():
    c = ChainConsumer()
    result = c.configure_truth()
    assert result is c
    assert c.config_truth["color"] == "#000000"
    assert c.config_truth["zorder"] == 100
    assert c.config_truth["ls"] == "--"
    c.configure_truth(color="#ff0000")
    assert c.config_truth["color"] == "#ff0000"
