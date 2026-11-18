import pytest
import numpy as np
import pandas as pd
from chainconsumer.chainconsumer import ChainConsumer

def test_init_defaults():
    cc = ChainConsumer()
    # Basic attributes present
    assert hasattr(cc, "chains")
    assert isinstance(cc.chains, list)
    assert cc._configured is False
    assert cc._num_configure_calls == 0

def test_add_chain_basic_numpy():
    cc = ChainConsumer()
    data = np.random.randn(100, 2)
    cc.add_chain(data, parameters=["a", "b"], name="testchain")
    assert len(cc.chains) == 1
    chain = cc.chains[0]
    assert chain.name == "testchain"
    assert chain.parameters == ["a", "b"]
    assert chain.chain.shape[0] == 100
    assert chain.chain.shape[1] == 2

def test_add_chain_dict_and_default_params():
    cc = ChainConsumer()
    data = {"x": np.arange(50), "y": np.arange(50) * 2}
    cc.add_chain(data)
    assert len(cc.chains) == 1
    assert cc._default_parameters == ["x", "y"]
    # Adding another chain without parameters uses default
    data2 = np.random.randn(30, 2)
    cc.add_chain(data2)
    assert cc.chains[-1].parameters == ["x", "y"]

def test_add_chain_dataframe_with_weights_and_posterior():
    cc = ChainConsumer()
    df = pd.DataFrame({
        "x": np.random.randn(40),
        "y": np.random.randn(40),
        "weight": np.ones(40) * 2,
        "posterior": np.linspace(0, 1, 40),
    })
    cc.add_chain(df)
    assert "weight" not in cc.chains[-1].parameters
    assert "posterior" not in cc.chains[-1].parameters
    assert cc.chains[-1].weights is not None
    assert cc.chains[-1].posterior is not None

def test_add_chain_grid_mode_with_meshgrid(monkeypatch):
    cc = ChainConsumer()
    # grid requires weights, no walkers
    chain = [np.linspace(0,1,3), np.linspace(0,1,2)]
    weights = np.array([[0.1, 0.2, 0.3],[0.4,0.5,0.6]])
    weights = weights.T.flatten()
    chain_array = np.array(chain).T  # Shape incorrectly flattened on purpose for test
    # Provide flattened chain for non-dict grid
    flat_chain = np.linspace(0,1,6).reshape(6,1)
    weights = np.ones(6)

    # This should trigger the meshgrid branch
    # We create a 2D grid to test meshgrid as per code
    chain = np.array(chain)
    weights = np.ones(chain.shape[1]*chain.shape[0])
    # monkeypatch logger to avoid unnecessary logging output
    cc._logger = None
    cc.add_chain(chain, grid=True, weights=weights)
    assert len(cc.chains) == 1

def test_add_chain_invalid_power_type():
    cc = ChainConsumer()
    data = np.random.randn(10,1)
    with pytest.raises(AssertionError):
        cc.add_chain(data, power="invalid")

def test_add_chain_shift_params_warning(caplog):
    cc = ChainConsumer()
    data = np.random.randn(10,2)
    with caplog.at_level("WARNING"):
        cc.add_chain(data, parameters=["a","b"], shift_params={"c": 1.0})
        assert "Warning, shift parameter c is not in list of parameters ['a', 'b']" in caplog.text

def test_add_covariance_adds_chain_and_sets_mcmc_false():
    cc = ChainConsumer()
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    cc.add_covariance(mean, cov, parameters=["p1", "p2"], name="covariance_chain")
    assert len(cc.chains) == 1
    c = cc.chains[0]
    assert c.name == "covariance_chain"
    assert c.mcmc_chain is False

def test_add_marker_creates_marker_chain():
    cc = ChainConsumer()
    location = [1.0, 2.0]
    cc.add_marker(location, parameters=["x", "y"], name="marker_chain")
    c = cc.chains[-1]
    assert c.name == "marker_chain"
    assert c.chain.shape[0] == 2
    assert c.chain[0,0] == 1.0
    assert c.mcmc_chain is False

def test_remove_chain_by_index_and_name():
    cc = ChainConsumer()
    data = np.random.randn(20, 1)
    cc.add_chain(data, parameters=["a"], name="first")
    cc.add_chain(data, parameters=["a"], name="second")
    # remove by index
    cc.remove_chain(0)
    assert len(cc.chains) == 1
    # remove by name
    cc.remove_chain("second")
    assert len(cc.chains) == 0

def test_remove_chain_multiple_and_assert_error_on_duplicates():
    cc = ChainConsumer()
    data = np.random.randn(10, 1)
    cc.add_chain(data, parameters=["a"], name="one")
    cc.add_chain(data, parameters=["a"], name="two")
    with pytest.raises(AssertionError):
        cc.remove_chain(["one", "one"])

def test_configure_basic_sets_parameters_and_chains():
    cc = ChainConsumer()
    data = np.random.randn(50, 2)
    cc.add_chain(data, parameters=["a", "b"])
    result = cc.configure(statistics="max", shade_alpha=0.5)
    assert result is cc
    assert cc._configured is True
    # Check that shade_alpha is set correctly in chain config
    for c in cc.chains:
        assert hasattr(c.config, "__getitem__")
        assert "shade_alpha" in c.config

def test_configure_rejects_invalid_statistics():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10,1), parameters=["a"])
    with pytest.raises(AssertionError):
        cc.configure(statistics="invalid_stat")

def test_configure_statistics_list_lowercased():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10,1), parameters=["a"])
    cc.add_chain(np.random.randn(10,1), parameters=["a"])
    stats = ["MAX", "mean"]
    cc.configure(statistics=stats)
    for c, stat in zip(cc.chains, [s.lower() for s in stats]):
        assert c.config["statistics"] == stat

def test_configure_color_and_linestyle_lists():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10,1), parameters=["a"])
    cc.add_chain(np.random.randn(10,1), parameters=["a"])
    colors = ["#ff0000", "#00ff00"]
    linestyles = ["-", "--"]
    cc.configure(colors=colors, linestyles=linestyles)
    for i, c in enumerate(cc.chains):
        assert c.config["color"] == colors[i]
        assert c.config["linestyle"] == linestyles[i]

def test_configure_raises_for_invalid_bins_type():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(25,2), parameters=["x","y"])
    with pytest.raises(ValueError):
        cc.configure(bins={"dict": "not valid"})

def test_divide_chain_splits_walkers():
    cc = ChainConsumer()
    # Create dummy chain with 4 walkers (each 5 samples)
    data = np.arange(20).reshape(20,1)
    # Set walkers=4 in chain
    cc.add_chain(data, walkers=4, parameters=["a"])
    new_cc = cc.divide_chain()
    assert isinstance(new_cc, ChainConsumer)
    # Should be 4 chains, one per walker
    assert len(new_cc.chains) == 4
    for c in new_cc.chains:
        assert c.chain.shape[0] == 5

def test_get_mcmc_chains_filters_non_mcmc():
    cc = ChainConsumer()
    data = np.random.randn(10, 2)
    cc.add_chain(data, parameters=["a","b"], name="mcmcchain")
    # Add non mcmc chain
    cc.add_covariance([0,0], [[1,0],[0,1]], parameters=["a","b"], name="covchain")
    mcmc_chains = cc.get_mcmc_chains()
    assert all(c.mcmc_chain for c in mcmc_chains)
    assert any(c.name=="mcmcchain" for c in mcmc_chains)
    assert all(c.name!="covchain" for c in mcmc_chains)

def test_get_chain_name_and_all_names():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10,1), parameters=["a"], name="first")
    cc.add_chain(np.random.randn(10,1), parameters=["a"], name="second")
    assert cc._get_chain_name(0) == "first"
    assert cc._all_names() == ["first", "second"]

def test_get_chain_invalid_type_raises():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10,1), parameters=["a"], name="onlychain")
    with pytest.raises(ValueError):
        cc._get_chain(3.14)

def test_get_chain_string_not_found_raises():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10,1), parameters=["a"], name="onlychain")
    with pytest.raises(AssertionError):
        cc._get_chain("notfound")

def test_get_chain_index_out_of_range_raises():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10,1), parameters=["a"], name="onlychain")
    with pytest.raises(AssertionError):
        cc._get_chain(5)
