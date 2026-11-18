import io
import os
import numpy as np
import pandas as pd
import pytest
from chainconsumer.chainconsumer import ChainConsumer


def test_add_chain_with_numpy_array():
    cc = ChainConsumer()
    data = np.random.randn(100, 3)
    cc.add_chain(data, parameters=["a", "b", "c"], name="test_chain")
    assert len(cc.chains) == 1
    chain = cc.chains[0]
    assert chain.name == "test_chain"
    assert chain.chain.shape == (100, 3)
    assert set(chain.parameters) == {"a", "b", "c"}


def test_add_chain_with_dict_and_auto_parameters():
    cc = ChainConsumer()
    data = {"x": np.random.randn(50), "y": np.random.randn(50)}
    cc.add_chain(data, name="dict_chain")
    assert len(cc.chains) == 1
    chain = cc.chains[0]
    assert chain.name == "dict_chain"
    assert set(chain.parameters) == {"x", "y"}
    assert chain.chain.shape == (50, 2)


def test_add_chain_with_list_and_transpose():
    cc = ChainConsumer()
    data = [[1, 2, 3], [4, 5, 6]]
    cc.add_chain(data, parameters=["p1", "p2", "p3"], name="list_chain")
    assert len(cc.chains) == 1
    chain = cc.chains[0]
    assert chain.name == "list_chain"
    assert chain.chain.shape == (2, 3)
    assert chain.parameters == ["p1", "p2", "p3"]


def test_add_chain_with_dataframe_and_extract_weights_posterior(tmp_path):
    cc = ChainConsumer()
    df = pd.DataFrame({
        "a": np.random.randn(10),
        "b": np.random.randn(10),
        "weight": np.linspace(0.1, 1.0, 10),
        "posterior": np.linspace(-10, 0, 10),
    })
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)

    cc.add_chain(str(path))
    chain = cc.chains[0]
    assert "weight" not in chain.parameters
    assert "posterior" not in chain.parameters
    np.testing.assert_array_equal(chain.weights, df["weight"])
    np.testing.assert_array_equal(chain.posterior, df["posterior"])
    assert chain.chain.shape[0] == 10
    assert chain.chain.shape[1] == 2
    assert set(chain.parameters) == {"a", "b"}


def test_add_chain_with_grid_assertions():
    cc = ChainConsumer()
    values = np.linspace(0, 1, 5)
    chain = np.array([values, values * 2])
    # weights must be passed, walkers must be None
    weights = np.ones(5)

    # Wrong: walkers provided with grid -> assert error
    with pytest.raises(AssertionError):
        cc.add_chain(chain, grid=True, weights=weights, walkers=2)

    # Wrong: no weights with grid -> assert error
    with pytest.raises(AssertionError):
        cc.add_chain(chain, grid=True)

    # Good case: grid with weights no walkers
    cc.add_chain(chain, grid=True, weights=weights)
    assert len(cc.chains) == 1
    chain_obj = cc.chains[0]
    assert np.allclose(np.unique(chain_obj.chain[:, 0]), values)


def test_add_chain_with_invalid_power_type():
    cc = ChainConsumer()
    data = np.random.randn(10, 2)
    with pytest.raises(AssertionError):
        cc.add_chain(data, parameters=["x", "y"], power="not_numeric")


def test_remove_chain_by_index_and_name():
    cc = ChainConsumer()
    data1 = np.random.randn(10, 2)
    data2 = np.random.randn(10, 2)
    cc.add_chain(data1, parameters=["x", "y"], name="chain1")
    cc.add_chain(data2, parameters=["x", "y"], name="chain2")

    # Remove by index
    cc.remove_chain(0)
    assert len(cc.chains) == 1
    assert cc.chains[0].name == "chain2"
    assert "x" in cc._all_parameters

    # Remove by name
    cc.remove_chain("chain2")
    assert len(cc.chains) == 0
    assert cc._all_parameters == []


def test_remove_chain_with_duplicates_raises():
    cc = ChainConsumer()
    data = np.random.randn(10, 2)
    cc.add_chain(data, parameters=["a", "b"], name="c1")
    # Try to remove the same chain twice by name and index
    with pytest.raises(AssertionError):
        cc.remove_chain(["c1", 0])


def test_add_covariance_creates_samples_and_chain():
    cc = ChainConsumer()
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]
    cc.add_covariance(mean, cov, parameters=["x", "y"], name="cov_chain", color="#ff0000")
    assert len(cc.chains) == 1
    chain = cc.chains[0]
    assert chain.name == "cov_chain"
    assert not chain.mcmc_chain
    assert chain.chain.shape == (1000000, 2)


def test_add_marker_creates_non_mcmc_chain():
    cc = ChainConsumer()
    location = [1, 2, 3]
    cc.add_marker(location, parameters=["a", "b", "c"], name="marker")
    assert len(cc.chains) == 1
    chain = cc.chains[0]
    # marker chains are not mcmc chains
    assert chain.mcmc_chain is False
    # chain should have 2 identical rows (the marker)
    assert chain.chain.shape == (2, 3)
    assert np.all(chain.chain[0] == location)
    assert np.all(chain.chain[1] == location)


@pytest.mark.parametrize("statistics", ["max", ["mean", "max"], ["mean"]])
def test_configure_sets_correct_statistics(statistics):
    cc = ChainConsumer()
    data = np.random.randn(10, 2)
    cc.add_chain(data, parameters=["x", "y"])

    if isinstance(statistics, list):
        with pytest.raises(AssertionError):
            cc.configure(statistics=statistics)
    else:
        cc.configure(statistics=statistics)
        chain_stats = [c.config["statistics"] for c in cc.chains]
        assert all(isinstance(s, str) and s in ["max", "mean", "cumulative", "max_symmetric", "max_closest", "max_central"] for s in chain_stats)


def test_configure_with_invalid_statistics_raises():
    cc = ChainConsumer()
    data = np.random.randn(10, 2)
    cc.add_chain(data, parameters=["x", "y"])
    with pytest.raises(AssertionError):
        cc.configure(statistics="invalid_stat")


def test_configure_with_colors_and_cmap_mutually_exclusive():
    cc = ChainConsumer()
    data = np.random.randn(10, 2)
    cc.add_chain(data, parameters=["x", "y"])
    with pytest.raises(AssertionError):
        cc.configure(colors="#000000", cmap="viridis")


def test_get_mcmc_chains_filters_correctly():
    cc = ChainConsumer()
    data = np.random.randn(20, 2)
    cc.add_chain(data, parameters=["x", "y"])
    cc.add_covariance([0,0], [[1, 0],[0, 1]], parameters=["x", "y"])
    mcmc_chains = cc.get_mcmc_chains()
    assert len(mcmc_chains) == 1  # only the first added chain is MCMC


def test_divide_chain_creates_walks():
    cc = ChainConsumer()
    nwalkers = 4
    nsteps = 20
    data = np.vstack([np.linspace(0, 1, nsteps) + i for i in range(nwalkers)])
    data = data.T  # shape (nsteps, nwalkers)
    # Add chain with walkers, shape (nsteps*nwalkers, dim) - here dim=1
    chain_data = np.hstack([data.flatten()[:, None]])
    # Reshape chain_data to (nsteps*nwalkers, 1)
    chain_data = chain_data.reshape(-1, 1)
    cc.add_chain(chain_data, parameters=["p"], name="walker_chain", walkers=nwalkers)
    # When divide_chain called, it splits chain into walkers
    divided = cc.divide_chain("walker_chain")
    assert isinstance(divided, ChainConsumer)
    assert len(divided.chains) == nwalkers
    for c in divided.chains:
        assert c.chain.shape[0] == nsteps


def test_get_chain_by_index_and_name_and_object():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(5, 1), parameters=["p"], name="first")
    cc.add_chain(np.random.randn(5, 1), parameters=["p"], name="second")
    idx = cc._get_chain(1)
    assert idx == [1]
    idx = cc._get_chain("first")
    assert idx == [0]
    chain_obj = cc.chains[0]
    idx = cc._get_chain(chain_obj)
    assert idx == [0]
    with pytest.raises(AssertionError):
        cc._get_chain("nonexistent")
    with pytest.raises(AssertionError):
        cc._get_chain(10)
    with pytest.raises(ValueError):
        cc._get_chain(1.234)


def test_configure_calls_multiple_warning(caplog):
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10, 2), parameters=["x", "y"])

    cc.configure()
    with caplog.at_level("WARNING"):
        cc.configure()
    assert any("Configure has been called" in msg for msg in caplog.text.splitlines())


def test_configure_with_invalid_bins_raises():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10, 2), parameters=["x", "y"])
    with pytest.raises(ValueError):
        cc.configure(bins="invalid_type")


def test_configure_with_invalid_contour_labels_raises():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10, 2), parameters=["x", "y"])
    with pytest.raises(AssertionError):
        cc.configure(contour_labels="invalid_value")


def test_configure_with_invalid_legend_kwargs_raises():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10, 2), parameters=["x", "y"])
    with pytest.raises(AssertionError):
        cc.configure(legend_kwargs="not_a_dict")


def test_configure_summary_area_validations():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10, 2), parameters=["x", "y"])

    # summary_area must be float
    with pytest.raises(AssertionError):
        cc.configure(summary_area="not_float")

    # summary_area must be > 0
    with pytest.raises(AssertionError):
        cc.configure(summary_area=0)

    # summary_area must be < 1
    with pytest.raises(AssertionError):
        cc.configure(summary_area=2.0)


def test_configure_truth_defaults():
    cc = ChainConsumer()
    cc.configure_truth()
    config = cc.config_truth
    # Defaults: linestyle == '--', linewidth==1, color == '#000000', zorder == 100
    assert ("ls" in config or "linestyle" in config)
    assert config.get("linewidth", None) == 1
    assert config.get("color") == "#000000"
    assert config.get("zorder") == 100

def test_configure_truth_custom():
    cc = ChainConsumer()
    cc.configure_truth(color="#123456", linestyle="-.", zorder=10, linewidth=2)
    config = cc.config_truth
    assert config["color"] == "#123456"
    assert config["ls"] == "-."
    assert config["zorder"] == 10
    assert config["linewidth"] == 2
