import numpy as np
import pandas as pd
import pytest
from chainconsumer.chainconsumer import ChainConsumer


def test_initial_state():
    cc = ChainConsumer()
    assert cc.__version__ == "0.31.1"
    assert isinstance(cc.chains, list)
    assert cc._configured is False
    # Check default parameters is None initially
    assert cc._default_parameters is None
    # Plotter, diagnostic, comparison, analysis assigned
    assert hasattr(cc, "plotter")
    assert hasattr(cc, "diagnostic")
    assert hasattr(cc, "comparison")
    assert hasattr(cc, "analysis")


def test_add_chain_numpy_array_sets_defaults():
    cc = ChainConsumer()
    data = np.random.randn(100, 2)
    cc.add_chain(data)
    assert len(cc.chains) == 1
    # Default parameters set from first add_chain
    assert cc._default_parameters is not None
    assert all(p in cc._all_parameters for p in cc.chains[0].parameters)


def test_add_chain_dict_and_parameters_none():
    cc = ChainConsumer()
    d = {"a": np.random.randn(10), "b": np.random.randn(10)}
    cc.add_chain(d)
    # parameters assigned from dict keys
    assert "a" in cc.chains[0].parameters
    assert "b" in cc.chains[0].parameters


def test_add_chain_invalid_dict_and_parameters_given_raises():
    cc = ChainConsumer()
    d = {"a": np.random.randn(10), "b": np.random.randn(10)}
    with pytest.raises(AssertionError):
        cc.add_chain(d, parameters=["a", "b"])  # parameters must be None when dict


def test_add_chain_list_transpose_to_ndarray():
    cc = ChainConsumer()
    lst = [np.arange(5), np.arange(5) + 10]
    cc.add_chain(lst, parameters=["x", "y"])
    assert cc.chains[0].chain.shape == (5, 2)
    assert "x" in cc.chains[0].parameters
    assert "y" in cc.chains[0].parameters


def test_add_chain_str_loads_csv(tmp_path):
    # Write example csv file
    df = pd.DataFrame({"x": np.arange(5), "y": np.arange(5) * 2})
    file = tmp_path / "chain.csv"
    df.to_csv(file, index=False)

    cc = ChainConsumer()
    cc.add_chain(str(file))
    assert len(cc.chains) == 1
    # parameters come from dataframe columns excluding weight/posterior
    for p in ["x", "y"]:
        assert p in cc.chains[0].parameters


def test_add_chain_str_loads_npy(tmp_path):
    filename = tmp_path / "chain.npy"
    data = np.random.rand(10, 2)
    np.save(filename, data)
    cc = ChainConsumer()
    cc.add_chain(str(filename))
    assert len(cc.chains) == 1
    assert np.allclose(cc.chains[0].chain, data)


def test_add_chain_with_grid_and_weights():
    cc = ChainConsumer()
    # grid chain: 2D parameters, weights provided
    data = np.array([[0, 1], [2, 3], [4, 5]])
    weights = np.array([0.1, 0.2, 0.3])
    cc.add_chain(data, weights=weights, grid=True)
    c = cc.chains[0]
    # KDE is disabled for grid chains
    assert c.kde is None
    assert np.allclose(c.weights, weights)


def test_add_covariance_creates_chain_with_shape_and_flag():
    cc = ChainConsumer()
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]
    cc.add_covariance(mean, cov)
    assert len(cc.chains) == 1
    chain = cc.chains[0]
    assert chain.chain.shape[1] == 2
    assert chain.mcmc_chain is False  # flag set in add_covariance


def test_add_marker_creates_chain_with_posterior_flag_and_plot_settings():
    cc = ChainConsumer()
    loc = [1.0, 2.0]
    cc.add_marker(loc, parameters=["p1", "p2"], name="marker")
    chain = cc.chains[-1]
    assert chain.name == "marker"
    assert chain.chain.shape[1] == 2
    assert np.array_equal(chain.posterior, np.array([0, 1]))
    assert chain.plot_point is True
    assert chain.plot_contour is False
    assert chain.mcmc_chain is False


def test_remove_chain_by_index_and_name():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10, 2), name="a")
    cc.add_chain(np.random.randn(10, 2), name="b")
    cc.add_chain(np.random.randn(10, 2), name="c")
    # Remove by name
    cc.remove_chain("b")
    names = [c.name for c in cc.chains]
    assert "b" not in names
    # Remove by index
    cc.remove_chain(0)
    names = [c.name for c in cc.chains]
    assert "a" not in names
    # Remove last remaining by default (chain=-1)
    cc.remove_chain()
    assert len(cc.chains) == 0


def test_remove_chain_raises_error_if_duplicate_remove():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10, 2), name="a")
    with pytest.raises(AssertionError):
        # Attempt to remove the same chain twice
        cc.remove_chain(["a", "a"])


def test_divide_chain_splits_into_walkers():
    cc = ChainConsumer()
    chain_data = np.vstack([np.arange(12), np.arange(12) + 1]).T  # shape (12, 2)
    # Add chain with 3 walkers (so divide into 3 splits of 4 rows)
    cc.add_chain(chain_data, parameters=["p1", "p2"], walkers=3)
    # Manually mark last chain MCMC for test
    cc.chains[-1].mcmc_chain = True
    con_new = cc.divide_chain(0)
    assert isinstance(con_new, ChainConsumer)
    # Should have 3 chains in new consumer (one per walker)
    assert len(con_new.chains) == 3
    for c in con_new.chains:
        assert c.chain.shape[0] == 4  # each walker has 4 samples
        assert c.chain.shape[1] == 2


def test_get_mcmc_chains_filters_correctly():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10, 1))
    cc.add_chain(np.random.randn(5, 1))
    # Mark second chain as non-MCMC
    cc.chains[1].mcmc_chain = False
    mcmc_chains = cc.get_mcmc_chains()
    assert len(mcmc_chains) == 1
    assert mcmc_chains[0] == cc.chains[0]


def test_configure_sets_chain_attributes_and_config_dict():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10, 2))
    cc.configure(statistics="max", colors="#FF0000", linewidths=2.3, shade_alpha=0.5)
    c = cc.chains[0]
    assert c.config["statistics"] == "max"
    assert c.config["color"].lower() == "#ff0000"
    assert c.config["linewidth"] == 2.3
    # shade_alpha should be set based on inputs (list with one element)
    assert isinstance(c.config["shade_alpha"], float)
    # Config dictionary on ChainConsumer updated
    assert cc.config["shade_alpha"] is None or "shade_alpha" not in cc.config


def test_configure_multiple_times_warns(caplog):
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10, 1))
    with caplog.at_level("WARNING", logger="chainconsumer"):
        cc.configure()
        cc.configure()
        assert any("Configure has been called 2 times" in m for m in caplog.messages)


def test_configure_invalid_bins_value_raises():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10, 1))
    with pytest.raises(ValueError):
        cc.configure(bins="invalid")  # bins must be None, int, float, or list


def test_configure_invalid_statistics_raises():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10, 1))
    with pytest.raises(AssertionError):
        cc.configure(statistics="not_a_stat")


def test_configure_invalid_contour_labels_raises():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10, 1))
    with pytest.raises(AssertionError):
        cc.configure(contour_labels="invalid_label")


def test_configure_invalid_legend_kwargs_type_raises():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10, 1))
    with pytest.raises(AssertionError):
        cc.configure(legend_kwargs="notadict")


def test_configure_invalid_summary_area_type_raises():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10, 1))
    with pytest.raises(AssertionError):
        cc.configure(summary_area="notafloat")


def test_configure_summary_area_bounds_raises():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10, 1))
    with pytest.raises(AssertionError):
        cc.configure(summary_area=0)  # zero, must be positive
    with pytest.raises(AssertionError):
        cc.configure(summary_area=1.5)  # must be less than 1


def test_configure_global_point_type_raises():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10, 1))
    with pytest.raises(AssertionError):
        cc.configure(global_point="notbool")


def test_configure_color_params_assertion():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10, 2), parameters=["a", "b"])
    # color_params must be in parameters for given chains
    with pytest.raises(AssertionError):
        cc.configure(color_params="c")


def test__get_chain_by_name_and_index():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10, 1), name="chain1")
    cc.add_chain(np.random.randn(10, 1), name="chain2")

    idx = cc._get_chain("chain1")
    assert idx == [0]
    idx = cc._get_chain(1)
    assert idx == [1]

    with pytest.raises(AssertionError):
        cc._get_chain("not_a_chain")

    with pytest.raises(AssertionError):
        cc._get_chain(10)

    with pytest.raises(ValueError):
        cc._get_chain(1.5)  # invalid type


def test__get_chain_name_returns_correct_name():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10, 1), name="abc")
    assert cc._get_chain_name(0) == "abc"


def test__all_names_returns_all_chain_names():
    cc = ChainConsumer()
    cc.add_chain(np.random.randn(10, 1), name="one")
    cc.add_chain(np.random.randn(10, 1), name="two")
    names = cc._all_names()
    assert names == ["one", "two"]

