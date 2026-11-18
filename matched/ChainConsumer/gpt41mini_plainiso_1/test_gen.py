import io
import numpy as np
import pandas as pd
import pytest
from chainconsumer.chainconsumer import ChainConsumer


def test_init_defaults():
    cc = ChainConsumer()
    assert cc._gauss_mode == "reflect"
    assert not cc._configured
    assert cc._num_configure_calls == 0
    # Should have empty chains list
    assert isinstance(cc.chains, list)
    assert len(cc.chains) == 0


def test_add_chain_with_numpy_array():
    cc = ChainConsumer()
    data = np.random.rand(100, 3)
    cc.add_chain(data, parameters=["a", "b", "c"], name="testchain")
    assert len(cc.chains) == 1
    chain = cc.chains[0]
    assert chain.parameters == ["a", "b", "c"]
    assert chain.name == "testchain"
    assert np.allclose(chain.chain, data)


def test_add_chain_with_dict():
    cc = ChainConsumer()
    data = {"a": np.arange(10), "b": np.arange(10) * 2}
    cc.add_chain(data)
    assert len(cc.chains) == 1
    chain = cc.chains[0]
    # Parameters should be keys of dict in insertion order (Py3.8+ guarantees insertion order)
    assert chain.parameters == ["a", "b"]
    assert chain.chain.shape == (10, 2)


def test_add_chain_with_list_transpose():
    cc = ChainConsumer()
    data = [[1, 2, 3], [4, 5, 6]]
    cc.add_chain(data, parameters=["a", "b", "c"])
    assert cc.chains[0].chain.shape == (2, 3)
    # Parameters correspond to 3 columns
    assert cc.chains[0].parameters == ["a", "b", "c"]


def test_add_chain_with_dataframe_and_weight_and_posterior():
    df = pd.DataFrame(
        {
            "a": np.arange(5),
            "b": np.arange(5) * 2,
            "weight": np.ones(5) * 0.5,
            "posterior": np.arange(5) * -1,
        }
    )
    cc = ChainConsumer()
    cc.add_chain(df)
    chain = cc.chains[0]
    # Parameters excludes weight and posterior
    assert "weight" not in chain.parameters
    assert "posterior" not in chain.parameters
    # weights and posterior should be correctly assigned
    assert np.allclose(chain.weights, 0.5 * np.ones(5))
    assert np.allclose(chain.posterior, np.arange(5) * -1)


def test_add_chain_with_grid_requires_weights_and_no_walkers():
    cc = ChainConsumer()
    data = np.array([[1, 2, 3], [4, 5, 6]])
    weights = np.array([0.1, 0.9])
    with pytest.raises(AssertionError):
        cc.add_chain(data, weights=weights, grid=True, walkers=10)
    with pytest.raises(AssertionError):
        cc.add_chain(data, grid=True, walkers=None)

    # But works if weights provided and walkers is None
    cc.add_chain(data, weights=weights, grid=True)
    assert len(cc.chains) == 1


def test_add_chain_with_power_type_check():
    cc = ChainConsumer()
    data = np.random.rand(10, 2)
    with pytest.raises(AssertionError):
        cc.add_chain(data, power="notanumber")
    cc.add_chain(data, power=2)
    cc.add_chain(data, power=3.14)


def test_add_chain_default_name_and_parameter_setting():
    cc = ChainConsumer()
    data1 = np.random.rand(10, 2)
    cc.add_chain(data1, parameters=["x", "y"])
    data2 = np.random.rand(10, 2)
    cc.add_chain(data2)
    # Default parameters once set should be reused
    assert cc.chains[1].parameters == ["x", "y"]
    assert cc.chains[0].name == "Chain 0"
    assert cc.chains[1].name == "Chain 1"


def test_add_chain_shift_params_warning(caplog):
    cc = ChainConsumer()
    data = np.random.rand(10, 2)
    with caplog.at_level("WARNING"):
        cc.add_chain(data, parameters=["x", "y"], shift_params={"z": 1.0})
    assert "Warning, shift parameter" in caplog.text


def test_add_covariance_adds_chain_with_generated_samples():
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]
    cc = ChainConsumer()
    cc.add_covariance(mean, cov, parameters=["a", "b"], name="covchain")
    assert len(cc.chains) == 1
    ch = cc.chains[0]
    assert ch.name == "covchain"
    assert ch.mcmc_chain is False
    assert ch.chain.shape[1] == 2


def test_add_marker_adds_chain_with_2_points_and_posterior():
    cc = ChainConsumer()
    loc = [1, 2]
    cc.add_marker(loc, parameters=["x", "y"], name="mymarker")
    chain = cc.chains[-1]
    assert chain.name == "mymarker"
    assert chain.chain.shape == (2, 2)
    assert np.allclose(chain.chain[0], loc)
    assert chain.posterior is not None
    assert chain.mcmc_chain is False


def test_remove_chain_by_index_and_name():
    cc = ChainConsumer()
    cc.add_chain(np.random.rand(5, 2), parameters=["a", "b"], name="first")
    cc.add_chain(np.random.rand(5, 2), parameters=["a", "b"], name="second")
    # Remove by index
    cc.remove_chain(0)
    assert len(cc.chains) == 1
    assert cc.chains[0].name == "second"
    cc.remove_chain("second")
    assert len(cc.chains) == 0


def test_remove_chain_with_duplicates_raises():
    cc = ChainConsumer()
    cc.add_chain(np.random.rand(5, 1), parameters=["a"], name="foo")
    with pytest.raises(AssertionError):
        cc.remove_chain(["foo", "foo"])


def test_configure_basic_flow_and_defaults():
    cc = ChainConsumer()
    cc.add_chain(np.random.rand(100, 2), parameters=["a", "b"])
    cc.configure()
    assert cc._configured
    # _num_configure_calls increments
    assert cc._num_configure_calls == 1
    # Config keys set
    assert "sigma2d" in cc.config
    assert "sigmas" in cc.config
    assert "summary_area" in cc.chains[0].config


def test_configure_multiple_calls_logs_warning(caplog):
    cc = ChainConsumer()
    cc.add_chain(np.random.rand(100, 1), parameters=["a"])
    cc.configure()
    with caplog.at_level("WARNING"):
        cc.configure()
    assert "Configure has been called 2 times" in caplog.text


def test_configure_type_error_on_invalid_statistics():
    cc = ChainConsumer()
    cc.add_chain(np.random.rand(100, 1), parameters=["a"])
    with pytest.raises(AssertionError):
        cc.configure(statistics="invalid_stat")


def test_configure_bins_variants():
    cc = ChainConsumer()
    cc.add_chain(np.random.rand(100, 2), parameters=["a", "b"])
    # bins as integer
    cc.configure(bins=5)
    # bins as float
    cc.configure(bins=1.5)
    # bins as list
    cc.configure(bins=[1.5, 2.0])
    # bins invalid type
    with pytest.raises(ValueError):
        cc.configure(bins="badtype")


def test_configure_smooth_variants():
    cc = ChainConsumer()
    cc.add_chain(np.random.rand(100, 2), parameters=["a", "b"])
    cc.configure(smooth=None)
    cc.configure(smooth=0)
    cc.configure(smooth=[1, 2])
    cc.configure(smooth=3)


def test_configure_color_params_and_cmaps():
    cc = ChainConsumer()
    cc.add_chain(np.random.rand(10, 2), parameters=["a", "b"])
    # color_params string for param present
    cc.configure(color_params="a")
    # color_params string not present -> None
    cc.configure(color_params="notpresent")
    # color_params list with valid parameters
    cc.configure(color_params=["a"])
    # Invalid color_param should raise
    with pytest.raises(AssertionError):
        cc.configure(color_params=["invalid_param"])


def test_configure_legend_kwargs_type_check():
    cc = ChainConsumer()
    cc.add_chain(np.random.rand(20, 1), parameters=["a"])
    with pytest.raises(AssertionError):
        cc.configure(legend_kwargs="notadict")
    # Proper dict should succeed
    cc.configure(legend_kwargs={"loc": "lower left"})


def test_configure_contour_labels_validation():
    cc = ChainConsumer()
    cc.add_chain(np.random.rand(20, 1), parameters=["a"])
    with pytest.raises(AssertionError):
        cc.configure(contour_labels=123)
    with pytest.raises(AssertionError):
        cc.configure(contour_labels="badstring")
    # valid ones succeed
    cc.configure(contour_labels="sigma")
    cc.configure(contour_labels="confidence")


def test_configure_summary_area_validation():
    cc = ChainConsumer()
    cc.add_chain(np.random.rand(20, 1), parameters=["a"])
    with pytest.raises(AssertionError):
        cc.configure(summary_area="string")
    with pytest.raises(AssertionError):
        cc.configure(summary_area=-0.1)
    with pytest.raises(AssertionError):
        cc.configure(summary_area=2.0)
    # valid value
    cc.configure(summary_area=0.9)


def test_configure_marker_lists_and_scalars():
    cc = ChainConsumer()
    cc.add_chain(np.random.rand(20, 2), parameters=["a", "b"])
    # marker style string
    cc.configure(marker_style="x")
    # marker_size int
    cc.configure(marker_size=10)
    # marker_alpha float
    cc.configure(marker_alpha=0.3)
    # marker style list length matched to chains
    cc.configure(marker_style=["x"])
    cc.configure(marker_size=[10])
    cc.configure(marker_alpha=[0.1])


def test_configure_cloud_and_num_cloud_cases():
    cc = ChainConsumer()
    cc.add_chain(np.random.rand(20, 2), parameters=["a", "b"])
    cc.configure(cloud=True)
    cc.configure(num_cloud=10000)
    cc.configure(num_cloud=[10000])
    with pytest.raises(AssertionError):
        cc.configure(num_cloud=[10000, 20000])


def test_get_mcmc_chains_returns_only_mcmc():
    cc = ChainConsumer()
    d1 = np.random.rand(10, 1)
    d2 = np.random.rand(10, 1)
    cc.add_chain(d1, parameters=["a"])
    cc.add_chain(d2, parameters=["a"])
    cc.chains[0].mcmc_chain = True
    cc.chains[1].mcmc_chain = False
    mcmc_chains = cc.get_mcmc_chains()
    assert len(mcmc_chains) == 1
    assert mcmc_chains[0] == cc.chains[0]


def test__get_chain_string_and_int_and_instance():
    cc = ChainConsumer()
    cc.add_chain(np.random.rand(5, 1), parameters=["a"], name="foo")
    # By name
    indices = cc._get_chain("foo")
    assert indices == [0]
    # By int
    indices = cc._get_chain(0)
    assert indices == [0]
    # By Chain object instance
    cobj = cc.chains[0]
    indices = cc._get_chain(cobj)
    assert indices == [0]
    # Invalid name
    with pytest.raises(AssertionError):
        cc._get_chain("bar")
    # Invalid int
    with pytest.raises(AssertionError):
        cc._get_chain(10)
    # Invalid type
    with pytest.raises(ValueError):
        cc._get_chain(3.14)


def test_divide_chain_splits_equally():
    cc = ChainConsumer()
    # Create a chain with 4 walkers, 8 samples each
    nwalkers = 4
    nsamples = 8
    data = np.arange(nwalkers * nsamples * 2).reshape(nwalkers * nsamples, 2)
    weights = np.ones(nwalkers * nsamples)
    cc.add_chain(data, parameters=["x", "y"], walkers=nwalkers, weights=weights)
    new_cc = cc.divide_chain(0)
    # Should have 4 chains now
    assert len(new_cc.chains) == nwalkers
    for ch in new_cc.chains:
        # Each divided chain has nsamples points
        assert ch.chain.shape[0] == nsamples


def test_configure_truth_defaults_and_overrides():
    cc = ChainConsumer()
    ret = cc.configure_truth()
    assert cc._configured_truth
    cfg = cc.config_truth
    # Defaults set for linestyle, linewidth, color, zorder
    assert cfg["ls"] == "--"
    assert cfg["linewidth"] == 1
    assert cfg["color"] == "#000000"
    assert cfg["zorder"] == 100
    # Override some kwargs
    cc.configure_truth(color="red", linestyle="-", zorder=5)
    cfg2 = cc.config_truth
    assert cfg2["color"] == "red"
    assert cfg2["ls"] == "-"
    assert cfg2["zorder"] == 5


def test_deprecated_methods_print_and_forward(monkeypatch):
    cc = ChainConsumer()
    # Patch underlying methods to mark calls
    called = {}

    def dummy(*args, **kwargs):
        called["called"] = True
        return 123

    monkeypatch.setattr(cc.plotter, "plot", dummy)
    monkeypatch.setattr(cc.plotter, "plot_walks", dummy)
    monkeypatch.setattr(cc.analysis, "get_latex_table", dummy)
    monkeypatch.setattr(cc.analysis, "get_parameter_text", dummy)
    monkeypatch.setattr(cc.analysis, "get_summary", dummy)
    monkeypatch.setattr(cc.analysis, "get_correlations", dummy)
    monkeypatch.setattr(cc.analysis, "get_correlation_table", dummy)
    monkeypatch.setattr(cc.analysis, "get_covariance", dummy)
    monkeypatch.setattr(cc.analysis, "get_covariance_table", dummy)
    monkeypatch.setattr(cc.diagnostic, "gelman_rubin", dummy)
    monkeypatch.setattr(cc.diagnostic, "geweke", dummy)
    monkeypatch.setattr(cc.comparison, "aic", dummy)
    monkeypatch.setattr(cc.comparison, "bic", dummy)
    monkeypatch.setattr(cc.comparison, "dic", dummy)
    monkeypatch.setattr(cc.comparison, "comparison_table", dummy)

    import builtins
    import sys

    output = []

    def fake_print(msg):
        output.append(msg)

    monkeypatch.setattr(builtins, "print", fake_print)

    assert cc.plot() == 123
    assert "deprecated" in output[-1]
    assert cc.plot_walks() == 123
    assert cc.get_latex_table() == 123
    assert cc.get_parameter_text() == 123
    assert cc.get_summary() == 123
    assert cc.get_correlations() == 123
    assert cc.get_correlation_table() == 123
    assert cc.get_covariance() == 123
    assert cc.get_covariance_table() == 123
    assert cc.diagnostic_gelman_rubin() == 123
    assert cc.diagnostic_geweke() == 123
    assert cc.comparison_aic() == 123
    assert cc.comparison_bic() == 123
    assert cc.comparison_dic() == 123
    assert cc.comparison_table() == 123


def test_add_chain_raises_assert_if_none_chain():
    cc = ChainConsumer()
    with pytest.raises(AssertionError):
        cc.add_chain(None)
