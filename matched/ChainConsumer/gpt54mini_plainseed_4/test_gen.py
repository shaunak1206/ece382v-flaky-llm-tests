import numpy as np
import pytest

import chainconsumer.chainconsumer as cc_mod
from chainconsumer.chainconsumer import ChainConsumer


class DummyChain:
    def __init__(
        self,
        chain,
        parameters,
        name,
        weights=None,
        posterior=None,
        walkers=None,
        grid=False,
        num_free_params=None,
        num_eff_data_points=None,
        color=None,
        linewidth=None,
        linestyle=None,
        kde=None,
        shade_alpha=None,
        power=None,
        marker_style=None,
        marker_size=None,
        marker_alpha=None,
        plot_contour=None,
        plot_point=None,
        show_as_1d_prior=None,
        statistics=None,
        cloud=None,
        shade=None,
        shade_gradient=None,
        bar_shade=None,
        bins=None,
        smooth=None,
        color_params=None,
        plot_color_params=None,
        cmap=None,
        num_cloud=None,
        zorder=None,
        shift_params=None,
    ):
        self.chain = np.asarray(chain)
        self.parameters = list(parameters)
        self.name = name
        self.weights = np.ones(self.chain.shape[0]) if weights is None else np.asarray(weights)
        self.posterior = None if posterior is None else np.asarray(posterior)
        self.walkers = walkers
        self.grid = grid
        self.num_free_params = num_free_params
        self.num_eff_data_points = num_eff_data_points
        self.color = color
        self.linewidth = linewidth
        self.linestyle = linestyle
        self.kde = kde
        self.shade_alpha = shade_alpha
        self.power = power
        self.marker_style = marker_style
        self.marker_size = marker_size
        self.marker_alpha = marker_alpha
        self.plot_contour = plot_contour
        self.plot_point = plot_point
        self.show_as_1d_prior = show_as_1d_prior
        self.statistics = statistics
        self.cloud = cloud
        self.shade = shade
        self.shade_gradient = shade_gradient
        self.bar_shade = bar_shade
        self.bins = bins
        self.smooth = smooth
        self.color_params = color_params
        self.plot_color_params = plot_color_params
        self.cmap = cmap
        self.num_cloud = num_cloud
        self.zorder = zorder
        self.shift_params = shift_params
        self.mcmc_chain = True
        self.config = {}
        self.update_calls = []

    def update_unset_config(self, key, value, override=None):
        self.update_calls.append((key, value, override))
        self.config[key] = value


@pytest.fixture(autouse=True)
def patch_chain_class(monkeypatch):
    monkeypatch.setattr(cc_mod, "Chain", DummyChain)


def test_init_sets_expected_defaults():
    consumer = ChainConsumer()
    assert consumer.chains == []
    assert consumer._all_parameters == []
    assert consumer._default_parameters is None
    assert consumer._configured is False
    assert consumer._configured_truth is False
    assert consumer._num_configure_calls == 0
    assert "viridis" in consumer._cmaps
    assert consumer.plotter is not None
    assert consumer.diagnostic is not None
    assert consumer.comparison is not None
    assert consumer.analysis is not None


def test_add_chain_from_ndarray_sets_default_parameters_and_appends_chain():
    consumer = ChainConsumer()
    consumer.add_chain(np.array([[1.0, 2.0], [3.0, 4.0]]), parameters=["x", "y"], name="c1")
    assert len(consumer.chains) == 1
    assert consumer._default_parameters is None  # reset by _init_params after adding
    assert consumer._all_parameters == []
    chain = consumer.chains[0]
    assert chain.name == "c1"
    assert chain.parameters == ["x", "y"]
    assert chain.chain.shape == (2, 2)


def test_add_chain_from_dict_requires_no_parameters():
    consumer = ChainConsumer()
    consumer.add_chain({"a": [1, 2], "b": [3, 4]})
    chain = consumer.chains[0]
    assert chain.parameters == ["a", "b"]
    assert chain.chain.shape == (2, 2)


def test_add_chain_from_dataframe_uses_columns_and_extracts_weight_and_posterior():
    consumer = ChainConsumer()
    df = __import__("pandas").DataFrame(
        {"x": [1, 2], "y": [3, 4], "weight": [0.1, 0.2], "posterior": [5.0, 6.0]}
    )
    consumer.add_chain(df)
    chain = consumer.chains[0]
    assert chain.parameters == ["x", "y"]
    assert np.allclose(chain.weights, [0.1, 0.2])
    assert np.allclose(chain.posterior, [5.0, 6.0])


def test_add_chain_from_list_is_transposed():
    consumer = ChainConsumer()
    consumer.add_chain([[1, 2, 3], [4, 5, 6]], parameters=["a", "b"])
    chain = consumer.chains[0]
    assert chain.chain.shape == (3, 2)
    assert np.array_equal(chain.chain[:, 0], [1, 2, 3])
    assert np.array_equal(chain.chain[:, 1], [4, 5, 6])


def test_add_chain_with_numpy_filename_loads_file(tmp_path):
    consumer = ChainConsumer()
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    path = tmp_path / "chain.npy"
    np.save(path, arr)
    consumer.add_chain(str(path), parameters=["x", "y"])
    assert len(consumer.chains) == 1
    assert np.array_equal(consumer.chains[0].chain, arr)


def test_add_chain_rejects_none_chain():
    consumer = ChainConsumer()
    with pytest.raises(AssertionError):
        consumer.add_chain(None)


def test_add_chain_grid_flattens_mesh_and_disables_kde():
    consumer = ChainConsumer()
    chain = np.array([[0, 1], [10, 20]])
    weights = np.array([[1, 2], [3, 4]])
    consumer.add_chain(chain, parameters=["x", "y"], grid=True, weights=weights, kde=True)
    c = consumer.chains[0]
    assert c.grid is True
    assert c.kde is None
    assert c.chain.shape == (4, 2)
    assert np.array_equal(c.weights, np.array([1, 2, 3, 4]))


def test_add_marker_creates_non_mcmc_chain():
    consumer = ChainConsumer()
    consumer.add_marker([1.5, 2.5], parameters=["x", "y"], name="pt", color="#123456")
    c = consumer.chains[0]
    assert c.mcmc_chain is False
    assert c.plot_point is True
    assert c.plot_contour is False
    assert np.array_equal(c.posterior, np.array([0, 1]))


def test_add_covariance_uses_multivariate_normal_and_marks_chain_non_mcmc(monkeypatch):
    consumer = ChainConsumer()

    sample = np.array([[1.0, 2.0], [1.5, 2.5]])

    def fake_mvn(mean, cov, size):
        assert size == 1000000
        return sample

    monkeypatch.setattr(cc_mod.np.random, "multivariate_normal", fake_mvn)
    consumer.add_covariance([0, 0], np.eye(2), parameters=["x", "y"], name="cov")
    c = consumer.chains[0]
    assert np.array_equal(c.chain, sample)
    assert c.mcmc_chain is False


def test_get_mcmc_chains_filters_non_mcmc_chains():
    consumer = ChainConsumer()
    consumer.add_chain([[1, 2], [3, 4]], parameters=["x", "y"], name="mcmc")
    consumer.add_marker([0, 0], parameters=["x", "y"], name="marker")
    mcmc = consumer.get_mcmc_chains()
    assert len(mcmc) == 1
    assert mcmc[0].name == "mcmc"


def test_get_chain_accepts_name_index_and_object():
    consumer = ChainConsumer()
    consumer.add_chain([[1, 2], [3, 4]], parameters=["x", "y"], name="a")
    consumer.add_chain([[5, 6], [7, 8]], parameters=["x", "y"], name="b")
    first = consumer.chains[0]
    assert consumer._get_chain(0) == [0]
    assert consumer._get_chain("b") == [1]
    assert consumer._get_chain(first) == [0]


def test_remove_chain_by_name_and_index_updates_chain_list():
    consumer = ChainConsumer()
    consumer.add_chain([[1, 2]], parameters=["x", "y"], name="a")
    consumer.add_chain([[3, 4]], parameters=["x", "y"], name="b")
    consumer.remove_chain("a")
    assert [c.name for c in consumer.chains] == ["b"]
    consumer.remove_chain(0)
    assert consumer.chains == []


def test_remove_chain_raises_when_removing_duplicate_selection():
    consumer = ChainConsumer()
    consumer.add_chain([[1, 2]], parameters=["x", "y"], name="a")
    with pytest.raises(AssertionError):
        consumer.remove_chain(["a", 0])


def test_configure_sets_defaults_and_updates_chain_config():
    consumer = ChainConsumer()
    consumer.add_chain(np.random.randn(20, 2), parameters=["x", "y"], name="a")
    consumer.configure(statistics="max", bins=5, colors="#ff0000", global_point=True)
    c = consumer.chains[0]
    assert consumer.config["sigma2d"] is False
    assert consumer.config["plot_hists"] is True
    assert consumer.config["global_point"] is True
    assert consumer.config["legend_kwargs"]["frameon"] is False
    assert c.config["statistics"] == "max"
    assert c.config["bins"] == 5
    assert c.config["color"] == "#ff0000"


def test_configure_rejects_cmap_and_colors_together():
    consumer = ChainConsumer()
    consumer.add_chain(np.random.randn(10, 2), parameters=["x", "y"])
    with pytest.raises(AssertionError):
        consumer.configure(statistics="max", cmap="viridis", colors=["#000000"])


def test_configure_truth_defaults_and_overrides():
    consumer = ChainConsumer()
    consumer.configure_truth(color="#123456", linewidth=2)
    assert consumer.config_truth["ls"] == "--"
    assert consumer.config_truth["linewidth"] == 2
    assert consumer.config_truth["color"] == "#123456"
    assert consumer.config_truth["zorder"] == 100


def test_divide_chain_splits_walkers_into_new_consumer():
    consumer = ChainConsumer()
    data = np.arange(16).reshape(8, 2)
    weights = np.arange(8)
    consumer.add_chain(data, parameters=["x", "y"], name="walks", walkers=4, weights=weights)
    divided = consumer.divide_chain(0)
    assert isinstance(divided, ChainConsumer)
    assert len(divided.chains) == 4
    for i, c in enumerate(divided.chains):
        assert c.name == f"Chain {i}"
        assert c.chain.shape == (2, 2)
        assert c.parameters == ["x", "y"]


def test_deprecated_plot_wrapper_delegates_to_plotter(monkeypatch):
    consumer = ChainConsumer()
    called = {}

    def fake_plot(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr(consumer.plotter, "plot", fake_plot)
    assert consumer.plot(1, foo="bar") == "ok"
    assert called["args"] == (1,)
    assert called["kwargs"] == {"foo": "bar"}
