import numpy as np
import pandas as pd
import pytest

import chainconsumer.chainconsumer as cc


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
        self.weights = None if weights is None else np.asarray(weights)
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
        self.config_updates = []

    def update_unset_config(self, key, value, override=None):
        self.config_updates.append((key, value, tuple(override) if override is not None else None))
        setattr(self, key, value)


@pytest.fixture
def consumer(monkeypatch):
    monkeypatch.setattr(cc, "Chain", DummyChain)
    return cc.ChainConsumer()


def test_init_sets_expected_defaults(consumer):
    assert consumer.chains == []
    assert consumer._all_parameters == []
    assert consumer._default_parameters is None
    assert consumer._configured is False
    assert consumer._num_configure_calls == 0
    assert consumer.config == {}
    assert consumer.config_truth == {}


def test_add_chain_from_numpy_sets_default_parameters_and_returns_self(consumer):
    chain = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = consumer.add_chain(chain, parameters=["x", "y"], name="c1")
    assert out is consumer
    assert len(consumer.chains) == 1
    assert consumer._default_parameters == ["x", "y"]
    assert consumer._all_parameters == ["x", "y"]
    assert consumer.chains[0].name == "c1"
    assert consumer.chains[0].parameters == ["x", "y"]


def test_add_chain_from_dict_uses_keys_as_parameters(consumer):
    chain = {"b": [2, 4, 6], "a": [1, 3, 5]}
    consumer.add_chain(chain, name="dict")
    c = consumer.chains[0]
    assert c.parameters == ["b", "a"]
    assert c.chain.shape == (3, 2)
    assert consumer._default_parameters == ["b", "a"]


def test_add_chain_from_dataframe_extracts_weight_and_posterior_columns(consumer):
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "weight": [0.1, 0.2],
            "posterior": [9.0, 8.0],
        }
    )
    consumer.add_chain(df, name="df")
    c = consumer.chains[0]
    assert c.parameters == ["x", "y"]
    assert np.allclose(c.weights, [0.1, 0.2])
    assert np.allclose(c.posterior, [9.0, 8.0])
    assert c.chain.shape == (2, 2)


def test_add_chain_with_grid_meshes_and_flattens_weights(consumer):
    chain = np.array([[0, 1], [10, 20]])
    weights = np.array([[0.1, 0.2], [0.3, 0.4]])
    consumer.add_chain(chain, parameters=["x", "y"], weights=weights, grid=True)
    c = consumer.chains[0]
    assert c.grid is True
    assert c.kde is None
    assert c.chain.shape == (4, 2)
    assert np.allclose(c.weights, [0.1, 0.2, 0.3, 0.4])


def test_add_chain_with_grid_and_dictionary_raises(monkeypatch, consumer):
    chain = {"x": [0, 1], "y": [10, 20]}
    weights = np.array([[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(AssertionError, match="dictionary"):
        consumer.add_chain(chain, weights=weights, grid=True)


def test_add_covariance_delegates_to_add_chain_and_marks_non_mcmc(monkeypatch, consumer):
    captured = {}

    def fake_multivariate_normal(mean, cov, size):
        captured["mean"] = mean
        captured["cov"] = cov
        captured["size"] = size
        return np.ones((size, len(mean)))

    monkeypatch.setattr(cc.np.random, "multivariate_normal", fake_multivariate_normal)
    consumer.add_covariance([1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]], parameters=["x", "y"], name="cov")
    assert captured["size"] == 1000000
    assert len(consumer.chains) == 1
    assert consumer.chains[0].mcmc_chain is False


def test_add_marker_creates_two_point_chain_and_marks_non_mcmc(consumer):
    consumer.add_marker([5.0, 6.0], parameters=["x", "y"], name="marker", color="#ff0000")
    c = consumer.chains[0]
    assert c.plot_point is True
    assert c.plot_contour is False
    assert c.mcmc_chain is False
    assert np.allclose(c.posterior, [0, 1])
    assert c.chain.shape == (2, 2)


def test_get_mcmc_chains_filters_non_mcmc(consumer):
    consumer.add_chain(np.array([[1.0], [2.0]]), parameters=["x"], name="m1")
    consumer.add_marker([3.0], parameters=["x"], name="m2")
    mcmc = consumer.get_mcmc_chains()
    assert len(mcmc) == 1
    assert mcmc[0].name == "m1"


def test_get_chain_by_index_name_and_object(consumer):
    consumer.add_chain(np.array([[1.0], [2.0]]), parameters=["x"], name="first")
    consumer.add_chain(np.array([[3.0], [4.0]]), parameters=["x"], name="second")
    assert consumer._get_chain(1) == [1]
    assert consumer._get_chain("first") == [0]
    assert consumer._get_chain(consumer.chains[0]) == [0]


def test_get_chain_invalid_inputs_raise(consumer):
    consumer.add_chain(np.array([[1.0], [2.0]]), parameters=["x"], name="first")
    with pytest.raises(AssertionError):
        consumer._get_chain(5)
    with pytest.raises(AssertionError):
        consumer._get_chain("missing")
    with pytest.raises(ValueError):
        consumer._get_chain(3.14)


def test_remove_chain_by_name_updates_remaining_chains_and_parameters(consumer):
    consumer.add_chain(np.array([[1.0], [2.0]]), parameters=["x"], name="a")
    consumer.add_chain(np.array([[3.0], [4.0]]), parameters=["y"], name="b")
    consumer.remove_chain("a")
    assert len(consumer.chains) == 1
    assert consumer.chains[0].name == "b"
    assert consumer._all_parameters == ["y"]
    assert consumer._configured is False


def test_remove_chain_duplicate_selection_raises(consumer):
    consumer.add_chain(np.array([[1.0], [2.0]]), parameters=["x"], name="a")
    with pytest.raises(AssertionError, match="more than once"):
        consumer.remove_chain(["a", 0])


def test_configure_sets_chain_configs_and_general_settings(consumer):
    consumer.add_chain(np.array([[1.0], [2.0]]), parameters=["x"], name="a")
    consumer.configure(
        statistics="max",
        colors=["#123456"],
        linestyles="-",
        linewidths=2,
        kde=False,
        bins=5,
        summary_area=0.5,
        global_point=False,
        legend_kwargs={"loc": "best"},
    )
    c = consumer.chains[0]
    assert c.statistics == "max"
    assert c.color == "#123456"
    assert c.linestyle == "-"
    assert c.linewidth == 2
    assert c.kde is False
    assert c.bins == 5
    assert consumer.config["summary_area"] == 0.5
    assert consumer.config["global_point"] is False
    assert consumer.config["legend_kwargs"]["loc"] == "best"


def test_configure_twice_logs_warning(monkeypatch, consumer, caplog):
    consumer.add_chain(np.array([[1.0], [2.0]]), parameters=["x"], name="a")
    consumer.configure(statistics="max")
    with caplog.at_level("WARNING", logger="chainconsumer"):
        consumer.configure(statistics="max")
    assert "Configure has been called 2 times" in caplog.text


def test_configure_truth_sets_defaults_and_preserves_explicit_values(consumer):
    consumer.configure_truth(color="#ff00ff", linewidth=3)
    assert consumer.config_truth["color"] == "#ff00ff"
    assert consumer.config_truth["linewidth"] == 3
    assert consumer.config_truth["ls"] == "--"
    assert consumer.config_truth["zorder"] == 100


def test_divide_chain_splits_walkers_into_new_consumer(consumer):
    data = np.arange(12.0).reshape(6, 2)
    weights = np.arange(6.0)
    consumer.add_chain(data, parameters=["x", "y"], walkers=3, weights=weights, name="walks")
    divided = consumer.divide_chain(0)
    assert isinstance(divided, cc.ChainConsumer)
    assert len(divided.chains) == 3
    assert [c.name for c in divided.chains] == ["Chain 0", "Chain 1", "Chain 2"]
    assert all(c.parameters == ["x", "y"] for c in divided.chains)
    assert all(c.chain.shape == (2, 2) for c in divided.chains)


def test_divide_chain_without_walkers_raises(consumer):
    consumer.add_chain(np.array([[1.0], [2.0]]), parameters=["x"], name="a")
    with pytest.raises(AssertionError, match="was not added with any walkers"):
        consumer.divide_chain(0)
