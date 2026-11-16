import numpy as np
import pandas as pd
import pytest

from chainconsumer import ChainConsumer


@pytest.fixture
def rng():
    return np.random.RandomState(1234)


@pytest.fixture
def consumer():
    return ChainConsumer()


def test_init_creates_empty_state(consumer):
    assert consumer.chains == []
    assert consumer.config == {}
    assert consumer.config_truth == {}
    assert consumer._configured is False
    assert consumer._num_configure_calls == 0
    assert consumer._all_names() == []


def test_version_attribute():
    assert ChainConsumer.__version__ == "0.31.1"


def test_add_chain_returns_self_and_uses_default_name_and_params(consumer, rng):
    data = rng.normal(size=(100, 2))
    result = consumer.add_chain(data)
    assert result is consumer
    assert len(consumer.chains) == 1
    assert consumer.chains[0].name == "Chain 0"
    assert list(consumer.chains[0].parameters) == ["0", "1"]


def test_add_chain_explicit_name_and_parameters(consumer, rng):
    data = rng.normal(size=(50, 2))
    consumer.add_chain(data, parameters=["a", "b"], name="mine")
    c = consumer.chains[0]
    assert c.name == "mine"
    assert list(c.parameters) == ["a", "b"]
    assert consumer._all_parameters == ["a", "b"]


def test_default_name_increments(consumer, rng):
    consumer.add_chain(rng.normal(size=(20, 2)))
    consumer.add_chain(rng.normal(size=(20, 2)))
    assert consumer._all_names() == ["Chain 0", "Chain 1"]


def test_add_chain_none_raises(consumer):
    with pytest.raises(AssertionError):
        consumer.add_chain(None)


def test_add_chain_1d_array_becomes_single_column(consumer):
    consumer.add_chain(np.arange(10.0))
    c = consumer.chains[0]
    assert len(c.parameters) == 1
    assert c.chain.shape == (10, 1)


def test_add_chain_from_dict_uses_keys_as_parameters(consumer):
    samples = {"omega": np.linspace(0, 1, 30), "sigma8": np.linspace(2, 3, 30)}
    consumer.add_chain(samples)
    c = consumer.chains[0]
    assert list(c.parameters) == ["omega", "sigma8"]
    assert c.chain.shape == (30, 2)


def test_add_chain_dict_with_parameters_raises(consumer):
    samples = {"x": np.zeros(5), "y": np.ones(5)}
    with pytest.raises(AssertionError):
        consumer.add_chain(samples, parameters=["a", "b"])


def test_add_chain_from_list_is_transposed(consumer):
    consumer.add_chain([[1, 2, 3], [4, 5, 6]], parameters=["p", "q"])
    c = consumer.chains[0]
    assert c.chain.shape == (3, 2)


def test_add_chain_dataframe_extracts_weight_and_posterior(consumer):
    df = pd.DataFrame(
        {
            "a": np.linspace(0, 1, 8),
            "b": np.linspace(1, 2, 8),
            "weight": np.ones(8),
            "posterior": np.arange(8.0),
        }
    )
    consumer.add_chain(df)
    c = consumer.chains[0]
    assert list(c.parameters) == ["a", "b"]
    assert c.chain.shape == (8, 2)
    assert np.allclose(c.posterior, np.arange(8.0))


def test_add_chain_dataframe_with_parameters_raises(consumer):
    df = pd.DataFrame({"a": np.zeros(4), "b": np.zeros(4)})
    with pytest.raises(AssertionError):
        consumer.add_chain(df, parameters=["x", "y"])


def test_add_chain_explicit_posterior_is_stored(consumer, rng):
    data = rng.normal(size=(15, 2))
    post = np.arange(15.0)
    consumer.add_chain(data, posterior=post)
    assert np.allclose(consumer.chains[0].posterior, post)


def test_default_parameters_reused_by_later_chain(consumer, rng):
    consumer.add_chain(rng.normal(size=(10, 2)), parameters=["x", "y"])
    consumer.add_chain(rng.normal(size=(10, 2)))
    assert list(consumer.chains[1].parameters) == ["x", "y"]


def test_default_parameters_dimension_mismatch_raises(consumer, rng):
    consumer.add_chain(rng.normal(size=(10, 2)), parameters=["x", "y"])
    with pytest.raises(AssertionError):
        consumer.add_chain(rng.normal(size=(10, 3)))


def test_more_parameter_names_than_columns_raises(consumer, rng):
    with pytest.raises(AssertionError):
        consumer.add_chain(rng.normal(size=(10, 2)), parameters=["a", "b", "c"])


def test_non_numeric_power_raises(consumer, rng):
    with pytest.raises(AssertionError):
        consumer.add_chain(rng.normal(size=(10, 2)), power="high")


def test_grid_with_walkers_raises(consumer, rng):
    with pytest.raises(AssertionError):
        consumer.add_chain(
            rng.normal(size=(10, 2)), grid=True, walkers=2, weights=np.ones(10)
        )


def test_grid_without_weights_raises(consumer, rng):
    with pytest.raises(AssertionError):
        consumer.add_chain(rng.normal(size=(10, 2)), grid=True)


def test_grid_2d_weights_builds_meshgrid_and_disables_kde(consumer):
    axes = np.zeros((4, 2))
    weights = np.ones((4, 4))
    consumer.add_chain(axes, grid=True, weights=weights, kde=True, parameters=["x", "y"])
    c = consumer.chains[0]
    assert len(c.weights) == 16
    assert c.kde is None


def test_remove_chain_by_name_and_updates_parameters(consumer, rng):
    consumer.add_chain(rng.normal(size=(10, 2)), parameters=["a", "b"], name="first")
    consumer.add_chain(rng.normal(size=(10, 2)), parameters=["c", "d"], name="second")
    result = consumer.remove_chain("first")
    assert result is consumer
    assert consumer._all_names() == ["second"]
    assert consumer._all_parameters == ["c", "d"]


def test_remove_chain_default_removes_last(consumer, rng):
    consumer.add_chain(rng.normal(size=(10, 1)), name="keep")
    consumer.add_chain(rng.normal(size=(10, 1)), name="drop")
    consumer.remove_chain()
    assert consumer._all_names() == ["keep"]


def test_remove_chain_duplicate_indices_raises(consumer, rng):
    consumer.add_chain(rng.normal(size=(10, 1)))
    consumer.add_chain(rng.normal(size=(10, 1)))
    with pytest.raises(AssertionError):
        consumer.remove_chain([0, 0])


def test_add_marker_is_not_mcmc_chain(consumer):
    consumer.add_chain(np.random.RandomState(0).normal(size=(20, 2)), parameters=["x", "y"])
    consumer.add_marker([0.5, 0.5], parameters=["x", "y"], name="marker")
    assert consumer.chains[-1].mcmc_chain is False
    assert len(consumer.get_mcmc_chains()) == 1
    assert consumer.get_mcmc_chains()[0].name == "Chain 0"


def test_configure_returns_self_and_marks_configured(consumer, rng):
    consumer.add_chain(rng.normal(size=(30, 2)), parameters=["x", "y"])
    result = consumer.configure()
    assert result is consumer
    assert consumer._configured is True
    assert consumer.config["flip"] is True
    assert consumer.config["usetex"] is True


def test_configure_prepends_zero_sigma(consumer, rng):
    consumer.add_chain(rng.normal(size=(30, 2)), parameters=["x", "y"])
    consumer.configure(sigmas=np.array([1, 2]))
    assert list(consumer.config["sigmas"]) == [0, 1, 2]


def test_configure_int_bins_applied_to_chain(consumer, rng):
    consumer.add_chain(rng.normal(size=(30, 2)), parameters=["x", "y"])
    consumer.configure(bins=25)
    assert consumer.chains[0].config["bins"] == 25


def test_configure_invalid_statistics_string_raises(consumer, rng):
    consumer.add_chain(rng.normal(size=(30, 2)), parameters=["x", "y"])
    with pytest.raises(AssertionError):
        consumer.configure(statistics="not_a_statistic")


def test_configure_numeric_statistics_raises_value_error(consumer, rng):
    consumer.add_chain(rng.normal(size=(30, 2)), parameters=["x", "y"])
    with pytest.raises(ValueError):
        consumer.configure(statistics=5)


def test_configure_summary_area_out_of_range_raises(consumer, rng):
    consumer.add_chain(rng.normal(size=(30, 2)), parameters=["x", "y"])
    with pytest.raises(AssertionError):
        consumer.configure(summary_area=1.5)


def test_configure_cmap_and_colors_together_raises(consumer, rng):
    consumer.add_chain(rng.normal(size=(30, 2)), parameters=["x", "y"])
    with pytest.raises(AssertionError):
        consumer.configure(cmap="viridis", colors="#ff0000")


def test_configure_truth_defaults(consumer):
    result = consumer.configure_truth()
    assert result is consumer
    assert consumer._configured_truth is True
    assert consumer.config_truth["ls"] == "--"
    assert consumer.config_truth["color"] == "#000000"
    assert consumer.config_truth["linewidth"] == 1
    assert consumer.config_truth["zorder"] == 100


def test_configure_truth_respects_linestyle(consumer):
    consumer.configure_truth(linestyle=":", color="red")
    assert "ls" not in consumer.config_truth
    assert consumer.config_truth["linestyle"] == ":"
    assert consumer.config_truth["color"] == "red"


def test_divide_chain_splits_walkers_into_new_consumer(consumer, rng):
    data = rng.normal(size=(10, 2))
    consumer.add_chain(data, parameters=["x", "y"], walkers=2, name="walk")
    divided = consumer.divide_chain("walk")
    assert divided is not consumer
    assert divided._all_names() == ["Chain 0", "Chain 1"]
    assert len(consumer.chains) == 1


def test_divide_chain_without_walkers_raises(consumer, rng):
    consumer.add_chain(rng.normal(size=(10, 2)), parameters=["x", "y"])
    with pytest.raises(AssertionError):
        consumer.divide_chain(0)


def test_get_chain_index_out_of_range_raises(consumer, rng):
    consumer.add_chain(rng.normal(size=(10, 1)))
    with pytest.raises(AssertionError):
        consumer._get_chain(5)


def test_get_chain_unknown_type_raises(consumer, rng):
    consumer.add_chain(rng.normal(size=(10, 1)))
    with pytest.raises(ValueError):
        consumer._get_chain(1.5)


def test_get_chain_by_object_and_name(consumer, rng):
    consumer.add_chain(rng.normal(size=(10, 1)), name="alpha")
    consumer.add_chain(rng.normal(size=(10, 1)), name="beta")
    assert consumer._get_chain(consumer.chains[1]) == [1]
    assert consumer._get_chain("alpha") == [0]
    assert consumer._get_chain_name(1) == "beta"
