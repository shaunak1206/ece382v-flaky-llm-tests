import numpy as np
import pandas as pd
import pytest

from chainconsumer import ChainConsumer


@pytest.fixture
def samples():
    rng = np.random.RandomState(0)
    return rng.normal(size=(200, 2))


@pytest.fixture
def consumer():
    return ChainConsumer()


def test_init_starts_empty(consumer):
    assert consumer.chains == []
    assert consumer.config == {}
    assert consumer._configured is False
    assert consumer._configured_truth is False


def test_add_chain_numpy_returns_self_and_stores_one_chain(consumer, samples):
    result = consumer.add_chain(samples)
    assert result is consumer
    assert len(consumer.chains) == 1
    assert consumer.chains[0].parameters == ["0", "1"]


def test_add_chain_with_explicit_parameters(consumer, samples):
    consumer.add_chain(samples, parameters=["alpha", "beta"])
    assert consumer.chains[0].parameters == ["alpha", "beta"]


def test_add_chain_default_names_are_sequential(consumer, samples):
    consumer.add_chain(samples)
    consumer.add_chain(samples)
    assert consumer.chains[0].name == "Chain 0"
    assert consumer.chains[1].name == "Chain 1"


def test_add_chain_uses_given_name(consumer, samples):
    consumer.add_chain(samples, name="my_chain")
    assert consumer.chains[0].name == "my_chain"


def test_add_chain_none_raises(consumer):
    with pytest.raises(AssertionError):
        consumer.add_chain(None)


def test_add_chain_1d_array_becomes_single_column(consumer):
    consumer.add_chain(np.arange(10.0), parameters=["x"])
    assert consumer.chains[0].chain.shape == (10, 1)


def test_add_chain_list_is_transposed(consumer):
    data = [list(range(5)), list(range(5, 10))]
    consumer.add_chain(data, parameters=["a", "b"])
    assert consumer.chains[0].chain.shape == (5, 2)


def test_add_chain_dict_takes_parameters_from_keys(consumer):
    chain = {"zeta": np.linspace(0, 1, 20), "eta": np.linspace(1, 2, 20)}
    consumer.add_chain(chain)
    assert consumer.chains[0].parameters == ["zeta", "eta"]
    assert consumer.chains[0].chain.shape == (20, 2)


def test_add_chain_dict_with_parameters_raises(consumer):
    chain = {"a": np.ones(5), "b": np.ones(5)}
    with pytest.raises(AssertionError):
        consumer.add_chain(chain, parameters=["a", "b"])


def test_add_chain_dataframe_extracts_weight_and_posterior(consumer):
    df = pd.DataFrame(
        {
            "a": np.linspace(0, 1, 10),
            "b": np.linspace(1, 2, 10),
            "weight": np.ones(10),
            "posterior": np.arange(10.0),
        }
    )
    consumer.add_chain(df)
    chain = consumer.chains[0]
    assert chain.parameters == ["a", "b"]
    assert chain.chain.shape == (10, 2)
    assert chain.posterior is not None


def test_add_chain_dataframe_with_parameters_raises(consumer):
    df = pd.DataFrame({"a": np.ones(5), "b": np.ones(5)})
    with pytest.raises(AssertionError):
        consumer.add_chain(df, parameters=["a", "b"])


def test_add_chain_too_many_parameter_names_raises(consumer, samples):
    with pytest.raises(AssertionError):
        consumer.add_chain(samples, parameters=["a", "b", "c"])


def test_add_chain_non_numeric_power_raises(consumer, samples):
    with pytest.raises(AssertionError):
        consumer.add_chain(samples, power="strong")


def test_second_chain_without_parameters_reuses_defaults(consumer, samples):
    consumer.add_chain(samples, parameters=["p", "q"])
    consumer.add_chain(samples)
    assert consumer.chains[1].parameters == ["p", "q"]


def test_second_chain_with_wrong_dimension_raises(consumer, samples):
    consumer.add_chain(samples, parameters=["p", "q"])
    with pytest.raises(AssertionError):
        consumer.add_chain(np.ones((10, 3)))


def test_all_parameters_accumulates_unique_names(consumer, samples):
    consumer.add_chain(samples, parameters=["a", "b"])
    consumer.add_chain(samples, parameters=["b", "c"])
    assert consumer._all_parameters == ["a", "b", "c"]


def test_remove_chain_default_removes_last(consumer, samples):
    consumer.add_chain(samples, name="first")
    consumer.add_chain(samples, name="second")
    consumer.remove_chain()
    assert [c.name for c in consumer.chains] == ["first"]


def test_remove_chain_by_name(consumer, samples):
    consumer.add_chain(samples, name="first")
    consumer.add_chain(samples, name="second")
    consumer.remove_chain("first")
    assert [c.name for c in consumer.chains] == ["second"]


def test_remove_chain_unknown_name_raises(consumer, samples):
    consumer.add_chain(samples, name="only")
    with pytest.raises(AssertionError):
        consumer.remove_chain("missing")


def test_remove_chain_same_index_twice_raises(consumer, samples):
    consumer.add_chain(samples)
    consumer.add_chain(samples)
    with pytest.raises(AssertionError):
        consumer.remove_chain([0, 0])


def test_remove_chain_updates_all_parameters(consumer, samples):
    consumer.add_chain(samples, parameters=["a", "b"])
    consumer.add_chain(samples, parameters=["c", "d"])
    consumer.remove_chain(1)
    assert consumer._all_parameters == ["a", "b"]


def test_add_marker_is_not_an_mcmc_chain(consumer, samples):
    consumer.add_chain(samples, parameters=["a", "b"], name="mcmc")
    consumer.add_marker([0.5, 0.5], parameters=["a", "b"], name="marker")
    assert len(consumer.chains) == 2
    mcmc = consumer.get_mcmc_chains()
    assert len(mcmc) == 1
    assert mcmc[0].name == "mcmc"


def test_grid_with_walkers_raises(consumer):
    weights = np.ones(12)
    with pytest.raises(AssertionError):
        consumer.add_chain(np.ones((12, 2)), grid=True, walkers=2, weights=weights)


def test_grid_without_weights_raises(consumer):
    with pytest.raises(AssertionError):
        consumer.add_chain(np.ones((12, 2)), grid=True)


def test_grid_chain_is_flagged_and_has_no_kde(consumer):
    consumer.add_chain(np.ones((12, 2)), grid=True, weights=np.ones(12), kde=True)
    chain = consumer.chains[0]
    assert chain.grid is True
    assert chain.kde is None


def test_configure_sets_default_config(consumer, samples):
    consumer.add_chain(samples, parameters=["a", "b"])
    result = consumer.configure()
    assert result is consumer
    assert consumer._configured is True
    assert consumer.config["flip"] is True
    assert consumer.config["serif"] is True
    assert np.array_equal(consumer.config["sigmas"], np.array([0, 1, 2]))
    assert consumer.config["summary_area"] if "summary_area" in consumer.config else True


def test_configure_invalid_statistics_string_raises(consumer, samples):
    consumer.add_chain(samples)
    with pytest.raises(AssertionError):
        consumer.configure(statistics="not_a_statistic")


def test_configure_non_string_statistics_raises_value_error(consumer, samples):
    consumer.add_chain(samples)
    with pytest.raises(ValueError):
        consumer.configure(statistics=5)


def test_configure_bins_integer_applied_to_chain(consumer, samples):
    consumer.add_chain(samples, parameters=["a", "b"])
    consumer.configure(bins=25)
    assert consumer.chains[0].config["bins"] == 25


def test_configure_truth_defaults(consumer):
    result = consumer.configure_truth()
    assert result is consumer
    assert consumer._configured_truth is True
    assert consumer.config_truth["ls"] == "--"
    assert consumer.config_truth["linewidth"] == 1
    assert consumer.config_truth["color"] == "#000000"
    assert consumer.config_truth["zorder"] == 100


def test_configure_truth_keeps_user_color(consumer):
    consumer.configure_truth(color="#ff0000")
    assert consumer.config_truth["color"] == "#ff0000"


def test_divide_chain_creates_one_chain_per_walker(consumer):
    data = np.arange(12.0).reshape(6, 2)
    consumer.add_chain(data, parameters=["a", "b"], walkers=2)
    divided = consumer.divide_chain(0)
    assert isinstance(divided, ChainConsumer)
    assert len(divided.chains) == 2
    assert divided.chains[0].chain.shape == (3, 2)


def test_divide_chain_without_walkers_raises(consumer, samples):
    consumer.add_chain(samples, parameters=["a", "b"])
    with pytest.raises(AssertionError):
        consumer.divide_chain(0)
