import numpy as np
import pandas as pd
import pytest

from chainconsumer import ChainConsumer
from chainconsumer.chain import Chain


@pytest.fixture
def rng():
    return np.random.RandomState(1234)


@pytest.fixture
def consumer():
    return ChainConsumer()


def _samples(rng, n=200, d=2):
    return rng.normal(size=(n, d))


def test_init_creates_empty_state(consumer):
    assert consumer.chains == []
    assert consumer.config == {}
    assert consumer.config_truth == {}
    assert consumer._configured is False
    assert consumer._configured_truth is False
    assert consumer._num_configure_calls == 0
    assert consumer.get_mcmc_chains() == []


def test_add_chain_ndarray_defaults(consumer, rng):
    data = _samples(rng)
    result = consumer.add_chain(data)
    assert len(consumer.chains) == 1
    chain = consumer.chains[0]
    assert chain.name == "Chain 0"
    assert list(chain.parameters) == ["0", "1"]


def test_add_chain_returns_self_for_chaining(consumer, rng):
    returned = consumer.add_chain(_samples(rng))
    assert returned is consumer


def test_add_chain_default_names_increment_and_custom_name(consumer, rng):
    consumer.add_chain(_samples(rng))
    consumer.add_chain(_samples(rng), name="custom")
    consumer.add_chain(_samples(rng))
    names = [c.name for c in consumer.chains]
    assert names == ["Chain 0", "custom", "Chain 2"]


def test_add_chain_1d_array_becomes_single_parameter(consumer, rng):
    consumer.add_chain(rng.normal(size=300), parameters=["x"])
    chain = consumer.chains[0]
    assert list(chain.parameters) == ["x"]
    assert np.asarray(chain.chain).shape[1] == 1


def test_add_chain_dict_takes_parameters_from_keys(consumer, rng):
    data = {"alpha": rng.normal(size=100), "beta": rng.normal(size=100)}
    consumer.add_chain(data)
    assert list(consumer.chains[0].parameters) == ["alpha", "beta"]


def test_add_chain_dict_with_explicit_parameters_raises(consumer, rng):
    data = {"alpha": rng.normal(size=100), "beta": rng.normal(size=100)}
    with pytest.raises(AssertionError):
        consumer.add_chain(data, parameters=["a", "b"])


def test_add_chain_none_raises(consumer):
    with pytest.raises(AssertionError):
        consumer.add_chain(None)


def test_add_chain_dataframe_extracts_weight_and_posterior(consumer, rng):
    df = pd.DataFrame(
        {
            "a": rng.normal(size=50),
            "b": rng.normal(size=50),
            "weight": np.ones(50),
            "posterior": rng.normal(size=50),
        }
    )
    consumer.add_chain(df)
    chain = consumer.chains[0]
    assert list(chain.parameters) == ["a", "b"]
    assert len(chain.weights) == 50


def test_add_chain_dataframe_with_parameters_raises(consumer, rng):
    df = pd.DataFrame({"a": rng.normal(size=10), "b": rng.normal(size=10)})
    with pytest.raises(AssertionError):
        consumer.add_chain(df, parameters=["a", "b"])


def test_add_chain_too_many_parameter_names_raises(consumer, rng):
    with pytest.raises(AssertionError):
        consumer.add_chain(_samples(rng, d=2), parameters=["a", "b", "c"])


def test_add_chain_non_numeric_power_raises(consumer, rng):
    with pytest.raises(AssertionError):
        consumer.add_chain(_samples(rng), power="strong")


def test_default_parameters_are_reused_by_later_chains(consumer, rng):
    consumer.add_chain(_samples(rng), parameters=["a", "b"])
    consumer.add_chain(_samples(rng))
    assert list(consumer.chains[1].parameters) == ["a", "b"]


def test_all_parameters_is_union_without_duplicates(consumer, rng):
    consumer.add_chain(_samples(rng, d=2), parameters=["a", "b"])
    consumer.add_chain(_samples(rng, d=2), parameters=["b", "c"])
    assert consumer._all_parameters == ["a", "b", "c"]


def test_add_chain_marks_mcmc_chain_by_default(consumer, rng):
    consumer.add_chain(_samples(rng))
    assert len(consumer.get_mcmc_chains()) == 1


def test_add_covariance_is_not_mcmc_chain(consumer):
    consumer.add_chain(np.zeros((10, 2)), parameters=["a", "b"], name="mcmc")
    result = consumer.add_covariance([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], parameters=["a", "b"], name="fisher")
    assert result is consumer
    assert len(consumer.chains) == 2
    assert consumer.chains[-1].mcmc_chain is False
    assert [c.name for c in consumer.get_mcmc_chains()] == ["mcmc"]


def test_remove_chain_defaults_to_last(consumer, rng):
    consumer.add_chain(_samples(rng), name="first")
    consumer.add_chain(_samples(rng), name="second")
    consumer.remove_chain()
    assert [c.name for c in consumer.chains] == ["first"]


def test_remove_chain_by_name(consumer, rng):
    consumer.add_chain(_samples(rng), name="first")
    consumer.add_chain(_samples(rng), name="second")
    consumer.remove_chain("first")
    assert [c.name for c in consumer.chains] == ["second"]


def test_remove_chain_same_chain_twice_raises(consumer, rng):
    consumer.add_chain(_samples(rng))
    consumer.add_chain(_samples(rng))
    with pytest.raises(AssertionError):
        consumer.remove_chain([0, 0])


def test_get_chain_invalid_type_raises_value_error(consumer, rng):
    consumer.add_chain(_samples(rng))
    with pytest.raises(ValueError):
        consumer._get_chain(1.5)


def test_get_chain_index_out_of_range_raises(consumer, rng):
    consumer.add_chain(_samples(rng))
    with pytest.raises(AssertionError):
        consumer._get_chain(5)


def test_configure_invalid_statistics_raises(consumer, rng):
    consumer.add_chain(_samples(rng))
    with pytest.raises(AssertionError):
        consumer.configure(statistics="not_a_statistic")


def test_configure_sets_state_and_sigmas(consumer, rng):
    consumer.add_chain(_samples(rng))
    result = consumer.configure(sigmas=np.array([1, 2]))
    assert result is consumer
    assert consumer._configured is True
    assert consumer._num_configure_calls == 1
    assert list(consumer.config["sigmas"]) == [0, 1, 2]


def test_configure_cmap_and_colors_together_raises(consumer, rng):
    consumer.add_chain(_samples(rng))
    with pytest.raises(AssertionError):
        consumer.configure(cmap="viridis", colors="#ff0000")


def test_configure_truth_defaults(consumer):
    result = consumer.configure_truth()
    assert result is consumer
    assert consumer._configured_truth is True
    assert consumer.config_truth["ls"] == "--"
    assert consumer.config_truth["color"] == "#000000"
    assert consumer.config_truth["zorder"] == 100


def test_divide_chain_splits_by_walkers(consumer, rng):
    data = _samples(rng, n=8, d=2)
    consumer.add_chain(data, parameters=["a", "b"], name="mc", walkers=2, weights=np.ones(8))
    divided = consumer.divide_chain("mc")
    assert isinstance(divided, ChainConsumer)
    assert divided is not consumer
    assert len(divided.chains) == 2
    assert [c.name for c in divided.chains] == ["Chain 0", "Chain 1"]
    assert len(consumer.chains) == 1


def test_divide_chain_without_walkers_raises(consumer, rng):
    consumer.add_chain(_samples(rng, n=8), weights=np.ones(8))
    with pytest.raises(AssertionError):
        consumer.divide_chain(0)
