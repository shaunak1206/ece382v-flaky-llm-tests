import logging

import numpy as np
import pandas as pd
import pytest

from chainconsumer.chainconsumer import ChainConsumer


def _samples(n=100, d=2, seed=0):
    return np.random.RandomState(seed).normal(size=(n, d))


@pytest.fixture
def consumer():
    return ChainConsumer()


def test_add_chain_returns_self_and_appends(consumer):
    result = consumer.add_chain(_samples())
    assert result is consumer
    assert len(consumer.chains) == 1


def test_default_names_are_sequential_and_custom_name_is_stored(consumer):
    consumer.add_chain(_samples())
    consumer.add_chain(_samples(), name="custom")
    consumer.add_chain(_samples())
    assert consumer.chains[0].name == "Chain 0"
    assert consumer.chains[1].name == "custom"
    assert consumer.chains[2].name == "Chain 2"


def test_default_parameter_names_are_column_indices(consumer):
    consumer.add_chain(_samples(d=3))
    assert consumer.chains[0].parameters == ["0", "1", "2"]


def test_list_input_is_transposed_to_samples_by_parameters(consumer):
    consumer.add_chain([[1, 2, 3], [4, 5, 6]])
    assert consumer.chains[0].chain.shape == (3, 2)


def test_1d_array_becomes_single_column_chain(consumer):
    consumer.add_chain(np.arange(5.0))
    assert consumer.chains[0].chain.shape == (5, 1)


def test_dict_input_uses_keys_as_parameters_in_order(consumer):
    data = {"alpha": np.arange(4.0), "beta": np.arange(4.0) * 2}
    consumer.add_chain(data)
    c = consumer.chains[0]
    assert c.parameters == ["alpha", "beta"]
    np.testing.assert_allclose(c.chain[:, 1], np.arange(4.0) * 2)


def test_dataframe_extracts_weight_and_posterior_columns(consumer):
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [4.0, 5.0, 6.0],
        "weight": [1.0, 2.0, 3.0],
        "posterior": [-1.0, -2.0, -3.0],
    })
    consumer.add_chain(df)
    c = consumer.chains[0]
    assert c.parameters == ["x", "y"]
    assert c.chain.shape == (3, 2)
    np.testing.assert_allclose(np.asarray(c.weights), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.asarray(c.posterior), [-1.0, -2.0, -3.0])


def test_npy_file_path_is_loaded(consumer, tmp_path):
    path = tmp_path / "chain.npy"
    arr = _samples(n=20, d=3)
    np.save(path, arr)
    consumer.add_chain(str(path))
    np.testing.assert_allclose(consumer.chains[0].chain, arr)


def test_csv_file_path_is_loaded_as_dataframe(consumer, tmp_path):
    path = tmp_path / "chain.csv"
    pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [4.0, 5.0, 6.0],
        "weight": [1.0, 1.0, 2.0],
        "posterior": [-1.0, -2.0, -3.0],
    }).to_csv(path, index=False)
    consumer.add_chain(str(path))
    c = consumer.chains[0]
    assert c.parameters == ["a", "b"]
    assert c.chain.shape == (3, 2)


def test_none_chain_raises(consumer):
    with pytest.raises(AssertionError):
        consumer.add_chain(None)


def test_dict_with_explicit_parameters_raises(consumer):
    with pytest.raises(AssertionError):
        consumer.add_chain({"a": np.arange(3.0)}, parameters=["a"])


def test_dataframe_with_explicit_parameters_raises(consumer):
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(AssertionError):
        consumer.add_chain(df, parameters=["a", "b"])


def test_more_parameter_names_than_columns_raises(consumer):
    with pytest.raises(AssertionError):
        consumer.add_chain(_samples(d=2), parameters=["a", "b", "c"])


def test_non_numeric_power_raises(consumer):
    with pytest.raises(AssertionError):
        consumer.add_chain(_samples(), power="strong")


def test_default_parameters_are_reused_and_dimensions_checked(consumer):
    consumer.add_chain(_samples(d=2), parameters=["a", "b"])
    consumer.add_chain(_samples(d=2))
    assert consumer.chains[1].parameters == ["a", "b"]
    with pytest.raises(AssertionError):
        consumer.add_chain(_samples(d=3))


def test_all_parameters_accumulate_without_duplicates(consumer):
    consumer.add_chain(_samples(d=2), parameters=["a", "b"])
    consumer.add_chain(_samples(d=2), parameters=["b", "c"])
    assert consumer._all_parameters == ["a", "b", "c"]


def test_grid_input_constraints_are_enforced(consumer):
    with pytest.raises(AssertionError):
        consumer.add_chain(_samples(n=3), grid=True)
    with pytest.raises(AssertionError):
        consumer.add_chain(_samples(n=3), grid=True, walkers=2, weights=np.ones(3))
    with pytest.raises(AssertionError):
        consumer.add_chain(
            {"a": np.arange(3.0), "b": np.arange(3.0)},
            grid=True,
            weights=np.ones((3, 3)),
        )


def test_grid_with_2d_weights_builds_meshgrid(consumer):
    grid_axes = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    weights = np.arange(9.0).reshape(3, 3)
    consumer.add_chain(grid_axes, grid=True, weights=weights)
    c = consumer.chains[0]
    assert c.chain.shape == (9, 2)
    assert c.grid


def test_add_covariance_produces_non_mcmc_chain_with_requested_mean(consumer):
    mean = [1.0, -2.0]
    cov = [[1.0, 0.0], [0.0, 0.25]]
    consumer.add_covariance(mean, cov, parameters=["x", "y"], name="fisher")
    c = consumer.chains[-1]
    assert c.name == "fisher"
    assert c.mcmc_chain is False
    np.testing.assert_allclose(c.chain.mean(axis=0), mean, atol=0.02)


def test_add_marker_creates_non_mcmc_point_chain(consumer):
    consumer.add_marker([1.0, 2.0], parameters=["a", "b"], name="truth")
    c = consumer.chains[-1]
    assert c.name == "truth"
    assert c.parameters == ["a", "b"]
    assert c.mcmc_chain is False


def test_get_mcmc_chains_excludes_markers_and_covariances(consumer):
    consumer.add_chain(_samples(), name="mcmc")
    consumer.add_marker([0.0, 0.0], parameters=["0", "1"], name="marker")
    mcmc = consumer.get_mcmc_chains()
    assert [c.name for c in mcmc] == ["mcmc"]


def test_remove_chain_defaults_to_last_chain(consumer):
    consumer.add_chain(_samples(), name="first")
    consumer.add_chain(_samples(), name="second")
    result = consumer.remove_chain()
    assert result is consumer
    assert consumer._all_names() == ["first"]


def test_remove_chain_by_name(consumer):
    consumer.add_chain(_samples(), name="a")
    consumer.add_chain(_samples(), name="b")
    consumer.remove_chain("a")
    assert consumer._all_names() == ["b"]


def test_remove_chain_duplicate_indices_raises(consumer):
    consumer.add_chain(_samples(), name="a")
    consumer.add_chain(_samples(), name="b")
    with pytest.raises(AssertionError):
        consumer.remove_chain([0, 0])


def test_remove_chain_recomputes_all_parameters(consumer):
    consumer.add_chain(_samples(d=2), parameters=["a", "b"])
    consumer.add_chain(_samples(d=1), parameters=["c"])
    consumer.remove_chain(0)
    assert consumer._all_parameters == ["c"]


def test_get_chain_by_name_returns_all_matching_indices(consumer):
    consumer.add_chain(_samples(), name="dup")
    consumer.add_chain(_samples(), name="other")
    consumer.add_chain(_samples(), name="dup")
    assert consumer._get_chain("dup") == [0, 2]
    assert consumer._get_chain(consumer.chains[1]) == [1]


def test_get_chain_with_unsupported_type_raises(consumer):
    consumer.add_chain(_samples())
    with pytest.raises(ValueError):
        consumer._get_chain(1.5)


def test_divide_chain_splits_each_walker_into_its_own_chain(consumer):
    data = np.arange(40.0 * 2).reshape(40, 2)
    consumer.add_chain(data, walkers=4)
    divided = consumer.divide_chain(0)
    assert len(divided.chains) == 4
    assert divided._all_names() == ["Chain 0", "Chain 1", "Chain 2", "Chain 3"]
    assert divided.chains[1].chain.shape == (10, 2)
    np.testing.assert_array_equal(divided.chains[1].chain, data[10:20])


def test_divide_chain_without_walkers_raises(consumer):
    consumer.add_chain(_samples(n=40))
    with pytest.raises(AssertionError):
        consumer.divide_chain(0)


def test_configure_default_sigmas_and_returns_self(consumer):
    consumer.add_chain(_samples())
    result = consumer.configure()
    assert result is consumer
    assert consumer._configured
    np.testing.assert_array_equal(consumer.config["sigmas"], [0, 1, 2])


def test_configure_prepends_zero_and_sorts_sigmas(consumer):
    consumer.add_chain(_samples())
    consumer.configure(sigmas=np.array([3, 1]))
    np.testing.assert_array_equal(consumer.config["sigmas"], [0, 1, 3])


def test_configure_summary_only_applies_to_single_chain(consumer):
    consumer.add_chain(_samples(), name="one")
    consumer.configure(summary=True)
    assert consumer.config["summary"] is True

    consumer.add_chain(_samples(), name="two")
    consumer.configure(summary=True)
    assert consumer.config["summary"] is False


def test_configure_invalid_statistics_raises(consumer):
    consumer.add_chain(_samples())
    with pytest.raises(AssertionError):
        consumer.configure(statistics="not_a_statistic")
    with pytest.raises(ValueError):
        consumer.configure(statistics=5)


def test_configure_cmap_and_explicit_colors_conflict_raises(consumer):
    consumer.add_chain(_samples())
    with pytest.raises(AssertionError):
        consumer.configure(cmap="viridis", colors="#ff0000")


def test_configure_summary_area_must_be_float(consumer):
    consumer.add_chain(_samples())
    with pytest.raises(AssertionError):
        consumer.configure(summary_area=1)
    with pytest.raises(AssertionError):
        consumer.configure(summary_area=1.5)


def test_configure_lowercases_contour_labels(consumer):
    consumer.add_chain(_samples())
    consumer.configure(contour_labels="Sigma")
    assert consumer.config["contour_labels"] == "sigma"


def test_configure_legend_kwargs_override_defaults(consumer):
    consumer.add_chain(_samples())
    consumer.configure(legend_kwargs={"loc": "lower left"})
    assert consumer.config["legend_kwargs"]["loc"] == "lower left"
    assert consumer.config["legend_kwargs"]["frameon"] is False


def test_configure_truth_sets_defaults_and_respects_linestyle(consumer):
    consumer.configure_truth()
    assert consumer._configured_truth
    assert consumer.config_truth["ls"] == "--"
    assert consumer.config_truth["color"] == "#000000"
    assert consumer.config_truth["zorder"] == 100

    consumer.configure_truth(linestyle=":")
    assert "ls" not in consumer.config_truth
    assert consumer.config_truth["linestyle"] == ":"


def test_add_chain_resets_previous_configuration(consumer):
    consumer.add_chain(_samples())
    consumer.configure()
    assert consumer._configured
    consumer.add_chain(_samples())
    assert consumer._configured is False
    assert consumer.config == {}


def test_shift_params_with_unknown_key_logs_warning(consumer, caplog):
    with caplog.at_level(logging.WARNING, logger="chainconsumer"):
        consumer.add_chain(
            _samples(d=2),
            parameters=["a", "b"],
            shift_params={"zz": 1.0},
        )
    assert any("shift parameter" in r.getMessage() for r in caplog.records)
