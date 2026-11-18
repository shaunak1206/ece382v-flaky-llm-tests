import numpy as np
import pandas as pd
import pytest

from chainconsumer.chainconsumer import ChainConsumer


def make_consumer_with_chain(data=None, parameters=None, **kwargs):
    c = ChainConsumer()
    if data is None:
        data = np.array([[0.1, 1.1], [0.2, 1.2], [0.3, 1.3], [0.4, 1.4]])
    if parameters is None:
        parameters = ["x", "y"]
    c.add_chain(data, parameters=parameters, **kwargs)
    return c


def test_init_sets_expected_defaults():
    c = ChainConsumer()
    assert c.chains == []
    assert c._all_parameters == []
    assert c._default_parameters is None
    assert c._configured is False
    assert c._configured_truth is False
    assert c._num_configure_calls == 0
    assert isinstance(c.config, dict)
    assert isinstance(c.config_truth, dict)


def test_add_chain_with_numpy_array_sets_default_parameters_and_name():
    c = ChainConsumer()
    chain = np.array([[1.0, 2.0], [3.0, 4.0]])
    c.add_chain(chain)

    assert len(c.chains) == 1
    assert c._default_parameters == ["0", "1"]
    assert c._all_parameters == ["0", "1"]
    assert c.chains[0].name == "Chain 0"
    assert c.chains[0].parameters == ["0", "1"]


def test_add_chain_with_list_transposes_input():
    c = ChainConsumer()
    c.add_chain([[1, 2, 3], [4, 5, 6]], parameters=["a", "b"])

    assert len(c.chains) == 1
    np.testing.assert_array_equal(c.chains[0].chain, np.array([[1, 4], [2, 5], [3, 6]]))
    assert c.chains[0].parameters == ["a", "b"]


def test_add_chain_from_dict_uses_keys_as_parameters_and_disallows_explicit_parameters():
    c = ChainConsumer()
    chain = {"x": [1, 2, 3], "y": [4, 5, 6]}
    c.add_chain(chain)

    assert c.chains[0].parameters == ["x", "y"]
    assert c._default_parameters == ["x", "y"]
    with pytest.raises(AssertionError):
        c.add_chain(chain, parameters=["x", "y"])


def test_add_chain_from_dataframe_extracts_weight_and_posterior_columns():
    c = ChainConsumer()
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0],
            "b": [3.0, 4.0],
            "weight": [0.2, 0.8],
            "posterior": [10.0, 20.0],
        }
    )
    c.add_chain(df)

    chain = c.chains[0]
    assert chain.parameters == ["a", "b"]
    np.testing.assert_array_equal(chain.weights, np.array([0.2, 0.8]))
    np.testing.assert_array_equal(chain.posterior, np.array([10.0, 20.0]))
    np.testing.assert_array_equal(chain.chain, np.array([[1.0, 3.0], [2.0, 4.0]]))


def test_add_chain_rejects_none_chain():
    c = ChainConsumer()
    with pytest.raises(AssertionError, match="You cannot have a chain of None"):
        c.add_chain(None)


def test_add_chain_requires_parameter_count_not_exceed_columns():
    c = ChainConsumer()
    data = np.array([[1.0, 2.0]])
    with pytest.raises(AssertionError):
        c.add_chain(data, parameters=["x", "y", "z"])


def test_add_chain_grid_with_2d_weights_constructs_meshgrid():
    c = ChainConsumer()
    data = np.array([[0.0, 1.0], [10.0, 20.0]])
    weights = np.array([[1.0, 2.0], [3.0, 4.0]])

    c.add_chain(data, parameters=["x", "y"], grid=True, weights=weights)

    chain = c.chains[0]
    assert chain.grid is True
    assert chain.kde is None
    assert chain.chain.shape == (4, 2)
    np.testing.assert_array_equal(chain.weights, np.array([1.0, 2.0, 3.0, 4.0]))
    assert set(map(tuple, chain.chain)) == {(0.0, 10.0), (0.0, 20.0), (1.0, 10.0), (1.0, 20.0)}


def test_add_chain_grid_rejects_walkers():
    c = ChainConsumer()
    data = np.array([[0.0, 1.0], [10.0, 20.0]])
    weights = np.array([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(AssertionError, match="If grid is set, walkers should not be"):
        c.add_chain(data, parameters=["x", "y"], grid=True, weights=weights, walkers=2)


def test_add_chain_grid_rejects_dict_meshgrid():
    c = ChainConsumer()
    chain = {"x": [0.0, 1.0], "y": [10.0, 20.0]}
    weights = np.array([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(AssertionError, match="flattened array instead"):
        c.add_chain(chain, grid=True, weights=weights)


def test_add_chain_with_shift_params_keeps_chain_and_warns_for_unknown_parameter(caplog):
    c = ChainConsumer()
    data = np.array([[1.0, 2.0], [3.0, 4.0]])

    with caplog.at_level("WARNING"):
        c.add_chain(data, parameters=["a", "b"], shift_params={"missing": 1.0})

    assert any("shift parameter missing is not in list of parameters" in rec.message for rec in caplog.records)


def test_add_covariance_marks_generated_chain_as_non_mcmc(monkeypatch):
    c = ChainConsumer()

    generated = np.array([[1.0, 2.0], [3.0, 4.0]])
    monkeypatch.setattr(np.random, "multivariate_normal", lambda mean, cov, size: generated)

    c.add_covariance([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], parameters=["x", "y"], name="cov")
    assert len(c.chains) == 1
    assert c.chains[0].mcmc_chain is False
    assert c.chains[0].name == "cov"
    np.testing.assert_array_equal(c.chains[0].chain, generated)


def test_add_marker_creates_two_point_chain_and_marks_non_mcmc():
    c = ChainConsumer()
    c.add_marker([1.5, -2.0], parameters=["x", "y"], name="m")

    assert len(c.chains) == 1
    chain = c.chains[0]
    assert chain.mcmc_chain is False
    assert chain.name == "m"
    np.testing.assert_array_equal(chain.chain, np.array([[1.5, -2.0], [1.5, -2.0]]))
    np.testing.assert_array_equal(chain.posterior, np.array([0, 1]))


def test_get_mcmc_chains_filters_out_marker_and_covariance_chains(monkeypatch):
    c = ChainConsumer()
    c.add_chain(np.array([[0.0, 1.0], [1.0, 2.0]]), parameters=["x", "y"])

    monkeypatch.setattr(np.random, "multivariate_normal", lambda mean, cov, size: np.array([[1.0, 2.0], [3.0, 4.0]]))
    c.add_covariance([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], parameters=["x", "y"], name="cov")

    c.add_marker([9.0, 9.0], parameters=["x", "y"], name="marker")

    mcmc = c.get_mcmc_chains()
    assert len(mcmc) == 1
    assert mcmc[0].mcmc_chain is True


def test_remove_chain_by_index_updates_internal_state():
    c = ChainConsumer()
    c.add_chain(np.array([[0.0, 1.0]]), parameters=["x", "y"], name="first")
    c.add_chain(np.array([[2.0, 3.0]]), parameters=["x", "z"], name="second")

    c.remove_chain(0)

    assert len(c.chains) == 1
    assert c.chains[0].name == "second"
    assert c._all_parameters == ["x", "z"]
    assert c._configured is False
    assert c._configured_truth is False


def test_remove_chain_by_name_removes_matching_chain():
    c = ChainConsumer()
    c.add_chain(np.array([[0.0, 1.0]]), parameters=["x", "y"], name="first")
    c.add_chain(np.array([[2.0, 3.0]]), parameters=["x", "z"], name="second")

    c.remove_chain("first")

    assert len(c.chains) == 1
    assert c.chains[0].name == "second"


def test_get_chain_supports_index_name_and_chain_object():
    c = ChainConsumer()
    c.add_chain(np.array([[0.0, 1.0]]), parameters=["x", "y"], name="first")
    c.add_chain(np.array([[2.0, 3.0]]), parameters=["x", "z"], name="second")

    assert c._get_chain(1) == [1]
    assert c._get_chain("first") == [0]
    assert c._get_chain(c.chains[1]) == [1]


def test_configure_sets_truth_defaults_and_returns_self():
    c = ChainConsumer()
    out = c.configure_truth()

    assert out is c
    assert c._configured_truth is True
    assert c.config_truth["ls"] == "--"
    assert c.config_truth["linewidth"] == 1
    assert c.config_truth["color"] == "#000000"
    assert c.config_truth["zorder"] == 100


def test_configure_sets_summary_and_legend_defaults_for_single_chain():
    c = make_consumer_with_chain()
    c.configure(statistics="max", summary=True, summary_area=0.6827, contour_labels="sigma")

    assert c._configured is True
    assert c.config["summary"] is True
    assert c.config["contour_labels"] == "sigma"
    assert c.config["global_point"] is True
    assert c.config["legend_kwargs"]["loc"] == "upper right"
    assert c.config["legend_artists"] is False
    assert c.chains[0].config["summary_area"] == 0.6827


def test_configure_rejects_invalid_statistics_and_invalid_summary_area():
    c = make_consumer_with_chain()

    with pytest.raises(AssertionError):
        c.configure(statistics="not-a-summary")

    with pytest.raises(AssertionError):
        c.configure(statistics="max", summary_area=1.0)
