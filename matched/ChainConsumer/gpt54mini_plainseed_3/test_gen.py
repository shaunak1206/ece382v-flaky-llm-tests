import numpy as np
import pandas as pd
import pytest

from chainconsumer.chainconsumer import ChainConsumer


def test_init_sets_expected_defaults():
    cc = ChainConsumer()
    assert cc.chains == []
    assert cc._all_parameters == []
    assert cc._default_parameters is None
    assert cc._configured is False
    assert cc._num_configure_calls == 0
    assert cc.config == {}
    assert cc.config_truth == {}
    assert cc._gauss_mode == "reflect"
    assert isinstance(cc._all_colours, list)
    assert len(cc._all_colours) > 0


def test_add_chain_from_numpy_sets_default_parameters_and_name():
    cc = ChainConsumer()
    cc.add_chain(np.array([[1.0, 2.0], [3.0, 4.0]]), parameters=["x", "y"])
    assert len(cc.chains) == 1
    chain = cc.chains[0]
    assert chain.name == "Chain 0"
    assert chain.parameters == ["x", "y"]
    assert cc._default_parameters == ["x", "y"]
    assert cc._all_parameters == ["x", "y"]


def test_add_chain_from_list_transposes_input():
    cc = ChainConsumer()
    cc.add_chain([[1, 2, 3], [4, 5, 6]], parameters=["a", "b"])
    chain = cc.chains[0]
    assert chain.chain.shape == (3, 2)
    assert np.allclose(chain.chain, np.array([[1, 4], [2, 5], [3, 6]]))
    assert chain.parameters == ["a", "b"]


def test_add_chain_from_dict_uses_keys_as_parameters():
    cc = ChainConsumer()
    cc.add_chain({"a": [1, 2], "b": [3, 4]})
    chain = cc.chains[0]
    assert chain.parameters == ["a", "b"]
    assert np.allclose(chain.chain, np.array([[1, 3], [2, 4]]))


def test_add_chain_from_dataframe_extracts_weight_and_posterior_columns():
    cc = ChainConsumer()
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "weight": [0.1, 0.9],
            "posterior": [10.0, 11.0],
        }
    )
    cc.add_chain(df)
    chain = cc.chains[0]
    assert chain.parameters == ["x", "y"]
    assert np.allclose(chain.chain, np.array([[1.0, 3.0], [2.0, 4.0]]))
    assert np.allclose(np.asarray(chain.weights), np.array([0.1, 0.9]))
    assert np.allclose(np.asarray(chain.posterior), np.array([10.0, 11.0]))


def test_add_chain_grid_with_matrix_weights_flattens_meshgrid():
    cc = ChainConsumer()
    chain = np.array([[0.0, 1.0], [10.0, 20.0]])  # two parameter ranges
    weights = np.array([[1.0, 2.0], [3.0, 4.0]])

    cc.add_chain(chain, parameters=["a", "b"], grid=True, weights=weights)

    c = cc.chains[0]
    assert c.grid is True
    assert c.kde is None
    assert c.chain.shape == (4, 2)
    assert np.allclose(c.chain, np.array([[0.0, 10.0], [0.0, 20.0], [1.0, 10.0], [1.0, 20.0]]))
    assert np.allclose(np.asarray(c.weights), np.array([1.0, 2.0, 3.0, 4.0]))


def test_add_chain_grid_rejects_walkers():
    cc = ChainConsumer()
    with pytest.raises(AssertionError):
        cc.add_chain(np.array([[0.0, 1.0], [2.0, 3.0]]), parameters=["a", "b"], grid=True, walkers=2, weights=np.array([1.0, 2.0]))


def test_add_chain_rejects_none_chain():
    cc = ChainConsumer()
    with pytest.raises(AssertionError, match="You cannot have a chain of None"):
        cc.add_chain(None, parameters=["x"])


def test_add_chain_rejects_dataframe_with_parameters():
    cc = ChainConsumer()
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    with pytest.raises(AssertionError, match="You cannot pass a DataFrame and use parameter names"):
        cc.add_chain(df, parameters=["x", "y"])


def test_add_chain_rejects_dict_with_parameters():
    cc = ChainConsumer()
    with pytest.raises(AssertionError, match="You cannot pass a dictionary and specify parameter names"):
        cc.add_chain({"a": [1, 2]}, parameters=["a"])


def test_add_chain_rejects_too_many_parameter_names():
    cc = ChainConsumer()
    with pytest.raises(AssertionError, match="Have only 2 columns in chain"):
        cc.add_chain(np.array([[1, 2], [3, 4]]), parameters=["a", "b", "c"])


def test_add_chain_accepts_numeric_power_and_rejects_non_numeric():
    cc = ChainConsumer()
    cc.add_chain(np.array([[1.0], [2.0]]), parameters=["x"], power=2.5)
    assert cc.chains[0].power == 2.5

    cc2 = ChainConsumer()
    with pytest.raises(AssertionError, match="Power should be numeric"):
        cc2.add_chain(np.array([[1.0], [2.0]]), parameters=["x"], power="bad")


def test_add_marker_creates_non_mcmc_chain_with_point_plotting():
    cc = ChainConsumer()
    cc.add_marker([1.0, 2.0], parameters=["x", "y"], name="m")
    assert len(cc.chains) == 1
    c = cc.chains[0]
    assert c.mcmc_chain is False
    assert c.posterior is not None
    assert np.allclose(c.posterior, np.array([0, 1]))
    assert c.config["plot_point"] is True
    assert c.config["plot_contour"] is False


def test_add_covariance_uses_multivariate_normal_and_marks_chain_non_mcmc(monkeypatch):
    cc = ChainConsumer()

    calls = {}

    def fake_mvn(mean, cov, size):
        calls["mean"] = mean
        calls["cov"] = cov
        calls["size"] = size
        return np.array([[1.0, 2.0], [3.0, 4.0]])

    monkeypatch.setattr(np.random, "multivariate_normal", fake_mvn)

    cc.add_covariance([0.0, 1.0], [[1.0, 0.0], [0.0, 1.0]], parameters=["x", "y"], name="cov")
    assert calls["mean"] == [0.0, 1.0]
    assert calls["cov"] == [[1.0, 0.0], [0.0, 1.0]]
    assert calls["size"] == 1000000
    assert len(cc.chains) == 1
    assert cc.chains[0].mcmc_chain is False


def test_remove_chain_by_index_and_name_updates_internal_state():
    cc = ChainConsumer()
    cc.add_chain(np.array([[1.0], [2.0]]), parameters=["a"], name="first")
    cc.add_chain(np.array([[3.0], [4.0]]), parameters=["b"], name="second")

    cc.remove_chain("first")
    assert len(cc.chains) == 1
    assert cc.chains[0].name == "second"
    assert cc._all_parameters == []
    assert cc._configured is False

    cc.remove_chain(0)
    assert cc.chains == []


def test_get_mcmc_chains_filters_out_marker_and_covariance_chains():
    cc = ChainConsumer()
    cc.add_chain(np.array([[1.0], [2.0]]), parameters=["a"], name="mcmc")
    cc.add_marker([1.0], parameters=["a"], name="marker")
    cc.add_covariance([0.0], [[1.0]], parameters=["a"], name="cov")

    mcmc_chains = cc.get_mcmc_chains()
    assert len(mcmc_chains) == 1
    assert mcmc_chains[0].name == "mcmc"


def test_get_chain_supports_index_name_and_chain_object():
    cc = ChainConsumer()
    cc.add_chain(np.array([[1.0], [2.0]]), parameters=["a"], name="first")
    cc.add_chain(np.array([[3.0], [4.0]]), parameters=["b"], name="second")

    assert cc._get_chain(0) == [0]
    assert cc._get_chain("second") == [1]
    assert cc._get_chain(cc.chains[0]) == [0]


def test_divide_chain_splits_each_walker_into_separate_chains():
    cc = ChainConsumer()
    data = np.arange(12.0).reshape(6, 2)
    weights = np.arange(6.0)
    cc.add_chain(data, parameters=["x", "y"], walkers=3, weights=weights, name="walk")

    divided = cc.divide_chain(0)
    assert isinstance(divided, ChainConsumer)
    assert len(divided.chains) == 3
    assert [c.name for c in divided.chains] == ["Chain 0", "Chain 1", "Chain 2"]
    assert all(c.parameters == ["x", "y"] for c in divided.chains)
    assert np.allclose(divided.chains[0].chain, data[:2])
    assert np.allclose(np.asarray(divided.chains[1].weights), weights[2:4])


def test_configure_sets_defaults_and_updates_chain_config():
    cc = ChainConsumer()
    cc.add_chain(np.array([[1.0], [2.0], [3.0]]), parameters=["x"], name="one")

    cc.configure(statistics="max", bins=10, shade=True, cloud=False, summary_area=0.6827)
    chain = cc.chains[0]

    assert cc._configured is True
    assert cc.config["summary_area"] == pytest.approx(0.6827)
    assert cc.config["global_point"] is True
    assert cc.config["legend_kwargs"]["loc"] == "upper right"
    assert chain.config["statistics"] == "max"
    assert chain.config["bins"] == 10
    assert chain.config["shade"] is True
    assert chain.config["cloud"] is False


def test_configure_truth_sets_defaults_and_preserves_overrides():
    cc = ChainConsumer()
    cc.configure_truth(color="red", linewidth=2, zorder=5)
    assert cc._configured_truth is True
    assert cc.config_truth["color"] == "red"
    assert cc.config_truth["linewidth"] == 2
    assert cc.config_truth["zorder"] == 5
    assert cc.config_truth["ls"] == "--"


def test_configure_rejects_invalid_statistics_and_summary_area():
    cc = ChainConsumer()
    cc.add_chain(np.array([[1.0], [2.0]]), parameters=["x"])

    with pytest.raises(AssertionError):
        cc.configure(statistics="not-a-summary")

    with pytest.raises(AssertionError):
        cc.configure(statistics="max", summary_area=1.5)


def test_deprecated_methods_delegate_to_underlying_components(capsys):
    cc = ChainConsumer()

    class Stub:
        def __init__(self):
            self.calls = []

        def plot(self, *args, **kwargs):
            self.calls.append(("plot", args, kwargs))
            return "plot-ok"

        def plot_walks(self, *args, **kwargs):
            self.calls.append(("plot_walks", args, kwargs))
            return "walks-ok"

        def get_latex_table(self, *args, **kwargs):
            self.calls.append(("latex", args, kwargs))
            return "latex-ok"

        def get_parameter_text(self, *args, **kwargs):
            self.calls.append(("text", args, kwargs))
            return "text-ok"

        def get_summary(self, *args, **kwargs):
            self.calls.append(("summary", args, kwargs))
            return "summary-ok"

        def get_correlations(self, *args, **kwargs):
            self.calls.append(("corr", args, kwargs))
            return "corr-ok"

        def get_correlation_table(self, *args, **kwargs):
            self.calls.append(("corr-table", args, kwargs))
            return "corr-table-ok"

        def get_covariance(self, *args, **kwargs):
            self.calls.append(("cov", args, kwargs))
            return "cov-ok"

        def get_covariance_table(self, *args, **kwargs):
            self.calls.append(("cov-table", args, kwargs))
            return "cov-table-ok"

        def gelman_rubin(self, *args, **kwargs):
            self.calls.append(("gr", args, kwargs))
            return "gr-ok"

        def geweke(self, *args, **kwargs):
            self.calls.append(("geweke", args, kwargs))
            return "geweke-ok"

        def aic(self):
            self.calls.append(("aic", (), {}))
            return "aic-ok"

        def bic(self):
            self.calls.append(("bic", (), {}))
            return "bic-ok"

        def dic(self):
            self.calls.append(("dic", (), {}))
            return "dic-ok"

        def comparison_table(self, *args, **kwargs):
            self.calls.append(("comparison-table", args, kwargs))
            return "comparison-table-ok"

    stub = Stub()
    cc.plotter = stub
    cc.analysis = stub
    cc.diagnostic = stub
    cc.comparison = stub

    assert cc.plot(1, x=2) == "plot-ok"
    assert cc.plot_walks(3, y=4) == "walks-ok"
    assert cc.get_latex_table() == "latex-ok"
    assert cc.get_parameter_text() == "text-ok"
    assert cc.get_summary() == "summary-ok"
    assert cc.get_correlations() == "corr-ok"
    assert cc.get_correlation_table() == "corr-table-ok"
    assert cc.get_covariance() == "cov-ok"
    assert cc.get_covariance_table() == "cov-table-ok"
    assert cc.diagnostic_gelman_rubin() == "gr-ok"
    assert cc.diagnostic_geweke() == "geweke-ok"
    assert cc.comparison_aic() == "aic-ok"
    assert cc.comparison_bic() == "bic-ok"
    assert cc.comparison_dic() == "dic-ok"
    assert cc.comparison_table() == "comparison-table-ok"

    out = capsys.readouterr().out
    assert "deprecated" in out.lower()
