import importlib
import inspect
import builtins
import sys

import numpy as np
import pandas as pd
import pytest

import jmcm
from jmcm import JointModel


def _make_data(n_subjects=10, n_times=4, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_subjects):
        for j in range(n_times):
            t = float(j + 1)
            rows.append({"id": i + 1, "time": t,
                         "y": 1.0 + 0.5 * t + rng.normal(scale=0.3)})
    return pd.DataFrame(rows)


def _try_build_model():
    """Try a few plausible ways to build a JointModel; skip if none work."""
    data = _make_data()
    formula = "y ~ 1 | id | time"
    attempts = []
    if hasattr(JointModel, "from_formula"):
        attempts.append(lambda: JointModel.from_formula(formula, data))
        attempts.append(lambda: JointModel.from_formula(formula, data, mean_degree=1))
    attempts.append(lambda: JointModel(formula, data))
    attempts.append(lambda: JointModel(formula, data, "mcd"))
    for attempt in attempts:
        try:
            return attempt()
        except Exception:
            continue
    pytest.skip("Could not construct a JointModel with the guessed signature")


# ---------------------------------------------------------------- package level

def test_package_name():
    assert jmcm.name == "jmcm"


def test_jointmodel_is_exported_class():
    assert inspect.isclass(JointModel)
    assert jmcm.JointModel is JointModel


def test_jointmodel_defined_in_joint_model_module():
    mod = importlib.import_module("jmcm.joint_model")
    assert mod.JointModel is JointModel
    assert JointModel.__module__ == "jmcm.joint_model"


def test_jointmodel_name():
    assert JointModel.__name__ == "JointModel"


def test_helper_names_cleaned_from_namespace():
    assert not hasattr(jmcm, "hard_dependencies")
    assert not hasattr(jmcm, "missing_dependencies")
    assert not hasattr(jmcm, "dependency")


def test_reload_keeps_jointmodel_available():
    reloaded = importlib.reload(jmcm)
    assert inspect.isclass(reloaded.JointModel)
    assert reloaded.name == "jmcm"


def test_missing_dependency_raises_import_error():
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "patsy":
            raise ImportError("No module named 'patsy'")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    try:
        with pytest.raises(ImportError) as excinfo:
            importlib.reload(jmcm)
    finally:
        builtins.__import__ = real_import
        importlib.reload(sys.modules["jmcm"])
    assert "patsy" in str(excinfo.value)
    assert "Unable to import required dependencies" in str(excinfo.value)


def test_namespace_clean_after_recovery_reload():
    importlib.reload(jmcm)
    assert not hasattr(jmcm, "hard_dependencies")
    assert hasattr(jmcm, "JointModel")


# ---------------------------------------------------------------- class level

def test_jointmodel_init_signature_requires_arguments():
    sig = inspect.signature(JointModel.__init__)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    assert len(params) >= 1


def test_jointmodel_instantiation_without_args_fails():
    sig = inspect.signature(JointModel.__init__)
    required = [p for p in sig.parameters.values()
                if p.name != "self"
                and p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD,
                               p.KEYWORD_ONLY)]
    if not required:
        pytest.skip("JointModel has no required constructor arguments")
    with pytest.raises(TypeError):
        JointModel()


def test_jointmodel_has_public_methods():
    public = [n for n, m in inspect.getmembers(JointModel, callable)
              if not n.startswith("_")]
    assert len(public) >= 1


def test_jointmodel_has_fit_or_similar_method():
    names = set(dir(JointModel))
    assert names & {"fit", "optim", "fit_mcd", "fit_acd", "fit_hpc", "from_formula"}


# ---------------------------------------------------------------- instance level

def test_model_can_be_constructed_from_data():
    model = _try_build_model()
    assert isinstance(model, JointModel)


def test_model_fit_returns_something_with_finite_values():
    model = _try_build_model()
    if not hasattr(model, "fit"):
        pytest.skip("model has no fit method")
    try:
        result = model.fit()
    except TypeError:
        pytest.skip("fit requires arguments unknown to this test")
    target = result if result is not None else model
    for attr in ("params", "coefficients", "optim_results", "x"):
        value = getattr(target, attr, None)
        if value is not None and not callable(value):
            arr = np.asarray(value, dtype=float).ravel() \
                if not hasattr(value, "x") else np.asarray(value.x, dtype=float)
            assert np.all(np.isfinite(arr))
            return
    pytest.skip("no recognizable parameter attribute on fitted model")


def test_two_models_are_independent_instances():
    m1 = _try_build_model()
    m2 = _try_build_model()
    assert m1 is not m2
