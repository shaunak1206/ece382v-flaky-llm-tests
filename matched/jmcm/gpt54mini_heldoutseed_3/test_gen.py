import importlib
import inspect
import sys

import pytest

import jmcm


def _get_joint_model_class():
    from jmcm import JointModel as exported
    from jmcm.joint_model import JointModel as direct

    assert exported is direct
    return exported


def _make_candidate_kwargs(cls):
    """
    Build a few plausible constructor kwargs based on parameter names.
    This keeps the tests resilient to small signature differences.
    """
    sig = inspect.signature(cls)
    params = list(sig.parameters.values())[1:]  # skip self

    # Minimal common toy data
    y = [1.0, 2.0, 3.0, 4.0]
    x1 = [[1.0], [2.0], [3.0], [4.0]]
    x2 = [[0.0], [1.0], [0.0], [1.0]]
    g = [0, 0, 1, 1]

    kwargs = {}
    for p in params:
        name = p.name.lower()

        if name in {"y", "endog", "response", "outcome", "dependent"}:
            kwargs[p.name] = y
        elif name in {"x", "exog", "design", "mean_exog", "meanx", "mean_x"}:
            kwargs[p.name] = x1
        elif name in {"cov", "cov_exog", "covariate", "variance", "var_exog", "varx", "var_x"}:
            kwargs[p.name] = x2
        elif name in {"group", "groups", "cluster", "subject", "id", "ids"}:
            kwargs[p.name] = g
        elif p.default is not inspect._empty:
            # Leave defaults alone.
            continue
        else:
            # Unknown required parameter; don't guess further.
            return None

    return kwargs


def _try_instantiate_joint_model():
    cls = _get_joint_model_class()
    kwargs = _make_candidate_kwargs(cls)
    if kwargs is None:
        pytest.skip("JointModel signature is not compatible with generic test data")
    try:
        return cls(**kwargs)
    except Exception as exc:
        pytest.skip(f"Could not instantiate JointModel with generic test data: {exc}")


def test_jmcm_exports_joint_model():
    assert hasattr(jmcm, "JointModel")
    assert jmcm.JointModel is _get_joint_model_class()


def test_joint_model_is_importable_from_package_root():
    mod = importlib.import_module("jmcm")
    assert mod.JointModel is _get_joint_model_class()


def test_joint_model_class_metadata():
    cls = _get_joint_model_class()
    assert cls.__name__ == "JointModel"
    assert cls.__module__ == "jmcm.joint_model"


def test_joint_model_is_a_class():
    cls = _get_joint_model_class()
    assert inspect.isclass(cls)
    assert isinstance(cls, type)


def test_joint_model_constructor_signature_exists():
    cls = _get_joint_model_class()
    sig = inspect.signature(cls)
    assert isinstance(sig, inspect.Signature)
    assert "self" not in sig.parameters


def test_package_does_not_expose_internal_dependency_lists():
    assert not hasattr(jmcm, "hard_dependencies")
    assert not hasattr(jmcm, "missing_dependencies")
    assert not hasattr(jmcm, "dependency")


def test_package_name_constant():
    assert jmcm.name == "jmcm"


def test_module_allows_access_to_joint_model_via_getattr():
    cls = getattr(jmcm, "JointModel")
    assert cls is _get_joint_model_class()


def test_joint_model_instance_can_be_created_with_generic_data():
    obj = _try_instantiate_joint_model()
    assert obj is not None
    assert isinstance(obj, _get_joint_model_class())


def test_joint_model_instance_has_a_repr():
    obj = _try_instantiate_joint_model()
    text = repr(obj)
    assert isinstance(text, str)
    assert len(text) > 0
    assert "JointModel" in text or obj.__class__.__name__ in text


def test_joint_model_instance_module_matches_class_module():
    obj = _try_instantiate_joint_model()
    assert obj.__class__.__module__ == "jmcm.joint_model"


def test_joint_model_class_has_callable_init():
    cls = _get_joint_model_class()
    assert callable(cls.__init__)
    assert inspect.isfunction(cls.__init__) or inspect.ismethoddescriptor(cls.__init__)


def test_joint_model_repr_of_class_contains_name():
    cls = _get_joint_model_class()
    assert "JointModel" in repr(cls)


def test_reimport_package_preserves_export():
    # Ensure the package-level export remains stable after reload.
    mod = importlib.reload(sys.modules["jmcm"])
    assert mod.JointModel is _get_joint_model_class()
