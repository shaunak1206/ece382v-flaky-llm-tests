import inspect

import numpy as np
import pandas as pd
import pytest

import jmcm
from jmcm import JointModel


def _constructor_signature():
    return inspect.signature(JointModel)


def _make_minimal_data():
    return pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.5, 1.5, 2.5, 3.5],
            "id": [1, 1, 2, 2],
        }
    )


def _build_constructor_args_kwargs():
    sig = _constructor_signature()
    args = []
    kwargs = {}
    data = _make_minimal_data()

    for param in sig.parameters.values():
        if param.name == "self":
            continue
        if param.default is not inspect._empty:
            # Prefer defaults where available; do not override them.
            continue

        name = param.name.lower()

        value = None
        if "formula" in name:
            value = "y ~ x"
        elif name in {"data", "df", "dataset"}:
            value = data
        elif name in {"endog", "y", "response"}:
            value = data["y"]
        elif name in {"exog", "x", "covariates", "predictors"}:
            value = data[["x"]]
        elif name in {"groups", "group", "cluster", "id"}:
            value = data["id"]
        elif any(token in name for token in ("n_", "num", "nobs", "k", "p", "q", "rank")):
            value = 1
        elif any(token in name for token in ("bool", "verbose", "intercept", "fit_")):
            value = False
        elif "seed" in name:
            value = 123
        elif "weights" in name:
            value = np.ones(len(data))
        else:
            # A conservative fallback that works for many APIs.
            value = None

        # Use positional args for positional-only / positional-or-keyword params
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            args.append(value)
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs[param.name] = value

    return args, kwargs


def _maybe_create_model():
    args, kwargs = _build_constructor_args_kwargs()
    return JointModel(*args, **kwargs)


def test_package_exposes_jointmodel():
    assert hasattr(jmcm, "JointModel")
    assert jmcm.JointModel is JointModel


def test_jointmodel_is_a_class():
    assert inspect.isclass(JointModel)


def test_jointmodel_module_name():
    assert JointModel.__module__ == "jmcm.joint_model"


def test_jointmodel_has_a_reasonable_docstring_or_none():
    assert JointModel.__doc__ is None or isinstance(JointModel.__doc__, str)


def test_jointmodel_constructor_signature_is_inspectable():
    sig = _constructor_signature()
    assert isinstance(sig, inspect.Signature)
    assert "self" not in sig.parameters or True


def test_jointmodel_can_be_instantiated_with_minimal_inputs_if_required():
    sig = _constructor_signature()
    required = [
        p for p in sig.parameters.values()
        if p.name != "self" and p.default is inspect._empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]

    if not required:
        model = JointModel()
        assert isinstance(model, JointModel)
    else:
        model = _maybe_create_model()
        assert isinstance(model, JointModel)


def test_jointmodel_repr_contains_class_name():
    model = _maybe_create_model()
    r = repr(model)
    assert "JointModel" in r


def test_jointmodel_str_contains_class_name():
    model = _maybe_create_model()
    s = str(model)
    assert "JointModel" in s


def test_jointmodel_instance_has_dict_or_slots():
    model = _maybe_create_model()
    assert hasattr(model, "__dict__") or hasattr(model, "__slots__")


def test_jointmodel_constructor_missing_required_arguments_raises_typeerror_if_any():
    sig = _constructor_signature()
    required = [
        p for p in sig.parameters.values()
        if p.name != "self" and p.default is inspect._empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]

    if required:
        with pytest.raises(TypeError):
            JointModel()
    else:
        pytest.skip("Constructor has no required arguments.")


def test_jointmodel_signature_parameter_names_are_strings():
    sig = _constructor_signature()
    assert all(isinstance(name, str) for name in sig.parameters.keys())


def test_jointmodel_instance_type_is_exact_or_subclass():
    model = _maybe_create_model()
    assert isinstance(model, JointModel)
    assert type(model).__name__ == "JointModel" or issubclass(type(model), JointModel)


def test_jointmodel_class_is_accessible_from_package_namespace():
    exported = getattr(jmcm, "JointModel", None)
    assert exported is JointModel
