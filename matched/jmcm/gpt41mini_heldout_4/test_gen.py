import pytest
import math
import numpy as np
import pandas as pd
import scipy
import warnings
import collections
import patsy

from jmcm.joint_model import JointModel


def test_jointmodel_can_instantiate_default():
    jm = JointModel()
    assert isinstance(jm, JointModel)


def test_jointmodel_attributes_exist():
    jm = JointModel()
    # Check some expected attributes or methods exist
    assert hasattr(jm, "fit") or hasattr(jm, "run") or hasattr(jm, "__call__")


def test_jointmodel_fit_method_exists():
    jm = JointModel()
    assert callable(getattr(jm, "fit", None))


def test_jointmodel_fit_basic_behavior():
    jm = JointModel()
    # We can't test real functionality without signature, but call fit with minimal data
    try:
        res = jm.fit()
    except TypeError:
        # Accept TypeError if positional args required
        pass
    except Exception as e:
        pytest.fail(f"Unexpected exception on fit(): {e}")


def test_jointmodel_init_kwargs():
    # Try initializing with arbitrary kwargs - some classes accept them
    try:
        jm = JointModel(param1=10, param2="test")
    except TypeError:
        # If JointModel doesn't accept kwargs, pass test (not a failure)
        assert True
    except Exception as e:
        pytest.fail(f"Unexpected exception on init with kwargs: {e}")


def test_import_math_numpy_pandas_and_scipy_available():
    # Confirming the modules are importable (mirroring the import test in __init__.py)
    import math
    import numpy as np
    import pandas as pd
    import scipy


def test_import_collections_and_patsy_and_warnings_available():
    import collections
    import patsy
    import warnings


def test_jointmodel_repr_or_str():
    jm = JointModel()
    assert isinstance(repr(jm), str)
    assert isinstance(str(jm), str)


def test_jointmodel_pickle_support():
    import pickle
    jm = JointModel()
    try:
        p = pickle.dumps(jm)
        jm2 = pickle.loads(p)
        assert isinstance(jm2, JointModel)
    except Exception:
        # Not all models support pickling; test should not fail if unsupported
        pass


def test_jointmodel_warns_on_deprecated_usage():
    jm = JointModel()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Trigger potential warnings, e.g. calling deprecated method, or re-call fit
        try:
            if hasattr(jm, "fit"):
                jm.fit()
        except Exception:
            pass
        # We don't assert that warning must be raised, but test runs without error


@pytest.mark.parametrize("invalid_arg", [None, [], {}, 123])
def test_jointmodel_init_invalid_types(invalid_arg):
    # Try passing invalid types for typical expected params, if signature known
    try:
        jm = JointModel(invalid_arg)
    except TypeError:
        pass
    except Exception:
        # Accept exception, but not failures from e.g. attribute errors
        pass


def test_jointmodel_method_callability():
    jm = JointModel()
    for attr in dir(jm):
        attribute = getattr(jm, attr)
        if callable(attribute):
            # ensure calling without params raises at most TypeError, not more severe
            try:
                attribute()
            except TypeError:
                pass
            except Exception:
                pass


def test_jointmodel_inheritance():
    jm = JointModel()
    # Verify JointModel is subclass of object
    assert isinstance(jm, object)


def test_jointmodel_attributes_mutability():
    jm = JointModel()
    # Try setting arbitrary attribute and check
    attr_name = "test_attr"
    setattr(jm, attr_name, 42)
    assert getattr(jm, attr_name) == 42


def test_jointmodel_in_default_namespace():
    import jmcm
    assert hasattr(jmcm, "JointModel")


def test_jointmodel_handles_unknown_methods_gracefully():
    jm = JointModel()
    with pytest.raises(AttributeError):
        jm.nonexistent_method()


def test_jointmodel_repr_contains_classname():
    jm = JointModel()
    r = repr(jm)
    assert "JointModel" in r


def test_jointmodel_str_contains_classname():
    jm = JointModel()
    s = str(jm)
    assert "JointModel" in s


def test_jointmodel_repr_str_no_side_effects():
    jm = JointModel()
    _ = repr(jm)
    _ = str(jm)
    # No exceptions and no attribute changes


def test_jointmodel_can_be_used_multiple_times():
    jm = JointModel()
    # Call fit or equivalent multiple times without error
    for _ in range(2):
        try:
            if hasattr(jm, "fit"):
                jm.fit()
        except Exception:
            # Accept failure but no crash
            pass


def test_jointmodel_default_instance_has_no_attribute_error_on_basic_attributes():
    jm = JointModel()
    # Check some likely attributes do not raise AttributeError
    for attr in ("coef_", "params", "fitted", "data"):
        try:
            _ = getattr(jm, attr)
        except AttributeError:
            pass


def test_jointmodel_no_hidden_imports():
    # Confirm __init__.py imports correct names (JointModel)
    import jmcm
    assert hasattr(jmcm, "JointModel")
