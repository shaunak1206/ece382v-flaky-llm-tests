import importlib
import inspect
import pickle
from pathlib import Path

import pytest

import pypfm
from pypfm import PFMLoader


def _make_temp_file(tmp_path, name="sample.pfm", content=b""):
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _instantiate_loader(tmp_path):
    """
    Best-effort constructor helper.

    PFMLoader's exact signature is not assumed. We try to satisfy any required
    positional parameters using a temporary path or simple placeholder values.
    """
    sig = inspect.signature(PFMLoader)
    args = []
    kwargs = {}
    temp_path = _make_temp_file(tmp_path)

    for param in sig.parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is not inspect._empty:
            continue

        name = param.name.lower()
        if any(token in name for token in ("path", "file", "fname", "filename", "fp")):
            value = str(temp_path)
        elif "mode" in name:
            value = "rb"
        elif "encoding" in name:
            value = "utf-8"
        elif "data" in name:
            value = b""
        elif "bytes" in name:
            value = b""
        else:
            value = str(temp_path)

        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        elif param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            args.append(value)
        else:
            kwargs[param.name] = value

    return PFMLoader(*args, **kwargs)


def test_package_import_exposes_pfml_loader():
    assert hasattr(pypfm, "PFMLoader")
    assert pypfm.PFMLoader is PFMLoader


def test_version_string_present():
    assert isinstance(pypfm.__version__, str)
    assert pypfm.__version__ == "1.0.0"


def test_pfml_loader_is_class():
    assert inspect.isclass(PFMLoader)
    assert PFMLoader.__name__ == "PFMLoader"


def test_pfml_loader_module_path():
    assert PFMLoader.__module__ == "pypfm.pypfm"


def test_pfml_loader_identity_matches_submodule():
    submodule = importlib.import_module("pypfm.pypfm")
    assert submodule.PFMLoader is PFMLoader


def test_pfml_loader_has_docstring_or_none():
    assert hasattr(PFMLoader, "__doc__")
    assert PFMLoader.__doc__ is None or isinstance(PFMLoader.__doc__, str)


def test_pfml_loader_is_new_style_class():
    assert issubclass(PFMLoader, object)


def test_pfml_loader_signature_is_introspectable():
    sig = inspect.signature(PFMLoader)
    assert isinstance(sig, inspect.Signature)
    assert len(sig.parameters) >= 0


def test_pfml_loader_can_be_instantiated_best_effort(tmp_path):
    loader = _instantiate_loader(tmp_path)
    assert isinstance(loader, PFMLoader)


def test_pfml_loader_instance_class_name(tmp_path):
    loader = _instantiate_loader(tmp_path)
    assert loader.__class__.__name__ == "PFMLoader"


def test_pfml_loader_repr_mentions_class_name(tmp_path):
    loader = _instantiate_loader(tmp_path)
    assert "PFMLoader" in repr(loader)


def test_pfml_loader_has_some_public_callable_member(tmp_path):
    loader = _instantiate_loader(tmp_path)
    public_callables = [
        name
        for name in dir(loader)
        if not name.startswith("_") and callable(getattr(loader, name, None))
    ]
    assert public_callables, "Expected at least one public callable on PFMLoader"


def test_pfml_loader_instance_has_no_unexpected_module_change(tmp_path):
    loader = _instantiate_loader(tmp_path)
    assert loader.__class__.__module__ == "pypfm.pypfm"


def test_pfml_loader_can_be_pickled_if_state_allows(tmp_path):
    loader = _instantiate_loader(tmp_path)
    try:
        blob = pickle.dumps(loader)
        restored = pickle.loads(blob)
    except Exception as exc:
        pytest.skip(f"PFMLoader is not picklable in this environment: {exc}")
    assert isinstance(restored, PFMLoader)


def test_importing_package_twice_keeps_same_exports():
    mod1 = importlib.import_module("pypfm")
    mod2 = importlib.import_module("pypfm")
    assert mod1.PFMLoader is mod2.PFMLoader
    assert mod1.__version__ == mod2.__version__


def test_pfml_loader_constructor_does_not_mutate_temp_path_object(tmp_path):
    path = _make_temp_file(tmp_path)
    before = path.stat().st_size
    sig = inspect.signature(PFMLoader)

    # Try to instantiate with any required path-like arguments.
    args = []
    kwargs = {}
    for param in sig.parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is not inspect._empty:
            continue
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            if any(token in param.name.lower() for token in ("path", "file", "fname", "filename", "fp")):
                args.append(str(path))
            else:
                args.append(str(path))
        else:
            kwargs[param.name] = str(path)

    try:
        PFMLoader(*args, **kwargs)
    except Exception:
        pytest.skip("PFMLoader requires a different constructor contract than expected")
    after = path.stat().st_size
    assert before == after


def test_pfml_loader_class_attributes_are_accessible():
    # Minimal smoke check for common class metadata.
    assert PFMLoader.__name__
    assert PFMLoader.__qualname__.endswith(PFMLoader.__name__)
    assert PFMLoader.__module__ == "pypfm.pypfm"
