import importlib
import inspect


def test_package_import_exposes_version():
    import torch_layer_normalization as tln

    assert hasattr(tln, "__version__")
    assert tln.__version__ == "0.11.0"


def test_package_import_exposes_layer_normalization():
    import torch_layer_normalization as tln

    assert hasattr(tln, "LayerNormalization")
    assert inspect.isclass(tln.LayerNormalization)


def test_layer_normalization_is_importable_from_package_directly():
    from torch_layer_normalization import LayerNormalization

    assert inspect.isclass(LayerNormalization)
    assert LayerNormalization.__name__ == "LayerNormalization"


def test_layer_normalization_class_is_same_object_as_package_attribute():
    import torch_layer_normalization as tln
    from torch_layer_normalization import LayerNormalization

    assert LayerNormalization is tln.LayerNormalization


def test_layer_normalization_is_same_as_submodule_symbol():
    import torch_layer_normalization as tln
    submodule = importlib.import_module("torch_layer_normalization.layer_normalization")

    assert hasattr(submodule, "LayerNormalization")
    assert tln.LayerNormalization is submodule.LayerNormalization


def test_package_dir_contains_layer_normalization():
    import torch_layer_normalization as tln

    names = dir(tln)
    assert "LayerNormalization" in names


def test_package_dir_contains_version():
    import torch_layer_normalization as tln

    names = dir(tln)
    assert "__version__" in names


def test_layer_normalization_class_module_name_looks_correct():
    import torch_layer_normalization as tln

    assert tln.LayerNormalization.__module__.startswith("torch_layer_normalization")


def test_layer_normalization_is_a_class_not_an_instance():
    import torch_layer_normalization as tln

    cls = tln.LayerNormalization
    assert inspect.isclass(cls)
    assert not isinstance(cls, cls)


def test_layer_normalization_class_has_a_docstring_or_none():
    import torch_layer_normalization as tln

    assert hasattr(tln.LayerNormalization, "__doc__")


def test_reimporting_package_preserves_version_and_class_identity():
    import torch_layer_normalization as tln1

    tln2 = importlib.reload(tln1)

    assert tln2.__version__ == "0.11.0"
    assert tln2.LayerNormalization is tln1.LayerNormalization


def test_from_import_star_exposes_layer_normalization_symbol():
    namespace = {}
    exec("from torch_layer_normalization import *", namespace)

    assert "LayerNormalization" in namespace
    assert inspect.isclass(namespace["LayerNormalization"])


def test_submodule_import_does_not_change_package_version():
    import torch_layer_normalization as tln

    _ = importlib.import_module("torch_layer_normalization.layer_normalization")

    assert tln.__version__ == "0.11.0"
