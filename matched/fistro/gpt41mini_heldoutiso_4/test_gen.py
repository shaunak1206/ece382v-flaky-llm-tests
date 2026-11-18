import pytest
from dataclasses import dataclass, field, fields, MISSING, asdict, is_dataclass
from typing import List, Dict, Any

from fistro.fistro import generate, enrich_fields, generate_from_json, inner_inspection, get_class_body_from_annotations
from fistro.generators import int_generator


@dataclass
class Simple:
    a: int
    b: str
    c: bool = True


@dataclass
class Mixed:
    x: int
    y: List[int]
    z: Dict[str, int]
    w: float = 1.23


def test_generate_returns_dataclass_type():
    result = generate(Simple)
    assert is_dataclass(result)
    assert result.__name__ == "Simple"


def test_generate_instance_fields_types():
    gen_cls = generate(Simple)
    flds = {f.name: f.type for f in fields(gen_cls)}
    assert flds['a'] == int
    assert flds['b'] == str
    assert flds['c'] == bool


def test_generate_with_as_dict_returns_dict_with_correct_keys():
    result = generate(Simple, as_dict=True)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"a", "b", "c"}


def test_generate_with_as_json_returns_json_string():
    result = generate(Simple, as_json=True)
    assert isinstance(result, str)
    import json
    loaded = json.loads(result)
    assert set(loaded.keys()) == {"a", "b", "c"}


def test_enrich_fields_returns_list_of_tuples():
    enriched = enrich_fields(Simple, generators=None)
    assert isinstance(enriched, list)
    assert all(isinstance(tpl, tuple) and len(tpl) == 3 for tpl in enriched)
    # Check the third element is a dataclasses.Field instance
    for _, _, fld in enriched:
        assert hasattr(fld, "default_factory") or fld.default is not MISSING


def test_enrich_fields_with_generators_affects_factory_choice():
    # Provide a dummy generator list that doesn't override Factory internally to see if no error
    enriched_default = enrich_fields(Simple, generators=None)
    enriched_with_gen = enrich_fields(Simple, generators=[int_generator])
    assert len(enriched_default) == len(enriched_with_gen)

    # The default_factory is callable
    for _, _, fld in enriched_with_gen:
        assert callable(getattr(fld, 'default_factory', None)) or fld.default is not MISSING


def test_generate_from_json_creates_dataclass_with_correct_fields():
    sample_json = {"foo": 123, "bar": {"sub": "val"}}
    gen_cls = generate_from_json(sample_json)
    assert is_dataclass(gen_cls)
    gen_fields = fields(gen_cls)
    # Should have fields "foo" and "bar"
    names = {f.name for f in gen_fields}
    assert "foo" in names and "bar" in names


def test_generate_from_json_raises_notimplemented_for_list():
    with pytest.raises(NotImplementedError):
        generate_from_json([1, 2, 3])


def test_inner_inspection_returns_field_and_type_for_dict():
    sample_dict = {"key": 1}
    type_, fld = inner_inspection(sample_dict)
    from typing import Dict
    assert hasattr(fld, "default_factory")
    # type_ should be typing.Dict of key and value types
    assert str(type_).startswith('typing.Dict') or str(type_).startswith('typing_extensions.Dict')


def test_inner_inspection_returns_field_and_type_for_list():
    sample_list = [1, 2, 3]
    type_, fld = inner_inspection(sample_list)
    from typing import List
    assert hasattr(fld, "default_factory")
    assert str(type_).startswith('typing.List') or str(type_).startswith('typing_extensions.List')


def test_inner_inspection_returns_field_and_type_for_scalar():
    sample_int = 42
    type_, fld = inner_inspection(sample_int)
    assert hasattr(fld, "default_factory")
    assert type_ is int


def test_get_class_body_from_annotations_with_simple_types():
    ann = {"a": int, "b": str, "c": float}
    result = get_class_body_from_annotations(ann)
    assert "a: int" in result
    assert "b: str" in result
    assert "c: float" in result


def test_get_class_body_from_annotations_with_dict_and_list_types():
    from typing import Dict, List
    ann = {"x": Dict[str, int], "y": List[int]}
    result = get_class_body_from_annotations(ann)
    assert "x: " in result and "Dict" in result
    assert "y: " in result and "List" in result


def test_get_class_body_from_annotations_with_none_type():
    ann = {"foo": None}
    result = get_class_body_from_annotations(ann)
    assert "foo: Any" in result


def test_generate_creates_distinct_classes_each_call():
    cls1 = generate(Simple)
    cls2 = generate(Simple)
    assert cls1 is not cls2
    # Instances from each are distinct types
    assert type(cls1()) is cls1
    assert type(cls2()) is cls2


def test_generate_with_default_generators_affects_field_factories():
    gen_list = [int_generator]
    cls = generate(Simple, generators=gen_list)
    inst = cls()
    # The fields have default values of correct types
    assert isinstance(inst.a, int)
    assert isinstance(inst.b, str)
    assert isinstance(inst.c, bool)


def test_generate_with_specified_field_with_default_value_respects_default():
    @dataclass
    class WithDefault:
        x: int = 10
        y: str = "hello"

    cls = generate(WithDefault)
    inst = cls()
    assert inst.x == 10
    assert inst.y == "hello"


def test_generate_with_complex_nested_fields():
    cls = generate(Mixed)
    inst = cls()
    # Check types of default generated values
    assert isinstance(inst.x, int)
    assert isinstance(inst.y, list)
    assert isinstance(inst.z, dict)
    assert isinstance(inst.w, float)
    # If list is non-empty, its contents are int (most likely)
    if inst.y:
        assert isinstance(inst.y[0], int)


def test_inner_inspection_field_default_factory_produces_same_value():
    test_val = {"a": 123}
    _, fld = inner_inspection(test_val)
    val1 = fld.default_factory()
    val2 = fld.default_factory()
    # Because lambda returns the same object literal, should be equal but not necessarily identical
    assert val1 == test_val
    assert val2 == test_val


def test_enrich_fields_respects_fields_with_defaults():
    @dataclass
    class WithDefaults:
        a: int = 5
        b: str = "hi"
        c: bool = True

    enriched = enrich_fields(WithDefaults, generators=None)
    for name, _typ, fld in enriched:
        if name in ("a", "b", "c"):
            assert fld.default != MISSING


def test_enrich_fields_respects_fields_without_defaults_use_factory():
    @dataclass
    class WithoutDefaults:
        a: int
        b: str

    enriched = enrich_fields(WithoutDefaults, generators=None)
    for name, _typ, fld in enriched:
        if name in ("a", "b"):
            assert callable(getattr(fld, 'default_factory', None))
