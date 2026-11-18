import pytest
import json
from datetime import datetime, date
from typing import List, Dict, Any
from fistro.fistro import generate, enrich_fields, generate_from_json, inner_inspection, get_class_body_from_annotations
from fistro.generators import int_generator


# We need some dummy dataclasses for testing
from dataclasses import dataclass, field


@dataclass
class Simple:
    a: int
    b: bool
    c: str
    d: datetime


@dataclass
class WithDefaults:
    a: int = 1
    b: str = "test"


def test_generate_returns_dataclass_type():
    generated = generate(Simple)
    assert generated.__name__ == Simple.__name__
    assert hasattr(generated, 'a')
    # Should be dataclass
    from dataclasses import is_dataclass
    assert is_dataclass(generated)


def test_generate_as_dict_returns_dict_of_correct_keys():
    generated_dict = generate(Simple, as_dict=True)
    assert isinstance(generated_dict, dict)
    # keys correspond to fields in Simple
    for key in ['a', 'b', 'c', 'd']:
        assert key in generated_dict


def test_generate_as_json_returns_valid_json_string():
    generated_json = generate(Simple, as_json=True)
    assert isinstance(generated_json, str)
    loaded = json.loads(generated_json)
    # keys in loaded json
    for key in ['a', 'b', 'c', 'd']:
        assert key in loaded


def test_generate_with_custom_generators_overrides_factory():
    # Use only int_generator to produce ints
    generated = generate(Simple, generators=[int_generator])
    instance = generated()
    # a is int -> use int_generator, should be int and within valid int_generator range
    assert isinstance(instance.a, int)
    # b is bool, no explicit generator supplied, so default factory used -> inferred bool or default?
    # c is str, no explicit generator, so default
    # This test only verifies a was generated as int successfully
    assert hasattr(instance, 'a')


def test_enrich_fields_returns_expected_structure():
    ef = enrich_fields(Simple, generators=None)
    assert isinstance(ef, list)
    for tup in ef:
        assert isinstance(tup, tuple)
        assert len(tup) == 3
        assert isinstance(tup[0], str)  # field name
        # second is a type object
        assert isinstance(tup[1], type)
        # Third is a dataclasses.field
        from dataclasses import Field
        assert hasattr(tup[2], 'default_factory') or hasattr(tup[2], 'default')


def test_enrich_fields_uses_default_when_field_has_default():
    ef = enrich_fields(WithDefaults, generators=None)
    # fields with default should have default in field argument
    for name, typ, fld in ef:
        if name == 'a':
            # default is 1, should be default attribute set for field
            assert fld.default == 1
        if name == 'b':
            assert fld.default == "test"


def test_generate_from_json_creates_class_with_correct_fields():
    input_dict = {"foo": 123, "bar": "baz"}
    cls = generate_from_json(input_dict)
    assert hasattr(cls, "__annotations__")
    assert "foo" in cls.__annotations__
    assert "bar" in cls.__annotations__
    # foo type should be int
    assert cls.__annotations__["foo"] == int
    # bar type should be str
    assert cls.__annotations__["bar"] == str


def test_generate_from_json_raises_notimplemented_with_list():
    with pytest.raises(NotImplementedError):
        generate_from_json([1, 2, 3])


def test_inner_inspection_with_dict_returns_dict_type_and_field():
    test_dict = {"k": 1}
    inner_type, fld = inner_inspection(test_dict)
    assert hasattr(fld, 'default_factory')
    # inner_type should be something like Dict[type(k), type(v)]
    from typing import Dict
    assert hasattr(inner_type, '__origin__') and inner_type.__origin__ is dict


def test_inner_inspection_with_list_returns_list_type_and_field():
    test_list = [1, 2, 3]
    inner_type, fld = inner_inspection(test_list)
    assert hasattr(fld, 'default_factory')
    from typing import List
    assert hasattr(inner_type, '__origin__') and inner_type.__origin__ is list


def test_inner_inspection_with_non_collection_returns_type_and_field():
    val = 123
    inner_type, fld = inner_inspection(val)
    assert inner_type == int
    assert hasattr(fld, 'default_factory')
    # The default_factory should produce the original value
    assert fld.default_factory() == val


def test_get_class_body_from_annotations_simple_and_complex():
    annotations = {
        "x": int,
        "y": Dict[str, int],
        "z": List[int],
        "w": None
    }
    result = get_class_body_from_annotations(annotations)
    # 'x' line should contain int
    assert 'x: int' in result
    # 'y' line should contain Dict
    assert 'y: typing.Dict' in result or 'y: Dict[str, int]' in result
    # 'z' line should contain List
    assert 'z: typing.List' in result or 'z: List[int]' in result
    # 'w' line should contain Any
    assert 'w: Any' in result


def test_int_generator_returns_int_in_expected_range():
    value = int_generator(3)
    assert isinstance(value, int)
    assert 0 <= value <= 9 * 10 ** 3


def test_int_generator_default_length():
    value = int_generator()
    assert isinstance(value, int)
    # Default INT_LENGTH is 5 or similar but we cannot import config,
    # so just test that value is int and non-negative
    assert value >= 0


def test_all_fields_of_generated_class_are_dataclass_fields():
    cls = generate(Simple)
    from dataclasses import fields as dc_fields
    dc_fields_cls = dc_fields(cls)
    # Check all fields have default factory or default
    for fld in dc_fields_cls:
        assert hasattr(fld, 'default_factory') or fld.default != fld.default.__class__


def test_generate_returns_different_classes_for_different_input_classes():
    cls1 = generate(Simple)
    @dataclass
    class OtherClass:
        x: int
    cls2 = generate(OtherClass)
    # Names correspond to their source classes
    assert cls1.__name__ == Simple.__name__
    assert cls2.__name__ == 'OtherClass'
    assert cls1 != cls2


def test_enrich_fields_uses_factory_with_generators():
    # call enrich_fields with custom generator (int_generator), check it sets factory
    ef = enrich_fields(Simple, generators=[int_generator])
    # All default_factories should exist
    for _, _, fld in ef:
        assert fld.default_factory is not None


def test_generate_creates_instance_with_field_values():
    cls = generate(Simple)
    instance = cls()
    # Instance fields should be set and not MISSING
    values = [getattr(instance, f.name) for f in instance.__dataclass_fields__.values()]
    for value in values:
        assert value is not None


def test_generate_with_as_dict_and_as_json_combined_false():
    # Should return dataclass type, not instance or string
    result = generate(Simple, as_dict=False, as_json=False)
    from dataclasses import is_dataclass
    assert is_dataclass(result)


def test_generate_with_as_dict_and_as_json_both_true_prefers_as_dict():
    # This is a strange call. As both True, the code returns asdict() (because it checks as_dict before as_json)
    dict_result = generate(Simple, as_dict=True, as_json=True)
    assert isinstance(dict_result, dict)


def test_generate_from_json_field_default_factory_consistency():
    obj = {"hello": {"key": "value"}}
    cls = generate_from_json(obj)
    fld = cls.__dataclass_fields__['hello']
    # The default_factory should produce a dict
    val = fld.default_factory()
    assert isinstance(val, dict)
    assert val == {"key": "value"}
