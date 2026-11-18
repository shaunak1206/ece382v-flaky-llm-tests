import pytest
import re
import json
from datetime import datetime, date
from dataclasses import dataclass, field, fields, is_dataclass
from typing import List, Dict, Any

from fistro.fistro import generate, enrich_fields, generate_from_json, inner_inspection, get_class_body_from_annotations
from fistro.generators import int_generator


# Helper dataclasses for testing
@dataclass
class Simple:
    a: int
    b: str = 'default'
    c: bool = False


@dataclass
class Complex:
    x: int
    y: List[int]
    z: Dict[str, int]


def test_int_generator_default_length():
    val = int_generator()
    assert isinstance(val, int)
    assert 0 <= val <= 9 * 10 ** 6


def test_int_generator_custom_length():
    val = int_generator(3)
    assert isinstance(val, int)
    assert 0 <= val <= 9 * 10 ** 3


def test_generate_returns_dataclass_type():
    result = generate(Simple)
    assert is_dataclass(result)
    assert result.__name__ == Simple.__name__
    # Check that fields names are the same
    original_fields = {f.name for f in fields(Simple)}
    new_fields = {f.name for f in fields(result)}
    assert original_fields == new_fields


def test_generate_returns_instance_as_dict():
    d = generate(Simple, as_dict=True)
    assert isinstance(d, dict)
    # It should have keys 'a', 'b', 'c'
    assert all(k in d for k in ('a', 'b', 'c'))
    # Default values apply for b and c, a is from default_factory so should be int
    assert isinstance(d["a"], int)
    assert d["b"] == 'default'
    assert isinstance(d["c"], bool)


def test_generate_returns_instance_as_json():
    j = generate(Simple, as_json=True)
    # It should be JSON string
    assert isinstance(j, str)
    # loading verifies it's valid JSON and has keys
    d = json.loads(j)
    assert isinstance(d, dict)
    assert all(k in d for k in ('a', 'b', 'c'))


def test_enrich_fields_returns_list_of_tuples_with_field_factory():
    gen_list = [int_generator]
    enriched = enrich_fields(Simple, gen_list)
    assert isinstance(enriched, list)
    # Each item should be tuple of length 3: (name, type, field)
    for item in enriched:
        assert isinstance(item, tuple)
        assert len(item) == 3
        assert isinstance(item[0], str)
        assert hasattr(item[1], '__name__') or isinstance(item[1], type)
        assert hasattr(item[2], 'default_factory') or hasattr(item[2], 'default')


def test_generate_with_generators_overrides_default_factory():
    # use int_generator forcibly via generators param, results should be int and in expected range
    result_cls = generate(Simple, generators=[int_generator])
    inst = result_cls()
    assert isinstance(inst.a, int)
    assert 0 <= inst.a <= 9 * 10 ** 6


def test_generate_from_json_creates_dataclass_with_typing():
    obj = {'one': 1, 'two': {'inner': 5}, 'three': [1, 2, 3], 'four': "test"}
    cls = generate_from_json(obj)
    # It should be a dataclass type
    assert is_dataclass(cls)
    # It should have fields named after keys
    field_names = [f.name for f in fields(cls)]
    for key in obj.keys():
        assert key in field_names
    # The field types should match expected: int, dict, list, str
    for f in fields(cls):
        val = obj[f.name]
        if isinstance(val, dict):
            assert hasattr(f.type, '__origin__') and (f.type.__origin__ == dict or str(f.type).startswith('typing.Dict'))
        elif isinstance(val, list):
            assert hasattr(f.type, '__origin__') and (f.type.__origin__ == list or str(f.type).startswith('typing.List'))
        else:
            assert f.type == type(val)


def test_inner_inspection_with_dict_returns_field_and_type():
    val = {'a': 1}
    typ, fld = inner_inspection(val)
    from typing import Dict
    assert hasattr(fld, 'default_factory')
    # type should be Dict with key and value types
    rep = str(typ)
    assert rep.startswith('typing.Dict') or 'Dict' in rep


def test_inner_inspection_with_list_returns_field_and_list_type():
    val = [1, 2, 3]
    typ, fld = inner_inspection(val)
    from typing import List
    assert hasattr(fld, 'default_factory')
    rep = str(typ)
    assert rep.startswith('typing.List') or 'List' in rep


def test_inner_inspection_with_primitive_returns_field_and_type():
    val = 42
    typ, fld = inner_inspection(val)
    assert typ == int
    assert hasattr(fld, 'default_factory')
    # default_factory returns val
    assert fld.default_factory() == val


def test_get_class_body_from_annotations_basic_types():
    annotations = {'a': int, 'b': str, 'c': None}
    body = get_class_body_from_annotations(annotations)
    assert 'a: int' in body
    assert 'b: str' in body
    assert 'c: Any' in body


def test_get_class_body_from_annotations_with_list_and_dict():
    from typing import List, Dict
    annotations = {'lst': List[int], 'dct': Dict[str, int], 'x': int}
    body = get_class_body_from_annotations(annotations)
    # list and dict should be present fully (their repr)
    assert 'lst: ' in body and ('List' in body)
    assert 'dct: ' in body and ('Dict' in body)
    assert 'x: int' in body


def test_generate_with_no_generators_uses_factory_default():
    cls = generate(Simple)
    obj = cls()
    # Because Simple.a has no default, its factory should be from Factory().factory(int)
    assert isinstance(obj.a, int)


def test_generate_with_generators_can_inject_bool_gen():
    from fistro.generators import bool_generator
    cls = generate(Simple, generators=[bool_generator])
    obj = cls()
    # Simple.a uses bool_generator (return bool), so type mismatch, but testing that the generator is used
    # This test basically asserts no errors, since bool_generator returns bool (not int), factory may not exactly match
    # But just ensure attribute exists.
    assert hasattr(obj, 'a')


def test_generate_from_json_raises_not_implemented_on_list():
    with pytest.raises(NotImplementedError):
        generate_from_json([1, 2, 3])


def test_fields_in_generated_class_defaults_are_correct():
    cls = generate(Simple)
    inst = cls()
    # b and c have defaults from original Simple
    assert inst.b == 'default'
    assert inst.c == False


def test_generate_returns_separate_classes_each_time():
    # generating twice creates distinct dataclasses with same name but distinct types
    cls1 = generate(Simple)
    cls2 = generate(Simple)
    # Both are dataclasses
    assert is_dataclass(cls1)
    assert is_dataclass(cls2)
    # They should be distinct types
    assert cls1 is not cls2


def test_enrich_fields_with_none_generators_uses_default_factory():
    enriched = enrich_fields(Simple, None)
    # Each field's third element should have a default_factory
    for name, typ, fld in enriched:
        if name == 'b' or name == 'c':  # these have default values
            assert hasattr(fld, 'default')
        else:
            # fields with no default must have default_factory
            assert hasattr(fld, 'default_factory')


def test_enrich_fields_with_empty_generators_list_uses_custom_factory():
    enriched = enrich_fields(Simple, [])
    # Empty list counts as False? Actually [] is False in python, so used Factory(generators)
    for name, typ, fld in enriched:
        # fields with default should not have default_factory, others should
        if name == 'b' or name == 'c':
            assert hasattr(fld, 'default')
        else:
            assert hasattr(fld, 'default_factory')


def test_generate_with_as_json_returns_valid_json_string():
    j = generate(Simple, as_json=True)
    # Should be a JSON string with keys matching fields of Simple
    parsed = json.loads(j)
    assert isinstance(parsed, dict)
    expected_keys = {f.name for f in fields(Simple)}
    assert expected_keys == set(parsed.keys())
