import random
import pytest
from datetime import datetime, date

from fistro.fistro import generate, enrich_fields, generate_from_json, inner_inspection, get_class_body_from_annotations
from fistro.generators import int_generator


def test_int_generator_seeded_returns_reproducible_int():
    random.seed(12345)
    val1 = int_generator(length=3)
    random.seed(12345)
    val2 = int_generator(length=3)
    assert val1 == val2
    assert isinstance(val1, int)
    # upper bound check
    assert 0 <= val1 <= 9 * 10 ** 3


def test_generate_returns_dataclass_instance_by_default():
    class Sample:
        a: int
        b: str

    # No generators passed
    obj = generate(Sample)
    assert obj.__class__.__name__ == Sample.__name__
    # Check fields exist and default factory produces expected types
    assert hasattr(obj, "a")
    assert hasattr(obj, "b")

    # Because default factory used Factory().factory(type) which is unknown here, just check types:
    assert isinstance(obj.a, int)
    assert isinstance(obj.b, str)

def test_generate_as_dict_and_as_json():
    class Sample:
        a: int
        b: bool

    # Seed random for reproducibility (Factory likely uses random too)
    random.seed(42)

    d = generate(Sample, as_dict=True)
    assert isinstance(d, dict)
    assert "a" in d and "b" in d
    assert isinstance(d['a'], int)
    assert isinstance(d['b'], bool)

    json_str = generate(Sample, as_json=True)
    assert isinstance(json_str, str)
    import json
    d2 = json.loads(json_str)
    assert set(d2.keys()) == {'a', 'b'}
    assert isinstance(d2['a'], int)
    assert isinstance(d2['b'], bool)

def test_enrich_fields_with_and_without_generators_affects_default_factories():
    from fistro.factory import Factory

    class Sample:
        x: int
        y: str

    # Without generators
    fields_without_gen = enrich_fields(Sample, None)
    assert all(len(f) == 3 for f in fields_without_gen)
    # The field default_factory should come from Factory().factory(type)
    for name, typ, fld in fields_without_gen:
        # default should be a dataclasses.Field object with default_factory set
        assert hasattr(fld, 'default_factory') and callable(fld.default_factory)

    # With generators: pass a custom generator list
    custom_gen = [lambda t: 123 if t == int else "test" if t == str else None]

    # Patch Factory to record calls to factory()
    # Using monkeypatch is better but here just test normal call
    fields_with_gen = enrich_fields(Sample, custom_gen)
    for name, typ, fld in fields_with_gen:
        assert hasattr(fld, 'default_factory') and callable(fld.default_factory)

def test_generate_from_json_creates_dataclass_with_keys_and_types():
    data = {"name": "test", "age": 42}
    cls = generate_from_json(data)
    cls_name = cls.__name__
    # Name should be a valid str (returned by valid_id)
    assert isinstance(cls_name, str) and len(cls_name) > 0
    # Fields should be 'name' and 'age'
    field_names = [f.name for f in getattr(cls, '__dataclass_fields__').values()]
    assert set(field_names) == set(data.keys())

    # For each field type in class, the type should be typing.List or Dict or type of value
    from typing import List, Dict
    # We cannot directly compare generic types, check name start
    for key in data:
        f = cls.__dataclass_fields__[key]
        # The field's type can be List or Dict if from inner_inspection, else type
        typ = f.type
        # For "name": str, "age": int
        assert (typ == type(data[key])) or (hasattr(typ, '__origin__') and (typ.__origin__ in (list, dict)))

def test_generate_from_json_raises_for_non_dict_input():
    with pytest.raises(NotImplementedError):
        generate_from_json([1, 2, 3])

def test_inner_inspection_returns_correct_type_and_field_for_dict_list_and_scalar():
    d = {"k": 42}
    typ, fld = inner_inspection(d)
    from typing import Dict, List
    # Typing check
    assert hasattr(typ, '__origin__') and typ.__origin__ == dict
    # default_factory produces the exact dict passed
    assert fld.default_factory() == d

    l = [1, 2, 3]
    typ2, fld2 = inner_inspection(l)
    assert hasattr(typ2, '__origin__') and typ2.__origin__ == list
    assert fld2.default_factory() == l

    scalar = 3.14
    typ3, fld3 = inner_inspection(scalar)
    assert typ3 == float
    assert fld3.default_factory() == scalar

def test_get_class_body_from_annotations_basic_types_and_any():
    from typing import Dict, List, Any

    ann = {
        "a": int,
        "b": List[int],
        "c": Dict[str, int],
        "d": None,
    }
    # We need to import is_dict and is_list not directly available here;
    # but since this just returns string body, check output lines contain expected substrings
    body = get_class_body_from_annotations(ann)
    # Check lines contain correct type info (type.__name__ or str of epi)
    assert "a: int" in body
    # For list and dict types, string representation with typing is included
    assert "b: " in body and "List" in body
    assert "c: " in body and "Dict" in body
    # None fallback to Any
    assert "d: Any" in body

def test_int_generator_bounds_and_type():
    random.seed(999)
    for length in [0, 1, 3, 5]:
        val = int_generator(length=length)
        assert isinstance(val, int)
        assert 0 <= val <= 9 * 10 ** length

def test_int_generator_length_zero_is_zero_to_nine():
    random.seed(0)
    val = int_generator(length=0)
    assert 0 <= val <= 9

def test_generate_with_field_defaults():
    from dataclasses import dataclass

    @dataclass
    class Sample:
        a: int = 7
        b: str = "hello"

    # enrich_fields should respect default values (no default_factory)
    rich = enrich_fields(Sample, None)
    for name, typ, fld in rich:
        if name == "a":
            assert fld.default == 7
            assert not hasattr(fld, 'default_factory') or fld.default_factory is None
        if name == "b":
            assert fld.default == "hello"
            assert not hasattr(fld, 'default_factory') or fld.default_factory is None

    # generate should produce object with those default values set
    NewClass = generate(Sample)
    obj = NewClass()
    assert obj.a == 7
    assert obj.b == "hello"

def test_generate_with_generators_integration_with_int_generator(monkeypatch):
    class Sample:
        a: int

    # Patch randint inside fistro.generators.int_generator to a constant value
    def fake_randint(low, high):
        return 42  # fixed number within bounds

    import fistro.generators
    monkeypatch.setattr(fistro.generators, "randint", fake_randint)

    # Provide int_generator as the generator in list
    obj_cls = generate(Sample, generators=[int_generator])
    obj = obj_cls()

    # Value of a should be 42 because of patched randint
    assert obj.a == 42

def test_generate_object_serialization_consistency():
    # Uses Sample with int and str fields
    class Sample:
        a: int
        b: str

    random.seed(123)
    obj_cls = generate(Sample)
    # Create instances multiple times
    obj1 = obj_cls()
    obj2 = obj_cls()
    # They should be different because factory generates new random values each time
    # So values shouldn't be equal necessarily, but their types should match
    assert isinstance(obj1.a, int) and isinstance(obj1.b, str)
    assert isinstance(obj2.a, int) and isinstance(obj2.b, str)

    # as_dict produces dict representation matching asdict()
    d1 = obj_cls().__dict__
    # Using generate with as_dict produces dict with same keys and appropriate types
    d2 = generate(Sample, as_dict=True)
    assert set(d1.keys()) == set(d2.keys())
    for k in d1:
        assert type(d1[k]) == type(d2[k])

def test_inner_inspection_field_default_factory_returns_fresh_objects():
    value = {"a": [1, 2, 3]}
    typ, fld = inner_inspection(value)
    v1 = fld.default_factory()
    v2 = fld.default_factory()
    assert v1 == value
    assert v2 == value
    # Ensure new calls produce distinct objects if possible
    if isinstance(v1, dict) or isinstance(v1, list):
        assert v1 is not v2

def test_get_class_body_from_annotations_allows_empty_annotations():
    # Empty dict in annotations
    body = get_class_body_from_annotations({})
    assert body == ''

def test_get_class_body_from_annotations_unknown_type_yields_any():
    ann = {"foo": None}
    body = get_class_body_from_annotations(ann)
    assert "foo: Any" in body

def test_generate_from_json_field_types_with_nested_structures():
    nested = {
        "key1": {"nested_key": 1},
        "key2": [1, 2, 3],
        "key3": "simple"
    }
    cls = generate_from_json(nested)
    fields = cls.__dataclass_fields__
    # key1 type should be a typing.Dict
    assert hasattr(fields['key1'].type, '__origin__') and fields['key1'].type.__origin__ == dict
    # key2 type should be a typing.List
    assert hasattr(fields['key2'].type, '__origin__') and fields['key2'].type.__origin__ == list
    # key3 type should be str (the type of "simple")
    assert fields['key3'].type == str

def test_generate_with_default_factory_custom_generator(monkeypatch):
    class Sample:
        a: int

    # Setup fake generator always returning 77
    def fake_gen(t):
        if t == int:
            return 77
        raise ValueError("Unsupported type")

    # Patch Factory's factory method to use fake_gen
    import fistro.fistro
    factory_instance = fistro.fistro.Factory([fake_gen])
    monkeypatch.setattr(fistro.fistro.Factory, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(fistro.fistro.Factory, "factory", lambda self, t: lambda: fake_gen(t))

    new_cls = generate(Sample, generators=[fake_gen])
    obj = new_cls()
    assert obj.a == 77

