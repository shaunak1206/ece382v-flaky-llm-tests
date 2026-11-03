import pytest
import json
from dataclasses import dataclass, field, fields, MISSING
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Callable

from fistro import fistro
from fistro.generators import int_generator


# --- Helpers for testing ---


@dataclass
class Simple:
    a: int
    b: str = "default"
    c: bool = False


@dataclass
class ComplexTypes:
    x: List[int]
    y: Dict[str, int]
    z: Optional[int] = None


# --- Tests for fistro.generate ---


def test_generate_returns_dataclass_type():
    result = fistro.generate(Simple)
    assert hasattr(result, "__dataclass_fields__")
    assert result.__name__ == "Simple"


def test_generate_instance_fields_and_defaults():
    GenClass = fistro.generate(Simple)
    instance = GenClass()
    # a has no default, so using Factory's factory for int type, which should produce an int
    assert isinstance(instance.a, int)
    # b and c have defaults copied exactly
    assert instance.b == "default"
    assert instance.c is False


def test_generate_with_generators_replaces_factories(monkeypatch):
    # Define a fake generator for int
    def fake_int_gen():
        return 42

    def fake_factory(cls):
        if cls is int:
            return fake_int_gen
        # fallback: use default Factory factory
        return lambda: None

    class DummyFactory:
        def __init__(self, generators=None):
            pass

        def factory(self, cls):
            return fake_factory(cls)

    # Patch Factory in fistro.fistro
    monkeypatch.setattr(fistro, "Factory", DummyFactory)
    GenClass = fistro.generate(Simple, generators=[fake_int_gen])
    inst = GenClass()
    assert inst.a == 42


def test_generate_as_dict_returns_dict():
    d = fistro.generate(Simple, as_dict=True)
    assert isinstance(d, dict)
    assert "a" in d and "b" in d and "c" in d


def test_generate_as_json_returns_json_string():
    json_str = fistro.generate(Simple, as_json=True)
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)
    assert "a" in parsed


def test_enrich_fields_with_default_and_missing():
    # Use dataclass with one default, one missing
    @dataclass
    class TestDC:
        x: int
        y: int = 10

    # Factory should be called for x, default used for y
    enriched = fistro.enrich_fields(TestDC, None)
    # enriched is list of tuples (name, type, field)
    names = [ef[0] for ef in enriched]
    assert "x" in names and "y" in names
    # Check default for y
    y_field = next(ef[2] for ef in enriched if ef[0] == "y")
    # default is 10
    assert y_field.default == 10


def test_generate_from_json_creates_dataclass_with_correct_fields():
    obj = {"foo": 1, "bar": [1, 2, 3], "baz": {"x": "y"}}
    Generated = fistro.generate_from_json(obj)
    instance = Generated()
    # Fields keys must match keys in original dict
    field_names = [f.name for f in fields(Generated)]
    for key in obj.keys():
        assert key in field_names
    # Types are derived roughly correctly: foo is int
    foo_type = next(f.type for f in fields(Generated) if f.name == "foo")
    assert foo_type == int
    # bar is a list (of int)
    bar_type = next(f.type for f in fields(Generated) if f.name == "bar")
    # list with inner type int
    assert hasattr(bar_type, "__origin__") and bar_type.__origin__ == list
    # baz is dict
    baz_type = next(f.type for f in fields(Generated) if f.name == "baz")
    assert hasattr(baz_type, "__origin__") and baz_type.__origin__ == dict


def test_generate_from_json_raises_for_list_input():
    with pytest.raises(NotImplementedError):
        fistro.generate_from_json([1, 2, 3])


def test_inner_inspection_on_primitive_returns_type_and_field():
    typ, fld = fistro.inner_inspection(42)
    assert typ is int
    # field default factory returns the original value
    val = fld.default_factory()
    assert val == 42


def test_inner_inspection_on_dict_returns_dict_type_and_field():
    test_dict = {"key": 123}
    typ, fld = fistro.inner_inspection(test_dict)
    assert hasattr(typ, "__origin__")
    assert typ.__origin__ == dict
    val = fld.default_factory()
    assert val == test_dict


def test_inner_inspection_on_list_returns_list_type_and_field():
    test_list = [1, 2, 3]
    typ, fld = fistro.inner_inspection(test_list)
    assert hasattr(typ, "__origin__")
    assert typ.__origin__ == list
    val = fld.default_factory()
    assert val == test_list


def test_get_class_body_from_annotations_for_various_types():
    annotations = {
        "a": int,
        "b": list,
        "c": dict,
        "d": None,
    }

    # Helpers from fistro to identify is_dict and is_list are used,
    # so dict and list here may be type or typing.List / Dict
    res = fistro.get_class_body_from_annotations(annotations)
    assert "a: int" in res
    assert "b: " in res
    assert "c: " in res
    assert "d: Any" in res


# --- Tests for fistro.generators.int_generator ---


def test_int_generator_returns_int_and_in_range():
    val = int_generator()
    assert isinstance(val, int)
    # Minimum 0, maximum roughly 9 * 10**length where length is default INT_LENGTH (10)
    assert 0 <= val <= 9 * 10 ** 10


@pytest.mark.parametrize("length", [0, 1, 5, 10])
def test_int_generator_with_various_lengths(length):
    val = int_generator(length)
    assert isinstance(val, int)
    assert 0 <= val <= 9 * 10 ** length


# --- Additional tests for generators (indirectly verifying integration) ---


def test_generate_with_int_generator_produces_int_field(monkeypatch):
    # Patch Factory().factory to always return int_generator for int
    class DummyFactory:
        def __init__(self, genlist=None):
            pass

        def factory(self, cls):
            if cls is int:
                return lambda: 12345
            return lambda: None

    monkeypatch.setattr(fistro, "Factory", DummyFactory)

    @dataclass
    class TestClass:
        val: int

    GenClass = fistro.generate(TestClass)
    inst = GenClass()
    assert inst.val == 12345


def test_generate_with_default_generators(monkeypatch):
    from fistro.generators import default_generators

    # Patch Factory to record what it receives and return a fixed int
    class DummyFactory:
        def __init__(self, gens=None):
            self.gens = gens

        def factory(self, cls):
            if cls is int:
                return lambda: 777
            else:
                return lambda: None

    monkeypatch.setattr(fistro, "Factory", DummyFactory)

    @dataclass
    class TestClass:
        val: int

    GenClass = fistro.generate(TestClass, generators=default_generators())
    inst = GenClass()
    assert inst.val == 777


def test_generate_preserves_field_type_in_new_dataclass():
    # generate produces a dataclass with same field names and types
    GenClass = fistro.generate(Simple)
    gen_fields = {f.name: f.type for f in fields(GenClass)}
    orig_fields = {f.name: f.type for f in fields(Simple)}
    assert gen_fields == orig_fields


def test_enrich_fields_with_custom_generator_factory(monkeypatch):
    called = {}

    def custom_factory(cls):
        called['cls'] = cls
        return lambda: 999

    class DummyFactory:
        def __init__(self, gens=None):
            pass

        def factory(self, cls):
            return custom_factory(cls)

    monkeypatch.setattr(fistro, "Factory", DummyFactory)

    enriched = fistro.enrich_fields(Simple, generators=[lambda: 1])
    # The factory should be called with int type (for field 'a')
    assert called.get('cls') == int
    # The field's default_factory() should return 999
    df = next(ef[2].default_factory for ef in enriched if ef[0] == 'a')
    assert df() == 999

