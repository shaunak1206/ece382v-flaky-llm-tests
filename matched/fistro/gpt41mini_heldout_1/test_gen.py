import pytest
import string
import json
from datetime import datetime, date
from typing import List, Optional, Dict, Any, Callable
from random import randint

from fistro.fistro import generate, enrich_fields, generate_from_json, inner_inspection, get_class_body_from_annotations
from fistro.generators import int_generator, bool_generator, str_generator, datetime_generator, date_generator, int_list_generator, int_dict_generator


# Helper dataclasses for testing
from dataclasses import dataclass, field, fields, Field, MISSING


@dataclass
class SimpleClass:
    x: int
    y: str = "default"
    b: bool = False


@dataclass
class NestedClass:
    a_list: List[int]
    a_dict: Dict[str, int]
    a_int: int


def test_int_generator_returns_int_and_within_expected_range():
    val = int_generator()
    assert isinstance(val, int)
    assert 0 <= val <= 9 * 10 ** 8  # Default INT_LENGTH presumably 8


def test_int_generator_custom_length():
    val = int_generator(3)
    assert isinstance(val, int)
    assert 0 <= val <= 9 * 10 ** 3


def test_bool_generator_returns_bool():
    val = bool_generator()
    assert isinstance(val, bool)


def test_str_generator_length_and_characters():
    val = str_generator()
    assert isinstance(val, str)
    assert len(val) == 20  # Default STR_LENGTH presumably 20
    for ch in val:
        assert ch in string.printable


def test_str_generator_custom_population_and_length():
    population = "abc"
    length = 10
    val = str_generator(population=population, length=length)
    assert isinstance(val, str)
    assert len(val) == length
    for ch in val:
        assert ch in population


def test_datetime_generator_returns_datetime_and_within_bounds():
    rand_date = {
        'min_year': 2000,
        'max_year': 2000,
        'min_month': 1,
        'max_month': 1,
        'min_day': 1,
        'max_day': 1,
        'min_hour': 0,
        'max_hour': 0,
        'min_minute': 0,
        'max_minute': 0,
        'min_second': 0,
        'max_second': 0,
    }
    dt = datetime_generator(rand_date)
    assert isinstance(dt, datetime)
    assert dt.year == 2000
    assert dt.month == 1
    assert dt.day == 1
    assert dt.hour == 0
    assert dt.minute == 0
    assert dt.second == 0


def test_date_generator_returns_date_and_within_bounds():
    rand_date = {
        'min_year': 2020,
        'max_year': 2020,
        'min_month': 2,
        'max_month': 2,
        'min_day': 15,
        'max_day': 15,
    }
    d = date_generator(rand_date)
    assert isinstance(d, date)
    assert d.year == 2020
    assert d.month == 2
    assert d.day == 15


def test_int_list_generator_returns_list_of_ints_correct_length():
    result = int_list_generator(list_length=5, int_length=2)
    assert isinstance(result, list)
    assert len(result) == 5
    for i in result:
        assert isinstance(i, int)
        assert 0 <= i <= 9 * 10 ** 2


def test_int_dict_generator_returns_dict_with_str_keys_and_int_values():
    result = int_dict_generator(dict_length=5, int_length=2)
    assert isinstance(result, dict)
    assert len(result) == 5
    for k, v in result.items():
        assert isinstance(k, str)
        assert isinstance(v, int)
        assert 0 <= v <= 9 * 10 ** 2


def test_enrich_fields_uses_default_factory_when_no_default_and_no_generators():
    enriched = enrich_fields(SimpleClass, None)
    field_names = [t[0] for t in enriched]
    assert "x" in field_names
    # The field corresponding to x should have default factory (because no default)
    for (name, typ, fld) in enriched:
        if name == "x":
            assert fld.default is MISSING
            assert callable(fld.default_factory)
        elif name in ("y", "b"):
            # Has defaults so default used instead of default factory
            assert fld.default != MISSING


def test_enrich_fields_uses_default_factory_with_generators_list():
    dummy_gen = [lambda t: lambda: 42]
    enriched = enrich_fields(SimpleClass, dummy_gen)
    for (name, typ, fld) in enriched:
        if name == "x":
            assert callable(fld.default_factory)
            assert fld.default == MISSING
        elif name == "y":
            assert fld.default == "default"
        elif name == "b":
            assert fld.default is False


def test_generate_returns_dataclass_type_by_default():
    cls = generate(SimpleClass)
    assert hasattr(cls, "__dataclass_fields__")
    # Check name unchanged
    assert cls.__name__ == SimpleClass.__name__


def test_generate_returns_dict_when_as_dict_true():
    result = generate(SimpleClass, as_dict=True)
    assert isinstance(result, dict)
    assert "x" in result and "y" in result and "b" in result


def test_generate_returns_json_when_as_json_true():
    result = generate(SimpleClass, as_json=True)
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert isinstance(parsed, dict)
    assert "x" in parsed and "y" in parsed and "b" in parsed


def test_inner_inspection_returns_correct_type_and_field_for_primitive():
    t, fld = inner_inspection(123)
    assert t == int
    assert fld.default_factory() == 123

    t, fld = inner_inspection("abc")
    assert t == str
    assert fld.default_factory() == "abc"


def test_inner_inspection_returns_list_type_and_field():
    t, fld = inner_inspection([1, 2, 3])
    from typing import List as TypingList
    assert hasattr(t, "__origin__")
    assert t.__origin__ == list
    assert fld.default_factory() == [1, 2, 3]

    # Empty list should default to List[Any]
    t2, fld2 = inner_inspection([])
    assert hasattr(t2, "__origin__")
    assert t2.__origin__ == list


def test_inner_inspection_returns_dict_type_and_field():
    sample_dict = {"a": 1}
    t, fld = inner_inspection(sample_dict)
    from typing import Dict as TypingDict
    assert hasattr(t, "__origin__")
    assert t.__origin__ == dict
    assert fld.default_factory() == sample_dict


def test_generate_from_json_creates_dataclass_from_dict():
    sample = {"field1": 1, "field2": {"subfield": 2}}
    cls = generate_from_json(sample)
    assert hasattr(cls, "__dataclass_fields__")
    flds = cls.__dataclass_fields__
    assert "field1" in flds and "field2" in flds
    # field1 type is int
    assert flds["field1"].type == int
    # field2 should be a Dict since sample["field2"] is dict
    assert hasattr(flds["field2"].type, "__origin__")
    assert flds["field2"].type.__origin__ == dict


def test_generate_from_json_raises_not_implemented_for_list():
    with pytest.raises(NotImplementedError):
        generate_from_json([1, 2, 3])


def test_get_class_body_from_annotations_simple_types():
    ann = {"a": int, "b": str, "c": None}
    result = get_class_body_from_annotations(ann)
    assert "a: int" in result
    assert "b: str" in result
    assert "c: Any" in result


def test_get_class_body_from_annotations_with_dict_and_list():
    from typing import List, Dict
    ann = {
        "ld": List[int],
        "dd": Dict[str, int],
    }
    result = get_class_body_from_annotations(ann)
    assert "ld: " in result
    assert "dd: " in result
    # Should include typing info as str
    assert "List" in result or "Dict" in result


def test_generate_with_generators_produces_fields_using_given_generator():
    # Provide custom generator list that always returns 999 for int
    def gen(_type):
        if _type == int:
            return lambda: 999
        return lambda: None
    cls = generate(SimpleClass, generators=[gen])
    inst = cls()
    assert inst.x == 999
    # Defaults remain same
    assert inst.y == "default"
    assert inst.b is False
