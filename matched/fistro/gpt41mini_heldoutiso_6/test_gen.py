import pytest
import json
from typing import List, Dict, Any
from datetime import datetime, date

from fistro.fistro import generate, enrich_fields, generate_from_json, inner_inspection, get_class_body_from_annotations
from fistro.generators import (
    int_generator,
    bool_generator,
    str_generator,
    datetime_generator,
    date_generator,
    int_list_generator,
    int_dict_generator,
)

from dataclasses import dataclass, field, fields, Field, is_dataclass, asdict
from typing import Optional, List as TypingList, Dict as TypingDict


@dataclass
class SimpleClass:
    x: int
    y: str = "default"


@dataclass
class ComplexClass:
    a: int
    b: List[int]
    c: Dict[str, str]
    d: bool = True


def test_generate_returns_dataclass_type():
    result = generate(SimpleClass)
    assert is_dataclass(result)
    assert result.__name__ == "SimpleClass"
    # Fields should match the input dataclass
    result_fields = {f.name for f in fields(result)}
    assert result_fields == {"x", "y"}


def test_generate_returns_dict_when_as_dict_true():
    obj_dict = generate(SimpleClass, as_dict=True)
    assert isinstance(obj_dict, dict)
    assert set(obj_dict.keys()) == {"x", "y"}
    assert obj_dict["y"] == "default"
    assert isinstance(obj_dict["x"], int)


def test_generate_returns_json_when_as_json_true():
    json_str = generate(SimpleClass, as_json=True)
    assert isinstance(json_str, str)
    loaded = json.loads(json_str)
    assert set(loaded.keys()) == {"x", "y"}
    assert loaded["y"] == "default"
    assert isinstance(loaded["x"], int)


def test_enrich_fields_uses_default_when_present():
    # y has default in SimpleClass
    enriched = enrich_fields(SimpleClass, generators=None)
    y_field = next(f for f in enriched if f[0] == "y")
    assert hasattr(y_field[2], "default")
    assert y_field[2].default == "default"


def test_enrich_fields_uses_factory_when_no_default():
    enriched = enrich_fields(SimpleClass, generators=None)
    x_field = next(f for f in enriched if f[0] == "x")
    # default_factory should be callable
    assert callable(x_field[2].default_factory)


def test_enrich_fields_with_generators_overrides_factory():
    # Provide a list of one generator that returns fixed int 42
    def fixed_int_generator():
        return 42

    enriched = enrich_fields(SimpleClass, generators=[fixed_int_generator])
    x_field = next(f for f in enriched if f[0] == "x")
    # The default_factory here is created from Factory([fixed_int_generator])
    # To test, call it and expect the fixed int 42 returned
    value = x_field[2].default_factory()
    assert value == 42


def test_inner_inspection_with_dict():
    test_dict = {"key": 123}
    typ, fld = inner_inspection(test_dict)
    assert hasattr(fld, "default_factory")
    # typ should be Dict[type(key), type(value)] = Dict[str, int]
    assert typ.__origin__ == dict
    assert typ.__args__[0] == str
    assert typ.__args__[1] == int
    # default_factory should produce equal dict
    assert fld.default_factory() == test_dict


def test_inner_inspection_with_list_of_ints():
    test_list = [1, 2, 3]
    typ, fld = inner_inspection(test_list)
    assert hasattr(fld, "default_factory")
    # typ should be List[int]
    assert typ.__origin__ == list
    assert typ.__args__[0] == int
    assert fld.default_factory() == test_list


def test_inner_inspection_with_list_of_empty_list():
    test_list = []
    typ, fld = inner_inspection(test_list)
    assert typ.__origin__ == list
    # When list empty, type is List[Any]
    from typing import Any
    assert typ.__args__[0] == Any
    assert fld.default_factory() == test_list


def test_inner_inspection_with_scalar_value():
    val = 42
    typ, fld = inner_inspection(val)
    assert typ == int
    assert fld.default_factory()() == 42


def test_generate_from_json_creates_dataclass():
    input_dict = {"a": 1, "b": [1, 2, 3]}
    generated_cls = generate_from_json(input_dict)
    assert is_dataclass(generated_cls)
    f_names = {f.name for f in fields(generated_cls)}
    assert f_names == {"a", "b"}
    # Types should be int and List[int]
    type_a = next(f for f in fields(generated_cls) if f.name == "a").type
    type_b = next(f for f in fields(generated_cls) if f.name == "b").type
    assert type_a == int
    assert type_b.__origin__ == list
    # default_factory of field b should produce [1,2,3]
    b_field = next(f for f in fields(generated_cls) if f.name == "b")
    assert b_field.default_factory() == [1, 2, 3]


def test_generate_from_json_raises_not_implemented_for_list():
    with pytest.raises(NotImplementedError):
        generate_from_json([{"a": 1}])


def test_get_class_body_from_annotations_works_for_simple_types():
    ann = {"a": int, "b": str, "c": None}
    body = get_class_body_from_annotations(ann)
    assert "a: int" in body
    assert "b: str" in body
    assert "c: Any" in body


def test_get_class_body_from_annotations_handles_dict_and_list():
    from typing import Dict, List

    ann = {
        "a": Dict[str, int],
        "b": List[int],
        "c": int,
    }
    body = get_class_body_from_annotations(ann)
    assert "a: " in body and "Dict" in body
    assert "b: " in body and "List" in body
    assert "c: int" in body


def test_int_generator_returns_int_within_range():
    val = int_generator(3)
    # The max possible is 9 * 10^3 = 9000 roughly
    assert isinstance(val, int)
    assert 0 <= val <= 9000


@pytest.mark.parametrize("count", [1, 10])
def test_int_list_generator_has_given_length(count):
    result = int_list_generator(list_length=count, int_length=2)
    assert isinstance(result, list)
    assert len(result) == count
    assert all(isinstance(x, int) for x in result)


def test_int_dict_generator_returns_dict_with_str_keys_and_int_values():
    result = int_dict_generator(dict_length=5, int_length=2)
    assert isinstance(result, dict)
    assert len(result) == 5
    for k, v in result.items():
        assert isinstance(k, str)
        assert isinstance(v, int)


def test_bool_generator_returns_bool():
    val = bool_generator()
    assert isinstance(val, bool)


def test_str_generator_returns_str_of_correct_length():
    val = str_generator(length=15)
    assert isinstance(val, str)
    assert len(val) == 15


def test_datetime_generator_returns_datetime_in_range():
    params = {"min_year": 2000, "max_year": 2001}
    val = datetime_generator(rand_date=params)
    assert isinstance(val, datetime)
    assert 2000 <= val.year <= 2001


def test_date_generator_returns_date_in_range():
    params = {"min_year": 1990, "max_year": 1992}
    val = date_generator(rand_date=params)
    assert isinstance(val, date)
    assert 1990 <= val.year <= 1992
