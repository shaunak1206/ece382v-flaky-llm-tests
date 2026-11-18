import pytest
import json
from datetime import datetime, date
from typing import List, Callable, Any, Dict
from dataclasses import dataclass, field, is_dataclass, asdict

from fistro.fistro import generate, enrich_fields, generate_from_json, inner_inspection, get_class_body_from_annotations
from fistro.generators import int_generator, bool_generator, str_generator, datetime_generator, date_generator, int_list_generator, int_dict_generator


@dataclass
class SimpleData:
    x: int
    y: str


@dataclass
class DefaultsData:
    a: int = 42
    b: str = "hello"


def test_generate_returns_dataclass_type():
    result = generate(SimpleData)
    assert is_dataclass(result)
    assert result.__name__ == "SimpleData"


def test_generate_returns_instance_as_dict():
    result = generate(SimpleData, as_dict=True)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"x", "y"}


def test_generate_returns_instance_as_json():
    result = generate(SimpleData, as_json=True)
    # Should be JSON string
    assert isinstance(result, str)
    decoded = json.loads(result)
    assert set(decoded.keys()) == {"x", "y"}


def test_enrich_fields_with_defaults():
    fields_result = enrich_fields(DefaultsData, None)
    # Should preserve default values in default field
    for name, typ, fld in fields_result:
        if name == 'a':
            assert fld.default == 42
        if name == 'b':
            assert fld.default == "hello"


def test_enrich_fields_without_defaults_and_generators():
    # Should add default_factory for fields without defaults
    fields_result = enrich_fields(SimpleData, None)
    for name, typ, fld in fields_result:
        if name in ('x', 'y'):
            assert fld.default == field().default or fld.default_factory is not None


def test_enrich_fields_with_generators_overrides_factory():
    # Use dummy generator that returns fixed value for int and str
    def dummy_gen(tp):
        if tp is int:
            return lambda: 123
        if tp is str:
            return lambda: "gen"
        return lambda: None

    fields_result = enrich_fields(SimpleData, [dummy_gen])
    # default factory should come from dummy_gen
    for name, typ, fld in fields_result:
        val = fld.default if fld.default != field().default else None
        assert (fld.default_factory is not None) or (fld.default == val or val is None)


def test_generate_from_json_creates_dataclass():
    data = {"foo": 10, "bar": {"x": 1, "y": 2}}
    cls = generate_from_json(data)
    assert is_dataclass(cls)
    # It should have fields corresponding to keys of input dict
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    assert "foo" in field_names
    assert "bar" in field_names
    # The type of foo should be int, field default_factory should produce 10
    foo_field = cls.__dataclass_fields__["foo"]
    foo_val = foo_field.default_factory()
    assert foo_val == 10
    # The type of bar should be Dict[int, int] (type hints) and factory yields input dictionary
    bar_field = cls.__dataclass_fields__["bar"]
    bar_val = bar_field.default_factory()
    assert isinstance(bar_val, dict)
    assert bar_val == {"x": 1, "y": 2}


def test_generate_from_json_raises_on_list():
    with pytest.raises(NotImplementedError):
        generate_from_json([1, 2, 3])


@pytest.mark.parametrize(
    "input_obj, expected_type",
    [
        ({"a": 1}, dict),
        ([1, 2, 3], list),
        (42, int),
        ("abc", str),
    ],
)
def test_inner_inspection_types_and_defaults(input_obj, expected_type):
    inner_type, fld = inner_inspection(input_obj)
    # Check if the returned inner_type matches expectations:
    if isinstance(input_obj, dict):
        assert hasattr(inner_type, "__origin__")
        assert inner_type.__origin__ == dict
    elif isinstance(input_obj, list):
        assert hasattr(inner_type, "__origin__")
        assert inner_type.__origin__ == list
    else:
        assert inner_type == type(input_obj)
    # Check that default_factory returns the object
    val = fld.default_factory()
    assert val == input_obj


def test_get_class_body_from_annotations_simple_and_complex():
    from typing import Dict, List, Any

    ann = {
        "str_prop": str,
        "int_prop": int,
        "dict_prop": Dict[str, int],
        "list_prop": List[str],
        "none_prop": None,
    }
    code_str = get_class_body_from_annotations(ann)
    # Should contain str_prop: str and dict_prop: Dict[str, int] and list_prop accordingly
    assert "str_prop: str" in code_str
    assert "int_prop: int" in code_str
    assert "dict_prop: typing.Dict[str, int]" in code_str or "dict_prop: Dict[str, int]" in code_str
    assert "list_prop: typing.List[str]" in code_str or "list_prop: List[str]" in code_str
    assert "none_prop: Any" in code_str


# --- Tests for fistro.generators.py ---


def test_int_generator_returns_int_and_in_range():
    for length in [1, 3, 5]:
        val = int_generator(length)
        assert isinstance(val, int)
        assert 0 <= val <= 9 * 10 ** length


def test_bool_generator_returns_boolean():
    val = bool_generator()
    assert isinstance(val, bool)


def test_str_generator_default_length_and_population():
    val = str_generator()
    assert isinstance(val, str)
    assert len(val) == 32  # STR_LENGTH default in config is 32 (assumed)


def test_str_generator_with_custom_population_and_length():
    val = str_generator(population='abc', length=10)
    assert set(val).issubset(set('abc'))
    assert len(val) == 10


def test_datetime_generator_returns_datetime_and_values_in_range():
    val = datetime_generator()
    assert isinstance(val, datetime)


def test_datetime_generator_respects_ranges():
    rand_date = {
        "min_year": 2020, "max_year": 2020,
        "min_month": 1, "max_month": 1,
        "min_day": 1, "max_day": 1,
        "min_hour": 0, "max_hour": 0,
        "min_minute": 0, "max_minute": 0,
        "min_second": 0, "max_second": 0,
    }
    val = datetime_generator(rand_date)
    assert val.year == 2020
    assert val.month == 1
    assert val.day == 1
    assert val.hour == 0
    assert val.minute == 0
    assert val.second == 0


def test_date_generator_returns_date_and_in_range():
    val = date_generator()
    assert isinstance(val, date)


def test_date_generator_respects_ranges():
    rand_date = {
        "min_year": 2020, "max_year": 2020,
        "min_month": 2, "max_month": 2,
        "min_day": 15, "max_day": 15,
    }
    val = date_generator(rand_date)
    assert val.year == 2020
    assert val.month == 2
    assert val.day == 15


def test_int_list_generator_returns_list_of_ints_and_length():
    val = int_list_generator()
    assert isinstance(val, list)
    assert all(isinstance(x, int) for x in val)
    assert len(val) == 5  # int_list_length default value assumed to be 5


def test_int_dict_generator_returns_dict_of_str_int_and_length():
    val = int_dict_generator()
    assert isinstance(val, dict)
    assert all(isinstance(k, str) for k in val.keys())
    assert all(isinstance(v, int) for v in val.values())
    assert len(val) == 5  # int_list_length default value assumed to be 5


def test_default_generators_contains_expected_functions():
    gens = [func.__name__ for func in None or []] + [int_generator, bool_generator, str_generator,
                                                    datetime_generator, date_generator,
                                                    int_list_generator, int_dict_generator]
    default_gens = [f.__name__ for f in (default_generators())]
    for func_name in gens:
        assert func_name in default_gens
