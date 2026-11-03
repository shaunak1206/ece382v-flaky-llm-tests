import json
import random
import string
from dataclasses import dataclass, is_dataclass, fields
from datetime import date, datetime
from typing import Any, Dict, List

import pytest

from fistro.config import INT_LENGTH, INT_LIST_LENGTH, STR_LENGTH
from fistro.fistro import (
    generate,
    generate_from_json,
    get_class_body_from_annotations,
    inner_inspection,
)
from fistro.generators import (
    bool_generator,
    date_generator,
    datetime_generator,
    default_generators,
    int_dict_generator,
    int_generator,
    int_list_generator,
    str_generator,
)


@pytest.fixture(autouse=True)
def seed_random():
    random.seed(1234)


@dataclass
class Person:
    name: str
    age: int
    active: bool


@dataclass
class WithDefault:
    count: int
    label: str = 'fixed'


@dataclass
class Collections:
    numbers: List[int]
    mapping: Dict[str, int]


# ---------------------------------------------------------------- generators


def test_int_generator_default_range():
    for _ in range(200):
        value = int_generator()
        assert isinstance(value, int)
        assert 0 <= value <= 9 * 10 ** INT_LENGTH


def test_int_generator_custom_length_bounds():
    for _ in range(200):
        assert 0 <= int_generator(3) <= 9000
        assert 0 <= int_generator(length=0) <= 9


def test_bool_generator_returns_both_values():
    values = {bool_generator() for _ in range(200)}
    assert values == {True, False}


def test_str_generator_defaults():
    value = str_generator()
    assert isinstance(value, str)
    assert len(value) == STR_LENGTH
    assert all(c in string.printable for c in value)


def test_str_generator_custom_population_and_length():
    value = str_generator(population='ab', length=25)
    assert len(value) == 25
    assert set(value) <= {'a', 'b'}


def test_str_generator_zero_length_is_empty():
    assert str_generator(length=0) == ''


def test_datetime_generator_fixed_bounds_gives_exact_value():
    bounds = {
        'min_year': 2020, 'max_year': 2020,
        'min_month': 2, 'max_month': 2,
        'min_day': 3, 'max_day': 3,
        'min_hour': 4, 'max_hour': 4,
        'min_minute': 5, 'max_minute': 5,
        'min_second': 6, 'max_second': 6,
    }
    assert datetime_generator(bounds) == datetime(2020, 2, 3, 4, 5, 6)


def test_datetime_generator_stays_within_bounds():
    bounds = {
        'min_year': 2000, 'max_year': 2010,
        'min_month': 1, 'max_month': 12,
        'min_day': 1, 'max_day': 28,
    }
    for _ in range(100):
        result = datetime_generator(bounds)
        assert type(result) is datetime
        assert 2000 <= result.year <= 2010
        assert 1 <= result.day <= 28
        assert result.microsecond == 0


def test_date_generator_fixed_bounds_gives_exact_value():
    bounds = {
        'min_year': 1999, 'max_year': 1999,
        'min_month': 12, 'max_month': 12,
        'min_day': 31, 'max_day': 31,
    }
    result = date_generator(bounds)
    assert result == date(1999, 12, 31)
    assert type(result) is date


def test_date_generator_stays_within_bounds():
    bounds = {
        'min_year': 2015, 'max_year': 2016,
        'min_month': 3, 'max_month': 6,
        'min_day': 1, 'max_day': 28,
    }
    for _ in range(100):
        result = date_generator(bounds)
        assert 2015 <= result.year <= 2016
        assert 3 <= result.month <= 6
        assert 1 <= result.day <= 28


def test_int_list_generator_default_and_custom_shapes():
    default = int_list_generator()
    assert len(default) == INT_LIST_LENGTH
    assert all(isinstance(i, int) and 0 <= i <= 9 * 10 ** INT_LENGTH for i in default)

    custom = int_list_generator(list_length=3, int_length=1)
    assert len(custom) == 3
    assert all(0 <= i <= 90 for i in custom)
    assert int_list_generator(list_length=0) == []


def test_int_dict_generator_shape():
    result = int_dict_generator(dict_length=4, int_length=2)
    assert isinstance(result, dict)
    assert 1 <= len(result) <= 4
    assert all(isinstance(k, str) and len(k) == STR_LENGTH for k in result)
    assert all(isinstance(v, int) and 0 <= v <= 900 for v in result.values())
    assert int_dict_generator(dict_length=0) == {}


def test_default_generators_contents():
    generators = default_generators()
    assert len(generators) == 7
    assert all(callable(g) for g in generators)
    names = {g.__name__ for g in generators}
    assert 'int_generator' in names
    assert 'str_generator' in names
    assert 'int_dict_generator' in names


# ------------------------------------------------------------------ fistro.py


def test_generate_returns_dataclass_with_same_fields():
    result = generate(Person)
    assert is_dataclass(result)
    assert result.__name__ == 'Person'
    assert [f.name for f in fields(result)] == ['name', 'age', 'active']


def test_generate_instance_has_valid_values():
    instance = generate(Person)()
    assert isinstance(instance.name, str)
    assert len(instance.name) == STR_LENGTH
    assert isinstance(instance.age, int)
    assert 0 <= instance.age <= 9 * 10 ** INT_LENGTH
    assert isinstance(instance.active, bool)


def test_generate_as_dict():
    result = generate(Person, as_dict=True)
    assert isinstance(result, dict)
    assert set(result) == {'name', 'age', 'active'}
    assert isinstance(result['age'], int)


def test_generate_as_json_is_parseable():
    text = generate(Person, as_json=True)
    assert isinstance(text, str)
    parsed = json.loads(text)
    assert set(parsed) == {'name', 'age', 'active'}
    assert isinstance(parsed['active'], bool)


def test_generate_respects_field_defaults():
    instance = generate(WithDefault)()
    assert instance.label == 'fixed'
    assert isinstance(instance.count, int)
    assert generate(WithDefault, as_dict=True)['label'] == 'fixed'


def test_generate_collection_fields():
    instance = generate(Collections)()
    assert len(instance.numbers) == INT_LIST_LENGTH
    assert all(isinstance(n, int) for n in instance.numbers)
    assert isinstance(instance.mapping, dict)
    assert all(isinstance(v, int) for v in instance.mapping.values())


def test_generate_with_custom_generator_overrides_int():
    def always_seven() -> int:
        return 7

    assert generate(Person, generators=[always_seven], as_dict=True)['age'] == 7


def test_generate_from_json_preserves_values_and_class_is_dataclass():
    cls = generate_from_json({'a': 1, 'b': 'text', 'c': [1, 2, 3]})
    assert is_dataclass(cls)
    instance = cls()
    assert instance.a == 1
    assert instance.b == 'text'
    assert instance.c == [1, 2, 3]


def test_generate_from_json_rejects_non_dict():
    with pytest.raises((TypeError, NotImplementedError)):
        generate_from_json([1, 2, 3])


def test_inner_inspection_types():
    assert inner_inspection(5)[0] is int
    assert inner_inspection('x')[0] is str
    assert inner_inspection([1, 2])[0] == List[int]
    assert inner_inspection([])[0] == List[Any]
    assert inner_inspection({'k': 1})[0] == Dict[str, int]


def test_inner_inspection_default_factory_returns_original():
    original = {'k': 1}
    _, a_field = inner_inspection(original)
    assert a_field.default_factory() is original


def test_get_class_body_from_annotations():
    body = get_class_body_from_annotations(
        {'a': int, 'b': List[int], 'c': None, 'd': Dict[str, int]}
    )
    lines = body.splitlines()
    assert lines[0] == 'a: int'
    assert lines[1] == f'b: {List[int]}'
    assert lines[2] == 'c: Any'
    assert lines[3] == f'd: {Dict[str, int]}'
    assert get_class_body_from_annotations({}) == ''
