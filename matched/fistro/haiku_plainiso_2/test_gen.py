import json
import random
import string
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime
from inspect import signature
from typing import Dict, List

import pytest

from fistro.config import INT_LENGTH, INT_LIST_LENGTH, STR_LENGTH
from fistro.exceptions import NotSupportedType
from fistro.fistro import generate
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
def seeded_random():
    random.seed(382)


@dataclass
class Simple:
    count: int
    flag: bool
    name: str


@dataclass
class WithDefault:
    count: int = 7
    name: str = 'fixed'


@dataclass
class DateOnly:
    when: datetime


@dataclass
class Listy:
    values: List[int]


@dataclass
class Floaty:
    ratio: float


def test_int_generator_returns_int_within_default_range():
    upper = 9 * 10 ** INT_LENGTH
    samples = [int_generator() for _ in range(200)]
    assert all(type(v) is int and 0 <= v <= upper for v in samples)


def test_int_generator_length_shrinks_range():
    samples = [int_generator(length=2) for _ in range(200)]
    assert all(type(v) is int and 0 <= v <= 900 for v in samples)


def test_bool_generator_yields_both_values():
    samples = [bool_generator() for _ in range(200)]
    assert all(type(v) is bool for v in samples)
    assert set(samples) == {True, False}


def test_str_generator_default_length_and_charset():
    for _ in range(50):
        value = str_generator()
        assert len(value) == STR_LENGTH
        assert set(value) <= set(string.printable)


def test_str_generator_custom_population_and_length():
    for _ in range(50):
        value = str_generator(population='ab', length=5)
        assert len(value) == 5
        assert set(value) <= {'a', 'b'}


def test_str_generator_zero_length_is_empty():
    assert str_generator(length=0) == ''


def test_datetime_generator_honours_bounds():
    bounds = {
        'min_year': 1990,
        'max_year': 1995,
        'min_month': 3,
        'max_month': 3,
        'min_day': 10,
        'max_day': 20,
        'min_hour': 5,
        'max_hour': 5,
        'min_minute': 6,
        'max_minute': 6,
        'min_second': 7,
        'max_second': 7,
    }
    for _ in range(50):
        result = datetime_generator(bounds)
        assert isinstance(result, datetime)
        assert 1990 <= result.year <= 1995
        assert result.month == 3
        assert 10 <= result.day <= 20
        assert (result.hour, result.minute, result.second) == (5, 6, 7)


def test_date_generator_returns_plain_date():
    bounds = {
        'min_year': 2001,
        'max_year': 2001,
        'min_month': 2,
        'max_month': 2,
        'min_day': 1,
        'max_day': 28,
    }
    for _ in range(50):
        result = date_generator(bounds)
        assert type(result) is date
        assert result.year == 2001
        assert result.month == 2
        assert 1 <= result.day <= 28


def test_date_generator_exact_day_and_month():
    bounds = {
        'min_year': 1999,
        'max_year': 1999,
        'min_month': 12,
        'max_month': 12,
        'min_day': 15,
        'max_day': 15,
    }
    assert date_generator(bounds) == date(1999, 12, 15)


def test_int_list_generator_default_length():
    values = int_list_generator()
    upper = 9 * 10 ** INT_LENGTH
    assert isinstance(values, list)
    assert len(values) == INT_LIST_LENGTH
    assert all(type(v) is int and 0 <= v <= upper for v in values)


def test_int_list_generator_custom_lengths():
    assert int_list_generator(list_length=0) == []
    values = int_list_generator(list_length=3, int_length=1)
    assert len(values) == 3
    assert all(type(v) is int and 0 <= v <= 90 for v in values)


def test_int_dict_generator_shape():
    upper = 9 * 10 ** INT_LENGTH
    result = int_dict_generator()
    assert isinstance(result, dict)
    assert len(result) == INT_LIST_LENGTH
    for key, value in result.items():
        assert isinstance(key, str) and len(key) == STR_LENGTH
        assert type(value) is int and 0 <= value <= upper


def test_default_generators_return_annotations():
    annotations = [signature(g).return_annotation for g in default_generators()]
    assert annotations == [
        int,
        bool,
        str,
        datetime,
        date,
        List[int],
        Dict[str, int],
    ]


def test_generate_returns_named_dataclass():
    cls = generate(Simple)
    assert is_dataclass(cls)
    assert cls.__name__ == 'Simple'
    assert [f.name for f in fields(cls)] == ['count', 'flag', 'name']


def test_generate_as_dict_value_types():
    values = generate(Simple, as_dict=True)
    assert set(values) == {'count', 'flag', 'name'}
    assert type(values['count']) is int
    assert type(values['flag']) is bool
    assert isinstance(values['name'], str) and len(values['name']) == STR_LENGTH


def test_generate_as_json_round_trip():
    text = generate(Simple, as_json=True)
    assert isinstance(text, str)
    data = json.loads(text)
    assert set(data) == {'count', 'flag', 'name'}
    assert type(data['count']) is int
    assert type(data['flag']) is bool
    assert isinstance(data['name'], str)


def test_generate_keeps_declared_defaults():
    assert generate(WithDefault, as_dict=True) == {'count': 7, 'name': 'fixed'}


def test_generate_instances_are_independent():
    cls = generate(Simple)
    first = cls()
    second = cls()
    assert first.name != second.name


def test_generate_custom_int_generator_overrides_only_int():
    def constant_int() -> int:
        return 42

    values = generate(Simple, generators=[constant_int], as_dict=True)
    assert values['count'] == 42
    assert type(values['flag']) is bool
    assert len(values['name']) == STR_LENGTH


def test_generate_custom_list_generator():
    def fixed_list() -> List[int]:
        return [1, 2, 3]

    values = generate(Listy, generators=[fixed_list], as_dict=True)
    assert values == {'values': [1, 2, 3]}


def test_generate_custom_datetime_generator():
    def fixed_dt() -> datetime:
        return datetime(2000, 1, 1, 12, 30, 0)

    values = generate(DateOnly, generators=[fixed_dt], as_dict=True)
    assert values == {'when': datetime(2000, 1, 1, 12, 30, 0)}


def test_generate_unsupported_type_raises():
    with pytest.raises(NotSupportedType):
        generate(Floaty)
