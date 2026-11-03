import json
import random
import string
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime
from typing import Dict, List

import pytest

import fistro.generators as gens
from fistro.config import INT_LENGTH, INT_LIST_LENGTH, STR_LENGTH
from fistro.exceptions import NotSupportedType
from fistro.fistro import generate
from fistro.generators import (
    bool_generator,
    date_generator,
    datetime_generator,
    int_dict_generator,
    int_generator,
    int_list_generator,
    str_generator,
)

# Note: datetime_generator and date_generator with default ranges can build
# impossible dates (Feb 30, year 0), so the date tests pin ranges that always
# form valid calendar dates.


@pytest.fixture(autouse=True)
def _seed_random():
    random.seed(20241102)


@dataclass
class Plain:
    count: int
    flag: bool
    name: str


@dataclass
class WithDefaults:
    label: str = 'fixed'
    count: int = 3


@dataclass
class Bag:
    scores: List[int]
    meta: Dict[str, int]


@dataclass
class Ratio:
    value: float


class NotADataclass:
    def __init__(self):
        self.x = 1


def fixed_int() -> int:
    return 7


# int_generator


def test_int_generator_returns_bounded_int():
    upper = 9 * 10 ** INT_LENGTH
    for _ in range(200):
        value = int_generator()
        assert type(value) is int
        assert 0 <= value <= upper


def test_int_generator_length_controls_upper_bound():
    for _ in range(100):
        assert 0 <= int_generator(0) <= 9
    for _ in range(100):
        assert 0 <= int_generator(2) <= 900


def test_int_generator_same_seed_same_value():
    random.seed(7)
    first = int_generator()
    random.seed(7)
    second = int_generator(length=INT_LENGTH)
    assert first == second


def test_int_generator_passes_expected_bounds_to_randint(monkeypatch):
    calls = []

    def fake_randint(low, high):
        calls.append((low, high))
        return low

    monkeypatch.setattr(gens, 'randint', fake_randint)
    assert int_generator() == 0
    assert int_generator(4) == 0
    assert calls == [(0, 9 * 10 ** INT_LENGTH), (0, 9 * 10 ** 4)]


# bool_generator


def test_bool_generator_returns_both_bool_values():
    values = [bool_generator() for _ in range(100)]
    assert all(type(v) is bool for v in values)
    assert set(values) == {True, False}


def test_bool_generator_chooses_from_true_and_false(monkeypatch):
    seen = []

    def fake_choice(seq):
        seen.append(list(seq))
        return seq[0]

    monkeypatch.setattr(gens, 'choice', fake_choice)
    assert bool_generator() is True
    assert seen == [[True, False]]


# str_generator


def test_str_generator_default_length_and_population():
    value = str_generator()
    assert isinstance(value, str)
    assert len(value) == STR_LENGTH
    assert set(value) <= set(string.printable)


def test_str_generator_custom_population_and_length():
    value = str_generator(population='ab', length=12)
    assert len(value) == 12
    assert set(value) <= {'a', 'b'}


def test_str_generator_zero_length_returns_empty_string():
    assert str_generator(length=0) == ''


# datetime_generator and date_generator


def test_datetime_generator_pinned_range_returns_exact_value():
    pinned = {
        'min_year': 2020, 'max_year': 2020,
        'min_month': 6, 'max_month': 6,
        'min_day': 1, 'max_day': 1,
        'min_hour': 12, 'max_hour': 12,
        'min_minute': 30, 'max_minute': 30,
        'min_second': 45, 'max_second': 45,
    }
    value = datetime_generator(pinned)
    assert type(value) is datetime
    assert value == datetime(2020, 6, 1, 12, 30, 45)


def test_datetime_generator_stays_within_custom_bounds():
    bounds = {
        'min_year': 1990, 'max_year': 1995,
        'min_month': 3, 'max_month': 4,
        'min_day': 1, 'max_day': 28,
        'min_hour': 5, 'max_hour': 6,
    }
    for _ in range(100):
        value = datetime_generator(bounds)
        assert isinstance(value, datetime)
        assert 1990 <= value.year <= 1995
        assert value.month in (3, 4)
        assert 1 <= value.day <= 28
        assert value.hour in (5, 6)


def test_date_generator_pinned_range_returns_exact_date():
    pinned = {
        'min_year': 2021, 'max_year': 2021,
        'min_month': 3, 'max_month': 3,
        'min_day': 15, 'max_day': 15,
    }
    value = date_generator(pinned)
    assert type(value) is date
    assert value == date(2021, 3, 15)


def test_date_generator_stays_within_custom_bounds():
    bounds = {
        'min_year': 2000, 'max_year': 2010,
        'min_month': 7, 'max_month': 8,
        'min_day': 10, 'max_day': 20,
    }
    for _ in range(100):
        value = date_generator(bounds)
        assert type(value) is date
        assert 2000 <= value.year <= 2010
        assert value.month in (7, 8)
        assert 10 <= value.day <= 20


# int_list_generator and int_dict_generator


def test_int_list_generator_default_and_custom_lengths():
    values = int_list_generator()
    assert len(values) == INT_LIST_LENGTH
    assert all(type(v) is int and 0 <= v <= 9 * 10 ** INT_LENGTH for v in values)

    small = int_list_generator(list_length=3, int_length=1)
    assert len(small) == 3
    assert all(type(v) is int and 0 <= v <= 90 for v in small)


def test_int_dict_generator_has_string_keys_and_int_values():
    result = int_dict_generator(dict_length=3, int_length=2)
    assert isinstance(result, dict)
    assert len(result) <= 3
    assert all(isinstance(k, str) and len(k) == STR_LENGTH for k in result)
    assert all(type(v) is int and 0 <= v <= 900 for v in result.values())


# generate


def test_generate_returns_named_dataclass():
    cls = generate(Plain)
    assert is_dataclass(cls)
    assert cls.__name__ == 'Plain'
    assert [f.name for f in fields(cls)] == ['count', 'flag', 'name']


def test_generate_default_values_match_annotations():
    instance = generate(Plain)()
    assert type(instance.count) is int
    assert type(instance.flag) is bool
    assert type(instance.name) is str
    assert len(instance.name) == STR_LENGTH


def test_generate_as_dict_returns_plain_values():
    result = generate(Plain, as_dict=True)
    assert set(result) == {'count', 'flag', 'name'}
    assert type(result['count']) is int
    assert type(result['flag']) is bool
    assert type(result['name']) is str


def test_generate_as_json_round_trips_values():
    text = generate(Plain, as_json=True)
    assert isinstance(text, str)
    data = json.loads(text)
    assert set(data) == {'count', 'flag', 'name'}
    assert type(data['count']) is int
    assert type(data['flag']) is bool
    assert isinstance(data['name'], str)
    assert len(data['name']) == STR_LENGTH


def test_generate_supplied_generator_overrides_matching_type():
    result = generate(Plain, generators=[fixed_int], as_dict=True)
    assert result['count'] == 7
    assert type(result['flag']) is bool
    assert type(result['name']) is str


def test_generate_keeps_explicit_field_defaults():
    result = generate(WithDefaults, as_dict=True)
    assert result == {'label': 'fixed', 'count': 3}


def test_generate_builds_list_and_dict_fields():
    result = generate(Bag, as_dict=True)
    assert len(result['scores']) == INT_LIST_LENGTH
    assert all(type(v) is int for v in result['scores'])
    assert len(result['meta']) <= INT_LIST_LENGTH
    assert all(isinstance(k, str) for k in result['meta'])
    assert all(type(v) is int for v in result['meta'].values())


def test_generate_unsupported_type_raises():
    with pytest.raises(NotSupportedType):
        generate(Ratio)


def test_generate_rejects_non_dataclass():
    with pytest.raises(TypeError):
        generate(NotADataclass)
