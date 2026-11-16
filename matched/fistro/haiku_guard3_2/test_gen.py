import json
import random
import string
from dataclasses import dataclass, fields
from datetime import date, datetime

from fistro import generators
from fistro.config import (
    INT_LENGTH,
    INT_LIST_LENGTH,
    MAX_YEAR,
    MIN_YEAR,
    STR_LENGTH,
)
from fistro.fistro import generate


@dataclass
class Person:
    name: str
    age: int


@dataclass
class WithDefault:
    role: str = 'admin'


@dataclass
class Point:
    x: int
    y: int


def fixed_int() -> int:
    return 7


def test_int_generator_calls_randint_with_bounds_from_length(monkeypatch):
    random.seed(0)
    calls = []

    def fake_randint(lo, hi):
        calls.append((lo, hi))
        return lo

    monkeypatch.setattr(generators, 'randint', fake_randint)
    assert generators.int_generator(3) == 0
    assert calls == [(0, 9 * 10 ** 3)]


def test_int_generator_default_length_stays_in_range():
    random.seed(0)
    upper = 9 * 10 ** INT_LENGTH
    for _ in range(50):
        value = generators.int_generator()
        assert isinstance(value, int)
        assert 0 <= value <= upper


def test_int_generator_zero_length_gives_single_digit():
    random.seed(0)
    for _ in range(50):
        value = generators.int_generator(0)
        assert 0 <= value <= 9


def test_int_generator_same_seed_gives_same_value():
    random.seed(1234)
    first = generators.int_generator(5)
    random.seed(1234)
    second = generators.int_generator(5)
    assert first == second


def test_bool_generator_returns_value_chosen_by_choice(monkeypatch):
    calls = []

    def fake_choice(seq):
        calls.append(list(seq))
        return True

    monkeypatch.setattr(generators, 'choice', fake_choice)
    assert generators.bool_generator() is True
    assert calls == [[True, False]]


def test_bool_generator_returns_bool():
    random.seed(0)
    for _ in range(50):
        assert isinstance(generators.bool_generator(), bool)


def test_str_generator_default_length_and_alphabet():
    random.seed(0)
    result = generators.str_generator()
    assert isinstance(result, str)
    assert len(result) == STR_LENGTH
    assert set(result) <= set(string.printable)


def test_str_generator_custom_population_and_length():
    random.seed(0)
    result = generators.str_generator('ab', 5)
    assert len(result) == 5
    assert set(result) <= {'a', 'b'}


def test_str_generator_zero_length_is_empty():
    random.seed(0)
    assert generators.str_generator('xyz', 0) == ''


def test_datetime_generator_defaults_read_config_bounds(monkeypatch):
    random.seed(0)
    calls = []

    def fake_randint(lo, hi):
        calls.append((lo, hi))
        return lo

    monkeypatch.setattr(generators, 'randint', fake_randint)
    generators.datetime_generator()
    assert len(calls) == 6
    assert calls[0] == (MIN_YEAR, MAX_YEAR)


def test_datetime_generator_min_values_build_expected_datetime(monkeypatch):
    random.seed(0)
    monkeypatch.setattr(generators, 'randint', lambda lo, hi: lo)
    rand_date = {
        'min_year': 2000,
        'min_month': 2,
        'min_day': 3,
        'min_hour': 4,
        'min_minute': 5,
        'min_second': 6,
    }
    result = generators.datetime_generator(rand_date)
    assert result == datetime(2000, 2, 3, 4, 5, 6)


def test_datetime_generator_max_values_build_expected_datetime(monkeypatch):
    random.seed(0)
    monkeypatch.setattr(generators, 'randint', lambda lo, hi: hi)
    rand_date = {
        'max_year': 2030,
        'max_month': 12,
        'max_day': 28,
        'max_hour': 23,
        'max_minute': 59,
        'max_second': 59,
    }
    result = generators.datetime_generator(rand_date)
    assert result == datetime(2030, 12, 28, 23, 59, 59)


def test_datetime_generator_random_within_constrained_range():
    random.seed(0)
    rand_date = {
        'min_year': 2000,
        'max_year': 2020,
        'min_month': 1,
        'max_month': 12,
        'min_day': 1,
        'max_day': 28,
        'min_hour': 0,
        'max_hour': 23,
        'min_minute': 0,
        'max_minute': 59,
        'min_second': 0,
        'max_second': 59,
    }
    for _ in range(20):
        result = generators.datetime_generator(rand_date)
        assert isinstance(result, datetime)
        assert 2000 <= result.year <= 2020
        assert 1 <= result.month <= 12
        assert 1 <= result.day <= 28


def test_date_generator_min_values_build_expected_date(monkeypatch):
    random.seed(0)
    monkeypatch.setattr(generators, 'randint', lambda lo, hi: lo)
    rand_date = {'min_year': 2000, 'min_month': 2, 'min_day': 3}
    assert generators.date_generator(rand_date) == date(2000, 2, 3)


def test_date_generator_returns_date_not_datetime():
    random.seed(0)
    rand_date = {
        'min_year': 2000,
        'max_year': 2020,
        'min_month': 1,
        'max_month': 12,
        'min_day': 1,
        'max_day': 28,
    }
    result = generators.date_generator(rand_date)
    assert isinstance(result, date)
    assert not isinstance(result, datetime)


def test_int_list_generator_length_and_bounds():
    random.seed(0)
    result = generators.int_list_generator(4, 1)
    assert isinstance(result, list)
    assert len(result) == 4
    for value in result:
        assert isinstance(value, int)
        assert 0 <= value <= 9


def test_int_list_generator_default_length():
    random.seed(0)
    assert len(generators.int_list_generator()) == INT_LIST_LENGTH


def test_int_dict_generator_keys_and_values():
    random.seed(0)
    # Random keys may collide, so the length is only bounded from above
    # and below by properties that hold for every outcome.
    result = generators.int_dict_generator(3, 1)
    assert isinstance(result, dict)
    assert 1 <= len(result) <= 3
    for key, value in result.items():
        assert isinstance(key, str)
        assert isinstance(value, int)
        assert 0 <= value <= 9


def test_int_dict_generator_zero_length_is_empty():
    random.seed(0)
    assert generators.int_dict_generator(0, 1) == {}


def test_default_generators_lists_all_generators():
    names = [g.__name__ for g in generators.default_generators()]
    assert names == [
        'int_generator',
        'bool_generator',
        'str_generator',
        'datetime_generator',
        'date_generator',
        'int_list_generator',
        'int_dict_generator',
    ]


def test_generate_returns_dataclass_with_same_name_and_fields():
    random.seed(0)
    cls = generate(Person)
    assert cls.__name__ == 'Person'
    assert tuple(f.name for f in fields(cls)) == ('name', 'age')


def test_generate_as_dict_has_field_keys_and_types():
    random.seed(0)
    result = generate(Person, as_dict=True)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'name', 'age'}
    assert isinstance(result['name'], str)
    assert isinstance(result['age'], int)


def test_generate_as_json_round_trips_to_field_keys():
    random.seed(0)
    result = generate(Person, as_json=True)
    assert isinstance(result, str)
    loaded = json.loads(result)
    assert set(loaded.keys()) == {'name', 'age'}


def test_generate_keeps_default_values():
    random.seed(0)
    assert generate(WithDefault, as_dict=True) == {'role': 'admin'}


def test_generate_uses_custom_generators():
    random.seed(0)
    assert generate(Point, generators=[fixed_int], as_dict=True) == {'x': 7, 'y': 7}
