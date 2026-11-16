import dataclasses
import json
import string
from dataclasses import dataclass
from datetime import date, datetime

import fistro.generators as generators
from fistro.config import INT_LENGTH, STR_LENGTH
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


@dataclass
class Person:
    name: str
    age: int


@dataclass
class WithDefault:
    count: int = 5


# ---------- int_generator ----------

def test_int_generator_default_bounds(monkeypatch):
    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return a

    monkeypatch.setattr(generators, 'randint', fake_randint)
    int_generator()
    assert calls == [(0, 9 * 10 ** INT_LENGTH)]


def test_int_generator_custom_length_bounds(monkeypatch):
    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return b

    monkeypatch.setattr(generators, 'randint', fake_randint)
    result = int_generator(3)
    assert calls == [(0, 9000)]
    assert result == 9000


def test_int_generator_returns_int_within_range():
    for _ in range(50):
        value = int_generator(1)
        assert isinstance(value, int)
        assert 0 <= value <= 90


# ---------- bool_generator ----------

def test_bool_generator_returns_only_bools():
    values = {bool_generator() for _ in range(100)}
    assert values <= {True, False}
    assert all(isinstance(v, bool) for v in values)


def test_bool_generator_delegates_to_choice(monkeypatch):
    calls = []

    def fake_choice(seq):
        calls.append(list(seq))
        return False

    monkeypatch.setattr(generators, 'choice', fake_choice)
    assert bool_generator() is False
    assert calls == [[True, False]]


# ---------- str_generator ----------

def test_str_generator_default_length_and_population():
    value = str_generator()
    assert isinstance(value, str)
    assert len(value) == STR_LENGTH
    assert set(value) <= set(string.printable)


def test_str_generator_custom_length_and_population():
    value = str_generator(population='ab', length=5)
    assert len(value) == 5
    assert set(value) <= {'a', 'b'}


# ---------- datetime_generator ----------

def test_datetime_generator_fixed_bounds():
    rand_date = {
        'min_year': 2001, 'max_year': 2001,
        'min_month': 2, 'max_month': 2,
        'min_day': 3, 'max_day': 3,
        'min_hour': 4, 'max_hour': 4,
        'min_minute': 5, 'max_minute': 5,
        'min_second': 6, 'max_second': 6,
    }
    assert datetime_generator(rand_date) == datetime(2001, 2, 3, 4, 5, 6)


def test_datetime_generator_none_returns_datetime():
    result = datetime_generator(None)
    assert isinstance(result, datetime)


# ---------- date_generator ----------

def test_date_generator_fixed_bounds():
    rand_date = {
        'min_year': 1999, 'max_year': 1999,
        'min_month': 12, 'max_month': 12,
        'min_day': 31, 'max_day': 31,
    }
    assert date_generator(rand_date) == date(1999, 12, 31)


def test_date_generator_returns_date_not_datetime():
    result = date_generator(None)
    assert type(result) is date


# ---------- int_list_generator / int_dict_generator ----------

def test_int_list_generator_length_and_range():
    result = int_list_generator(list_length=4, int_length=1)
    assert len(result) == 4
    assert all(isinstance(x, int) and 0 <= x <= 90 for x in result)


def test_int_list_generator_zero_length():
    assert int_list_generator(list_length=0) == []


def test_int_dict_generator_shape():
    result = int_dict_generator(dict_length=3)
    assert isinstance(result, dict)
    assert 1 <= len(result) <= 3  # random keys may collide
    assert all(isinstance(k, str) for k in result)
    assert all(isinstance(v, int) for v in result.values())


# ---------- default_generators ----------

def test_default_generators_contents():
    gens = default_generators()
    assert len(gens) == 7
    assert all(callable(g) for g in gens)
    for g in (int_generator, bool_generator, str_generator,
              datetime_generator, date_generator,
              int_list_generator, int_dict_generator):
        assert g in gens


# ---------- generate ----------

def test_generate_returns_dataclass_named_like_input():
    result = generate(Person)
    assert dataclasses.is_dataclass(result)
    assert result.__name__ == 'Person'


def test_generate_default_instance_has_typed_fields():
    instance = generate(Person)()
    assert isinstance(instance.name, str)
    assert isinstance(instance.age, int)


def test_generate_as_dict_returns_dict_with_field_keys():
    result = generate(Person, as_dict=True)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'name', 'age'}
    assert isinstance(result['name'], str)
    assert isinstance(result['age'], int)


def test_generate_as_json_returns_valid_json_string():
    result = generate(Person, as_json=True)
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert set(parsed.keys()) == {'name', 'age'}


def test_generate_preserves_default_values():
    assert generate(WithDefault, as_dict=True) == {'count': 5}
    assert generate(WithDefault)().count == 5


def test_generate_with_custom_generators():
    result = generate(Person, generators=[int_generator, str_generator], as_dict=True)
    assert isinstance(result['name'], str)
    assert isinstance(result['age'], int)


def test_generate_as_dict_takes_precedence_over_as_json():
    result = generate(Person, as_dict=True, as_json=True)
    assert isinstance(result, dict)
