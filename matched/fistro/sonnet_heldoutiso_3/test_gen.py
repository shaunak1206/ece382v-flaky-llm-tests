import json
import string
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime

import pytest

from fistro import generators
from fistro.fistro import generate
from fistro.generators import (
    int_generator,
    bool_generator,
    str_generator,
    datetime_generator,
    date_generator,
    int_list_generator,
    int_dict_generator,
    default_generators,
)


@dataclass
class Person:
    name: str
    age: int
    active: bool


@dataclass
class WithDefault:
    name: str
    age: int = 7


# ---------------- generate ----------------

def test_generate_returns_dataclass():
    result = generate(Person)
    assert is_dataclass(result)
    assert isinstance(result, type)


def test_generate_keeps_class_name_and_fields():
    result = generate(Person)
    assert result.__name__ == 'Person'
    assert [f.name for f in fields(result)] == ['name', 'age', 'active']


def test_generate_instance_has_typed_values():
    instance = generate(Person)()
    assert isinstance(instance.name, str)
    assert isinstance(instance.age, int)
    assert isinstance(instance.active, bool)


def test_generate_as_dict_returns_dict_with_keys():
    result = generate(Person, as_dict=True)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'name', 'age', 'active'}
    assert isinstance(result['age'], int)


def test_generate_as_json_returns_valid_json():
    result = generate(Person, as_json=True)
    assert isinstance(result, str)
    loaded = json.loads(result)
    assert set(loaded.keys()) == {'name', 'age', 'active'}


def test_generate_preserves_default_values():
    assert generate(WithDefault)().age == 7
    assert generate(WithDefault, as_dict=True)['age'] == 7


def test_generate_default_value_not_overridden_but_others_generated():
    result = generate(WithDefault, as_dict=True)
    assert isinstance(result['name'], str)


def test_generate_with_custom_generator():
    def constant_int() -> int:
        return 42

    result = generate(Person, generators=[constant_int], as_dict=True)
    assert result['age'] == 42


def test_generate_instances_are_independent_classes_each_call():
    first = generate(Person)
    second = generate(Person)
    assert first is not second


# ---------------- int_generator ----------------

@pytest.mark.parametrize('length', [0, 1, 2, 5])
def test_int_generator_within_bounds(length):
    for _ in range(50):
        value = int_generator(length)
        assert isinstance(value, int)
        assert 0 <= value <= 9 * 10 ** length


def test_int_generator_default_returns_int():
    assert isinstance(int_generator(), int)


def test_int_generator_upper_bound_uses_length(monkeypatch):
    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return b

    monkeypatch.setattr(generators, 'randint', fake_randint)
    assert int_generator(3) == 9000
    assert calls == [(0, 9000)]


def test_int_generator_lower_bound(monkeypatch):
    monkeypatch.setattr(generators, 'randint', lambda a, b: a)
    assert int_generator(4) == 0


# ---------------- other generators ----------------

def test_bool_generator_returns_bool():
    values = {bool_generator() for _ in range(100)}
    assert values <= {True, False}
    assert all(isinstance(v, bool) for v in values)


def test_str_generator_length_and_population():
    value = str_generator(population='abc', length=20)
    assert isinstance(value, str)
    assert len(value) == 20
    assert set(value) <= set('abc')


def test_str_generator_default_population_printable():
    value = str_generator(length=30)
    assert len(value) == 30
    assert set(value) <= set(string.printable)


def test_datetime_generator_fixed_range():
    rand = {
        'min_year': 2020, 'max_year': 2020,
        'min_month': 5, 'max_month': 5,
        'min_day': 10, 'max_day': 10,
        'min_hour': 3, 'max_hour': 3,
        'min_minute': 4, 'max_minute': 4,
        'min_second': 5, 'max_second': 5,
    }
    assert datetime_generator(rand) == datetime(2020, 5, 10, 3, 4, 5)


def test_date_generator_fixed_range():
    rand = {
        'min_year': 2001, 'max_year': 2001,
        'min_month': 2, 'max_month': 2,
        'min_day': 3, 'max_day': 3,
    }
    result = date_generator(rand)
    assert result == date(2001, 2, 3)
    assert not isinstance(result, datetime)


def test_int_list_generator_length_and_bounds():
    result = int_list_generator(list_length=6, int_length=2)
    assert len(result) == 6
    assert all(isinstance(i, int) and 0 <= i <= 900 for i in result)


def test_int_dict_generator_types():
    result = int_dict_generator(dict_length=3, int_length=2)
    assert 1 <= len(result) <= 3
    assert all(isinstance(k, str) for k in result)
    assert all(isinstance(v, int) and 0 <= v <= 900 for v in result.values())


def test_default_generators_contents():
    gens = default_generators()
    assert int_generator in gens
    assert str_generator in gens
    assert bool_generator in gens
    assert len(gens) == 7
    assert all(callable(g) for g in gens)
