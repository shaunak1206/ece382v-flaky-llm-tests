import dataclasses
import json
import random
from dataclasses import dataclass, fields, is_dataclass

import pytest

import fistro.generators as generators
from fistro.config import INT_LENGTH
from fistro.fistro import generate
from fistro.generators import int_generator


@dataclass
class Person:
    name: str
    age: int
    active: bool


@dataclass
class WithDefault:
    age: int
    city: str = 'Paris'


@dataclass
class Empty:
    pass


@dataclass
class OnlyInt:
    value: int


def fixed_int() -> int:
    return 7


# ---------------------------------------------------------------------------
# int_generator
# ---------------------------------------------------------------------------

def test_int_generator_returns_int_type():
    random.seed(1001)
    value = int_generator(2)
    assert type(value) is int


def test_int_generator_value_within_bounds_length_zero():
    random.seed(1002)
    for _ in range(500):
        value = int_generator(0)
        assert 0 <= value <= 9


def test_int_generator_value_within_bounds_length_three():
    random.seed(1003)
    for _ in range(500):
        value = int_generator(3)
        assert 0 <= value <= 9 * 10 ** 3


def test_int_generator_passes_expected_range_to_randint(monkeypatch):
    random.seed(1004)
    calls = []

    def fake_randint(low, high):
        calls.append((low, high))
        return 0

    monkeypatch.setattr(generators, 'randint', fake_randint)
    int_generator(4)
    assert calls == [(0, 9 * 10 ** 4)]


def test_int_generator_default_length_uses_int_length(monkeypatch):
    random.seed(1005)
    calls = []

    def fake_randint(low, high):
        calls.append((low, high))
        return 0

    monkeypatch.setattr(generators, 'randint', fake_randint)
    int_generator()
    assert calls == [(0, 9 * 10 ** INT_LENGTH)]


def test_int_generator_returns_randint_result(monkeypatch):
    random.seed(1006)

    def fake_randint(low, high):
        return 42

    monkeypatch.setattr(generators, 'randint', fake_randint)
    assert int_generator(1) == 42


def test_int_generator_is_reproducible_with_seed():
    random.seed(1007)
    first = [int_generator(5) for _ in range(20)]
    random.seed(1007)
    second = [int_generator(5) for _ in range(20)]
    assert first == second


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def test_generate_returns_dataclass_type_named_after_class():
    random.seed(2001)
    result = generate(Person)
    assert isinstance(result, type)
    assert is_dataclass(result)
    assert result.__name__ == 'Person'


def test_generate_field_names_preserve_order():
    random.seed(2002)
    result = generate(Person)
    assert [f.name for f in fields(result)] == ['name', 'age', 'active']


def test_generate_field_types_match_original():
    random.seed(2003)
    result = generate(Person)
    assert {f.name: f.type for f in fields(result)} == {
        'name': str,
        'age': int,
        'active': bool,
    }


def test_generate_as_dict_returns_dict_with_expected_keys_and_types():
    random.seed(2004)
    result = generate(Person, as_dict=True)
    assert isinstance(result, dict)
    assert set(result) == {'name', 'age', 'active'}
    assert isinstance(result['name'], str)
    assert type(result['age']) is int
    assert type(result['active']) is bool


def test_generate_as_json_returns_string_matching_dict_keys():
    random.seed(2005)
    result = generate(Person, as_json=True)
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert isinstance(parsed, dict)
    assert set(parsed) == {'name', 'age', 'active'}
    assert type(parsed['age']) is int
    assert type(parsed['active']) is bool


def test_generate_as_json_is_json_of_same_seeded_dict():
    random.seed(2006)
    as_dict = generate(Person, as_dict=True)
    random.seed(2006)
    as_json = generate(Person, as_json=True)
    assert json.loads(as_json) == as_dict


def test_generate_is_reproducible_with_seed():
    random.seed(2007)
    first = generate(Person, as_dict=True)
    random.seed(2007)
    second = generate(Person, as_dict=True)
    assert first == second


def test_generate_as_dict_takes_precedence_over_as_json():
    random.seed(2008)
    result = generate(Person, as_dict=True, as_json=True)
    assert isinstance(result, dict)


def test_generate_preserves_field_default_value():
    random.seed(2009)
    result = generate(WithDefault, as_dict=True)
    assert result['city'] == 'Paris'
    assert type(result['age']) is int


def test_generate_empty_class_has_no_fields():
    random.seed(2010)
    result = generate(Empty)
    assert is_dataclass(result)
    assert fields(result) == ()
    assert generate(Empty, as_dict=True) == {}


def test_generate_with_custom_generator_uses_it():
    random.seed(2011)
    result = generate(OnlyInt, generators=[fixed_int], as_dict=True)
    assert result == {'value': 7}


def test_generate_does_not_mutate_original_class():
    random.seed(2012)
    before = [(f.name, f.type) for f in fields(Person)]
    generate(Person, as_dict=True)
    after = [(f.name, f.type) for f in fields(Person)]
    assert before == after
    assert Person.__name__ == 'Person'
