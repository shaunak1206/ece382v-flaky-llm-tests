import dataclasses
import json
import string
from dataclasses import MISSING, dataclass, fields, is_dataclass
from datetime import date, datetime

import fistro.generators as generators_module
from fistro.config import INT_LENGTH, INT_LIST_LENGTH, MAX_YEAR, MIN_YEAR, STR_LENGTH
from fistro.fistro import enrich_fields, generate
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


# ---------- helper dataclasses used by generate / enrich_fields tests ----------

@dataclass
class Simple:
    count: int
    label: str
    flag: bool


@dataclass
class WithDefault:
    retries: int = 5


@dataclass
class OnlyInt:
    x: int


def fixed_int() -> int:
    return 42


# ---------- int_generator ----------

def test_int_generator_returns_int_within_default_range():
    value = int_generator()
    assert isinstance(value, int)
    assert 0 <= value <= 9 * 10 ** INT_LENGTH


def test_int_generator_passes_length_to_randint(monkeypatch):
    calls = []

    def fake_randint(low, high):
        calls.append((low, high))
        return 123

    monkeypatch.setattr(generators_module, 'randint', fake_randint)
    assert int_generator(3) == 123
    assert calls == [(0, 9 * 10 ** 3)]


def test_int_generator_length_zero_upper_bound_is_nine(monkeypatch):
    calls = []

    def fake_randint(low, high):
        calls.append((low, high))
        return 0

    monkeypatch.setattr(generators_module, 'randint', fake_randint)
    int_generator(0)
    assert calls == [(0, 9)]


# ---------- other primitive generators ----------

def test_bool_generator_returns_bool():
    for _ in range(50):
        assert isinstance(bool_generator(), bool)


def test_bool_generator_returns_choice_result(monkeypatch):
    monkeypatch.setattr(generators_module, 'choice', lambda seq: seq[0])
    assert bool_generator() is True


def test_str_generator_default_length_and_population():
    value = str_generator()
    assert isinstance(value, str)
    assert len(value) == STR_LENGTH
    assert set(value) <= set(string.printable)


def test_str_generator_custom_population_and_length():
    value = str_generator(population='ab', length=7)
    assert len(value) == 7
    assert set(value) <= {'a', 'b'}


# ---------- datetime_generator / date_generator ----------

def test_datetime_generator_fixed_bounds_gives_exact_value():
    bounds = {
        'min_year': 2000, 'max_year': 2000,
        'min_month': 1, 'max_month': 1,
        'min_day': 2, 'max_day': 2,
        'min_hour': 3, 'max_hour': 3,
        'min_minute': 4, 'max_minute': 4,
        'min_second': 5, 'max_second': 5,
    }
    result = datetime_generator(bounds)
    assert isinstance(result, datetime)
    assert result == datetime(2000, 1, 2, 3, 4, 5)


def test_datetime_generator_default_bounds_returns_datetime_in_year_range():
    result = datetime_generator()
    assert isinstance(result, datetime)
    assert MIN_YEAR <= result.year <= MAX_YEAR


def test_date_generator_fixed_bounds_gives_exact_value():
    bounds = {
        'min_year': 2001, 'max_year': 2001,
        'min_month': 2, 'max_month': 2,
        'min_day': 3, 'max_day': 3,
    }
    result = date_generator(bounds)
    assert type(result) is date
    assert result == date(2001, 2, 3)


def test_date_generator_default_returns_plain_date_in_year_range():
    result = date_generator()
    assert type(result) is date
    assert MIN_YEAR <= result.year <= MAX_YEAR


# ---------- list / dict generators ----------

def test_int_list_generator_respects_length_and_bounds():
    result = int_list_generator(list_length=4, int_length=2)
    assert isinstance(result, list)
    assert len(result) == 4
    assert all(isinstance(v, int) and 0 <= v <= 900 for v in result)


def test_int_list_generator_zero_length_returns_empty_list():
    assert int_list_generator(0) == []


def test_int_dict_generator_keys_are_str_and_values_are_int(monkeypatch):
    counter = iter(range(1000))
    monkeypatch.setattr(
        generators_module, 'str_generator', lambda: f'key{next(counter)}'
    )
    result = int_dict_generator(dict_length=3, int_length=1)
    assert len(result) == 3
    assert all(isinstance(k, str) for k in result)
    assert all(isinstance(v, int) and 0 <= v <= 90 for v in result.values())


def test_default_generators_contains_all_builtin_generators():
    generators = default_generators()
    assert len(generators) == 7
    for expected in (
        int_generator,
        bool_generator,
        str_generator,
        datetime_generator,
        date_generator,
        int_list_generator,
        int_dict_generator,
    ):
        assert expected in generators


# ---------- generate ----------

def test_generate_returns_dataclass_with_same_name_and_fields():
    result = generate(Simple)
    assert is_dataclass(result)
    assert result.__name__ == 'Simple'
    assert [f.name for f in fields(result)] == ['count', 'label', 'flag']


def test_generate_default_returns_instantiable_class_with_typed_values():
    cls = generate(Simple)
    instance = cls()
    assert isinstance(instance.count, int)
    assert isinstance(instance.label, str)
    assert isinstance(instance.flag, bool)


def test_generate_as_dict_returns_dict_with_field_names():
    result = generate(Simple, as_dict=True)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'count', 'label', 'flag'}
    assert isinstance(result['count'], int)
    assert isinstance(result['label'], str)
    assert isinstance(result['flag'], bool)


def test_generate_as_json_returns_parseable_json_string():
    result = generate(Simple, as_json=True)
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert set(parsed.keys()) == {'count', 'label', 'flag'}


def test_generate_as_dict_takes_precedence_over_as_json():
    result = generate(Simple, as_dict=True, as_json=True)
    assert isinstance(result, dict)


def test_generate_preserves_default_values():
    result = generate(WithDefault, as_dict=True)
    assert result == {'retries': 5}


def test_generate_uses_custom_generators():
    result = generate(OnlyInt, generators=[fixed_int], as_dict=True)
    assert result == {'x': 42}


def test_generate_with_empty_generators_list_falls_back_to_defaults():
    result = generate(OnlyInt, generators=[], as_dict=True)
    assert isinstance(result['x'], int)


def test_enrich_fields_returns_name_type_field_tuples():
    enriched = enrich_fields(WithDefault, None)
    assert len(enriched) == 1
    name, ftype, fld = enriched[0]
    assert name == 'retries'
    assert ftype is int
    assert isinstance(fld, dataclasses.Field)
    assert fld.default == 5
    assert fld.default_factory is MISSING
