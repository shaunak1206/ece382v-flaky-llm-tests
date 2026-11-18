import json
import random
from dataclasses import dataclass, field
from datetime import date, datetime

import pytest

import fistro.fistro as fistro_mod
from fistro.fistro import generate, enrich_fields
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
class Sample:
    x: int
    y: str
    z: bool = True


def test_int_generator_is_deterministic_with_seed():
    random.seed(12345)
    first = int_generator()
    random.seed(12345)
    second = int_generator()
    assert first == second


def test_int_generator_respects_length_and_bounds():
    random.seed(1)
    value = int_generator(length=4)
    assert 0 <= value <= 9 * 10 ** 4
    assert isinstance(value, int)


def test_bool_generator_is_deterministic_with_seed():
    random.seed(2024)
    first = bool_generator()
    random.seed(2024)
    second = bool_generator()
    assert first == second
    assert isinstance(first, bool)


def test_str_generator_is_deterministic_with_seed_and_length():
    random.seed(7)
    first = str_generator(length=8)
    random.seed(7)
    second = str_generator(length=8)
    assert first == second
    assert len(first) == 8
    assert all(ch in str_generator.__defaults__[0] for ch in first)


def test_datetime_generator_is_deterministic_with_seed():
    random.seed(99)
    first = datetime_generator()
    random.seed(99)
    second = datetime_generator()
    assert first == second
    assert isinstance(first, datetime)


def test_datetime_generator_respects_custom_bounds():
    random.seed(5)
    result = datetime_generator(
        {
            "min_year": 2020,
            "max_year": 2020,
            "min_month": 2,
            "max_month": 2,
            "min_day": 3,
            "max_day": 3,
            "min_hour": 4,
            "max_hour": 4,
            "min_minute": 5,
            "max_minute": 5,
            "min_second": 6,
            "max_second": 6,
        }
    )
    assert result == datetime(2020, 2, 3, 4, 5, 6)


def test_date_generator_is_deterministic_with_seed():
    random.seed(101)
    first = date_generator()
    random.seed(101)
    second = date_generator()
    assert first == second
    assert isinstance(first, date)


def test_date_generator_respects_custom_bounds():
    random.seed(12)
    result = date_generator(
        {
            "min_year": 2019,
            "max_year": 2019,
            "min_month": 11,
            "max_month": 11,
            "min_day": 22,
            "max_day": 22,
        }
    )
    assert result == date(2019, 11, 22)


def test_int_list_generator_is_deterministic_with_seed_and_length():
    random.seed(77)
    first = int_list_generator(list_length=5, int_length=2)
    random.seed(77)
    second = int_list_generator(list_length=5, int_length=2)
    assert first == second
    assert len(first) == 5
    assert all(isinstance(v, int) and 0 <= v <= 9 * 10 ** 2 for v in first)


def test_int_dict_generator_is_deterministic_with_seed():
    random.seed(123)
    first = int_dict_generator(dict_length=3, int_length=1)
    random.seed(123)
    second = int_dict_generator(dict_length=3, int_length=1)
    assert first == second
    assert len(first) == 3
    assert all(isinstance(k, str) for k in first.keys())
    assert all(isinstance(v, int) and 0 <= v <= 9 * 10 ** 1 for v in first.values())


def test_default_generators_contains_expected_names():
    names = [gen.__name__ for gen in default_generators()]
    assert names == [
        "int_generator",
        "bool_generator",
        "str_generator",
        "datetime_generator",
        "date_generator",
        "int_list_generator",
        "int_dict_generator",
    ]


def test_enrich_fields_uses_default_values_without_factory(monkeypatch):
    calls = []

    class FakeFactory:
        def __init__(self, generators=None):
            calls.append(tuple(generators) if generators is not None else None)

        def factory(self, tp):
            def _gen():
                return f"generated-{tp.__name__}"

            return _gen

    monkeypatch.setattr(fistro_mod, "Factory", FakeFactory)
    enriched = enrich_fields(Sample, generators=None)

    assert [name for name, _, _ in enriched] == ["x", "y", "z"]
    assert enriched[2][2].default is True
    assert enriched[0][2].default_factory() == "generated-int"
    assert enriched[1][2].default_factory() == "generated-str"
    assert calls == [None, None]


def test_generate_returns_dataclass_type(monkeypatch):
    class FakeFactory:
        def __init__(self, generators=None):
            self.generators = generators

        def factory(self, tp):
            return lambda: 1 if tp is int else "abc"

    monkeypatch.setattr(fistro_mod, "Factory", FakeFactory)

    generated = generate(Sample)
    instance = generated()
    assert generated.__name__ == "Sample"
    assert instance.x == 1
    assert instance.y == "abc"
    assert instance.z is True


def test_generate_as_dict(monkeypatch):
    class FakeFactory:
        def __init__(self, generators=None):
            self.generators = generators

        def factory(self, tp):
            return lambda: 7 if tp is int else "hello"

    monkeypatch.setattr(fistro_mod, "Factory", FakeFactory)

    result = generate(Sample, as_dict=True)
    assert result == {"x": 7, "y": "hello", "z": True}


def test_generate_as_json(monkeypatch):
    class FakeFactory:
        def __init__(self, generators=None):
            self.generators = generators

        def factory(self, tp):
            return lambda: 3 if tp is int else "world"

    monkeypatch.setattr(fistro_mod, "Factory", FakeFactory)

    result = generate(Sample, as_json=True)
    assert json.loads(result) == {"x": 3, "y": "world", "z": True}
