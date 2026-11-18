import json
from dataclasses import dataclass
from datetime import date, datetime

import pytest

import fistro.fistro as fistro_module
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


def test_int_generator_uses_randint_and_default_length(monkeypatch):
    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return 123

    monkeypatch.setattr("fistro.generators.randint", fake_randint)

    result = int_generator()
    assert result == 123
    assert calls == [(0, 9 * 10 ** fistro_module.INT_LENGTH)] if hasattr(fistro_module, "INT_LENGTH") else [(0, 9 * 10 ** 1)]


def test_int_generator_custom_length(monkeypatch):
    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return 7

    monkeypatch.setattr("fistro.generators.randint", fake_randint)

    result = int_generator(length=3)
    assert result == 7
    assert calls == [(0, 9000)]


def test_bool_generator_uses_choice(monkeypatch):
    seen = {}

    def fake_choice(options):
        seen["options"] = options
        return True

    monkeypatch.setattr("fistro.generators.choice", fake_choice)

    result = bool_generator()
    assert result is True
    assert seen["options"] == [True, False]


def test_str_generator_uses_choices_and_length(monkeypatch):
    seen = {}

    def fake_choices(population, k):
        seen["population"] = population
        seen["k"] = k
        return ["a", "b", "c"]

    monkeypatch.setattr("fistro.generators.choices", fake_choices)

    result = str_generator(population="abc", length=3)
    assert result == "abc"
    assert seen["population"] == "abc"
    assert seen["k"] == 3


def test_datetime_generator_returns_datetime_and_uses_defaults(monkeypatch):
    values = iter([2020, 1, 2, 3, 4, 5])
    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return next(values)

    monkeypatch.setattr("fistro.generators.randint", fake_randint)

    result = datetime_generator()
    assert isinstance(result, datetime)
    assert result == datetime(2020, 1, 2, 3, 4, 5)
    assert len(calls) == 6


def test_datetime_generator_respects_custom_bounds(monkeypatch):
    values = iter([2021, 6, 15, 12, 30, 45])
    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return next(values)

    monkeypatch.setattr("fistro.generators.randint", fake_randint)

    result = datetime_generator(
        {
            "min_year": 2021,
            "max_year": 2021,
            "min_month": 6,
            "max_month": 6,
            "min_day": 15,
            "max_day": 15,
            "min_hour": 12,
            "max_hour": 12,
            "min_minute": 30,
            "max_minute": 30,
            "min_second": 45,
            "max_second": 45,
        }
    )
    assert result == datetime(2021, 6, 15, 12, 30, 45)
    assert calls == [
        (2021, 2021),
        (6, 6),
        (15, 15),
        (12, 12),
        (30, 30),
        (45, 45),
    ]


def test_date_generator_returns_date_and_uses_defaults(monkeypatch):
    values = iter([2022, 7, 8])
    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return next(values)

    monkeypatch.setattr("fistro.generators.randint", fake_randint)

    result = date_generator()
    assert isinstance(result, date)
    assert result == date(2022, 7, 8)
    assert len(calls) == 3


def test_date_generator_respects_custom_bounds(monkeypatch):
    values = iter([2030, 12, 31])

    def fake_randint(a, b):
        return next(values)

    monkeypatch.setattr("fistro.generators.randint", fake_randint)

    result = date_generator({"min_year": 2030, "max_year": 2030, "min_month": 12, "max_month": 12, "min_day": 31, "max_day": 31})
    assert result == date(2030, 12, 31)


def test_int_list_generator_length_and_values(monkeypatch):
    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return len(calls)

    monkeypatch.setattr("fistro.generators.randint", fake_randint)

    result = int_list_generator(list_length=4, int_length=2)
    assert result == [1, 2, 3, 4]
    assert calls == [(0, 900)] * 4


def test_int_dict_generator_length_and_keys(monkeypatch):
    keys = iter(["k1", "k2", "k3"])

    def fake_str_generator(*args, **kwargs):
        return next(keys)

    values = iter([10, 11, 12])

    def fake_randint(a, b):
        return next(values)

    monkeypatch.setattr("fistro.generators.str_generator", fake_str_generator)
    monkeypatch.setattr("fistro.generators.randint", fake_randint)

    result = int_dict_generator(dict_length=3, int_length=1)
    assert result == {"k1": 10, "k2": 11, "k3": 12}


def test_default_generators_contains_expected_callables():
    generators = default_generators()
    assert generators == [
        int_generator,
        bool_generator,
        str_generator,
        datetime_generator,
        date_generator,
        int_list_generator,
        int_dict_generator,
    ]


def test_generate_returns_dataclass_type_and_instantiates_defaults(monkeypatch):
    @dataclass
    class Sample:
        x: int
        y: str = "hello"

    class FakeFactory:
        def __init__(self, generators=None):
            self.generators = generators

        def factory(self, typ):
            if typ is int:
                return lambda: 42
            if typ is str:
                return lambda: "generated"
            return lambda: None

    monkeypatch.setattr(fistro_module, "Factory", FakeFactory)

    result_cls = fistro_module.generate(Sample)
    instance = result_cls()

    assert result_cls.__name__ == "Sample"
    assert instance.x == 42
    assert instance.y == "hello"


def test_generate_as_dict_returns_dictionary(monkeypatch):
    @dataclass
    class Sample:
        x: int
        y: str = "hello"

    class FakeFactory:
        def __init__(self, generators=None):
            self.generators = generators

        def factory(self, typ):
            return lambda: 7 if typ is int else "generated"

    monkeypatch.setattr(fistro_module, "Factory", FakeFactory)

    result = fistro_module.generate(Sample, as_dict=True)
    assert result == {"x": 7, "y": "hello"}


def test_generate_as_json_returns_json_string(monkeypatch):
    @dataclass
    class Sample:
        x: int
        y: str = "hello"

    class FakeFactory:
        def __init__(self, generators=None):
            self.generators = generators

        def factory(self, typ):
            return lambda: 99 if typ is int else "world"

    monkeypatch.setattr(fistro_module, "Factory", FakeFactory)

    result = fistro_module.generate(Sample, as_json=True)
    assert json.loads(result) == {"x": 99, "y": "hello"}


def test_generate_passes_custom_generators_to_factory(monkeypatch):
    @dataclass
    class Sample:
        x: int

    captured = {}

    class FakeFactory:
        def __init__(self, generators=None):
            captured["generators"] = generators

        def factory(self, typ):
            return lambda: 1

    custom_generators = [lambda: 123]
    monkeypatch.setattr(fistro_module, "Factory", FakeFactory)

    result_cls = fistro_module.generate(Sample, generators=custom_generators)
    assert result_cls().x == 1
    assert captured["generators"] == custom_generators
