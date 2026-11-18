import json
from dataclasses import dataclass, fields
from datetime import date, datetime

import pytest

import fistro.fistro as fistro_module
import fistro.generators as generators_module


@dataclass
class SampleModel:
    number: int
    text: str = "hello"


class DummyFactory:
    init_args = []

    def __init__(self, generators=None):
        DummyFactory.init_args.append(generators)
        self.generators = generators or []

    def factory(self, typ):
        # Return deterministic callables based on the requested field type.
        if typ is int:
            return lambda: 42
        if typ is str:
            return lambda: "generated"
        if typ is bool:
            return lambda: True
        if typ is list:
            return lambda: [1, 2, 3]
        if typ is dict:
            return lambda: {"a": 1}
        return lambda: None


def test_int_generator_uses_randint_upper_bound(monkeypatch):
    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return 7

    monkeypatch.setattr(generators_module, "randint", fake_randint)
    result = generators_module.int_generator(length=3)

    assert result == 7
    assert calls == [(0, 9 * 10 ** 3)]


def test_int_generator_default_length(monkeypatch):
    monkeypatch.setattr(generators_module, "randint", lambda a, b: b)
    result = generators_module.int_generator()

    assert result == 9 * 10 ** generators_module.INT_LENGTH


def test_bool_generator_returns_choice(monkeypatch):
    monkeypatch.setattr(generators_module, "choice", lambda seq: seq[1])

    assert generators_module.bool_generator() is False


def test_str_generator_respects_length_and_population(monkeypatch):
    seen = {}

    def fake_choices(population, k):
        seen["population"] = population
        seen["k"] = k
        return ["x"] * k

    monkeypatch.setattr(generators_module, "choices", fake_choices)
    result = generators_module.str_generator(population="abc", length=4)

    assert result == "xxxx"
    assert seen == {"population": "abc", "k": 4}


def test_datetime_generator_uses_defaults_and_parses_datetime(monkeypatch):
    values = iter([2020, 1, 2, 3, 4, 5])

    monkeypatch.setattr(generators_module, "randint", lambda a, b: next(values))
    result = generators_module.datetime_generator()

    assert isinstance(result, datetime)
    assert result == datetime(2020, 1, 2, 3, 4, 5)


def test_datetime_generator_accepts_custom_ranges(monkeypatch):
    seen = []

    def fake_randint(a, b):
        seen.append((a, b))
        return a

    monkeypatch.setattr(generators_module, "randint", fake_randint)
    result = generators_module.datetime_generator(
        {
            "min_year": 2001,
            "max_year": 2002,
            "min_month": 3,
            "max_month": 4,
            "min_day": 5,
            "max_day": 6,
            "min_hour": 7,
            "max_hour": 8,
            "min_minute": 9,
            "max_minute": 10,
            "min_second": 11,
            "max_second": 12,
        }
    )

    assert result == datetime(2001, 3, 5, 7, 9, 11)
    assert seen == [
        (2001, 2002),
        (3, 4),
        (5, 6),
        (7, 8),
        (9, 10),
        (11, 12),
    ]


def test_date_generator_uses_defaults_and_parses_date(monkeypatch):
    values = iter([2021, 6, 7])

    monkeypatch.setattr(generators_module, "randint", lambda a, b: next(values))
    result = generators_module.date_generator()

    assert isinstance(result, date)
    assert result == date(2021, 6, 7)


def test_int_list_generator_length_and_values(monkeypatch):
    monkeypatch.setattr(generators_module, "randint", lambda a, b: 9)

    result = generators_module.int_list_generator(list_length=3, int_length=2)

    assert result == [9, 9, 9]


def test_int_dict_generator_uses_str_generator_and_randint(monkeypatch):
    keys = iter(["k1", "k2"])
    values = iter([11, 22])

    monkeypatch.setattr(generators_module, "str_generator", lambda: next(keys))
    monkeypatch.setattr(generators_module, "randint", lambda a, b: next(values))

    result = generators_module.int_dict_generator(dict_length=2, int_length=1)

    assert result == {"k1": 11, "k2": 22}


def test_default_generators_contains_expected_callables():
    names = [gen.__name__ for gen in generators_module.default_generators()]

    assert names == [
        "int_generator",
        "bool_generator",
        "str_generator",
        "datetime_generator",
        "date_generator",
        "int_list_generator",
        "int_dict_generator",
    ]


def test_generate_returns_dataclass_with_generated_defaults(monkeypatch):
    monkeypatch.setattr(fistro_module, "Factory", DummyFactory)

    Generated = fistro_module.generate(SampleModel)

    assert Generated.__name__ == "SampleModel"
    assert [f.name for f in fields(Generated)] == ["number", "text"]

    instance = Generated()
    assert instance.number == 42
    assert instance.text == "generated"


def test_generate_as_dict_returns_generated_values(monkeypatch):
    monkeypatch.setattr(fistro_module, "Factory", DummyFactory)

    result = fistro_module.generate(SampleModel, as_dict=True)

    assert result == {"number": 42, "text": "generated"}


def test_generate_as_json_returns_json_string(monkeypatch):
    monkeypatch.setattr(fistro_module, "Factory", DummyFactory)

    result = fistro_module.generate(SampleModel, as_json=True)

    assert json.loads(result) == {"number": 42, "text": "generated"}


def test_generate_uses_provided_generators_list(monkeypatch):
    DummyFactory.init_args = []
    monkeypatch.setattr(fistro_module, "Factory", DummyFactory)

    custom_generators = [lambda: "custom"]
    Generated = fistro_module.generate(SampleModel, generators=custom_generators)

    assert Generated().number == 42
    assert Generated().text == "generated"
    assert DummyFactory.init_args[-1] == custom_generators


def test_generate_preserves_explicit_field_default(monkeypatch):
    monkeypatch.setattr(fistro_module, "Factory", DummyFactory)

    @dataclass
    class WithDefault:
        x: int
        y: str = "fixed"

    Generated = fistro_module.generate(WithDefault)

    instance = Generated()
    assert instance.y == "fixed"
    assert instance.x == 42


def test_generate_from_json_builds_dataclass_from_mapping(monkeypatch):
    monkeypatch.setattr(fistro_module, "valid_id", lambda: "json_model")

    Generated = fistro_module.generate_from_json({"name": "alice", "age": 30})

    assert Generated.__name__ == "json_model"
    assert [f.name for f in fields(Generated)] == ["name", "age"]

    instance = Generated()
    assert instance.name == "alice"
    assert instance.age == 30
