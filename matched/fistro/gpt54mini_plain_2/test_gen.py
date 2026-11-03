import json
from dataclasses import dataclass, field
from datetime import datetime, date

import pytest

import fistro.fistro as fistro_module
from fistro.fistro import generate, generate_from_json, inner_inspection, get_class_body_from_annotations
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
class SimpleModel:
    a: int
    b: str = "hello"
    c: bool = True


@dataclass
class MixedModel:
    x: int
    y: str
    z: bool = False


def test_int_generator_uses_randint_and_respects_length(monkeypatch):
    calls = []

    def fake_randint(low, high):
        calls.append((low, high))
        return high

    monkeypatch.setattr(fistro_module, "randint", fake_randint)
    value = int_generator(length=3)

    assert value == 9 * 10 ** 3
    assert calls == [(0, 9 * 10 ** 3)]


def test_bool_generator_uses_choice(monkeypatch):
    monkeypatch.setattr(fistro_module, "choice", lambda seq: seq[1])

    assert bool_generator() is False


def test_str_generator_uses_choices_and_length(monkeypatch):
    def fake_choices(population, k):
        assert population == "abc"
        assert k == 4
        return ["a", "b", "c", "a"]

    monkeypatch.setattr(fistro_module, "choices", fake_choices)

    assert str_generator(population="abc", length=4) == "abca"


def test_datetime_generator_returns_expected_datetime(monkeypatch):
    values = iter([2020, 5, 6, 7, 8, 9])

    monkeypatch.setattr(fistro_module, "randint", lambda low, high: next(values))

    value = datetime_generator(
        {
            "min_year": 2000,
            "max_year": 2025,
            "min_month": 1,
            "max_month": 12,
            "min_day": 1,
            "max_day": 31,
            "min_hour": 0,
            "max_hour": 23,
            "min_minute": 0,
            "max_minute": 59,
            "min_second": 0,
            "max_second": 59,
        }
    )

    assert value == datetime(2020, 5, 6, 7, 8, 9)


def test_date_generator_returns_expected_date(monkeypatch):
    values = iter([2022, 12, 31])

    monkeypatch.setattr(fistro_module, "randint", lambda low, high: next(values))

    value = date_generator({"min_year": 2000, "max_year": 2025})

    assert value == date(2022, 12, 31)


def test_int_list_generator_length_and_values(monkeypatch):
    monkeypatch.setattr(fistro_module, "randint", lambda low, high: high)

    value = int_list_generator(list_length=3, int_length=2)

    assert value == [9 * 10 ** 2, 9 * 10 ** 2, 9 * 10 ** 2]


def test_int_dict_generator_length_and_keys(monkeypatch):
    keys = iter(["k1", "k2"])
    monkeypatch.setattr(fistro_module, "str_generator", lambda: next(keys))
    monkeypatch.setattr(fistro_module, "randint", lambda low, high: high)

    value = int_dict_generator(dict_length=2, int_length=1)

    assert value == {"k1": 9 * 10 ** 1, "k2": 9 * 10 ** 1}


def test_default_generators_contains_expected_functions():
    gens = default_generators()
    names = [g.__name__ for g in gens]

    assert names == [
        "int_generator",
        "bool_generator",
        "str_generator",
        "datetime_generator",
        "date_generator",
        "int_list_generator",
        "int_dict_generator",
    ]


def test_inner_inspection_for_scalar_value_returns_type_and_default_factory():
    inner_type, fld = inner_inspection(123)

    assert inner_type is int
    assert fld.default_factory() == 123


def test_inner_inspection_for_list_and_dict_values():
    list_type, list_field = inner_inspection([1, 2, 3])
    dict_type, dict_field = inner_inspection({"a": 1})

    assert list_type.__origin__ is list
    assert list_field.default_factory() == [1, 2, 3]
    assert dict_type.__origin__ is dict
    assert dict_field.default_factory() == {"a": 1}


def test_generate_returns_dataclass_type_with_defaults(monkeypatch):
    class FakeFactory:
        def __init__(self, generators=None):
            self.generators = generators

        def factory(self, typ):
            return {
                int: lambda: 42,
                str: lambda: "generated",
                bool: lambda: False,
            }[typ]

    monkeypatch.setattr(fistro_module, "Factory", FakeFactory)

    generated_cls = generate(MixedModel)

    assert generated_cls.__name__ == "MixedModel"
    instance = generated_cls()
    assert instance.x == 42
    assert instance.y == "generated"
    assert instance.z is False
    assert generated_cls.__dataclass_fields__["z"].default is False


def test_generate_with_as_dict_returns_dictionary(monkeypatch):
    class FakeFactory:
        def __init__(self, generators=None):
            self.generators = generators

        def factory(self, typ):
            return {int: lambda: 1, str: lambda: "x"}[typ]

    monkeypatch.setattr(fistro_module, "Factory", FakeFactory)

    result = generate(SimpleModel, as_dict=True)

    assert result == {"a": 1, "b": "hello", "c": True}


def test_generate_with_as_json_returns_json_string(monkeypatch):
    class FakeFactory:
        def __init__(self, generators=None):
            self.generators = generators

        def factory(self, typ):
            return {int: lambda: 7, str: lambda: "json", bool: lambda: False}[typ]

    monkeypatch.setattr(fistro_module, "Factory", FakeFactory)

    result = generate(MixedModel, as_json=True)

    assert json.loads(result) == {"x": 7, "y": "json", "z": False}


def test_generate_passes_custom_generators_to_factory(monkeypatch):
    recorded = {}

    class FakeFactory:
        def __init__(self, generators=None):
            recorded["generators"] = generators

        def factory(self, typ):
            return {int: lambda: 5, str: lambda: "s", bool: lambda: True}[typ]

    custom_generators = [lambda: "custom"]
    monkeypatch.setattr(fistro_module, "Factory", FakeFactory)

    generated_cls = generate(SimpleModel, generators=custom_generators)

    assert recorded["generators"] == custom_generators
    assert generated_cls().a == 5


def test_generate_from_json_creates_dataclass_with_expected_fields(monkeypatch):
    monkeypatch.setattr(fistro_module, "valid_id", lambda: "MyGeneratedModel")

    generated_cls = generate_from_json({"alpha": 1, "beta": {"nested": "x"}, "gamma": [1, 2]})
    field_names = list(generated_cls.__dataclass_fields__.keys())

    assert generated_cls.__name__ == "MyGeneratedModel"
    assert field_names == ["alpha", "beta", "gamma"]
    instance = generated_cls()
    assert instance.alpha == 1
    assert instance.beta == {"nested": "x"}
    assert instance.gamma == [1, 2]
