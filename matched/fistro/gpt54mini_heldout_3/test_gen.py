import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

import fistro.fistro as fistro_mod
import fistro.generators as gen_mod


@dataclass
class Sample:
    a: int
    b: str = "fixed"
    c: bool = True


@dataclass
class Mixed:
    x: int
    y: List[int]
    z: Dict[str, int]
    w: Any


def test_int_generator_uses_default_length_and_randint(monkeypatch):
    calls = []

    def fake_randint(low, high):
        calls.append((low, high))
        return 7

    monkeypatch.setattr(gen_mod, "randint", fake_randint)
    result = gen_mod.int_generator()

    assert result == 7
    assert calls == [(0, 9 * 10 ** gen_mod.INT_LENGTH)]


def test_int_generator_honors_custom_length(monkeypatch):
    calls = []

    def fake_randint(low, high):
        calls.append((low, high))
        return high

    monkeypatch.setattr(gen_mod, "randint", fake_randint)
    result = gen_mod.int_generator(length=3)

    assert result == 9 * 10 ** 3
    assert calls == [(0, 9 * 10 ** 3)]


def test_bool_generator_returns_choice_value(monkeypatch):
    monkeypatch.setattr(gen_mod, "choice", lambda seq: seq[1])
    assert gen_mod.bool_generator() is False


def test_str_generator_uses_choices_and_length(monkeypatch):
    seen = {}

    def fake_choices(population, k):
        seen["population"] = population
        seen["k"] = k
        return ["a", "b", "c"]

    monkeypatch.setattr(gen_mod, "choices", fake_choices)
    result = gen_mod.str_generator(population="abc", length=3)

    assert result == "abc"
    assert seen == {"population": "abc", "k": 3}


def test_datetime_generator_uses_defaults_and_returns_datetime(monkeypatch):
    calls = []

    def fake_randint(low, high):
        calls.append((low, high))
        return low

    monkeypatch.setattr(gen_mod, "randint", fake_randint)
    result = gen_mod.datetime_generator()

    assert isinstance(result, datetime)
    assert calls[0] == (gen_mod.MIN_YEAR, gen_mod.MAX_YEAR)
    assert calls[1] == (gen_mod.MIN_MONTH, gen_mod.MAX_MONTH)
    assert calls[2] == (gen_mod.MIN_DAY, gen_mod.MAX_DAY)
    assert calls[3] == (gen_mod.MIN_HOUR, gen_mod.MAX_HOUR)
    assert calls[4] == (gen_mod.MIN_MINUTE, gen_mod.MAX_MINUTE)
    assert calls[5] == (gen_mod.MIN_SECOND, gen_mod.MAX_SECOND)


def test_datetime_generator_respects_overrides(monkeypatch):
    calls = []

    def fake_randint(low, high):
        calls.append((low, high))
        return low

    monkeypatch.setattr(gen_mod, "randint", fake_randint)
    result = gen_mod.datetime_generator(
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

    assert isinstance(result, datetime)
    assert calls == [(2001, 2002), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]


def test_date_generator_returns_date_and_uses_defaults(monkeypatch):
    calls = []

    def fake_randint(low, high):
        calls.append((low, high))
        return low

    monkeypatch.setattr(gen_mod, "randint", fake_randint)
    result = gen_mod.date_generator()

    assert isinstance(result, date)
    assert calls == [
        (gen_mod.MIN_YEAR, gen_mod.MAX_YEAR),
        (gen_mod.MIN_MONTH, gen_mod.MAX_MONTH),
        (gen_mod.MIN_DAY, gen_mod.MAX_DAY),
    ]


def test_int_list_generator_length_and_bounds(monkeypatch):
    values = iter([1, 2, 3, 4])

    def fake_randint(low, high):
        return next(values)

    monkeypatch.setattr(gen_mod, "randint", fake_randint)
    result = gen_mod.int_list_generator(list_length=4, int_length=2)

    assert result == [1, 2, 3, 4]


def test_int_dict_generator_length_and_keys(monkeypatch):
    keys = iter(["k1", "k2", "k3"])
    values = iter([10, 20, 30])

    monkeypatch.setattr(gen_mod, "str_generator", lambda: next(keys))
    monkeypatch.setattr(gen_mod, "randint", lambda low, high: next(values))

    result = gen_mod.int_dict_generator(dict_length=3, int_length=1)

    assert result == {"k1": 10, "k2": 20, "k3": 30}


def test_default_generators_contains_expected_callables():
    gens = gen_mod.default_generators()

    assert [g.__name__ for g in gens] == [
        "int_generator",
        "bool_generator",
        "str_generator",
        "datetime_generator",
        "date_generator",
        "int_list_generator",
        "int_dict_generator",
    ]


def test_inner_inspection_for_primitive_returns_type_and_factory():
    inner_type, f = fistro_mod.inner_inspection(123)
    assert inner_type is int
    assert f.default_factory() == 123


def test_inner_inspection_for_list_uses_first_item_type():
    inner_type, f = fistro_mod.inner_inspection([1, 2, 3])

    assert getattr(inner_type, "__origin__", None) is list
    assert f.default_factory() == [1, 2, 3]


def test_inner_inspection_for_dict_uses_key_and_value_types():
    inner_type, f = fistro_mod.inner_inspection({"a": 1})

    assert getattr(inner_type, "__origin__", None) is dict
    assert f.default_factory() == {"a": 1}


def test_generate_returns_dataclass_type_and_default_fields(monkeypatch):
    class DummyFactory:
        def __init__(self, generators=None):
            self.generators = generators

        def factory(self, typ):
            mapping = {int: lambda: 99, str: lambda: "hello", bool: lambda: False}
            return mapping[typ]

    monkeypatch.setattr(fistro_mod, "Factory", DummyFactory)

    generated = fistro_mod.generate(Sample)

    assert generated.__name__ == "Sample"
    obj = generated()
    assert (obj.a, obj.b, obj.c) == (99, "fixed", True)


def test_generate_as_dict_returns_instance_as_dict(monkeypatch):
    class DummyFactory:
        def __init__(self, generators=None):
            pass

        def factory(self, typ):
            return {int: lambda: 5, str: lambda: "x", bool: lambda: False}[typ]

    monkeypatch.setattr(fistro_mod, "Factory", DummyFactory)

    result = fistro_mod.generate(Sample, as_dict=True)

    assert result == {"a": 5, "b": "fixed", "c": True}


def test_generate_as_json_returns_json_string(monkeypatch):
    class DummyFactory:
        def __init__(self, generators=None):
            pass

        def factory(self, typ):
            return {int: lambda: 7, str: lambda: "json", bool: lambda: False}[typ]

    monkeypatch.setattr(fistro_mod, "Factory", DummyFactory)

    result = fistro_mod.generate(Sample, as_json=True)

    assert json.loads(result) == {"a": 7, "b": "fixed", "c": True}


def test_generate_uses_custom_generators_when_provided(monkeypatch):
    fake_factory = Mock()

    class DummyFactory:
        def __init__(self, generators=None):
            fake_factory.generators = generators

        def factory(self, typ):
            fake_factory(typ)
            return lambda: {"int": 1, "str": "s", "bool": True}[typ.__name__]

    monkeypatch.setattr(fistro_mod, "Factory", DummyFactory)
    custom = [lambda: None]

    generated = fistro_mod.generate(Sample, generators=custom)

    assert fake_factory.generators == custom
    assert generated().a == 1


def test_generate_from_json_builds_dataclass_from_dict(monkeypatch):
    monkeypatch.setattr(fistro_mod, "valid_id", lambda: "GeneratedName")

    generated = fistro_mod.generate_from_json({"x": 1, "y": "abc"})

    assert generated.__name__ == "GeneratedName"
    obj = generated()
    assert obj.x == 1
    assert obj.y == "abc"


def test_generate_from_json_rejects_list_input():
    with pytest.raises(NotImplementedError):
        fistro_mod.generate_from_json([1, 2, 3])


def test_get_class_body_from_annotations_formats_primitives_and_containers():
    body = fistro_mod.get_class_body_from_annotations(
        {"a": int, "b": List[int], "c": Dict[str, int], "d": None}
    )

    assert "a: int" in body
    assert "b: typing.List[int]" in body or "b: List[int]" in body
    assert "c: typing.Dict[str, int]" in body or "c: Dict[str, int]" in body
    assert "d: Any" in body
