import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List

import pytest

import fistro.fistro as fistro_mod
import fistro.generators as gen_mod
from fistro.fistro import (
    enrich_fields,
    generate,
    generate_from_json,
    get_class_body_from_annotations,
    inner_inspection,
)
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
class SimpleInput:
    a: int
    b: str
    c: bool = True


@dataclass
class WithDefaultsInput:
    x: int = 3
    y: str = "hello"


class DummyFactory:
    def __init__(self, generators=None):
        self.generators = generators

    def factory(self, typ):
        mapping = {
            int: lambda: 11,
            str: lambda: "generated",
            bool: lambda: False,
            list: lambda: [1, 2, 3],
            dict: lambda: {"k": "v"},
            datetime: lambda: datetime(2020, 1, 2, 3, 4, 5),
            date: lambda: date(2020, 1, 2),
        }
        return mapping.get(typ, lambda: f"unexpected:{typ}")


def test_int_generator_uses_randint_and_length(monkeypatch):
    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return 42

    monkeypatch.setattr(gen_mod, "randint", fake_randint)
    assert int_generator(length=4) == 42
    assert calls == [(0, 9 * 10 ** 4)]


def test_int_generator_default_length_uses_config_and_bounds(monkeypatch):
    monkeypatch.setattr(gen_mod, "randint", lambda a, b: b)
    result = int_generator()
    assert result == 9 * 10 ** gen_mod.INT_LENGTH


def test_bool_generator_uses_choice(monkeypatch):
    seen = []

    def fake_choice(seq):
        seen.append(tuple(seq))
        return True

    monkeypatch.setattr(gen_mod, "choice", fake_choice)
    assert bool_generator() is True
    assert seen == [(True, False)]


def test_str_generator_uses_choices_and_length(monkeypatch):
    seen = {}

    def fake_choices(population, k):
        seen["population"] = population
        seen["k"] = k
        return ["a", "b", "c"]

    monkeypatch.setattr(gen_mod, "choices", fake_choices)
    assert str_generator(population="abc", length=3) == "abc"
    assert seen == {"population": "abc", "k": 3}


def test_datetime_generator_returns_expected_datetime(monkeypatch):
    values = iter([2021, 2, 3, 4, 5, 6])

    def fake_randint(a, b):
        return next(values)

    monkeypatch.setattr(gen_mod, "randint", fake_randint)
    result = datetime_generator(
        {
            "min_year": 2020,
            "max_year": 2022,
            "min_month": 1,
            "max_month": 12,
            "min_day": 1,
            "max_day": 28,
            "min_hour": 0,
            "max_hour": 23,
            "min_minute": 0,
            "max_minute": 59,
            "min_second": 0,
            "max_second": 59,
        }
    )
    assert result == datetime(2021, 2, 3, 4, 5, 6)


def test_datetime_generator_uses_defaults_when_none(monkeypatch):
    seen = []

    def fake_randint(a, b):
        seen.append((a, b))
        return a

    monkeypatch.setattr(gen_mod, "randint", fake_randint)
    result = datetime_generator()
    assert isinstance(result, datetime)
    assert len(seen) == 6


def test_date_generator_returns_expected_date(monkeypatch):
    values = iter([2022, 7, 8])

    def fake_randint(a, b):
        return next(values)

    monkeypatch.setattr(gen_mod, "randint", fake_randint)
    result = date_generator({"min_year": 2020, "max_year": 2023})
    assert result == date(2022, 7, 8)


def test_int_list_generator_uses_randint_for_each_item(monkeypatch):
    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return 7

    monkeypatch.setattr(gen_mod, "randint", fake_randint)
    result = int_list_generator(list_length=4, int_length=2)
    assert result == [7, 7, 7, 7]
    assert calls == [(0, 9 * 10 ** 2)] * 4


def test_int_dict_generator_uses_str_generator_keys_and_randint(monkeypatch):
    key_iter = iter(["k1", "k2", "k3"])

    def fake_str_generator(*args, **kwargs):
        return next(key_iter)

    monkeypatch.setattr(gen_mod, "str_generator", fake_str_generator)
    monkeypatch.setattr(gen_mod, "randint", lambda a, b: 99)

    result = int_dict_generator(dict_length=3, int_length=1)
    assert result == {"k1": 99, "k2": 99, "k3": 99}


def test_default_generators_contains_expected_callables():
    gens = default_generators()
    assert gens == [
        int_generator,
        bool_generator,
        str_generator,
        datetime_generator,
        date_generator,
        int_list_generator,
        int_dict_generator,
    ]


def test_generate_returns_dataclass_type_and_uses_factories(monkeypatch):
    monkeypatch.setattr(fistro_mod, "Factory", DummyFactory)
    generated_cls = generate(SimpleInput)
    assert generated_cls.__name__ == "SimpleInput"
    instance = generated_cls()
    assert instance.a == 11
    assert instance.b == "generated"
    assert instance.c is True


def test_generate_as_dict_and_as_json(monkeypatch):
    monkeypatch.setattr(fistro_mod, "Factory", DummyFactory)

    as_dict_result = generate(SimpleInput, as_dict=True)
    assert as_dict_result == {"a": 11, "b": "generated", "c": True}

    as_json_result = generate(SimpleInput, as_json=True)
    assert json.loads(as_json_result) == {"a": 11, "b": "generated", "c": True}


def test_enrich_fields_preserves_default_and_uses_factory_for_missing(monkeypatch):
    monkeypatch.setattr(fistro_mod, "Factory", DummyFactory)
    fields_info = enrich_fields(WithDefaultsInput, None)

    assert fields_info[0][0] == "x"
    assert fields_info[1][0] == "y"

    generated_cls = generate(WithDefaultsInput)
    instance = generated_cls()
    assert instance.x == 3
    assert instance.y == "hello"


def test_inner_inspection_for_dict_list_and_scalar():
    inner_type, inner_field = inner_inspection({"name": "alice"})
    assert str(inner_type).startswith("typing.Dict")
    assert inner_field.default_factory() == {"name": "alice"}

    list_type, list_field = inner_inspection([1, 2, 3])
    assert str(list_type).startswith("typing.List")
    assert list_field.default_factory() == [1, 2, 3]

    scalar_type, scalar_field = inner_inspection(10)
    assert scalar_type is int
    assert scalar_field.default_factory() == 10


def test_generate_from_json_builds_dataclass_from_dict(monkeypatch):
    monkeypatch.setattr(fistro_mod, "valid_id", lambda: "generated_id")

    cls = generate_from_json({"name": "alice", "age": 30})
    assert cls.__name__ == "generated_id"

    instance = cls()
    assert instance.name == "alice"
    assert instance.age == 30


def test_generate_from_json_with_list_raises():
    with pytest.raises(TypeError):
        generate_from_json([1, 2, 3])


def test_get_class_body_from_annotations_formats_dict_list_and_scalar():
    body = get_class_body_from_annotations(
        {
            "a": int,
            "b": Dict[str, int],
            "c": List[str],
            "d": None,
        }
    )
    assert "a: int" in body
    assert "b: typing.Dict[str, int]" in body
    assert "c: typing.List[str]" in body
    assert "d: Any" in body
