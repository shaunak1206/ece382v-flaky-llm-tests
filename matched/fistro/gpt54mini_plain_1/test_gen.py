import json
from dataclasses import dataclass
from datetime import date, datetime

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
class SimpleModel:
    a: int
    b: str


@dataclass
class DefaultedModel:
    a: int = 7
    b: str = "hello"


@dataclass
class MixedModel:
    a: int
    b: str = "x"


class DummyFactory:
    def __init__(self, generators=None):
        self.generators = generators or []

    def factory(self, typ):
        if typ is int:
            return lambda: 11
        if typ is str:
            return lambda: "generated"
        if typ is bool:
            return lambda: True
        if typ is list:
            return lambda: [1, 2, 3]
        return lambda: f"fallback:{getattr(typ, '__name__', str(typ))}"


def test_int_generator_returns_int_in_expected_range(monkeypatch):
    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return b

    monkeypatch.setattr(gen_mod, "randint", fake_randint)
    result = int_generator(3)
    assert result == 9 * 10 ** 3
    assert calls == [(0, 9 * 10 ** 3)]


def test_int_generator_uses_default_length(monkeypatch):
    monkeypatch.setattr(gen_mod, "randint", lambda a, b: b)
    assert int_generator() == 9 * 10 ** gen_mod.INT_LENGTH


def test_str_generator_uses_choosen_population_and_length(monkeypatch):
    monkeypatch.setattr(gen_mod, "choices", lambda population, k: list("abc")[:k])
    assert str_generator(population="abcd", length=3) == "abc"


def test_bool_generator_returns_value_from_choice(monkeypatch):
    monkeypatch.setattr(gen_mod, "choice", lambda seq: seq[1])
    assert bool_generator() is False


def test_int_list_generator_uses_requested_length(monkeypatch):
    monkeypatch.setattr(gen_mod, "randint", lambda a, b: 7)
    result = int_list_generator(list_length=4, int_length=2)
    assert result == [7, 7, 7, 7]


def test_int_dict_generator_uses_string_keys_and_int_values(monkeypatch):
    keys = iter(["k1", "k2", "k3"])
    monkeypatch.setattr(gen_mod, "str_generator", lambda: next(keys))
    monkeypatch.setattr(gen_mod, "randint", lambda a, b: 5)
    result = int_dict_generator(dict_length=3, int_length=2)
    assert result == {"k1": 5, "k2": 5, "k3": 5}


def test_datetime_generator_uses_bounds_and_returns_datetime(monkeypatch):
    values = iter([2020, 1, 2, 3, 4, 5])

    def fake_randint(a, b):
        return next(values)

    monkeypatch.setattr(gen_mod, "randint", fake_randint)
    result = datetime_generator(
        {
            "min_year": 2000,
            "max_year": 2025,
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
    assert isinstance(result, datetime)
    assert result == datetime(2020, 1, 2, 3, 4, 5)


def test_date_generator_uses_bounds_and_returns_date(monkeypatch):
    values = iter([2021, 6, 7])

    def fake_randint(a, b):
        return next(values)

    monkeypatch.setattr(gen_mod, "randint", fake_randint)
    result = date_generator(
        {"min_year": 2000, "max_year": 2025, "min_month": 1, "max_month": 12, "min_day": 1, "max_day": 28}
    )
    assert isinstance(result, date)
    assert result == date(2021, 6, 7)


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


def test_enrich_fields_uses_factory_for_missing_defaults(monkeypatch):
    monkeypatch.setattr(fistro_mod, "Factory", DummyFactory)
    enriched = enrich_fields(SimpleModel, generators=None)
    assert [name for name, _, _ in enriched] == ["a", "b"]
    assert enriched[0][2].default_factory() == 11
    assert enriched[1][2].default_factory() == "generated"


def test_enrich_fields_preserves_explicit_defaults(monkeypatch):
    monkeypatch.setattr(fistro_mod, "Factory", DummyFactory)
    enriched = enrich_fields(DefaultedModel, generators=None)
    assert enriched[0][2].default == 7
    assert enriched[1][2].default == "hello"
    assert enriched[0][2].default_factory is pytest.raises  # sentinel-like check replaced below


def test_generate_returns_dataclass_type_and_instantiates_with_factories(monkeypatch):
    monkeypatch.setattr(fistro_mod, "Factory", DummyFactory)
    Generated = generate(SimpleModel)
    assert Generated.__name__ == "SimpleModel"
    obj = Generated()
    assert obj.a == 11
    assert obj.b == "generated"
    assert obj.__dataclass_fields__["a"].default is not pytest.FROZENSET


def test_generate_as_dict_returns_dict(monkeypatch):
    monkeypatch.setattr(fistro_mod, "Factory", DummyFactory)
    result = generate(SimpleModel, as_dict=True)
    assert result == {"a": 11, "b": "generated"}


def test_generate_as_json_returns_json_string(monkeypatch):
    monkeypatch.setattr(fistro_mod, "Factory", DummyFactory)
    result = generate(SimpleModel, as_json=True)
    assert json.loads(result) == {"a": 11, "b": "generated"}


def test_generate_prefers_as_dict_over_as_json(monkeypatch):
    monkeypatch.setattr(fistro_mod, "Factory", DummyFactory)
    result = generate(SimpleModel, as_dict=True, as_json=True)
    assert isinstance(result, dict)
    assert result == {"a": 11, "b": "generated"}


def test_generate_with_custom_generators_passes_them_into_factory(monkeypatch):
    seen = {}

    class RecordingFactory(DummyFactory):
        def __init__(self, generators=None):
            seen["generators"] = generators
            super().__init__(generators=generators)

    monkeypatch.setattr(fistro_mod, "Factory", RecordingFactory)
    custom = [lambda: 1]
    result = generate(MixedModel, generators=custom, as_dict=True)
    assert seen["generators"] == custom
    assert result == {"a": 11, "b": "x"}


def test_generate_from_json_creates_dataclass_with_fields(monkeypatch):
    monkeypatch.setattr(fistro_mod, "valid_id", lambda: "generated_name")
    Generated = generate_from_json({"a": 1, "b": "x"})
    assert Generated.__name__ == "generated_name"
    obj = Generated()
    assert obj.a == 1
    assert obj.b == "x"


def test_inner_inspection_handles_dict_list_and_scalar():
    inner_type, fld = inner_inspection({"x": 1})
    assert "dict" in str(inner_type).lower()
    assert fld.default_factory() == {"x": 1}

    inner_type, fld = inner_inspection([1, 2, 3])
    assert "list" in str(inner_type).lower()
    assert fld.default_factory() == [1, 2, 3]

    inner_type, fld = inner_inspection(5)
    assert inner_type is int
    assert fld.default_factory() == 5


def test_get_class_body_from_annotations_formats_basic_types():
    body = get_class_body_from_annotations({"a": int, "b": str})
    assert body == "a: int\nb: str\n"


def test_get_class_body_from_annotations_handles_dict_list_and_none():
    body = get_class_body_from_annotations({"a": dict, "b": list, "c": None})
    assert "a: dict" in body
    assert "b: list" in body
    assert "c: Any" in body
