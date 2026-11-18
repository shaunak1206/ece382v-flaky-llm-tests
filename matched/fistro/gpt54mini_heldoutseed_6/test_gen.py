import json
from dataclasses import dataclass, is_dataclass, fields
from datetime import date, datetime

import pytest

import fistro.fistro as fistro_module
from fistro.fistro import (
    generate,
    enrich_fields,
    inner_inspection,
    get_class_body_from_annotations,
)
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
class SampleModel:
    a: int
    b: str
    c: bool = True


@dataclass
class MixedModel:
    x: int
    y: str = "hello"
    z: list = None


class FakeFactory:
    def __init__(self, generators=None):
        self.generators = generators or []

    def factory(self, typ):
        # Deterministic test-only replacement for fistro.factory.Factory
        mapping = {
            int: lambda: 123,
            str: lambda: "abc",
            bool: lambda: False,
            list: lambda: [1, 2, 3],
            dict: lambda: {"k": "v"},
            datetime: lambda: datetime(2020, 1, 2, 3, 4, 5),
            date: lambda: date(2020, 1, 2),
        }
        return mapping.get(typ, lambda: None)


def test_int_generator_returns_int_and_within_expected_upper_bound():
    value = int_generator(3)
    assert isinstance(value, int)
    assert 0 <= value <= 9 * 10**3


def test_bool_generator_returns_boolean():
    value = bool_generator()
    assert isinstance(value, bool)


def test_str_generator_returns_string_of_requested_length():
    value = str_generator(population="ab", length=7)
    assert isinstance(value, str)
    assert len(value) == 7
    assert set(value).issubset({"a", "b"})


def test_datetime_generator_returns_datetime_within_custom_range():
    value = datetime_generator(
        {
            "min_year": 2021,
            "max_year": 2021,
            "min_month": 5,
            "max_month": 5,
            "min_day": 17,
            "max_day": 17,
            "min_hour": 8,
            "max_hour": 8,
            "min_minute": 9,
            "max_minute": 9,
            "min_second": 10,
            "max_second": 10,
        }
    )
    assert isinstance(value, datetime)
    assert value == datetime(2021, 5, 17, 8, 9, 10)


def test_date_generator_returns_date_within_custom_range():
    value = date_generator(
        {
            "min_year": 2022,
            "max_year": 2022,
            "min_month": 12,
            "max_month": 12,
            "min_day": 31,
            "max_day": 31,
        }
    )
    assert isinstance(value, date)
    assert value == date(2022, 12, 31)


def test_int_list_generator_returns_list_of_ints_with_requested_length():
    value = int_list_generator(list_length=4, int_length=2)
    assert isinstance(value, list)
    assert len(value) == 4
    assert all(isinstance(item, int) for item in value)
    assert all(0 <= item <= 9 * 10**2 for item in value)


def test_int_dict_generator_returns_dict_of_strings_to_ints_with_requested_length():
    value = int_dict_generator(dict_length=5, int_length=1)
    assert isinstance(value, dict)
    assert len(value) == 5
    assert all(isinstance(k, str) for k in value.keys())
    assert all(isinstance(v, int) for v in value.values())
    assert all(0 <= v <= 9 * 10**1 for v in value.values())


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


def test_enrich_fields_uses_default_factory_for_missing_fields(monkeypatch):
    monkeypatch.setattr(fistro_module, "Factory", FakeFactory)
    enriched = enrich_fields(SampleModel, generators=None)

    assert len(enriched) == 3
    assert enriched[0][0] == "a"
    assert enriched[1][0] == "b"
    assert enriched[2][0] == "c"

    generated_cls = generate(SampleModel)
    instance = generated_cls()
    assert instance.a == 123
    assert instance.b == "abc"
    assert instance.c is True


def test_generate_returns_dataclass_type(monkeypatch):
    monkeypatch.setattr(fistro_module, "Factory", FakeFactory)
    generated_cls = generate(MixedModel)
    assert is_dataclass(generated_cls)
    assert generated_cls.__name__ == "MixedModel"
    assert [f.name for f in fields(generated_cls)] == ["x", "y", "z"]


def test_generate_as_dict_returns_dictionary(monkeypatch):
    monkeypatch.setattr(fistro_module, "Factory", FakeFactory)
    result = generate(SampleModel, as_dict=True)
    assert result == {"a": 123, "b": "abc", "c": True}


def test_generate_as_json_returns_json_string(monkeypatch):
    monkeypatch.setattr(fistro_module, "Factory", FakeFactory)
    result = generate(SampleModel, as_json=True)
    assert isinstance(result, str)
    assert json.loads(result) == {"a": 123, "b": "abc", "c": True}


def test_generate_with_custom_generators_passes_them_to_factory(monkeypatch):
    captured = {}

    class CapturingFactory(FakeFactory):
        def __init__(self, generators=None):
            captured["generators"] = generators
            super().__init__(generators=generators)

    custom_generators = [lambda: 1]
    monkeypatch.setattr(fistro_module, "Factory", CapturingFactory)

    generated_cls = generate(SampleModel, generators=custom_generators)
    instance = generated_cls()

    assert captured["generators"] == custom_generators
    assert instance.a == 123
    assert instance.b == "abc"


def test_inner_inspection_for_scalar_value_returns_type_and_default_factory():
    inner_type, fld = inner_inspection(42)
    assert inner_type is int
    assert callable(fld.default_factory)
    assert fld.default_factory() == 42


def test_inner_inspection_for_list_and_dict_values_detects_container_types():
    list_type, list_field = inner_inspection([1, 2, 3])
    dict_type, dict_field = inner_inspection({"x": 1})

    assert str(list_type) == "typing.List[int]"
    assert "typing.Dict" in str(dict_type)
    assert list_field.default_factory() == [1, 2, 3]
    assert dict_field.default_factory() == {"x": 1}


def test_get_class_body_from_annotations_formats_simple_dict_and_list_types():
    body = get_class_body_from_annotations(
        {
            "a": int,
            "b": dict,
            "c": list,
            "d": None,
        }
    )
    assert "a: int" in body
    assert "b: <class 'dict'>" in body or "b: dict" in body
    assert "c: <class 'list'>" in body or "c: list" in body
    assert "d: Any" in body
