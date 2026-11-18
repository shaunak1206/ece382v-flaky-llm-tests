import json
from dataclasses import dataclass, field
from datetime import date, datetime

import pytest

import fistro.fistro as fio
import fistro.generators as gen


@dataclass
class Sample:
    a: int
    b: str = "fixed"


@dataclass
class Sample2:
    x: int
    y: bool


def test_int_generator_uses_randint_and_respects_length(monkeypatch):
    calls = {}

    def fake_randint(lo, hi):
        calls["args"] = (lo, hi)
        return 123

    monkeypatch.setattr(gen, "randint", fake_randint)
    result = gen.int_generator(length=3)

    assert result == 123
    assert calls["args"] == (0, 9 * 10**3)


def test_bool_generator_returns_choice_result(monkeypatch):
    monkeypatch.setattr(gen, "choice", lambda seq: seq[1])
    assert gen.bool_generator() is False


def test_str_generator_uses_choices_and_length(monkeypatch):
    calls = {}

    def fake_choices(population, k):
        calls["args"] = (population, k)
        return list("abc")

    monkeypatch.setattr(gen, "choices", fake_choices)
    result = gen.str_generator(population="xyz", length=3)

    assert result == "abc"
    assert calls["args"] == ("xyz", 3)


def test_int_list_generator_returns_expected_length_and_values(monkeypatch):
    vals = [1, 2, 3, 4]

    def fake_randint(lo, hi):
        return vals.pop(0)

    monkeypatch.setattr(gen, "randint", fake_randint)
    result = gen.int_list_generator(list_length=4, int_length=2)

    assert result == [1, 2, 3, 4]


def test_int_dict_generator_uses_str_generator_keys_and_randint_values(monkeypatch):
    keys = ["k1", "k2"]
    vals = [10, 20]

    monkeypatch.setattr(gen, "str_generator", lambda: keys.pop(0))

    def fake_randint(lo, hi):
        return vals.pop(0)

    monkeypatch.setattr(gen, "randint", fake_randint)
    result = gen.int_dict_generator(dict_length=2, int_length=1)

    assert result == {"k1": 10, "k2": 20}


def test_datetime_generator_uses_defaults_and_returns_datetime(monkeypatch):
    seq = [2020, 6, 15, 12, 30, 45]

    def fake_randint(lo, hi):
        return seq.pop(0)

    monkeypatch.setattr(gen, "randint", fake_randint)
    result = gen.datetime_generator()

    assert isinstance(result, datetime)
    assert result == datetime(2020, 6, 15, 12, 30, 45)


def test_datetime_generator_honors_custom_bounds(monkeypatch):
    seen = []

    def fake_randint(lo, hi):
        seen.append((lo, hi))
        return lo

    monkeypatch.setattr(gen, "randint", fake_randint)
    result = gen.datetime_generator(
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
    assert seen == [(2001, 2002), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]


def test_date_generator_uses_defaults_and_returns_date(monkeypatch):
    seq = [2021, 7, 8]

    def fake_randint(lo, hi):
        return seq.pop(0)

    monkeypatch.setattr(gen, "randint", fake_randint)
    result = gen.date_generator()

    assert isinstance(result, date)
    assert result == date(2021, 7, 8)


def test_date_generator_honors_custom_bounds(monkeypatch):
    seen = []

    def fake_randint(lo, hi):
        seen.append((lo, hi))
        return lo

    monkeypatch.setattr(gen, "randint", fake_randint)
    result = gen.date_generator({"min_year": 1999, "max_year": 2000, "min_month": 2, "max_month": 3, "min_day": 4, "max_day": 5})

    assert result == date(1999, 2, 4)
    assert seen == [(1999, 2000), (2, 3), (4, 5)]


def test_default_generators_contains_expected_callables():
    gens = gen.default_generators()

    assert len(gens) == 7
    assert [g.__name__ for g in gens] == [
        "int_generator",
        "bool_generator",
        "str_generator",
        "datetime_generator",
        "date_generator",
        "int_list_generator",
        "int_dict_generator",
    ]


def test_enrich_fields_uses_factory_without_custom_generators(monkeypatch):
    seen = {}

    class FakeFactory:
        def __init__(self, generators=None):
            seen["generators"] = generators

        def factory(self, typ):
            return lambda: f"value-for-{typ.__name__}"

    monkeypatch.setattr(fio, "Factory", FakeFactory)
    enriched = fio.enrich_fields(Sample2, generators=None)

    assert seen["generators"] is None
    assert [name for name, *_ in enriched] == ["x", "y"]
    assert all(callable(f.default_factory) for _, _, f in enriched)


def test_enrich_fields_passes_custom_generators(monkeypatch):
    seen = {}

    class FakeFactory:
        def __init__(self, generators=None):
            seen["generators"] = generators

        def factory(self, typ):
            return lambda: 99 if typ is int else True

    custom_gens = [lambda: None]
    monkeypatch.setattr(fio, "Factory", FakeFactory)
    enriched = fio.enrich_fields(Sample2, generators=custom_gens)

    assert seen["generators"] == custom_gens
    assert enriched[0][2].default_factory() == 99
    assert enriched[1][2].default_factory() is True


def test_generate_returns_dataclass_type(monkeypatch):
    class FakeFactory:
        def __init__(self, generators=None):
            pass

        def factory(self, typ):
            return lambda: 7 if typ is int else "hi"

    monkeypatch.setattr(fio, "Factory", FakeFactory)
    generated = fio.generate(Sample2)

    assert generated.__name__ == "Sample2"
    instance = generated()
    assert instance.x == 7
    assert instance.y == "hi"


def test_generate_as_dict_returns_instance_as_dict(monkeypatch):
    class FakeFactory:
        def __init__(self, generators=None):
            pass

        def factory(self, typ):
            return lambda: 1 if typ is int else "abc"

    monkeypatch.setattr(fio, "Factory", FakeFactory)
    result = fio.generate(Sample2, as_dict=True)

    assert result == {"x": 1, "y": "abc"}


def test_generate_as_json_returns_json_string(monkeypatch):
    class FakeFactory:
        def __init__(self, generators=None):
            pass

        def factory(self, typ):
            return lambda: 2 if typ is int else "z"

    monkeypatch.setattr(fio, "Factory", FakeFactory)
    result = fio.generate(Sample2, as_json=True)

    assert json.loads(result) == {"x": 2, "y": "z"}


def test_inner_inspection_for_dict_list_and_scalar():
    inner_type, inner_field = fio.inner_inspection({"a": 1})
    assert str(inner_type) == "typing.Dict[str, int]"
    assert inner_field.default_factory() == {"a": 1}

    inner_type, inner_field = fio.inner_inspection([1, 2])
    assert str(inner_type) == "typing.List[int]"
    assert inner_field.default_factory() == [1, 2]

    inner_type, inner_field = fio.inner_inspection("x")
    assert inner_type is str
    assert inner_field.default_factory() == "x"


def test_get_class_body_from_annotations_formats_primitives_dicts_and_lists():
    body = fio.get_class_body_from_annotations(
        {
            "a": int,
            "b": dict,
            "c": list,
            "d": None,
        }
    )

    assert "a: int" in body
    assert "b: <class 'dict'>" in body
    assert "c: <class 'list'>" in body
    assert "d: Any" in body


def test_generate_from_json_raises_typeerror_for_list_input():
    with pytest.raises(TypeError):
        fio.generate_from_json([1, 2, 3])
