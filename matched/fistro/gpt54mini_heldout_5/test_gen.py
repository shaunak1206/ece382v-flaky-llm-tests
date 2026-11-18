import json
from dataclasses import dataclass, field, fields
from datetime import date, datetime

import pytest

import fistro.fistro as fistro_module
import fistro.generators as gen_module


@dataclass
class Sample:
    x: int
    y: str
    z: bool = True


@dataclass
class WithDefaults:
    a: int = 5
    b: str = "hello"


def test_int_generator_uses_randint_with_expected_bounds(monkeypatch):
    calls = []

    def fake_randint(low, high):
        calls.append((low, high))
        return 7

    monkeypatch.setattr(gen_module, "randint", fake_randint)

    result = gen_module.int_generator(length=3)

    assert result == 7
    assert calls == [(0, 9 * 10 ** 3)]


def test_int_generator_default_length_uses_config_constant(monkeypatch):
    monkeypatch.setattr(gen_module, "INT_LENGTH", 4)
    monkeypatch.setattr(gen_module, "randint", lambda low, high: high)

    assert gen_module.int_generator() == 9 * 10 ** 4


def test_generate_returns_dataclass_type(monkeypatch):
    class DummyFactory:
        def __init__(self, generators=None):
            self.generators = generators

        def factory(self, typ):
            return lambda: 1 if typ is int else ("abc" if typ is str else False)

    monkeypatch.setattr(fistro_module, "Factory", DummyFactory)

    generated = fistro_module.generate(Sample)
    assert isinstance(generated, type)
    assert generated.__name__ == "Sample"
    instance = generated()
    assert instance.x == 1
    assert instance.y == "abc"
    assert instance.z is True


def test_generate_as_dict_returns_instance_dict(monkeypatch):
    class DummyFactory:
        def __init__(self, generators=None):
            pass

        def factory(self, typ):
            return lambda: 2 if typ is int else "x"

    monkeypatch.setattr(fistro_module, "Factory", DummyFactory)

    result = fistro_module.generate(Sample, as_dict=True)

    assert result == {"x": 2, "y": "x", "z": True}


def test_generate_as_json_returns_json_string(monkeypatch):
    class DummyFactory:
        def __init__(self, generators=None):
            pass

        def factory(self, typ):
            return lambda: 3 if typ is int else "y"

    monkeypatch.setattr(fistro_module, "Factory", DummyFactory)

    result = fistro_module.generate(Sample, as_json=True)
    assert json.loads(result) == {"x": 3, "y": "y", "z": True}


def test_generate_uses_default_values_without_factory_for_defaulted_fields(monkeypatch):
    calls = []

    class DummyFactory:
        def __init__(self, generators=None):
            self.generators = generators

        def factory(self, typ):
            calls.append(typ)
            return lambda: 99

    monkeypatch.setattr(fistro_module, "Factory", DummyFactory)

    generated = fistro_module.generate(WithDefaults)
    instance = generated()

    assert instance.a == 5
    assert instance.b == "hello"
    assert calls == []


def test_generate_passes_custom_generators_to_factory(monkeypatch):
    received = {}

    class DummyFactory:
        def __init__(self, generators=None):
            received["generators"] = generators

        def factory(self, typ):
            return lambda: 1

    custom_generators = [lambda: 42]
    monkeypatch.setattr(fistro_module, "Factory", DummyFactory)

    fistro_module.generate(Sample, generators=custom_generators)

    assert received["generators"] is custom_generators


def test_enrich_fields_builds_default_factories_for_missing_fields(monkeypatch):
    class DummyFactory:
        def __init__(self, generators=None):
            self.generators = generators

        def factory(self, typ):
            return lambda: f"value-for-{typ.__name__}"

    monkeypatch.setattr(fistro_module, "Factory", DummyFactory)

    rich_fields = fistro_module.enrich_fields(Sample, generators=None)

    assert [name for name, _, _ in rich_fields] == ["x", "y", "z"]
    assert rich_fields[0][2].default_factory() == "value-for-int"
    assert rich_fields[1][2].default_factory() == "value-for-str"
    assert rich_fields[2][2].default == True


def test_generate_from_json_builds_dataclass_from_dict(monkeypatch):
    monkeypatch.setattr(fistro_module, "valid_id", lambda: "GeneratedName")

    generated = fistro_module.generate_from_json({"a": 1, "b": "two"})
    instance = generated()

    assert generated.__name__ == "GeneratedName"
    assert instance.a == 1
    assert instance.b == "two"


def test_generate_from_json_raises_for_list_input():
    with pytest.raises(NotImplementedError):
        fistro_module.generate_from_json([1, 2, 3])


def test_inner_inspection_for_scalar_returns_type_and_default_factory():
    inner_type, fld = fistro_module.inner_inspection(123)

    assert inner_type is int
    assert fld.default_factory() == 123


def test_inner_inspection_for_empty_list_returns_list_any_type():
    inner_type, fld = fistro_module.inner_inspection([])

    assert str(inner_type) in {"typing.List[typing.Any]", "list[typing.Any]"}
    assert fld.default_factory() == []


def test_inner_inspection_for_nonempty_list_uses_first_element_type():
    inner_type, fld = fistro_module.inner_inspection([1, 2, 3])

    assert str(inner_type) in {"typing.List[int]", "list[int]"}
    assert fld.default_factory() == [1, 2, 3]


def test_inner_inspection_for_dict_uses_key_and_value_types():
    inner_type, fld = fistro_module.inner_inspection({"k": 5})

    assert "Dict" in str(inner_type) or "dict" in str(inner_type)
    assert fld.default_factory() == {"k": 5}


def test_get_class_body_from_annotations_handles_scalar_dict_and_list_types():
    annotations = {
        "a": int,
        "b": dict,
        "c": list,
        "d": None,
    }

    body = fistro_module.get_class_body_from_annotations(annotations)

    assert "a: int" in body
    assert "b: dict" in body
    assert "c: list" in body
    assert "d: Any" in body


def test_bool_generator_returns_boolean(monkeypatch):
    monkeypatch.setattr(gen_module, "choice", lambda seq: seq[0])

    assert gen_module.bool_generator() is True


def test_str_generator_uses_choices_and_join(monkeypatch):
    monkeypatch.setattr(gen_module, "choices", lambda population, k: ["a", "b", "c"][:k])

    assert gen_module.str_generator(population="abcd", length=3) == "abc"


def test_date_generator_returns_date_instance(monkeypatch):
    values = iter([2020, 1, 2])

    monkeypatch.setattr(gen_module, "randint", lambda low, high: next(values))

    result = gen_module.date_generator({"min_year": 2020, "max_year": 2020, "min_month": 1, "max_month": 1, "min_day": 2, "max_day": 2})

    assert isinstance(result, date)
    assert result == date(2020, 1, 2)


def test_datetime_generator_returns_datetime_instance(monkeypatch):
    values = iter([2021, 12, 31, 23, 59, 58])

    monkeypatch.setattr(gen_module, "randint", lambda low, high: next(values))

    result = gen_module.datetime_generator(
        {
            "min_year": 2021,
            "max_year": 2021,
            "min_month": 12,
            "max_month": 12,
            "min_day": 31,
            "max_day": 31,
            "min_hour": 23,
            "max_hour": 23,
            "min_minute": 59,
            "max_minute": 59,
            "min_second": 58,
            "max_second": 58,
        }
    )

    assert isinstance(result, datetime)
    assert result == datetime(2021, 12, 31, 23, 59, 58)


def test_int_list_generator_uses_randint_for_each_item(monkeypatch):
    values = iter([1, 2, 3, 4])

    monkeypatch.setattr(gen_module, "randint", lambda low, high: next(values))

    result = gen_module.int_list_generator(list_length=4, int_length=2)

    assert result == [1, 2, 3, 4]


def test_int_dict_generator_builds_dictionary_with_string_keys(monkeypatch):
    keys = iter(["k1", "k2"])
    values = iter([10, 20])

    monkeypatch.setattr(gen_module, "str_generator", lambda: next(keys))
    monkeypatch.setattr(gen_module, "randint", lambda low, high: next(values))

    result = gen_module.int_dict_generator(dict_length=2, int_length=1)

    assert result == {"k1": 10, "k2": 20}


def test_default_generators_contains_expected_callables():
    defaults = gen_module.default_generators()

    assert defaults[0] is gen_module.int_generator
    assert defaults[1] is gen_module.bool_generator
    assert defaults[2] is gen_module.str_generator
    assert defaults[3] is gen_module.datetime_generator
    assert defaults[4] is gen_module.date_generator
    assert defaults[5] is gen_module.int_list_generator
    assert defaults[6] is gen_module.int_dict_generator
