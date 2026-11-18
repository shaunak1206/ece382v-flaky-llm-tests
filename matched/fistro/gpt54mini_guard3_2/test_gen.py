import json
import random
from dataclasses import dataclass
from datetime import date, datetime

import pytest

import fistro.fistro as fistro_mod
import fistro.generators as gen_mod


@dataclass
class SampleModel:
    x: int
    y: str
    z: bool = True


def test_int_generator_deterministic_with_seed():
    random.seed(12345)
    value1 = gen_mod.int_generator(length=3)
    random.seed(12345)
    value2 = gen_mod.int_generator(length=3)
    assert value1 == value2
    assert 0 <= value1 <= 9000


def test_int_generator_uses_length_boundaries():
    random.seed(1)
    value = gen_mod.int_generator(length=0)
    assert 0 <= value <= 9


def test_bool_generator_is_boolean_with_seed():
    random.seed(2024)
    value = gen_mod.bool_generator()
    assert isinstance(value, bool)


def test_str_generator_respects_length_and_charset():
    random.seed(7)
    value = gen_mod.str_generator(length=8, population="abc")
    assert len(value) == 8
    assert set(value) <= set("abc")


def test_datetime_generator_default_returns_datetime():
    random.seed(11)
    value = gen_mod.datetime_generator()
    assert isinstance(value, datetime)


def test_datetime_generator_custom_bounds():
    random.seed(22)
    value = gen_mod.datetime_generator(
        {
            "min_year": 2020,
            "max_year": 2020,
            "min_month": 5,
            "max_month": 5,
            "min_day": 17,
            "max_day": 17,
            "min_hour": 13,
            "max_hour": 13,
            "min_minute": 14,
            "max_minute": 14,
            "min_second": 15,
            "max_second": 15,
        }
    )
    assert value == datetime(2020, 5, 17, 13, 14, 15)


def test_date_generator_custom_bounds():
    random.seed(33)
    value = gen_mod.date_generator(
        {
            "min_year": 2021,
            "max_year": 2021,
            "min_month": 12,
            "max_month": 12,
            "min_day": 31,
            "max_day": 31,
        }
    )
    assert value == date(2021, 12, 31)


def test_int_list_generator_length_and_range():
    random.seed(44)
    value = gen_mod.int_list_generator(list_length=5, int_length=2)
    assert len(value) == 5
    assert all(isinstance(x, int) for x in value)
    assert all(0 <= x <= 900 for x in value)


def test_int_dict_generator_uses_requested_length(monkeypatch):
    random.seed(55)
    monkeypatch.setattr(gen_mod, "str_generator", lambda population=gen_mod.string.printable, length=gen_mod.STR_LENGTH: "k")
    value = gen_mod.int_dict_generator(dict_length=3, int_length=1)
    assert value == {"k": value["k"]}


def test_default_generators_contains_expected_callables():
    generators = gen_mod.default_generators()
    assert generators == [
        gen_mod.int_generator,
        gen_mod.bool_generator,
        gen_mod.str_generator,
        gen_mod.datetime_generator,
        gen_mod.date_generator,
        gen_mod.int_list_generator,
        gen_mod.int_dict_generator,
    ]


def test_enrich_fields_uses_default_factory_when_no_generators(monkeypatch):
    class FakeFactory:
        def __init__(self, generators=None):
            self.generators = generators

        def factory(self, typ):
            return lambda: f"default-for-{typ.__name__}"

    monkeypatch.setattr(fistro_mod, "Factory", FakeFactory)
    fields = fistro_mod.enrich_fields(SampleModel, generators=None)

    assert [name for name, *_ in fields] == ["x", "y", "z"]
    assert fields[0][2].default_factory() == "default-for-int"
    assert fields[1][2].default_factory() == "default-for-str"
    assert fields[2][2].default == True


def test_enrich_fields_passes_custom_generators_to_factory(monkeypatch):
    captured = {}

    class FakeFactory:
        def __init__(self, generators=None):
            captured["generators"] = generators

        def factory(self, typ):
            return lambda: typ.__name__

    custom_generators = [lambda: None]
    monkeypatch.setattr(fistro_mod, "Factory", FakeFactory)

    fields = fistro_mod.enrich_fields(SampleModel, generators=custom_generators)

    assert captured["generators"] == custom_generators
    assert fields[0][2].default_factory() == "int"
    assert fields[1][2].default_factory() == "str"


def test_generate_returns_dataclass_type_with_instance_defaults(monkeypatch):
    class FakeFactory:
        def __init__(self, generators=None):
            self.generators = generators

        def factory(self, typ):
            mapping = {int: lambda: 10, str: lambda: "abc", bool: lambda: False}
            return mapping[typ]

    monkeypatch.setattr(fistro_mod, "Factory", FakeFactory)
    cls = fistro_mod.generate(SampleModel)

    assert isinstance(cls, type)
    obj = cls()
    assert obj.x == 10
    assert obj.y == "abc"
    assert obj.z is True
    assert obj.__class__.__name__ == "SampleModel"


def test_generate_as_dict(monkeypatch):
    class FakeFactory:
        def __init__(self, generators=None):
            pass

        def factory(self, typ):
            return {int: lambda: 7, str: lambda: "hi", bool: lambda: False}[typ]

    monkeypatch.setattr(fistro_mod, "Factory", FakeFactory)
    result = fistro_mod.generate(SampleModel, as_dict=True)

    assert result == {"x": 7, "y": "hi", "z": True}


def test_generate_as_json(monkeypatch):
    class FakeFactory:
        def __init__(self, generators=None):
            pass

        def factory(self, typ):
            return {int: lambda: 1, str: lambda: "x", bool: lambda: False}[typ]

    monkeypatch.setattr(fistro_mod, "Factory", FakeFactory)
    result = fistro_mod.generate(SampleModel, as_json=True)

    assert json.loads(result) == {"x": 1, "y": "x", "z": True}


def test_generate_uses_custom_generators_argument(monkeypatch):
    class FakeFactory:
        init_args = []

        def __init__(self, generators=None):
            FakeFactory.init_args.append(generators)

        def factory(self, typ):
            return {int: lambda: 3, str: lambda: "s", bool: lambda: False}[typ]

    custom_generators = [lambda: "unused"]
    monkeypatch.setattr(fistro_mod, "Factory", FakeFactory)

    cls = fistro_mod.generate(SampleModel, generators=custom_generators)
    assert FakeFactory.init_args == [custom_generators]
    assert cls().x == 3
    assert cls().y == "s"


def test_generate_from_json_creates_dataclass(monkeypatch):
    monkeypatch.setattr(fistro_mod, "valid_id", lambda: "GeneratedFromJson")
    cls = fistro_mod.generate_from_json({"a": 1, "b": "text", "c": [1, 2]})

    obj = cls()
    assert obj.a == 1
    assert obj.b == "text"
    assert obj.c == [1, 2]


def test_inner_inspection_for_scalar_dict_and_list():
    typ, fld = fistro_mod.inner_inspection(5)
    assert typ is int
    assert fld.default_factory() == 5

    typ, fld = fistro_mod.inner_inspection({"k": 1})
    assert str(typ) == "typing.Dict[str, int]"
    assert fld.default_factory() == {"k": 1}

    typ, fld = fistro_mod.inner_inspection([1, 2, 3])
    assert str(typ) == "typing.List[int]"
    assert fld.default_factory() == [1, 2, 3]


def test_get_class_body_from_annotations_formats_basic_types():
    body = fistro_mod.get_class_body_from_annotations(
        {"a": int, "b": dict, "c": list, "d": None}
    )
    assert "a: int" in body
    assert "b: <class 'dict'>" in body
    assert "c: <class 'list'>" in body
    assert "d: Any" in body
