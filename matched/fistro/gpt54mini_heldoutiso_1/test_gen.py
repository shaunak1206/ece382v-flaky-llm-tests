import json
from dataclasses import dataclass, field, is_dataclass
from datetime import date, datetime

import pytest

import fistro.fistro as ff
import fistro.generators as gen


@dataclass
class Sample:
    a: int
    b: str = "x"
    c: bool = field(default=True)


@dataclass
class SampleWithList:
    items: list


def test_generate_returns_dataclass_type():
    result = ff.generate(Sample)
    assert isinstance(result, type)
    assert is_dataclass(result)
    assert result.__name__ == "Sample"


def test_generate_instance_has_expected_defaults():
    result = ff.generate(Sample)
    obj = result()
    assert obj.a == 0
    assert obj.b == "x"
    assert obj.c is True


def test_generate_as_dict_uses_default_values():
    result = ff.generate(Sample, as_dict=True)
    assert result == {"a": 0, "b": "x", "c": True}


def test_generate_as_json_serializes_defaults():
    result = ff.generate(Sample, as_json=True)
    assert json.loads(result) == {"a": 0, "b": "x", "c": True}


def test_generate_preserves_field_names_and_annotations():
    result = ff.generate(Sample)
    assert list(result.__dataclass_fields__.keys()) == ["a", "b", "c"]
    assert result.__dataclass_fields__["a"].type is int
    assert result.__dataclass_fields__["b"].type is str
    assert result.__dataclass_fields__["c"].type is bool


def test_generate_uses_custom_factory_generators(monkeypatch):
    called = []

    class DummyFactory:
        def __init__(self, generators=None):
            self.generators = generators

        def factory(self, field_type):
            called.append((self.generators, field_type))

            def _factory():
                return f"generated:{field_type.__name__}"

            return _factory

    monkeypatch.setattr(ff, "Factory", DummyFactory)
    result = ff.generate(Sample, generators=["custom"])
    obj = result()
    assert obj.a == "generated:int"
    assert obj.b == "x"  # default is preserved
    assert obj.c is True  # default is preserved
    assert called == [(["custom"], int)]


def test_enrich_fields_uses_default_factory_for_missing_defaults(monkeypatch):
    class DummyFactory:
        def __init__(self, generators=None):
            pass

        def factory(self, field_type):
            return lambda: f"value-for-{field_type.__name__}"

    monkeypatch.setattr(ff, "Factory", DummyFactory)
    rich_fields = ff.enrich_fields(Sample, generators=None)
    assert len(rich_fields) == 3
    a_name, a_type, a_field = rich_fields[0]
    assert a_name == "a"
    assert a_type is int
    assert a_field.default_factory() == "value-for-int"
    assert rich_fields[1][2].default == "x"
    assert rich_fields[2][2].default is True


def test_generate_from_json_creates_dataclass_with_inferred_field_types():
    obj = {"name": "alice", "age": 10, "active": True}
    cls = ff.generate_from_json(obj)
    assert is_dataclass(cls)
    assert sorted(cls.__dataclass_fields__.keys()) == ["active", "age", "name"]
    assert cls.__dataclass_fields__["name"].type is str
    assert cls.__dataclass_fields__["age"].type is int
    assert cls.__dataclass_fields__["active"].type is bool


def test_generate_from_json_instance_uses_original_values():
    obj = {"name": "alice", "age": 10}
    cls = ff.generate_from_json(obj)
    inst = cls()
    assert inst.name == "alice"
    assert inst.age == 10


def test_generate_from_json_raises_for_list_input():
    with pytest.raises(NotImplementedError):
        ff.generate_from_json([1, 2, 3])


def test_inner_inspection_for_scalar_returns_type_and_factory():
    t, fld = ff.inner_inspection(123)
    assert t is int
    assert fld.default_factory() == 123


def test_inner_inspection_for_list_with_elements_uses_first_element_type():
    t, fld = ff.inner_inspection([1, 2, 3])
    assert t == list[int] or str(t) == "typing.List[int]"
    assert fld.default_factory() == [1, 2, 3]


def test_inner_inspection_for_dict_returns_dict_type():
    t, fld = ff.inner_inspection({"k": 1})
    assert "Dict" in str(t) or "dict" in str(t)
    assert fld.default_factory() == {"k": 1}


def test_get_class_body_from_annotations_formats_scalars_lists_and_dicts():
    body = ff.get_class_body_from_annotations(
        {"x": int, "y": dict, "z": list, "w": None}
    )
    assert "x: int" in body
    assert "y: dict" in body
    assert "z: list" in body
    assert "w: Any" in body


def test_int_generator_respects_randint_result(monkeypatch):
    monkeypatch.setattr(gen, "randint", lambda a, b: 7)
    assert gen.int_generator(length=3) == 7


def test_int_generator_uses_expected_upper_bound(monkeypatch):
    seen = {}

    def fake_randint(a, b):
        seen["args"] = (a, b)
        return b

    monkeypatch.setattr(gen, "randint", fake_randint)
    assert gen.int_generator(length=4) == 9 * 10 ** 4
    assert seen["args"] == (0, 9 * 10 ** 4)


def test_str_generator_uses_choices_and_population(monkeypatch):
    captured = {}

    def fake_choices(population, k):
        captured["population"] = population
        captured["k"] = k
        return ["a", "b", "c"]

    monkeypatch.setattr(gen, "choices", fake_choices)
    assert gen.str_generator(population="xyz", length=3) == "abc"
    assert captured == {"population": "xyz", "k": 3}


def test_bool_generator_returns_choice_result(monkeypatch):
    monkeypatch.setattr(gen, "choice", lambda seq: seq[1])
    assert gen.bool_generator() is False


def test_datetime_generator_uses_bounds_and_returns_datetime(monkeypatch):
    values = iter([2020, 2, 3, 4, 5, 6])

    def fake_randint(a, b):
        return next(values)

    monkeypatch.setattr(gen, "randint", fake_randint)
    result = gen.datetime_generator()
    assert isinstance(result, datetime)
    assert result == datetime(2020, 2, 3, 4, 5, 6)


def test_date_generator_uses_bounds_and_returns_date(monkeypatch):
    values = iter([2021, 12, 25])

    def fake_randint(a, b):
        return next(values)

    monkeypatch.setattr(gen, "randint", fake_randint)
    result = gen.date_generator()
    assert isinstance(result, date)
    assert result == date(2021, 12, 25)


def test_int_list_generator_creates_expected_length(monkeypatch):
    monkeypatch.setattr(gen, "randint", lambda a, b: 5)
    assert gen.int_list_generator(list_length=4, int_length=2) == [5, 5, 5, 5]


def test_int_dict_generator_creates_expected_number_of_items(monkeypatch):
    keys = iter(["k1", "k2", "k3"])

    monkeypatch.setattr(gen, "str_generator", lambda *args, **kwargs: next(keys))
    monkeypatch.setattr(gen, "randint", lambda a, b: 11)
    result = gen.int_dict_generator(dict_length=3, int_length=2)
    assert result == {"k1": 11, "k2": 11, "k3": 11}


def test_default_generators_contains_expected_callables():
    gens = gen.default_generators()
    assert len(gens) == 7
    assert gen.int_generator in gens
    assert gen.bool_generator in gens
    assert gen.str_generator in gens
    assert gen.datetime_generator in gens
    assert gen.date_generator in gens
    assert gen.int_list_generator in gens
    assert gen.int_dict_generator in gens
