import dataclasses
import inspect
import json
from dataclasses import dataclass, field as dc_field

from fistro.config import INT_LENGTH
from fistro.fistro import enrich_fields, generate
from fistro.generators import default_generators, int_generator


@dataclass
class Person:
    name: str
    age: int
    active: bool


@dataclass
class WithDefaults:
    count: int = 7
    label: str = "hello"


@dataclass
class Empty:
    pass


@dataclass
class OnlyInt:
    n: int


def fixed_int() -> int:
    return 42


# ---------- generate ----------

def test_generate_returns_dataclass_class_by_default():
    result = generate(Person)
    assert inspect.isclass(result)
    assert dataclasses.is_dataclass(result)


def test_generate_keeps_class_name():
    result = generate(Person)
    assert result.__name__ == "Person"


def test_generate_result_has_same_field_names():
    result = generate(Person)
    names = [f.name for f in dataclasses.fields(result)]
    assert names == ["name", "age", "active"]


def test_generate_instance_has_correct_value_types():
    instance = generate(Person)()
    assert isinstance(instance.name, str)
    assert type(instance.age) is int
    assert type(instance.active) is bool


def test_generate_as_dict_returns_dict_with_field_keys():
    result = generate(Person, as_dict=True)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"name", "age", "active"}
    assert isinstance(result["name"], str)
    assert type(result["age"]) is int
    assert type(result["active"]) is bool


def test_generate_as_json_returns_valid_json_string():
    result = generate(Person, as_json=True)
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert set(parsed.keys()) == {"name", "age", "active"}
    assert isinstance(parsed["name"], str)
    assert isinstance(parsed["age"], int)
    assert isinstance(parsed["active"], bool)


def test_generate_as_dict_takes_precedence_over_as_json():
    result = generate(Person, as_dict=True, as_json=True)
    assert isinstance(result, dict)


def test_generate_preserves_declared_defaults():
    instance = generate(WithDefaults)()
    assert instance.count == 7
    assert instance.label == "hello"


def test_generate_as_dict_preserves_declared_defaults():
    result = generate(WithDefaults, as_dict=True)
    assert result == {"count": 7, "label": "hello"}


def test_generate_empty_class_yields_no_fields():
    result = generate(Empty)
    assert dataclasses.fields(result) == ()
    assert generate(Empty, as_dict=True) == {}


def test_generate_uses_custom_generator_by_return_type():
    result = generate(OnlyInt, generators=[fixed_int], as_dict=True)
    assert result == {"n": 42}


def test_generate_with_default_generators_list_runs():
    instance = generate(Person, generators=default_generators())()
    assert isinstance(instance.name, str)
    assert type(instance.age) is int


# ---------- enrich_fields ----------

def test_enrich_fields_returns_one_tuple_per_field():
    rich = enrich_fields(Person, None)
    assert len(rich) == 3
    assert [t[0] for t in rich] == ["name", "age", "active"]
    assert [t[1] for t in rich] == [str, int, bool]


def test_enrich_fields_uses_default_factory_for_fields_without_defaults():
    rich = enrich_fields(Person, None)
    for _, _, fld in rich:
        assert isinstance(fld, dc_field().__class__)
        assert fld.default_factory is not dataclasses.MISSING


def test_enrich_fields_keeps_defaults_for_fields_with_defaults():
    rich = enrich_fields(WithDefaults, None)
    defaults = {name: fld.default for name, _, fld in rich}
    assert defaults == {"count": 7, "label": "hello"}


# ---------- int_generator ----------

def test_int_generator_returns_int():
    assert type(int_generator()) is int


def test_int_generator_default_length_range():
    upper = 9 * 10 ** INT_LENGTH
    for _ in range(50):
        value = int_generator()
        assert 0 <= value <= upper


def test_int_generator_length_zero_gives_single_digit():
    for _ in range(50):
        value = int_generator(0)
        assert 0 <= value <= 9


def test_int_generator_passes_bounds_to_randint(monkeypatch):
    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return b

    monkeypatch.setattr("fistro.generators.randint", fake_randint)
    result = int_generator(2)
    assert calls == [(0, 900)]
    assert result == 900


def test_int_generator_default_argument_uses_int_length(monkeypatch):
    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return a

    monkeypatch.setattr("fistro.generators.randint", fake_randint)
    int_generator()
    assert calls == [(0, 9 * 10 ** INT_LENGTH)]


def test_int_generator_is_registered_in_default_generators():
    assert int_generator in default_generators()
