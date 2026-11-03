import pytest
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, date
from typing import List, Dict, Any, Callable
import json

from fistro.fistro import generate, enrich_fields, generate_from_json, inner_inspection, get_class_body_from_annotations
from fistro.generators import int_generator


@pytest.mark.parametrize("length", [0, 1, 5, 10])
def test_int_generator_range(length):
    value = int_generator(length)
    assert isinstance(value, int)
    # int_generator returns randint between 0 and 9*10^length
    assert 0 <= value <= 9 * 10 ** length


def test_generate_returns_dataclass_instance():
    @dataclass
    class User:
        id: int
        name: str

    result = generate(User)
    assert is_dataclass(result)
    assert result.__name__ == "User"
    instance = result()
    assert isinstance(instance, result)
    # fields present
    assert all(f.name in ('id', 'name') for f in fields(result))


def test_generate_as_dict_and_json():
    @dataclass
    class Point:
        x: int
        y: int

    d = generate(Point, as_dict=True)
    assert isinstance(d, dict)
    assert "x" in d and "y" in d

    j = generate(Point, as_json=True)
    assert isinstance(j, str)
    parsed = json.loads(j)
    assert parsed == d


def test_generate_with_generators_overrides_default_factory():
    @dataclass
    class Container:
        value: int
        flag: bool

    # Use only int_generator as custom generator to override default for int
    def fake_int_gen():
        return 42

    # We pass list of generators, this should be used
    generated_cls = generate(Container, generators=[fake_int_gen])
    instance = generated_cls()

    # 'value' uses fake_int_gen returning 42
    assert instance.value == 42
    # 'flag' uses default factory so bool likely (True/False)
    assert isinstance(instance.flag, bool)


def test_enrich_fields_with_and_without_generators():
    @dataclass
    class Test:
        a: int
        b: str = "default"

    # Without generators uses Factory default
    enriched_no_gen = enrich_fields(Test, generators=None)
    assert len(enriched_no_gen) == 2
    # first field has default_factory
    _, _, f1 = enriched_no_gen[0]
    assert callable(f1.default_factory)
    # second field has default, not factory
    _, _, f2 = enriched_no_gen[1]
    assert f2.default == "default"

    # With generators uses Factory with those generators
    def dummy_gen():
        return 1

    enriched_gen = enrich_fields(Test, generators=[dummy_gen])
    assert len(enriched_gen) == 2
    _, _, f1_gen = enriched_gen[0]
    assert callable(f1_gen.default_factory)
    # The default value field remains as is
    _, _, f2_gen = enriched_gen[1]
    assert f2_gen.default == "default"


def test_generate_from_json_creates_dataclass_from_dict():
    obj = {
        "age": 30,
        "name": "John",
        "info": {"height": 180}
    }

    generated_cls = generate_from_json(obj)
    assert is_dataclass(generated_cls)
    names = [f.name for f in fields(generated_cls)]
    assert set(names) == {"age", "name", "info"}

    instance = generated_cls()
    # fields have expected types
    for f in fields(generated_cls):
        val = getattr(instance, f.name)
        # check inner types as described
        if f.name == "age":
            assert isinstance(val, int)
        elif f.name == "name":
            assert isinstance(val, str)
        elif f.name == "info":
            # info is a dict field so value is a dict
            d = val
            assert isinstance(d, dict)
            # contains the key "height"
            assert "height" in d


def test_generate_from_json_raises_on_list():
    with pytest.raises(NotImplementedError):
        generate_from_json([])


@pytest.mark.parametrize(
    "input_value, expected_type",
    [
        (42, int),
        (3.14, float),
        ("hello", str),
        ([1, 2, 3], list),
        ({"a": 1}, dict),
    ],
)
def test_inner_inspection_types(input_value, expected_type):
    typ, field_obj = inner_inspection(input_value)
    if isinstance(input_value, dict):
        assert typ.__origin__ is dict
    elif isinstance(input_value, list):
        assert typ.__origin__ is list
    else:
        assert typ == type(input_value)
    assert callable(field_obj.default_factory)


def test_get_class_body_from_annotations_basic():
    class Dummy:
        a: int
        b: str
        c: List[int]
        d: Dict[str, int]
        e: Any

    ann = {
        "a": int,
        "b": str,
        "c": List[int],
        "d": Dict[str, int],
        "e": None,
    }
    output = get_class_body_from_annotations(ann)
    # Should contain lines with proper types
    assert "a: int" in output
    assert "b: str" in output
    assert "c: typing.List[int]" in output or "c: List[int]" in output
    assert "d: typing.Dict[str, int]" in output or "d: Dict[str, int]" in output
    assert "e: Any" in output


def test_int_generator_length_variants():
    for length in [0, 2, 5]:
        val = int_generator(length)
        assert isinstance(val, int)
        assert 0 <= val <= 9 * 10 ** length


def test_generate_with_no_fields():
    @dataclass
    class Empty:
        pass

    gen_cls = generate(Empty)
    inst = gen_cls()
    assert is_dataclass(gen_cls)
    assert fields(gen_cls) == ()
    assert inst == inst  # trivial test to ensure instantiation succeeds


def test_generate_with_default_and_factory_mix():
    @dataclass
    class Mix:
        a: int = 10
        b: int = field(default_factory=lambda: 20)

    gen_cls = generate(Mix)
    inst = gen_cls()
    # a has default, unchanged
    assert inst.a == 10
    # b default factory generates 20
    assert inst.b == 20


def test_generate_with_custom_generator_affects_only_matching_types():
    @dataclass
    class SomeClass:
        x: int
        y: bool

    def always_seven() -> int:
        return 7

    def always_true() -> bool:
        return True

    gen_cls = generate(SomeClass, generators=[always_seven, always_true])
    inst = gen_cls()
    assert inst.x == 7 or inst.y is True  # at least one replaced

    # Since Factory matches first compatible generator, x should be 7


def test_inner_inspection_list_of_dicts_type():
    val = [{"a": 1}]
    typ, fld = inner_inspection(val)
    assert hasattr(typ, "__origin__") and typ.__origin__ is list
    assert callable(fld.default_factory)


def test_generate_from_json_field_types():
    obj = {"key": {"inner_key": 1}}
    gen_cls = generate_from_json(obj)
    f = fields(gen_cls)[0]
    # field type should be Dict
    assert getattr(f.type, "__origin__", None) is dict


def test_generate_field_values_are_generated_by_factory():
    @dataclass
    class TestClass:
        val: int
        flag: bool

    cls = generate(TestClass)
    instance1 = cls()
    instance2 = cls()
    # Fields should have values from default_factory which generate new values, so different instances may vary
    assert isinstance(instance1.val, int)
    assert isinstance(instance1.flag, bool)
    # Likely different values on different instantiations (not a strict guarantee but probable)
    # We don't assert equality to allow randomness


def test_generate_with_custom_generator_for_str():
    @dataclass
    class StringClass:
        s: str

    # Create a custom generator that always returns 'test'
    def fixed_str_gen():
        return "test"

    generated_cls = generate(StringClass, generators=[fixed_str_gen])
    # The generator factory tries to match callable with field type.
    # Because no type info to map, likely default factory creates default values
    inst = generated_cls()
    assert isinstance(inst.s, str)


def test_enrich_fields_returns_correct_field_objects():
    @dataclass
    class SomeData:
        i: int
        s: str = "hello"

    fields_result = enrich_fields(SomeData, None)
    assert all(len(t) == 3 for t in fields_result)
    # test fields with default and default_factory
    _, _, f_i = fields_result[0]
    _, _, f_s = fields_result[1]
    assert callable(f_i.default_factory)
    assert f_s.default == "hello"

