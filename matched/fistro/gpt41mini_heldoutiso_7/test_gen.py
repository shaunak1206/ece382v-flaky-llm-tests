import pytest
import json
from dataclasses import dataclass, field, is_dataclass
from typing import List, Dict, Any

from fistro.fistro import generate, enrich_fields, inner_inspection, generate_from_json, get_class_body_from_annotations
from fistro.generators import int_generator


# Sample dataclasses for testing
@dataclass
class Simple:
    x: int
    y: str = "default"
    z: bool = field(default_factory=lambda: True)


@dataclass
class Complex:
    a: int
    b: List[int]
    c: Dict[str, str]


def test_generate_returns_dataclass_instance():
    result = generate(Simple)
    assert is_dataclass(result)
    assert result.__name__ == Simple.__name__


def test_generate_as_dict_returns_correct_dict():
    d = generate(Simple, as_dict=True)
    assert isinstance(d, dict)
    # keys match fields
    assert set(d.keys()) == {'x', 'y', 'z'}
    # default values honored for y and z
    assert d['y'] == "default"
    assert d['z'] is True


def test_generate_as_json_returns_valid_json():
    j = generate(Simple, as_json=True)
    assert isinstance(j, str)
    decoded = json.loads(j)
    assert decoded['y'] == "default"
    assert decoded['z'] is True


def test_enrich_fields_returns_expected_structure():
    enriched = enrich_fields(Simple, generators=None)
    assert isinstance(enriched, list)
    assert len(enriched) == 3
    for name, typ, fld in enriched:
        assert isinstance(name, str)
        # Types should match Simple fields
        assert any(f.name == name for f in Simple.__dataclass_fields__.values())
        # field object returned for fld
        assert hasattr(fld, 'default') or hasattr(fld, 'default_factory')


def test_enrich_fields_with_generators_override_factory():
    # Provide a dummy generator list that returns constant lambda to test usage
    class DummyFactory:
        def factory(self, typ):
            return lambda: 42

    # Monkeypatch Factory inside enrich_fields temporarily
    import fistro.fistro as fistro_module
    orig_factory_class = fistro_module.Factory
    try:
        fistro_module.Factory = lambda g=None: DummyFactory()
        enriched = enrich_fields(Simple, generators=[lambda: 1])
        # The default_factory field returns 42 always
        for _, _, fld in enriched:
            val = fld.default_factory()
            assert val == 42
    finally:
        fistro_module.Factory = orig_factory_class


def test_inner_inspection_with_dict_returns_dict_type_and_field():
    sample = {"key": 1}
    typ, fld = inner_inspection(sample)
    assert typ.__origin__ == dict
    val = fld.default_factory()
    assert val == sample


def test_inner_inspection_with_list_returns_list_type_and_field():
    sample = [1, 2, 3]
    typ, fld = inner_inspection(sample)
    assert typ.__origin__ == list
    val = fld.default_factory()
    assert val == sample


def test_inner_inspection_with_empty_list_returns_list_of_any():
    sample = []
    typ, fld = inner_inspection(sample)
    assert typ.__origin__ == list
    assert typ.__args__[0] == Any


def test_inner_inspection_with_scalar_returns_type_and_field():
    sample = 123
    typ, fld = inner_inspection(sample)
    assert typ == int
    val = fld.default_factory()
    assert val == sample


def test_generate_from_json_creates_dataclass_from_dict():
    data = {"a": 1, "b": "str"}
    cls = generate_from_json(data)
    assert hasattr(cls, '__dataclass_fields__')
    assert set(cls.__dataclass_fields__.keys()) == set(data.keys())
    inst = cls()
    # The default_factory creates the field values fixed to original dict values
    assert inst.a == 1
    assert inst.b == "str"


def test_generate_from_json_raises_for_non_dict():
    with pytest.raises(NotImplementedError):
        generate_from_json([{"a": 1}])


def test_get_class_body_from_annotations_basic_types():
    annotations = {"x": int, "y": str, "z": None}
    body = get_class_body_from_annotations(annotations)
    assert "x: int" in body
    assert "y: str" in body
    assert "z: Any" in body


def test_get_class_body_from_annotations_for_list_and_dict():
    from typing import List, Dict
    annotations = {
        "a": List[int],
        "b": Dict[str, int],
        "c": int,
    }
    body = get_class_body_from_annotations(annotations)
    assert "a: " in body and "List" in body
    assert "b: " in body and "Dict" in body
    assert "c: int" in body


# -- Tests for int_generator in generators.py --

def test_int_generator_return_type_and_range():
    value = int_generator()
    assert isinstance(value, int)
    # Default INT_LENGTH = 6 (typically); upper bound: 9 * 10^length
    max_val = 9 * 10**6
    assert 0 <= value <= max_val


def test_int_generator_custom_length_affects_range():
    length = 2
    value = int_generator(length=length)
    max_val = 9 * 10**length
    assert 0 <= value <= max_val


def test_int_generator_zero_length():
    value = int_generator(length=0)
    assert isinstance(value, int)
    # max would be 9 * 10**0 = 9
    assert 0 <= value <= 9


def test_int_generator_negative_length_behavior():
    # Negative length is nonsensical, test that it returns int and does not error
    value = int_generator(length=-1)
    assert isinstance(value, int)
    # It might fall back to less or more, just check no exception and int type


# Extra: test generate uses int_generator as default factory indirectly (integration)
def test_generate_with_int_generator_default_value():
    @dataclass
    class WithIntField:
        num: int

    # generate returns dataclass type with default factory from Factory (which uses int_generator)
    gen_cls = generate(WithIntField)
    inst = gen_cls()
    assert isinstance(inst.num, int)

