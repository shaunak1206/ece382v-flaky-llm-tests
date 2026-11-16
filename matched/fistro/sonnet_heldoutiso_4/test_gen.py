import json
from dataclasses import dataclass, is_dataclass, fields
from unittest import mock

import pytest

from fistro.config import INT_LENGTH
from fistro.fistro import generate
from fistro.generators import int_generator


@dataclass
class Simple:
    number: int
    text: str
    flag: bool


@dataclass
class WithDefault:
    number: int
    fixed: int = 5
    name: str = 'constant'


@dataclass
class OnlyInts:
    a: int
    b: int


# ---------------------------------------------------------------- int_generator

def test_int_generator_returns_int():
    assert isinstance(int_generator(), int)


def test_int_generator_default_range():
    for _ in range(200):
        value = int_generator()
        assert 0 <= value <= 9 * 10 ** INT_LENGTH


def test_int_generator_custom_length_range():
    for _ in range(200):
        value = int_generator(2)
        assert 0 <= value <= 900


def test_int_generator_length_zero():
    for _ in range(100):
        assert 0 <= int_generator(0) <= 9


def test_int_generator_calls_randint_with_expected_bounds():
    with mock.patch('fistro.generators.randint', return_value=7) as fake:
        assert int_generator(3) == 7
    fake.assert_called_once_with(0, 9 * 10 ** 3)


def test_int_generator_default_calls_randint_with_default_length():
    with mock.patch('fistro.generators.randint', return_value=1) as fake:
        int_generator()
    fake.assert_called_once_with(0, 9 * 10 ** INT_LENGTH)


# --------------------------------------------------------------------- generate

def test_generate_returns_dataclass_type():
    result = generate(Simple)
    assert isinstance(result, type)
    assert is_dataclass(result)


def test_generate_preserves_class_name():
    assert generate(Simple).__name__ == 'Simple'


def test_generate_preserves_field_names_and_types():
    result = generate(Simple)
    original = {f.name: f.type for f in fields(Simple)}
    generated = {f.name: f.type for f in fields(result)}
    assert generated == original


def test_generate_instance_has_values_of_right_types():
    instance = generate(Simple)()
    assert isinstance(instance.number, int)
    assert isinstance(instance.text, str)
    assert isinstance(instance.flag, bool)


def test_generate_as_dict_returns_dict_with_all_fields():
    result = generate(Simple, as_dict=True)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'number', 'text', 'flag'}
    assert isinstance(result['number'], int)
    assert isinstance(result['text'], str)
    assert isinstance(result['flag'], bool)


def test_generate_as_json_returns_valid_json_string():
    result = generate(Simple, as_json=True)
    assert isinstance(result, str)
    loaded = json.loads(result)
    assert set(loaded.keys()) == {'number', 'text', 'flag'}
    assert isinstance(loaded['number'], int)
    assert isinstance(loaded['flag'], bool)


def test_generate_keeps_explicit_defaults():
    result = generate(WithDefault, as_dict=True)
    assert result['fixed'] == 5
    assert result['name'] == 'constant'
    assert isinstance(result['number'], int)


def test_generate_default_values_on_instance():
    instance = generate(WithDefault)()
    assert instance.fixed == 5
    assert instance.name == 'constant'


def test_generate_with_custom_generator_overrides_default():
    def my_int() -> int:
        return 42

    result = generate(OnlyInts, generators=[my_int], as_dict=True)
    assert result == {'a': 42, 'b': 42}


def test_generate_non_dataclass_raises_type_error():
    class NotADataclass:
        pass

    with pytest.raises(TypeError):
        generate(NotADataclass)
