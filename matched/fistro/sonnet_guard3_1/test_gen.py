import json
import random
from dataclasses import dataclass, fields, is_dataclass

import pytest

from fistro import generators
from fistro.config import INT_LENGTH
from fistro.fistro import generate
from fistro.generators import int_generator


@dataclass
class OnlyInt:
    number: int


@dataclass
class Mixed:
    number: int
    flag: bool
    text: str


@dataclass
class WithDefault:
    number: int
    fixed: int = 7


def _forty_two() -> int:
    return 42


# ---------------------------- int_generator ----------------------------


def test_int_generator_returns_int():
    random.seed(1)
    assert isinstance(int_generator(), int)


def test_int_generator_default_range():
    random.seed(2)
    upper = 9 * 10 ** INT_LENGTH
    for _ in range(200):
        value = int_generator()
        assert 0 <= value <= upper


def test_int_generator_length_zero_range():
    random.seed(3)
    for _ in range(100):
        assert 0 <= int_generator(0) <= 9


def test_int_generator_length_one_range():
    random.seed(4)
    for _ in range(200):
        assert 0 <= int_generator(1) <= 90


def test_int_generator_custom_length_bound():
    random.seed(5)
    for _ in range(200):
        assert 0 <= int_generator(length=3) <= 9000


def test_int_generator_same_seed_same_value():
    random.seed(123)
    first = [int_generator() for _ in range(5)]
    random.seed(123)
    second = [int_generator() for _ in range(5)]
    assert first == second


def test_int_generator_calls_randint_with_expected_bounds(monkeypatch):
    random.seed(6)
    calls = []

    def fake_randint(low, high):
        calls.append((low, high))
        return 11

    monkeypatch.setattr(generators, "randint", fake_randint)
    assert int_generator(2) == 11
    assert calls == [(0, 900)]


def test_int_generator_default_uses_config_length(monkeypatch):
    random.seed(7)
    calls = []

    def fake_randint(low, high):
        calls.append((low, high))
        return 0

    monkeypatch.setattr(generators, "randint", fake_randint)
    int_generator()
    assert calls == [(0, 9 * 10 ** INT_LENGTH)]


# ------------------------------- generate -------------------------------


def test_generate_returns_dataclass_type():
    random.seed(10)
    result = generate(OnlyInt)
    assert isinstance(result, type)
    assert is_dataclass(result)


def test_generate_preserves_name_and_field_names():
    random.seed(11)
    result = generate(Mixed)
    assert result.__name__ == "Mixed"
    assert [f.name for f in fields(result)] == ["number", "flag", "text"]


def test_generate_instance_has_values_of_right_types():
    random.seed(12)
    instance = generate(Mixed)()
    assert isinstance(instance.number, int)
    assert isinstance(instance.flag, bool)
    assert isinstance(instance.text, str)


def test_generate_instance_allows_overriding_values():
    random.seed(13)
    instance = generate(OnlyInt)(number=5)
    assert instance.number == 5


def test_generate_as_dict_returns_dict_with_field_keys():
    random.seed(14)
    result = generate(Mixed, as_dict=True)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"number", "flag", "text"}
    assert isinstance(result["number"], int)
    assert isinstance(result["flag"], bool)
    assert isinstance(result["text"], str)


def test_generate_as_json_returns_parseable_string():
    random.seed(15)
    result = generate(Mixed, as_json=True)
    assert isinstance(result, str)
    loaded = json.loads(result)
    assert set(loaded.keys()) == {"number", "flag", "text"}
    assert isinstance(loaded["number"], int)
    assert isinstance(loaded["flag"], bool)
    assert isinstance(loaded["text"], str)


def test_generate_as_dict_takes_precedence_over_as_json():
    random.seed(16)
    result = generate(OnlyInt, as_dict=True, as_json=True)
    assert isinstance(result, dict)


def test_generate_keeps_explicit_default_values():
    random.seed(17)
    as_dict = generate(WithDefault, as_dict=True)
    assert as_dict["fixed"] == 7
    assert isinstance(as_dict["number"], int)
    instance = generate(WithDefault)()
    assert instance.fixed == 7


def test_generate_with_custom_generator_overrides_int():
    random.seed(18)
    result = generate(OnlyInt, generators=[_forty_two], as_dict=True)
    assert result == {"number": 42}


def test_generate_same_seed_gives_same_dict():
    random.seed(99)
    first = generate(Mixed, as_dict=True)
    random.seed(99)
    second = generate(Mixed, as_dict=True)
    assert first == second
