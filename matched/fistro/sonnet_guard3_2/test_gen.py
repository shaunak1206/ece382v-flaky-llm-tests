import json
import random
import string
from dataclasses import dataclass, is_dataclass, fields, asdict

from fistro import generators
from fistro.fistro import generate
from fistro.generators import int_generator, str_generator, default_generators


@dataclass
class IntOnly:
    a: int
    b: int


@dataclass
class Mixed:
    number: int
    text: str


@dataclass
class BoolOnly:
    flag: bool


@dataclass
class WithDefault:
    a: int
    b: int = 5


# ---------------- int_generator ----------------

def test_int_generator_returns_int():
    random.seed(0)
    assert isinstance(int_generator(), int)


def test_int_generator_length_zero_range():
    random.seed(1)
    for _ in range(200):
        value = int_generator(0)
        assert 0 <= value <= 9


def test_int_generator_length_three_range():
    random.seed(2)
    for _ in range(200):
        value = int_generator(3)
        assert 0 <= value <= 9000


def test_int_generator_default_is_non_negative():
    random.seed(3)
    for _ in range(50):
        assert int_generator() >= 0


def test_int_generator_same_seed_same_value():
    random.seed(42)
    first = [int_generator(4) for _ in range(5)]
    random.seed(42)
    second = [int_generator(4) for _ in range(5)]
    assert first == second


def test_int_generator_passes_expected_bounds_to_randint(monkeypatch):
    random.seed(0)
    calls = []

    def fake_randint(low, high):
        calls.append((low, high))
        return 123

    monkeypatch.setattr(generators, 'randint', fake_randint)
    assert int_generator(2) == 123
    assert calls == [(0, 900)]


# ---------------- generate ----------------

def test_generate_returns_dataclass_with_same_name():
    random.seed(0)
    result = generate(IntOnly)
    assert is_dataclass(result)
    assert result.__name__ == 'IntOnly'


def test_generate_keeps_field_names_in_order():
    random.seed(0)
    result = generate(Mixed)
    assert [f.name for f in fields(result)] == ['number', 'text']


def test_generate_instance_has_generated_values():
    random.seed(0)
    instance = generate(Mixed)()
    assert isinstance(instance.number, int)
    assert isinstance(instance.text, str)


def test_generate_bool_field():
    random.seed(0)
    for _ in range(10):
        assert isinstance(generate(BoolOnly)().flag, bool)


def test_generate_as_dict_returns_dict_with_keys():
    random.seed(0)
    result = generate(Mixed, as_dict=True)
    assert isinstance(result, dict)
    assert list(result.keys()) == ['number', 'text']
    assert isinstance(result['number'], int)
    assert isinstance(result['text'], str)


def test_generate_as_json_returns_valid_json():
    random.seed(0)
    result = generate(Mixed, as_json=True)
    assert isinstance(result, str)
    loaded = json.loads(result)
    assert set(loaded.keys()) == {'number', 'text'}
    assert isinstance(loaded['number'], int)
    assert isinstance(loaded['text'], str)


def test_generate_preserves_default_values():
    random.seed(0)
    result = generate(WithDefault, as_dict=True)
    assert result['b'] == 5
    assert isinstance(result['a'], int)


def test_generate_with_custom_generator_overrides_type():
    random.seed(0)

    def constant_int() -> int:
        return 7

    result = generate(IntOnly, generators=[constant_int], as_dict=True)
    assert result == {'a': 7, 'b': 7}


def test_generate_with_default_generators_list():
    random.seed(0)
    instance = generate(Mixed, generators=default_generators())()
    assert isinstance(instance.number, int)
    assert isinstance(instance.text, str)
    assert asdict(instance).keys() == {'number', 'text'}


# ---------------- str_generator (supporting) ----------------

def test_str_generator_length_and_population():
    random.seed(0)
    value = str_generator(population='a', length=5)
    assert value == 'aaaaa'
    value = str_generator(population=string.digits, length=8)
    assert len(value) == 8
    assert all(c in string.digits for c in value)
