import json
from dataclasses import dataclass, fields, is_dataclass, field as dc_field
from datetime import datetime, date
from typing import Dict, List
from unittest import mock

import pytest

from fistro.config import INT_LENGTH, STR_LENGTH, INT_LIST_LENGTH
from fistro.fistro import (
    generate,
    enrich_fields,
    inner_inspection,
    get_class_body_from_annotations,
)
from fistro.generators import (
    int_generator,
    bool_generator,
    str_generator,
    datetime_generator,
    date_generator,
    int_list_generator,
    int_dict_generator,
    default_generators,
)


@dataclass
class Simple:
    number: int
    flag: bool
    text: str


@dataclass
class WithDefault:
    number: int
    label: str = 'fixed'


@dataclass
class Collections:
    numbers: List[int]
    mapping: Dict[str, int]


def constant_int() -> int:
    return 42


def test_int_generator_default_range():
    for _ in range(50):
        value = int_generator()
        assert isinstance(value, int)
        assert 0 <= value <= 9 * 10 ** INT_LENGTH


def test_int_generator_small_length_bounds():
    for _ in range(50):
        assert 0 <= int_generator(0) <= 9


def test_int_generator_passes_bounds_to_randint():
    with mock.patch('fistro.generators.randint', return_value=7) as fake:
        assert int_generator(3) == 7
    fake.assert_called_once_with(0, 9000)


def test_bool_generator_returns_bool():
    for _ in range(20):
        assert isinstance(bool_generator(), bool)


def test_str_generator_length_and_population():
    value = str_generator(population='ab', length=15)
    assert len(value) == 15
    assert set(value) <= {'a', 'b'}


def test_str_generator_default_length():
    assert len(str_generator()) == STR_LENGTH


def test_datetime_generator_fixed_range():
    rand_date = {
        'min_year': 2020, 'max_year': 2020,
        'min_month': 5, 'max_month': 5,
        'min_day': 10, 'max_day': 10,
        'min_hour': 1, 'max_hour': 1,
        'min_minute': 2, 'max_minute': 2,
        'min_second': 3, 'max_second': 3,
    }
    assert datetime_generator(rand_date) == datetime(2020, 5, 10, 1, 2, 3)


def test_date_generator_fixed_range():
    rand_date = {
        'min_year': 1999, 'max_year': 1999,
        'min_month': 12, 'max_month': 12,
        'min_day': 31, 'max_day': 31,
    }
    result = date_generator(rand_date)
    assert result == date(1999, 12, 31)
    assert type(result) is date


def test_int_list_generator_length_and_values():
    result = int_list_generator(list_length=4, int_length=2)
    assert len(result) == 4
    assert all(isinstance(v, int) and 0 <= v <= 900 for v in result)


def test_int_list_generator_default_length():
    assert len(int_list_generator()) == INT_LIST_LENGTH


def test_int_dict_generator_shape():
    result = int_dict_generator(dict_length=3, int_length=1)
    assert 1 <= len(result) <= 3
    assert all(isinstance(k, str) for k in result)
    assert all(isinstance(v, int) and 0 <= v <= 90 for v in result.values())


def test_default_generators_cover_expected_functions():
    gens = default_generators()
    assert len(gens) == 7
    assert int_generator in gens
    assert str_generator in gens
    assert int_dict_generator in gens


def test_generate_returns_dataclass_with_same_name_and_fields():
    result = generate(Simple)
    assert is_dataclass(result)
    assert result.__name__ == 'Simple'
    assert [f.name for f in fields(result)] == ['number', 'flag', 'text']
    assert [f.type for f in fields(result)] == [int, bool, str]


def test_generate_class_instances_have_typed_values():
    instance = generate(Simple)()
    assert isinstance(instance.number, int)
    assert isinstance(instance.flag, bool)
    assert isinstance(instance.text, str)
    assert len(instance.text) == STR_LENGTH


def test_generate_as_dict():
    result = generate(Simple, as_dict=True)
    assert isinstance(result, dict)
    assert set(result) == {'number', 'flag', 'text'}
    assert isinstance(result['number'], int)


def test_generate_as_json_is_parseable():
    result = generate(Simple, as_json=True)
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert set(parsed) == {'number', 'flag', 'text'}
    assert isinstance(parsed['flag'], bool)


def test_generate_keeps_declared_defaults():
    result = generate(WithDefault, as_dict=True)
    assert result['label'] == 'fixed'
    assert isinstance(result['number'], int)


def test_generate_with_custom_generator_overrides_default():
    assert generate(Simple, generators=[constant_int], as_dict=True)['number'] == 42


def test_generate_collections():
    result = generate(Collections, as_dict=True)
    assert len(result['numbers']) == INT_LIST_LENGTH
    assert all(isinstance(v, int) for v in result['numbers'])
    assert isinstance(result['mapping'], dict)


def test_generate_unsupported_type_raises():
    from fistro.exceptions import NotSupportedType

    @dataclass
    class Bad:
        value: float

    with pytest.raises(NotSupportedType):
        generate(Bad)


def test_enrich_fields_structure():
    enriched = enrich_fields(WithDefault, None)
    assert [(n, t) for n, t, _ in enriched] == [('number', int), ('label', str)]
    assert enriched[1][2].default == 'fixed'
    assert callable(enriched[0][2].default_factory)


def test_inner_inspection_and_class_body():
    inner_type, a_field = inner_inspection([1, 2])
    assert inner_type == List[int]
    assert a_field.default_factory() == [1, 2]
    assert get_class_body_from_annotations({'a': int, 'b': str}) == 'a: int\nb: str\n'
