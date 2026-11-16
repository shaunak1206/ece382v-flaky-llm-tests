"""Unit tests for data_partitioner.data_suplier.DataSuplier."""

import types

import pytest

from data_partitioner.data_suplier import DataSuplier


def _exhaust_both(suplier):
    """Consume both generators fully and return (training_pairs, test_pairs)."""
    training_pairs = list(suplier.training_set())
    test_pairs = list(suplier.test_set())
    return training_pairs, test_pairs


def _make_dataset(n):
    return ["item-{}".format(i) for i in range(n)]


def test_init_stores_dataset_and_percent():
    dataset = _make_dataset(10)
    suplier = DataSuplier(dataset, training_percent=0.6)
    assert suplier.dataset is dataset
    assert suplier.training_percent == 0.6


def test_init_default_training_percent_is_0_8():
    suplier = DataSuplier(_make_dataset(5))
    assert suplier.training_percent == 0.8


def test_init_records_number_of_elements():
    suplier = DataSuplier(_make_dataset(17))
    assert suplier.n_of_elements == 17


def test_init_starts_with_empty_index_lists_and_zero_index():
    suplier = DataSuplier(_make_dataset(8))
    assert suplier.training_indexes == []
    assert suplier.test_indexes == []
    assert suplier.index == 0


def test_init_creates_partitioner():
    suplier = DataSuplier(_make_dataset(8))
    assert suplier.partitioner is not None


def test_training_set_returns_generator():
    suplier = DataSuplier(_make_dataset(4))
    assert isinstance(suplier.training_set(), types.GeneratorType)


def test_test_set_returns_generator():
    suplier = DataSuplier(_make_dataset(4))
    assert isinstance(suplier.test_set(), types.GeneratorType)


def test_generators_are_lazy_and_do_not_advance_index_on_creation():
    suplier = DataSuplier(_make_dataset(6))
    suplier.training_set()
    suplier.test_set()
    assert suplier.index == 0
    assert suplier.training_indexes == []
    assert suplier.test_indexes == []


def test_yielded_items_are_index_and_dataset_value_pairs():
    dataset = _make_dataset(12)
    suplier = DataSuplier(dataset)
    for index, value in suplier.training_set():
        assert value == dataset[index]
    for index, value in suplier.test_set():
        assert value == dataset[index]


def test_yielded_values_are_same_objects_as_dataset_elements():
    dataset = [object() for _ in range(9)]
    suplier = DataSuplier(dataset)
    training_pairs, test_pairs = _exhaust_both(suplier)
    for index, value in training_pairs + test_pairs:
        assert value is dataset[index]


def test_training_and_test_sets_partition_all_indexes_without_overlap():
    n = 25
    suplier = DataSuplier(_make_dataset(n))
    training_pairs, test_pairs = _exhaust_both(suplier)
    training_idx = [i for i, _ in training_pairs]
    test_idx = [i for i, _ in test_pairs]
    assert set(training_idx).isdisjoint(test_idx)
    assert sorted(training_idx + test_idx) == list(range(n))


def test_training_indexes_are_yielded_in_ascending_order():
    suplier = DataSuplier(_make_dataset(30))
    training_pairs, test_pairs = _exhaust_both(suplier)
    training_idx = [i for i, _ in training_pairs]
    test_idx = [i for i, _ in test_pairs]
    assert training_idx == sorted(training_idx)
    assert test_idx == sorted(test_idx)


def test_yield_counts_match_recorded_index_lists():
    suplier = DataSuplier(_make_dataset(20))
    training_pairs, test_pairs = _exhaust_both(suplier)
    assert len(training_pairs) == len(suplier.training_indexes)
    assert len(test_pairs) == len(suplier.test_indexes)


def test_index_reaches_dataset_length_after_full_consumption():
    n = 14
    suplier = DataSuplier(_make_dataset(n))
    _exhaust_both(suplier)
    assert suplier.index == n


def test_repeated_training_set_calls_return_same_results():
    suplier = DataSuplier(_make_dataset(18))
    first = list(suplier.training_set())
    second = list(suplier.training_set())
    assert first == second


def test_partial_consumption_then_resume_yields_complete_set():
    dataset = _make_dataset(16)
    suplier = DataSuplier(dataset)
    gen = suplier.test_set()
    partial = [next(gen) for _ in range(1)]
    rest = list(gen)
    full = partial + rest
    assert [i for i, _ in full] == suplier.test_indexes
    assert all(value == dataset[i] for i, value in full)


def test_same_dataset_produces_same_split_across_instances():
    dataset = _make_dataset(40)
    first = DataSuplier(dataset)
    second = DataSuplier(dataset)
    assert _exhaust_both(first) == _exhaust_both(second)


def test_private_next_advances_index_and_records_one_index():
    suplier = DataSuplier(_make_dataset(5))
    suplier._DataSuplier__next()
    assert suplier.index == 1
    assert len(suplier.training_indexes) + len(suplier.test_indexes) == 1
    recorded = suplier.training_indexes + suplier.test_indexes
    assert recorded == [0]


def test_single_element_dataset_is_assigned_to_exactly_one_set():
    dataset = ["only"]
    suplier = DataSuplier(dataset)
    training_pairs, test_pairs = _exhaust_both(suplier)
    assert len(training_pairs) + len(test_pairs) == 1
    assert (training_pairs + test_pairs) == [(0, "only")]


def test_training_percent_one_puts_everything_in_training():
    dataset = _make_dataset(10)
    suplier = DataSuplier(dataset, training_percent=1.0)
    training_pairs, test_pairs = _exhaust_both(suplier)
    assert sorted(i for i, _ in training_pairs) == list(range(10))
    assert test_pairs == []


def test_training_percent_zero_puts_everything_in_test():
    dataset = _make_dataset(10)
    suplier = DataSuplier(dataset, training_percent=0.0)
    training_pairs, test_pairs = _exhaust_both(suplier)
    assert training_pairs == []
    assert sorted(i for i, _ in test_pairs) == list(range(10))
