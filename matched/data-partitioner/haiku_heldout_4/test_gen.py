"""Unit tests for data_partitioner.data_suplier.DataSuplier."""

import pytest

from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import pseudorandom_function


def _exhaust(generator):
    """Consume a generator and return its items as a list."""
    return list(generator)


def _make_dataset(n):
    """Build a simple dataset where each element is a distinct, checkable value."""
    return ["item-{}".format(i) for i in range(n)]


def test_init_stores_dataset_and_training_percent():
    dataset = _make_dataset(10)
    suplier = DataSuplier(dataset, training_percent=0.7)

    assert suplier.dataset is dataset
    assert suplier.training_percent == 0.7


def test_init_default_training_percent_is_point_eight():
    suplier = DataSuplier(_make_dataset(10))

    assert suplier.training_percent == 0.8


def test_init_records_number_of_elements():
    suplier = DataSuplier(_make_dataset(37))

    assert suplier.n_of_elements == 37


def test_init_starts_with_empty_index_lists_and_zero_index():
    suplier = DataSuplier(_make_dataset(10))

    assert suplier.training_indexes == []
    assert suplier.test_indexes == []
    assert suplier.index == 0


def test_init_does_not_consume_partitioner():
    suplier = DataSuplier(_make_dataset(20))

    assert suplier.index == 0
    assert suplier.training_indexes == []
    assert suplier.test_indexes == []


def test_creating_generators_does_not_advance_state():
    suplier = DataSuplier(_make_dataset(20))

    suplier.training_set()
    suplier.test_set()

    assert suplier.index == 0
    assert suplier.training_indexes == []
    assert suplier.test_indexes == []


def test_training_set_yields_index_and_value_pairs():
    dataset = _make_dataset(25)
    suplier = DataSuplier(dataset)

    items = _exhaust(suplier.training_set())

    assert len(items) > 0
    for index, value in items:
        assert isinstance(index, int)
        assert value == dataset[index]


def test_test_set_yields_index_and_value_pairs():
    dataset = _make_dataset(25)
    suplier = DataSuplier(dataset)

    items = _exhaust(suplier.test_set())

    assert len(items) > 0
    for index, value in items:
        assert value == dataset[index]


def test_training_and_test_sets_together_cover_every_index_exactly_once():
    n = 50
    suplier = DataSuplier(_make_dataset(n))

    train_indices = [index for index, _ in _exhaust(suplier.training_set())]
    test_indices = [index for index, _ in _exhaust(suplier.test_set())]

    combined = sorted(train_indices + test_indices)
    assert combined == list(range(n))
    assert set(train_indices).isdisjoint(test_indices)


def test_training_set_yields_exactly_the_training_indexes_in_ascending_order():
    suplier = DataSuplier(_make_dataset(40))

    yielded = [index for index, _ in _exhaust(suplier.training_set())]

    assert yielded == suplier.training_indexes
    assert yielded == sorted(yielded)


def test_test_set_yields_exactly_the_test_indexes_in_ascending_order():
    suplier = DataSuplier(_make_dataset(40))

    yielded = [index for index, _ in _exhaust(suplier.test_set())]

    assert yielded == suplier.test_indexes
    assert yielded == sorted(yielded)


def test_full_consumption_advances_index_to_length_of_dataset():
    n = 30
    suplier = DataSuplier(_make_dataset(n))

    _exhaust(suplier.training_set())

    assert suplier.index == n


def test_calling_training_set_twice_yields_same_pairs():
    suplier = DataSuplier(_make_dataset(30))

    first = _exhaust(suplier.training_set())
    second = _exhaust(suplier.training_set())

    assert first == second


def test_calling_test_set_twice_yields_same_pairs():
    suplier = DataSuplier(_make_dataset(30))

    first = _exhaust(suplier.test_set())
    second = _exhaust(suplier.test_set())

    assert first == second


def test_partial_consumption_only_advances_as_far_as_needed():
    n = 100
    suplier = DataSuplier(_make_dataset(n))

    generator = suplier.training_set()
    index, _ = next(generator)

    assert suplier.index >= 1
    assert suplier.index <= n
    assert suplier.training_indexes[0] == index
    assert len(suplier.training_indexes) == 1


def test_interleaving_both_sets_still_covers_every_index_once():
    n = 60
    dataset = _make_dataset(n)
    suplier = DataSuplier(dataset)

    train_gen = suplier.training_set()
    test_gen = suplier.test_set()
    seen = []
    train_done = test_done = False
    while not (train_done and test_done):
        if not train_done:
            try:
                seen.append(next(train_gen)[0])
            except StopIteration:
                train_done = True
        if not test_done:
            try:
                seen.append(next(test_gen)[0])
            except StopIteration:
                test_done = True

    assert sorted(seen) == list(range(n))


def test_empty_dataset_yields_nothing_for_both_sets():
    suplier = DataSuplier([])

    assert suplier.n_of_elements == 0
    assert _exhaust(suplier.training_set()) == []
    assert _exhaust(suplier.test_set()) == []
    assert suplier.training_indexes == []
    assert suplier.test_indexes == []


def test_single_element_dataset_is_assigned_to_exactly_one_set():
    dataset = ["only"]
    suplier = DataSuplier(dataset)

    train = _exhaust(suplier.training_set())
    test = _exhaust(suplier.test_set())

    assert len(train) + len(test) == 1
    assert (train + test) == [(0, "only")]


def test_explicit_pseudorandom_function_matches_default():
    dataset = _make_dataset(45)

    default_suplier = DataSuplier(dataset)
    explicit_suplier = DataSuplier(dataset, 0.8, pseudorandom_function)

    assert _exhaust(default_suplier.training_set()) == _exhaust(explicit_suplier.training_set())
    assert _exhaust(default_suplier.test_set()) == _exhaust(explicit_suplier.test_set())


def test_split_is_consistent_across_instances_with_same_inputs():
    dataset = _make_dataset(80)

    first = DataSuplier(dataset, 0.6)
    second = DataSuplier(dataset, 0.6)

    assert _exhaust(first.training_set()) == _exhaust(second.training_set())
    assert _exhaust(first.test_set()) == _exhaust(second.test_set())


def test_training_fraction_is_roughly_the_requested_percentage():
    n = 1000
    suplier = DataSuplier(_make_dataset(n), training_percent=0.8)

    train_count = len(_exhaust(suplier.training_set()))

    assert abs(train_count / float(n) - 0.8) < 0.1


def test_works_with_non_string_elements():
    dataset = [{"id": i, "payload": [i, i * 2]} for i in range(20)]
    suplier = DataSuplier(dataset)

    for index, value in _exhaust(suplier.training_set()) + _exhaust(suplier.test_set()):
        assert value is dataset[index]
