"""Unit tests for data_partitioner.data_suplier.DataSuplier."""

import random
import types

import pytest

from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import LinearFakeRandomFunction

VALUES = [0.0, 0.9, 0.1, 0.5, 0.79, 0.85, 0.99]


def fixed_values(values):
    """Partitioning function that returns values[index] for element index."""

    def _fn(index):
        return values[index]

    return _fn


def indexes_of(pairs):
    return [index for index, _ in pairs]


def interleave(gen_a, gen_b):
    """Alternately advance two generators until both are exhausted."""
    out_a, out_b = [], []
    done_a = done_b = False
    while not (done_a and done_b):
        if not done_a:
            try:
                out_a.append(next(gen_a))
            except StopIteration:
                done_a = True
        if not done_b:
            try:
                out_b.append(next(gen_b))
            except StopIteration:
                done_b = True
    return out_a, out_b


def test_constructor_sets_attributes():
    data = list(range(10))
    suplier = DataSuplier(
        data, training_percent=0.6, partitioning_function=fixed_values([0.0] * 10)
    )
    assert suplier.dataset is data
    assert suplier.training_percent == 0.6
    assert suplier.n_of_elements == 10
    assert suplier.index == 0
    assert suplier.training_indexes == []
    assert suplier.test_indexes == []


def test_default_training_percent_is_point_eight():
    suplier = DataSuplier(list(range(5)))
    assert suplier.training_percent == 0.8


def test_constructor_does_not_call_partitioning_function():
    calls = []

    def recorder(index):
        calls.append(index)
        return 0.0

    suplier = DataSuplier(list(range(5)), partitioning_function=recorder)
    assert calls == []
    assert suplier.index == 0


def test_fixed_values_assign_elements_to_expected_sets():
    data = list("abcdefg")
    suplier = DataSuplier(
        data, training_percent=0.8, partitioning_function=fixed_values(VALUES)
    )
    assert list(suplier.training_set()) == [(0, "a"), (2, "c"), (3, "d"), (4, "e")]
    assert list(suplier.test_set()) == [(1, "b"), (5, "f"), (6, "g")]


def test_index_lists_record_partition_after_iteration():
    suplier = DataSuplier(
        list("abcdefg"), training_percent=0.8, partitioning_function=fixed_values(VALUES)
    )
    list(suplier.training_set())
    list(suplier.test_set())
    assert suplier.training_indexes == [0, 2, 3, 4]
    assert suplier.test_indexes == [1, 5, 6]


def test_linear_fake_function_splits_in_half():
    suplier = DataSuplier(
        list(range(10)),
        training_percent=0.5,
        partitioning_function=LinearFakeRandomFunction(10),
    )
    assert indexes_of(suplier.training_set()) == [0, 1, 2, 3, 4]
    assert indexes_of(suplier.test_set()) == [5, 6, 7, 8, 9]


@pytest.mark.parametrize(
    "training_percent, expected_train",
    [(0.25, 5), (0.5, 10), (0.75, 15)],
)
def test_linear_partition_matches_requested_fraction(training_percent, expected_train):
    n = 20
    suplier = DataSuplier(
        list(range(n)), training_percent, LinearFakeRandomFunction(n)
    )
    assert indexes_of(suplier.training_set()) == list(range(expected_train))
    assert indexes_of(suplier.test_set()) == list(range(expected_train, n))


def test_pairs_hold_index_and_matching_item():
    data = ["zero", "one", "two", "three", "four"]
    suplier = DataSuplier(data, 0.5, LinearFakeRandomFunction(5))
    pairs = list(suplier.training_set()) + list(suplier.test_set())
    assert len(pairs) == 5
    for pair in pairs:
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        index, item = pair
        assert item == data[index]


@pytest.mark.parametrize("size", [1, 2, 7, 100])
@pytest.mark.parametrize("training_percent", [0.3, 0.8])
def test_default_partition_is_disjoint_and_covers_dataset(size, training_percent):
    suplier = DataSuplier(list(range(size)), training_percent=training_percent)
    train = indexes_of(suplier.training_set())
    test = indexes_of(suplier.test_set())
    assert sorted(train + test) == list(range(size))
    assert set(train).isdisjoint(test)


def test_yielded_indexes_are_ascending():
    values = [(i * 7 % 10) / 10 for i in range(30)]
    suplier = DataSuplier(
        list(range(30)), training_percent=0.4, partitioning_function=fixed_values(values)
    )
    train = indexes_of(suplier.training_set())
    test = indexes_of(suplier.test_set())
    assert train == sorted(train)
    assert test == sorted(test)


def test_empty_dataset_yields_nothing():
    suplier = DataSuplier([])
    assert suplier.n_of_elements == 0
    assert list(suplier.training_set()) == []
    assert list(suplier.test_set()) == []


def test_training_percent_one_sends_everything_to_training():
    suplier = DataSuplier(
        list(range(4)),
        training_percent=1.0,
        partitioning_function=fixed_values([0.0, 0.25, 0.5, 0.999]),
    )
    assert indexes_of(suplier.training_set()) == [0, 1, 2, 3]
    assert list(suplier.test_set()) == []


def test_training_percent_zero_sends_everything_to_test():
    suplier = DataSuplier(
        list(range(4)),
        training_percent=0.0,
        partitioning_function=fixed_values([0.0, 0.25, 0.5, 0.999]),
    )
    assert list(suplier.training_set()) == []
    assert indexes_of(suplier.test_set()) == [0, 1, 2, 3]


@pytest.mark.parametrize("bad_percent", [-0.1, 1.5])
def test_out_of_range_training_percent_fails_on_first_iteration(bad_percent):
    suplier = DataSuplier(
        list(range(3)),
        training_percent=bad_percent,
        partitioning_function=fixed_values([0.0, 0.0, 0.0]),
    )
    with pytest.raises(AssertionError):
        list(suplier.training_set())


def test_partitioning_function_called_once_per_element_in_order():
    calls = []

    def recorder(index):
        calls.append(index)
        return 0.0 if index % 2 == 0 else 0.9

    suplier = DataSuplier(list(range(6)), 0.5, partitioning_function=recorder)
    list(suplier.training_set())
    list(suplier.test_set())
    assert calls == [0, 1, 2, 3, 4, 5]


def test_partitioning_is_lazy_and_stops_at_first_needed_element():
    calls = []
    draws = [0.9, 0.9, 0.1, 0.9, 0.2]

    def recorder(index):
        calls.append(index)
        return draws[index]

    suplier = DataSuplier(list("abcde"), 0.8, partitioning_function=recorder)
    first = next(suplier.training_set())
    assert first == (2, "c")
    assert calls == [0, 1, 2]
    assert suplier.index == 3


def test_interleaved_iteration_matches_sequential_iteration():
    data = list(range(12))
    values = [(i * 5 % 12) / 12 for i in range(12)]
    interleaved = DataSuplier(data, 0.6, fixed_values(values))
    train_interleaved, test_interleaved = interleave(
        interleaved.training_set(), interleaved.test_set()
    )
    sequential = DataSuplier(data, 0.6, fixed_values(values))
    assert train_interleaved == list(sequential.training_set())
    assert test_interleaved == list(sequential.test_set())


def test_repeated_iteration_returns_same_pairs():
    suplier = DataSuplier(list("abcdefghij"), 0.5, LinearFakeRandomFunction(10))
    first_train = list(suplier.training_set())
    second_train = list(suplier.training_set())
    first_test = list(suplier.test_set())
    second_test = list(suplier.test_set())
    assert first_train == second_train
    assert first_test == second_test


def test_set_methods_return_generators():
    suplier = DataSuplier(list(range(3)), 0.5, LinearFakeRandomFunction(3))
    assert isinstance(suplier.training_set(), types.GeneratorType)
    assert isinstance(suplier.test_set(), types.GeneratorType)


def test_default_partitioning_function_uses_random_random(monkeypatch):
    draws = iter([0.1, 0.9, 0.2, 0.95])
    monkeypatch.setattr(random, "random", lambda: next(draws))
    suplier = DataSuplier(list("abcd"))
    assert list(suplier.training_set()) == [(0, "a"), (2, "c")]
    assert list(suplier.test_set()) == [(1, "b"), (3, "d")]


def test_index_reaches_dataset_length_after_full_iteration():
    suplier = DataSuplier(list(range(9)), 0.5, LinearFakeRandomFunction(9))
    list(suplier.test_set())
    assert suplier.index == 9
    assert len(suplier.training_indexes) + len(suplier.test_indexes) == 9


def test_string_dataset_is_split_by_character():
    suplier = DataSuplier("abcdef", 0.5, LinearFakeRandomFunction(6))
    assert suplier.n_of_elements == 6
    assert list(suplier.training_set()) == [(0, "a"), (1, "b"), (2, "c")]
    assert list(suplier.test_set()) == [(3, "d"), (4, "e"), (5, "f")]


def test_separate_instances_do_not_share_state():
    data = list(range(4))
    first = DataSuplier(data, 0.5, fixed_values([0.0, 0.0, 0.0, 0.0]))
    second = DataSuplier(data, 0.5, fixed_values([0.9, 0.9, 0.9, 0.9]))
    assert indexes_of(first.training_set()) == [0, 1, 2, 3]
    assert indexes_of(second.training_set()) == []
    assert indexes_of(second.test_set()) == [0, 1, 2, 3]
    assert first.test_indexes == []
