"""Unit tests for data_partitioner.data_suplier.DataSuplier."""

import inspect
import random

import pytest

from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import LinearFakeRandomFunction, pseudorandom_function


class ScriptedRandom(object):
    """Partitioning function that returns scripted values and records each index it sees."""

    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    def __call__(self, index):
        self.calls.append(index)
        return self.values.pop(0)


def split_indexes(supplier):
    """Return (training_indexes, test_indexes) as lists, consuming both sets."""
    train = [index for index, _ in supplier.training_set()]
    test = [index for index, _ in supplier.test_set()]
    return train, test


def test_construction_stores_dataset_and_defaults():
    data = list(range(7))
    ds = DataSuplier(data)
    assert ds.dataset is data
    assert ds.n_of_elements == 7
    assert ds.training_percent == 0.8
    assert ds.index == 0
    assert ds.training_indexes == []
    assert ds.test_indexes == []


def test_construction_does_not_call_partitioning_function():
    rand = ScriptedRandom([0.1] * 4)
    DataSuplier(list(range(4)), 0.5, rand)
    assert rand.calls == []


def test_generators_are_lazy_until_iterated():
    rand = ScriptedRandom([0.1] * 4)
    ds = DataSuplier(list(range(4)), 0.5, rand)
    ds.training_set()
    ds.test_set()
    assert rand.calls == []
    assert ds.index == 0


def test_partitioning_function_called_once_per_element_in_order():
    rand = ScriptedRandom([0.1, 0.2, 0.3, 0.4, 0.6])
    ds = DataSuplier(list(range(5)), 0.5, rand)
    list(ds.training_set())
    assert rand.calls == [0, 1, 2, 3, 4]
    list(ds.test_set())
    assert rand.calls == [0, 1, 2, 3, 4]
    assert ds.index == 5


def test_scripted_values_route_elements_to_sets():
    data = list("abcdef")
    rand = ScriptedRandom([0.1, 0.9, 0.2, 0.95, 0.4, 0.99])
    ds = DataSuplier(data, 0.5, rand)
    assert list(ds.training_set()) == [(0, "a"), (2, "c"), (4, "e")]
    assert list(ds.test_set()) == [(1, "b"), (3, "d"), (5, "f")]


def test_linear_half_split_of_ten():
    ds = DataSuplier(list(range(10)), 0.5, LinearFakeRandomFunction(10))
    train, test = split_indexes(ds)
    assert train == [0, 1, 2, 3, 4]
    assert test == [5, 6, 7, 8, 9]


def test_linear_quarter_split_of_eight():
    ds = DataSuplier(list(range(8)), 0.25, LinearFakeRandomFunction(8))
    train, test = split_indexes(ds)
    assert train == [0, 1]
    assert test == [2, 3, 4, 5, 6, 7]


def test_default_training_percent_is_eighty():
    ds = DataSuplier(list(range(10)), partitioning_function=LinearFakeRandomFunction(10))
    train, test = split_indexes(ds)
    assert train == [0, 1, 2, 3, 4, 5, 6, 7]
    assert test == [8, 9]


def test_zero_percent_sends_everything_to_test():
    ds = DataSuplier(list(range(5)), 0.0, LinearFakeRandomFunction(5))
    train, test = split_indexes(ds)
    assert train == []
    assert test == [0, 1, 2, 3, 4]


def test_full_percent_sends_everything_to_training():
    ds = DataSuplier(list(range(5)), 1.0, LinearFakeRandomFunction(5))
    train, test = split_indexes(ds)
    assert train == [0, 1, 2, 3, 4]
    assert test == []


def test_single_element_goes_to_training():
    ds = DataSuplier(["only"], 0.5, LinearFakeRandomFunction(1))
    assert list(ds.training_set()) == [(0, "only")]
    assert list(ds.test_set()) == []


def test_empty_dataset_yields_nothing():
    ds = DataSuplier([])
    assert ds.n_of_elements == 0
    assert list(ds.training_set()) == []
    assert list(ds.test_set()) == []


def test_yielded_pairs_hold_index_and_same_element():
    data = ["zero", "one", "two"]
    ds = DataSuplier(data, 0.5, LinearFakeRandomFunction(3))
    pairs = list(ds.training_set())
    assert pairs == [(0, "zero"), (1, "one")]
    for index, item in pairs:
        assert item is data[index]
    (test_pair,) = list(ds.test_set())
    assert test_pair == (2, "two")


def test_indexes_are_yielded_in_ascending_order():
    random.seed(11)
    ds = DataSuplier(list(range(200)))
    train, test = split_indexes(ds)
    assert train == sorted(train)
    assert test == sorted(test)


def test_sets_partition_dataset_without_overlap():
    random.seed(5)
    ds = DataSuplier(list(range(300)))
    train, test = split_indexes(ds)
    assert len(train) + len(test) == 300
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == list(range(300))


def test_repeat_iteration_yields_same_items():
    random.seed(17)
    ds = DataSuplier(list(range(60)), 0.6)
    first_train = list(ds.training_set())
    first_test = list(ds.test_set())
    assert list(ds.training_set()) == first_train
    assert list(ds.test_set()) == first_test


def test_interleaved_consumption_matches_sequential():
    values = [0.1, 0.9, 0.2, 0.95, 0.4, 0.99, 0.3, 0.6]
    data = list(range(8))
    ref_train, ref_test = split_indexes(DataSuplier(data, 0.5, ScriptedRandom(values)))

    ds = DataSuplier(data, 0.5, ScriptedRandom(values))
    test_gen = ds.test_set()
    train_gen = ds.training_set()
    got_test, got_train = [], []
    for _ in range(4):
        got_test.append(next(test_gen)[0])
        got_train.append(next(train_gen)[0])

    assert got_train == ref_train == [0, 2, 4, 6]
    assert got_test == ref_test == [1, 3, 5, 7]
    with pytest.raises(StopIteration):
        next(test_gen)
    with pytest.raises(StopIteration):
        next(train_gen)


def test_pseudorandom_split_reproducible_with_seed():
    data = list(range(100))
    random.seed(31337)
    first = split_indexes(DataSuplier(data))
    random.seed(31337)
    second = split_indexes(DataSuplier(data))
    assert first == second


def test_default_partitioning_function_is_pseudorandom():
    params = inspect.signature(DataSuplier.__init__).parameters
    assert params["partitioning_function"].default is pseudorandom_function
    assert params["training_percent"].default == 0.8


def test_pseudorandom_training_fraction_near_percent():
    random.seed(2024)
    data = list(range(2000))
    train, test = split_indexes(DataSuplier(data, 0.8))
    assert 1500 <= len(train) <= 1700
    assert len(train) + len(test) == 2000


def test_out_of_range_percent_fails_on_iteration():
    ds = DataSuplier([1, 2, 3], training_percent=1.5)
    assert ds.n_of_elements == 3
    with pytest.raises(AssertionError):
        list(ds.training_set())


def test_set_methods_return_generators():
    ds = DataSuplier([1, 2, 3, 4], 0.5, LinearFakeRandomFunction(4))
    assert inspect.isgenerator(ds.training_set())
    assert inspect.isgenerator(ds.test_set())
