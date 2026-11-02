import random

import pytest

from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import LinearFakeRandomFunction


def make_supplier(dataset, percent=0.5):
    return DataSuplier(
        dataset,
        training_percent=percent,
        partitioning_function=LinearFakeRandomFunction(len(dataset)),
    )


def test_init_stores_attributes():
    data = list("abcdef")
    supplier = DataSuplier(data, training_percent=0.7)
    assert supplier.dataset is data
    assert supplier.training_percent == 0.7
    assert supplier.n_of_elements == 6
    assert supplier.index == 0
    assert supplier.training_indexes == []
    assert supplier.test_indexes == []


def test_default_training_percent_is_80():
    supplier = DataSuplier([1, 2, 3])
    assert supplier.training_percent == 0.8


def test_init_is_lazy():
    supplier = make_supplier(list(range(10)))
    assert supplier.index == 0
    assert supplier.training_indexes == []
    assert supplier.test_indexes == []


def test_training_set_with_linear_function():
    supplier = make_supplier(["a", "b", "c", "d"], 0.5)
    assert list(supplier.training_set()) == [(0, "a"), (1, "b")]


def test_test_set_with_linear_function():
    supplier = make_supplier(["a", "b", "c", "d"], 0.5)
    assert list(supplier.test_set()) == [(2, "c"), (3, "d")]


def test_yielded_items_are_index_value_pairs():
    data = [10, 20, 30, 40]
    supplier = make_supplier(data, 0.5)
    for index, value in supplier.training_set():
        assert data[index] == value
    for index, value in supplier.test_set():
        assert data[index] == value


def test_sets_are_disjoint_and_cover_dataset():
    random.seed(1234)
    data = list(range(50))
    supplier = DataSuplier(data, training_percent=0.6)
    train = [i for i, _ in supplier.training_set()]
    test = [i for i, _ in supplier.test_set()]
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == data


def test_indexes_are_increasing_within_each_set():
    random.seed(7)
    supplier = DataSuplier(list(range(40)), training_percent=0.5)
    train = [i for i, _ in supplier.training_set()]
    test = [i for i, _ in supplier.test_set()]
    assert train == sorted(train)
    assert test == sorted(test)


def test_repeated_calls_are_consistent():
    random.seed(99)
    supplier = DataSuplier(list(range(30)), training_percent=0.5)
    first_train = list(supplier.training_set())
    first_test = list(supplier.test_set())
    assert list(supplier.training_set()) == first_train
    assert list(supplier.test_set()) == first_test


def test_test_set_first_then_training_set_is_consistent():
    supplier = make_supplier(list(range(8)), 0.5)
    test = list(supplier.test_set())
    train = list(supplier.training_set())
    assert [i for i, _ in train] == [0, 1, 2, 3]
    assert [i for i, _ in test] == [4, 5, 6, 7]


def test_interleaved_generators():
    supplier = make_supplier(list(range(6)), 0.5)
    train_gen = supplier.training_set()
    test_gen = supplier.test_set()
    assert next(train_gen) == (0, 0)
    assert next(test_gen) == (3, 3)
    assert next(train_gen) == (1, 1)
    assert next(test_gen) == (4, 4)
    assert next(train_gen) == (2, 2)
    assert next(test_gen) == (5, 5)
    with pytest.raises(StopIteration):
        next(train_gen)
    with pytest.raises(StopIteration):
        next(test_gen)


def test_partial_consumption_is_lazy():
    supplier = make_supplier(list(range(10)), 0.5)
    gen = supplier.training_set()
    assert next(gen) == (0, 0)
    assert supplier.index == 1
    assert supplier.training_indexes == [0]
    assert supplier.test_indexes == []


def test_all_training_with_custom_function():
    supplier = DataSuplier(list("xyz"), 0.5, partitioning_function=lambda i: 0.0)
    assert list(supplier.training_set()) == [(0, "x"), (1, "y"), (2, "z")]
    assert list(supplier.test_set()) == []


def test_all_test_with_custom_function():
    supplier = DataSuplier(list("xyz"), 0.5, partitioning_function=lambda i: 0.99)
    assert list(supplier.training_set()) == []
    assert list(supplier.test_set()) == [(0, "x"), (1, "y"), (2, "z")]


def test_full_training_percent_puts_everything_in_training():
    random.seed(5)
    supplier = DataSuplier(list(range(20)), training_percent=1.0)
    assert len(list(supplier.training_set())) == 20
    assert list(supplier.test_set()) == []


def test_zero_training_percent_puts_everything_in_test():
    random.seed(5)
    supplier = DataSuplier(list(range(20)), training_percent=0)
    assert list(supplier.training_set()) == []
    assert len(list(supplier.test_set())) == 20


def test_works_with_non_list_sequences():
    supplier = make_supplier("abcd", 0.5)
    assert list(supplier.training_set()) == [(0, "a"), (1, "b")]
    assert list(supplier.test_set()) == [(2, "c"), (3, "d")]


def test_empty_dataset_yields_nothing():
    supplier = DataSuplier([])
    assert supplier.n_of_elements == 0
    assert list(supplier.training_set()) == []
    assert list(supplier.test_set()) == []
