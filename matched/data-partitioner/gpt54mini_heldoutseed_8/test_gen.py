import pytest

import data_partitioner.data_suplier as ds_module
from data_partitioner.data_suplier import DataSuplier


class FakePartitioner:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def __next__(self):
        if self.calls >= len(self.outputs):
            raise AssertionError("Partitioner called more times than expected")
        value = self.outputs[self.calls]
        self.calls += 1
        return value


def make_dataset(n):
    return [f"item-{i}" for i in range(n)]


def test_initial_state_uses_dataset_length_and_defaults():
    dataset = make_dataset(5)
    supplier = DataSuplier(dataset)

    assert supplier.dataset is dataset
    assert supplier.training_percent == 0.8
    assert supplier.n_of_elements == 5
    assert supplier.test_indexes == []
    assert supplier.training_indexes == []
    assert supplier.index == 0


def test_custom_training_percent_is_stored_and_partitioner_is_created_with_complement(monkeypatch):
    captured = {}

    def fake_partitioner(n, probs, func):
        captured["n"] = n
        captured["probs"] = probs
        captured["func"] = func
        return FakePartitioner([])

    monkeypatch.setattr(ds_module, "partitioner", fake_partitioner)

    dataset = make_dataset(4)
    supplier = DataSuplier(dataset, training_percent=0.3, partitioning_function="marker")

    assert supplier.training_percent == 0.3
    assert captured["n"] == 4
    assert captured["probs"] == [0.3, 0.7]
    assert captured["func"] == "marker"


def test_training_set_empty_dataset_yields_nothing():
    supplier = DataSuplier([], partitioning_function=lambda *args, **kwargs: FakePartitioner([]))

    assert list(supplier.training_set()) == []
    assert supplier.index == 0
    assert supplier.training_indexes == []
    assert supplier.test_indexes == []


def test_test_set_empty_dataset_yields_nothing():
    supplier = DataSuplier([], partitioning_function=lambda *args, **kwargs: FakePartitioner([]))

    assert list(supplier.test_set()) == []
    assert supplier.index == 0
    assert supplier.training_indexes == []
    assert supplier.test_indexes == []


def test_training_set_yields_training_items_in_index_order():
    dataset = make_dataset(4)
    supplier = DataSuplier(dataset, partitioning_function=lambda *args, **kwargs: FakePartitioner([0, 0, 1, 0]))

    assert list(supplier.training_set()) == [(0, "item-0"), (1, "item-1"), (3, "item-3")]
    assert supplier.training_indexes == [0, 1, 3]
    assert supplier.test_indexes == [2]
    assert supplier.index == 4


def test_test_set_yields_test_items_in_index_order():
    dataset = make_dataset(4)
    supplier = DataSuplier(dataset, partitioning_function=lambda *args, **kwargs: FakePartitioner([0, 1, 1, 0]))

    assert list(supplier.test_set()) == [(1, "item-1"), (2, "item-2")]
    assert supplier.training_indexes == [0, 3]
    assert supplier.test_indexes == [1, 2]
    assert supplier.index == 4


def test_generators_can_be_consumed_in_either_order():
    dataset = make_dataset(5)
    supplier = DataSuplier(dataset, partitioning_function=lambda *args, **kwargs: FakePartitioner([0, 1, 0, 1, 0]))

    assert list(supplier.test_set()) == [(1, "item-1"), (3, "item-3")]
    assert list(supplier.training_set()) == [(0, "item-0"), (2, "item-2"), (4, "item-4")]

    assert supplier.training_indexes == [0, 2, 4]
    assert supplier.test_indexes == [1, 3]
    assert supplier.index == 5


def test_repeated_calls_return_fresh_generators_with_same_results():
    dataset = make_dataset(3)
    supplier = DataSuplier(dataset, partitioning_function=lambda *args, **kwargs: FakePartitioner([0, 1, 0]))

    first = list(supplier.training_set())
    second = list(supplier.training_set())

    assert first == [(0, "item-0"), (2, "item-2")]
    assert second == first


def test_repeated_test_set_calls_return_same_results():
    dataset = make_dataset(3)
    supplier = DataSuplier(dataset, partitioning_function=lambda *args, **kwargs: FakePartitioner([1, 0, 1]))

    first = list(supplier.test_set())
    second = list(supplier.test_set())

    assert first == [(0, "item-0"), (2, "item-2")]
    assert second == first


def test_consuming_training_set_then_test_set_after_partial_progress():
    dataset = make_dataset(6)
    supplier = DataSuplier(dataset, partitioning_function=lambda *args, **kwargs: FakePartitioner([0, 1, 0, 1, 1, 0]))

    training_iter = supplier.training_set()
    assert next(training_iter) == (0, "item-0")
    assert next(training_iter) == (2, "item-2")

    assert list(supplier.test_set()) == [(1, "item-1"), (3, "item-3"), (4, "item-4")]
    assert list(training_iter) == [(5, "item-5")]
    assert supplier.index == 6


def test_consuming_test_set_then_training_set_after_partial_progress():
    dataset = make_dataset(5)
    supplier = DataSuplier(dataset, partitioning_function=lambda *args, **kwargs: FakePartitioner([1, 0, 1, 0, 0]))

    test_iter = supplier.test_set()
    assert next(test_iter) == (0, "item-0")

    assert list(supplier.training_set()) == [(1, "item-1"), (3, "item-3"), (4, "item-4")]
    assert list(test_iter) == [(2, "item-2")]
    assert supplier.index == 5


def test_internal_state_after_exhausting_all_items_is_consistent():
    dataset = make_dataset(4)
    supplier = DataSuplier(dataset, partitioning_function=lambda *args, **kwargs: FakePartitioner([0, 0, 1, 1]))

    assert list(supplier.training_set()) == [(0, "item-0"), (1, "item-1")]
    assert list(supplier.test_set()) == [(2, "item-2"), (3, "item-3")]

    assert supplier.index == 4
    assert supplier.training_indexes == [0, 1]
    assert supplier.test_indexes == [2, 3]


def test_all_items_can_go_to_training_set():
    dataset = make_dataset(3)
    supplier = DataSuplier(dataset, partitioning_function=lambda *args, **kwargs: FakePartitioner([0, 0, 0]))

    assert list(supplier.training_set()) == [(0, "item-0"), (1, "item-1"), (2, "item-2")]
    assert list(supplier.test_set()) == []
    assert supplier.training_indexes == [0, 1, 2]
    assert supplier.test_indexes == []


def test_all_items_can_go_to_test_set():
    dataset = make_dataset(3)
    supplier = DataSuplier(dataset, partitioning_function=lambda *args, **kwargs: FakePartitioner([1, 1, 1]))

    assert list(supplier.test_set()) == [(0, "item-0"), (1, "item-1"), (2, "item-2")]
    assert list(supplier.training_set()) == []
    assert supplier.training_indexes == []
    assert supplier.test_indexes == [0, 1, 2]


def test_partitioning_function_is_used_during_iteration_not_initialization(monkeypatch):
    called = {"count": 0}

    def fake_partitioner(n, probs, func):
        called["count"] += 1
        return FakePartitioner([0, 1])

    monkeypatch.setattr(ds_module, "partitioner", fake_partitioner)

    supplier = DataSuplier(make_dataset(2), partitioning_function="sentinel")
    assert called["count"] == 1

    assert supplier.index == 0
    assert supplier.training_indexes == []
    assert supplier.test_indexes == []

    assert list(supplier.training_set()) == [(0, "item-0")]
    assert list(supplier.test_set()) == [(1, "item-1")]
