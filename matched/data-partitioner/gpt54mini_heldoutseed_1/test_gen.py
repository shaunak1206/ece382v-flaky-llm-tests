import pytest

import data_partitioner.data_suplier as data_suplier_module
from data_partitioner.data_suplier import DataSuplier


class DummyPartitioner:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def __next__(self):
        if self.calls >= len(self.values):
            raise StopIteration
        value = self.values[self.calls]
        self.calls += 1
        return value


def make_supplier(monkeypatch, dataset, partition_values, training_percent=0.8):
    dummy = DummyPartitioner(partition_values)
    captured = {}

    def fake_partitioner(n_of_elements, probs, partitioning_function):
        captured["n_of_elements"] = n_of_elements
        captured["probs"] = probs
        captured["partitioning_function"] = partitioning_function
        return dummy

    monkeypatch.setattr(data_suplier_module, "partitioner", fake_partitioner)
    supplier = DataSuplier(dataset, training_percent=training_percent)
    return supplier, dummy, captured


def test_init_sets_basic_attributes(monkeypatch):
    dataset = ["a", "b", "c"]
    supplier, dummy, captured = make_supplier(monkeypatch, dataset, [0, 1, 0])

    assert supplier.dataset is dataset
    assert supplier.training_percent == 0.8
    assert supplier.n_of_elements == 3
    assert supplier.partitioner is dummy
    assert supplier.test_indexes == []
    assert supplier.training_indexes == []
    assert supplier.index == 0
    assert captured["n_of_elements"] == 3
    assert captured["probs"] == [0.8, 0.2]
    assert captured["partitioning_function"] is data_suplier_module.pseudorandom_function


def test_init_uses_custom_training_percent_in_partition_probs(monkeypatch):
    _, _, captured = make_supplier(monkeypatch, [1, 2], [0, 1], training_percent=0.3)
    assert captured["probs"] == [0.3, 0.7]


def test_training_set_yields_training_items_in_original_order(monkeypatch):
    dataset = ["a", "b", "c", "d"]
    supplier, _, _ = make_supplier(monkeypatch, dataset, [0, 1, 0, 1])

    assert list(supplier.training_set()) == [(0, "a"), (2, "c")]
    assert supplier.training_indexes == [0, 2]
    assert supplier.test_indexes == [1, 3]


def test_test_set_yields_test_items_in_original_order(monkeypatch):
    dataset = ["a", "b", "c", "d"]
    supplier, _, _ = make_supplier(monkeypatch, dataset, [0, 1, 0, 1])

    assert list(supplier.test_set()) == [(1, "b"), (3, "d")]
    assert supplier.training_indexes == [0, 2]
    assert supplier.test_indexes == [1, 3]


def test_training_set_and_test_set_can_be_consumed_independently(monkeypatch):
    dataset = ["a", "b", "c", "d"]
    supplier, _, _ = make_supplier(monkeypatch, dataset, [0, 1, 0, 1])

    assert list(supplier.training_set()) == [(0, "a"), (2, "c")]
    assert list(supplier.test_set()) == [(1, "b"), (3, "d")]


def test_yield_set_populates_indexes_lazily(monkeypatch):
    dataset = ["a", "b", "c", "d"]
    supplier, dummy, _ = make_supplier(monkeypatch, dataset, [0, 1, 0, 1])

    gen = supplier.training_set()
    assert next(gen) == (0, "a")
    assert dummy.calls == 1
    assert supplier.index == 1

    assert next(gen) == (2, "c")
    assert dummy.calls == 3
    assert supplier.index == 3

    with pytest.raises(StopIteration):
        next(gen)
    assert dummy.calls == 4
    assert supplier.index == 4


def test_partial_training_consumption_does_not_require_full_dataset(monkeypatch):
    dataset = ["a", "b", "c", "d", "e"]
    supplier, dummy, _ = make_supplier(monkeypatch, dataset, [0, 1, 0, 1, 0])

    gen = supplier.training_set()
    assert next(gen) == (0, "a")
    assert supplier.index == 1
    assert dummy.calls == 1
    assert supplier.training_indexes == [0]
    assert supplier.test_indexes == []

    assert next(gen) == (2, "c")
    assert supplier.index == 3
    assert dummy.calls == 3
    assert supplier.training_indexes == [0, 2]
    assert supplier.test_indexes == [1]


def test_next_method_updates_training_indexes(monkeypatch):
    supplier, _, _ = make_supplier(monkeypatch, ["a", "b"], [0, 1])

    res = supplier._DataSuplier__next()
    assert res == 0
    assert supplier.training_indexes == [0]
    assert supplier.test_indexes == []
    assert supplier.index == 1


def test_next_method_updates_test_indexes(monkeypatch):
    supplier, _, _ = make_supplier(monkeypatch, ["a", "b"], [1, 0])

    res = supplier._DataSuplier__next()
    assert res == 1
    assert supplier.training_indexes == []
    assert supplier.test_indexes == [0]
    assert supplier.index == 1


def test_empty_dataset_yields_no_items(monkeypatch):
    supplier, dummy, _ = make_supplier(monkeypatch, [], [])

    assert list(supplier.training_set()) == []
    assert list(supplier.test_set()) == []
    assert dummy.calls == 0
    assert supplier.index == 0


def test_single_item_training_set(monkeypatch):
    supplier, dummy, _ = make_supplier(monkeypatch, ["x"], [0])

    assert list(supplier.training_set()) == [(0, "x")]
    assert supplier.training_indexes == [0]
    assert supplier.test_indexes == []
    assert supplier.index == 1
    assert dummy.calls == 1


def test_single_item_test_set(monkeypatch):
    supplier, dummy, _ = make_supplier(monkeypatch, ["x"], [1])

    assert list(supplier.test_set()) == [(0, "x")]
    assert supplier.training_indexes == []
    assert supplier.test_indexes == [0]
    assert supplier.index == 1
    assert dummy.calls == 1


def test_training_and_test_sets_return_generators(monkeypatch):
    supplier, _, _ = make_supplier(monkeypatch, ["a", "b"], [0, 1])

    training_gen = supplier.training_set()
    test_gen = supplier.test_set()

    assert hasattr(training_gen, "__iter__")
    assert hasattr(training_gen, "__next__")
    assert hasattr(test_gen, "__iter__")
    assert hasattr(test_gen, "__next__")


def test_repeated_calls_to_training_set_continue_from_state(monkeypatch):
    dataset = ["a", "b", "c", "d"]
    supplier, _, _ = make_supplier(monkeypatch, dataset, [0, 1, 0, 1])

    gen1 = supplier.training_set()
    assert next(gen1) == (0, "a")

    gen2 = supplier.training_set()
    assert list(gen2) == [(0, "a"), (2, "c")]
    assert list(gen1) == [(2, "c")]


def test_repeated_calls_to_test_set_continue_from_state(monkeypatch):
    dataset = ["a", "b", "c", "d"]
    supplier, _, _ = make_supplier(monkeypatch, dataset, [0, 1, 0, 1])

    gen1 = supplier.test_set()
    assert next(gen1) == (1, "b")

    gen2 = supplier.test_set()
    assert list(gen2) == [(1, "b"), (3, "d")]
    assert list(gen1) == [(3, "d")]
