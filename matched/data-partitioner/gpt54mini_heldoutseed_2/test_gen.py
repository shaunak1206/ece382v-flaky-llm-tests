import pytest

import data_partitioner.data_suplier as data_suplier_module
from data_partitioner.data_suplier import DataSuplier


class FakePartitioner:
    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def __next__(self):
        if self.calls >= len(self._results):
            raise AssertionError("FakePartitioner exhausted unexpectedly")
        value = self._results[self.calls]
        self.calls += 1
        return value


def make_partitioner_factory(results, captured=None):
    def factory(n_of_elements, probs, partitioning_function):
        if captured is not None:
            captured["n_of_elements"] = n_of_elements
            captured["probs"] = probs
            captured["partitioning_function"] = partitioning_function
        return FakePartitioner(results)

    return factory


def test_initialization_sets_basic_attributes(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        data_suplier_module,
        "partitioner",
        make_partitioner_factory([0, 1, 0], captured),
    )

    dataset = ["a", "b", "c"]
    ds = DataSuplier(dataset, training_percent=0.7, partitioning_function="pf")

    assert ds.dataset is dataset
    assert ds.training_percent == 0.7
    assert ds.n_of_elements == 3
    assert ds.index == 0
    assert ds.training_indexes == []
    assert ds.test_indexes == []
    assert captured["n_of_elements"] == 3
    assert captured["probs"] == [0.7, 0.30000000000000004]
    assert captured["partitioning_function"] == "pf"


def test_training_and_test_set_return_generators(monkeypatch):
    monkeypatch.setattr(
        data_suplier_module,
        "partitioner",
        make_partitioner_factory([0, 1]),
    )
    ds = DataSuplier([10, 20], training_percent=0.5)

    train_gen = ds.training_set()
    test_gen = ds.test_set()

    assert hasattr(train_gen, "__iter__")
    assert hasattr(test_gen, "__iter__")
    assert train_gen is not ds.training_set()
    assert test_gen is not ds.test_set()


def test_training_set_yields_only_training_items_in_original_order(monkeypatch):
    monkeypatch.setattr(
        data_suplier_module,
        "partitioner",
        make_partitioner_factory([0, 1, 0, 1]),
    )
    dataset = ["a", "b", "c", "d"]
    ds = DataSuplier(dataset)

    assert list(ds.training_set()) == [(0, "a"), (2, "c")]
    assert ds.training_indexes == [0, 2]
    assert ds.test_indexes == [1, 3]
    assert ds.index == 4


def test_test_set_yields_only_test_items_in_original_order(monkeypatch):
    monkeypatch.setattr(
        data_suplier_module,
        "partitioner",
        make_partitioner_factory([0, 1, 0, 1]),
    )
    dataset = ["a", "b", "c", "d"]
    ds = DataSuplier(dataset)

    assert list(ds.test_set()) == [(1, "b"), (3, "d")]
    assert ds.training_indexes == [0, 2]
    assert ds.test_indexes == [1, 3]
    assert ds.index == 4


def test_generators_are_independent_but_share_partitioning_state(monkeypatch):
    monkeypatch.setattr(
        data_suplier_module,
        "partitioner",
        make_partitioner_factory([0, 1, 0, 1]),
    )
    ds = DataSuplier(["a", "b", "c", "d"])

    train_gen = ds.training_set()
    test_gen = ds.test_set()

    assert next(train_gen) == (0, "a")
    assert next(test_gen) == (1, "b")
    assert next(train_gen) == (2, "c")
    assert next(test_gen) == (3, "d")


def test_training_set_after_test_set_still_returns_all_training_items(monkeypatch):
    monkeypatch.setattr(
        data_suplier_module,
        "partitioner",
        make_partitioner_factory([0, 1, 0, 1]),
    )
    ds = DataSuplier(["a", "b", "c", "d"])

    assert list(ds.test_set()) == [(1, "b"), (3, "d")]
    assert list(ds.training_set()) == [(0, "a"), (2, "c")]


def test_test_set_after_training_set_still_returns_all_test_items(monkeypatch):
    monkeypatch.setattr(
        data_suplier_module,
        "partitioner",
        make_partitioner_factory([0, 1, 0, 1]),
    )
    ds = DataSuplier(["a", "b", "c", "d"])

    assert list(ds.training_set()) == [(0, "a"), (2, "c")]
    assert list(ds.test_set()) == [(1, "b"), (3, "d")]


def test_empty_dataset_yields_empty_generators(monkeypatch):
    monkeypatch.setattr(
        data_suplier_module,
        "partitioner",
        make_partitioner_factory([]),
    )
    ds = DataSuplier([])

    assert list(ds.training_set()) == []
    assert list(ds.test_set()) == []
    assert ds.index == 0
    assert ds.training_indexes == []
    assert ds.test_indexes == []


def test_all_training_items(monkeypatch):
    monkeypatch.setattr(
        data_suplier_module,
        "partitioner",
        make_partitioner_factory([0, 0, 0]),
    )
    ds = DataSuplier(["a", "b", "c"], training_percent=1.0)

    assert list(ds.training_set()) == [(0, "a"), (1, "b"), (2, "c")]
    assert list(ds.test_set()) == []
    assert ds.training_indexes == [0, 1, 2]
    assert ds.test_indexes == []


def test_all_test_items(monkeypatch):
    monkeypatch.setattr(
        data_suplier_module,
        "partitioner",
        make_partitioner_factory([1, 1, 1]),
    )
    ds = DataSuplier(["a", "b", "c"], training_percent=0.0)

    assert list(ds.training_set()) == []
    assert list(ds.test_set()) == [(0, "a"), (1, "b"), (2, "c")]
    assert ds.training_indexes == []
    assert ds.test_indexes == [0, 1, 2]


def test_iteration_consumes_partitioner_only_once_per_dataset_item(monkeypatch):
    fake = FakePartitioner([0, 1, 0, 1])
    captured = {}

    def factory(n_of_elements, probs, partitioning_function):
        captured["factory_called"] = True
        return fake

    monkeypatch.setattr(data_suplier_module, "partitioner", factory)

    ds = DataSuplier(["a", "b", "c", "d"])
    assert list(ds.training_set()) == [(0, "a"), (2, "c")]
    assert fake.calls == 4
    assert ds.index == 4
    assert captured["factory_called"] is True


def test_partial_consumption_then_further_iteration_continues_correctly(monkeypatch):
    monkeypatch.setattr(
        data_suplier_module,
        "partitioner",
        make_partitioner_factory([0, 1, 0, 1]),
    )
    ds = DataSuplier(["a", "b", "c", "d"])

    train_gen = ds.training_set()
    assert next(train_gen) == (0, "a")

    test_gen = ds.test_set()
    assert next(test_gen) == (1, "b")

    remaining_train = list(train_gen)
    remaining_test = list(test_gen)

    assert remaining_train == [(2, "c")]
    assert remaining_test == [(3, "d")]


def test_yield_set_preserves_index_values_even_if_dataset_values_repeat(monkeypatch):
    monkeypatch.setattr(
        data_suplier_module,
        "partitioner",
        make_partitioner_factory([0, 1, 0, 1]),
    )
    ds = DataSuplier(["x", "x", "x", "x"])

    assert list(ds.training_set()) == [(0, "x"), (2, "x")]
    assert list(ds.test_set()) == [(1, "x"), (3, "x")]


def test_training_and_test_indexes_are_populated_in_partition_order(monkeypatch):
    monkeypatch.setattr(
        data_suplier_module,
        "partitioner",
        make_partitioner_factory([1, 0, 1, 0, 0]),
    )
    ds = DataSuplier(["a", "b", "c", "d", "e"])

    list(ds.training_set())
    list(ds.test_set())

    assert ds.test_indexes == [0, 2]
    assert ds.training_indexes == [1, 3, 4]
