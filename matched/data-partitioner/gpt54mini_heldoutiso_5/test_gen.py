import pytest

import data_partitioner.data_suplier as data_suplier_mod
from data_partitioner.data_suplier import DataSuplier


class FakePartitioner:
    def __init__(self, results):
        self._results = list(results)
        self._i = 0

    def __next__(self):
        if self._i >= len(self._results):
            raise StopIteration
        value = self._results[self._i]
        self._i += 1
        return value


def make_partitioner_factory(results):
    captured = {}

    def factory(n_of_elements, probabilities, partitioning_function):
        captured["n_of_elements"] = n_of_elements
        captured["probabilities"] = probabilities
        captured["partitioning_function"] = partitioning_function
        return FakePartitioner(results)

    factory.captured = captured
    return factory


def drain(generator):
    return list(generator)


def test_init_stores_dataset_and_training_percent():
    dataset = ["a", "b", "c"]
    factory = make_partitioner_factory([0, 1, 0])
    ds = DataSuplier(dataset, training_percent=0.7, partitioning_function=lambda x: x)

    # monkeypatch by direct assignment after construction not possible here; use class from module
    # so this test only checks state set by constructor inputs.
    assert ds.dataset is dataset
    assert ds.training_percent == 0.7
    assert ds.n_of_elements == 3


def test_init_calls_partitioner_with_expected_arguments(monkeypatch):
    dataset = [1, 2, 3, 4]
    factory = make_partitioner_factory([0, 1, 0, 1])
    monkeypatch.setattr(data_suplier_mod, "partitioner", factory)

    marker = object()
    ds = DataSuplier(dataset, training_percent=0.25, partitioning_function=marker)

    assert factory.captured["n_of_elements"] == 4
    assert factory.captured["probabilities"] == [0.25, 0.75]
    assert factory.captured["partitioning_function"] is marker
    assert ds.partitioner is not None


def test_training_set_returns_generator(monkeypatch):
    monkeypatch.setattr(data_suplier_mod, "partitioner", make_partitioner_factory([0]))
    ds = DataSuplier([10], training_percent=1.0)

    gen = ds.training_set()
    assert hasattr(gen, "__iter__")
    assert hasattr(gen, "__next__")


def test_test_set_returns_generator(monkeypatch):
    monkeypatch.setattr(data_suplier_mod, "partitioner", make_partitioner_factory([1]))
    ds = DataSuplier([10], training_percent=0.0)

    gen = ds.test_set()
    assert hasattr(gen, "__iter__")
    assert hasattr(gen, "__next__")


def test_training_set_yields_expected_pairs(monkeypatch):
    dataset = ["a", "b", "c", "d"]
    monkeypatch.setattr(data_suplier_mod, "partitioner", make_partitioner_factory([0, 1, 0, 1]))
    ds = DataSuplier(dataset)

    assert drain(ds.training_set()) == [(0, "a"), (2, "c")]


def test_test_set_yields_expected_pairs(monkeypatch):
    dataset = ["a", "b", "c", "d"]
    monkeypatch.setattr(data_suplier_mod, "partitioner", make_partitioner_factory([0, 1, 0, 1]))
    ds = DataSuplier(dataset)

    assert drain(ds.test_set()) == [(1, "b"), (3, "d")]


def test_training_and_test_sets_cover_all_elements_in_order(monkeypatch):
    dataset = ["a", "b", "c", "d", "e"]
    monkeypatch.setattr(data_suplier_mod, "partitioner", make_partitioner_factory([0, 1, 0, 1, 0]))
    ds = DataSuplier(dataset)

    training = drain(ds.training_set())
    test = drain(ds.test_set())

    assert training + test == [(0, "a"), (2, "c"), (4, "e"), (1, "b"), (3, "d")]


def test_training_set_can_be_drained_before_test_set(monkeypatch):
    dataset = [10, 20, 30, 40]
    monkeypatch.setattr(data_suplier_mod, "partitioner", make_partitioner_factory([0, 0, 1, 1]))
    ds = DataSuplier(dataset)

    assert drain(ds.training_set()) == [(0, 10), (1, 20)]
    assert drain(ds.test_set()) == [(2, 30), (3, 40)]


def test_test_set_can_be_drained_before_training_set(monkeypatch):
    dataset = [10, 20, 30, 40]
    monkeypatch.setattr(data_suplier_mod, "partitioner", make_partitioner_factory([0, 0, 1, 1]))
    ds = DataSuplier(dataset)

    assert drain(ds.test_set()) == [(2, 30), (3, 40)]
    assert drain(ds.training_set()) == [(0, 10), (1, 20)]


def test_empty_dataset_yields_empty_sets(monkeypatch):
    monkeypatch.setattr(data_suplier_mod, "partitioner", make_partitioner_factory([]))
    ds = DataSuplier([])

    assert drain(ds.training_set()) == []
    assert drain(ds.test_set()) == []


def test_all_training_dataset(monkeypatch):
    dataset = ["x", "y", "z"]
    monkeypatch.setattr(data_suplier_mod, "partitioner", make_partitioner_factory([0, 0, 0]))
    ds = DataSuplier(dataset, training_percent=1.0)

    assert drain(ds.training_set()) == [(0, "x"), (1, "y"), (2, "z")]
    assert drain(ds.test_set()) == []


def test_all_test_dataset(monkeypatch):
    dataset = ["x", "y", "z"]
    monkeypatch.setattr(data_suplier_mod, "partitioner", make_partitioner_factory([1, 1, 1]))
    ds = DataSuplier(dataset, training_percent=0.0)

    assert drain(ds.training_set()) == []
    assert drain(ds.test_set()) == [(0, "x"), (1, "y"), (2, "z")]


def test_indexes_are_cached_after_first_iteration(monkeypatch):
    dataset = ["a", "b", "c", "d"]
    monkeypatch.setattr(data_suplier_mod, "partitioner", make_partitioner_factory([0, 1, 0, 1]))
    ds = DataSuplier(dataset)

    first_training = drain(ds.training_set())
    first_test = drain(ds.test_set())

    assert ds.index == 4
    assert ds.training_indexes == [0, 2]
    assert ds.test_indexes == [1, 3]
    assert first_training == [(0, "a"), (2, "c")]
    assert first_test == [(1, "b"), (3, "d")]


def test_multiple_calls_return_same_results(monkeypatch):
    dataset = ["a", "b", "c", "d"]
    monkeypatch.setattr(data_suplier_mod, "partitioner", make_partitioner_factory([0, 1, 0, 1]))
    ds = DataSuplier(dataset)

    first = drain(ds.training_set())
    second = drain(ds.training_set())

    assert first == second == [(0, "a"), (2, "c")]
