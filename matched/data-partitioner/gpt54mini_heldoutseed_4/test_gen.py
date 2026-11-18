import pytest

import data_partitioner.data_suplier as ds_module
from data_partitioner.data_suplier import DataSuplier


class FakePartitioner:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def __next__(self):
        if self.calls >= len(self.values):
            raise StopIteration
        value = self.values[self.calls]
        self.calls += 1
        return value


def make_partition_factory(sequence, captured=None):
    """
    Returns a partitioner factory that yields sequence and optionally captures args.
    """
    def factory(n_of_elements, probabilities, partitioning_function):
        if captured is not None:
            captured["n_of_elements"] = n_of_elements
            captured["probabilities"] = probabilities
            captured["partitioning_function"] = partitioning_function
        return FakePartitioner(sequence)
    return factory


def test_init_sets_expected_attributes(monkeypatch):
    monkeypatch.setattr(ds_module, "partitioner", make_partition_factory([0, 1, 0]))
    dataset = ["a", "b", "c"]

    suplier = DataSuplier(dataset, training_percent=0.7, partitioning_function="pf")

    assert suplier.dataset == dataset
    assert suplier.training_percent == 0.7
    assert suplier.n_of_elements == 3
    assert suplier.test_indexes == []
    assert suplier.training_indexes == []
    assert suplier.index == 0
    assert isinstance(suplier.partitioner, FakePartitioner)


def test_init_passes_expected_arguments_to_partitioner(monkeypatch):
    captured = {}
    monkeypatch.setattr(ds_module, "partitioner", make_partition_factory([0], captured))
    dataset = [1, 2, 3, 4]

    DataSuplier(dataset, training_percent=0.25, partitioning_function="custom_pf")

    assert captured["n_of_elements"] == 4
    assert captured["probabilities"] == [0.25, 0.75]
    assert captured["partitioning_function"] == "custom_pf"


def test_training_set_returns_generator(monkeypatch):
    monkeypatch.setattr(ds_module, "partitioner", make_partition_factory([0]))
    suplier = DataSuplier([1])

    gen = suplier.training_set()

    assert hasattr(gen, "__iter__")
    assert hasattr(gen, "__next__")
    assert gen is not suplier.training_set()


def test_test_set_returns_generator(monkeypatch):
    monkeypatch.setattr(ds_module, "partitioner", make_partition_factory([1]))
    suplier = DataSuplier([1])

    gen = suplier.test_set()

    assert hasattr(gen, "__iter__")
    assert hasattr(gen, "__next__")


def test_training_set_yields_only_training_items(monkeypatch):
    monkeypatch.setattr(ds_module, "partitioner", make_partition_factory([0, 1, 0, 1]))
    dataset = ["a", "b", "c", "d"]
    suplier = DataSuplier(dataset)

    assert list(suplier.training_set()) == [(0, "a"), (2, "c")]
    assert suplier.training_indexes == [0, 2]
    assert suplier.test_indexes == [1, 3]


def test_test_set_yields_only_test_items(monkeypatch):
    monkeypatch.setattr(ds_module, "partitioner", make_partition_factory([0, 1, 0, 1]))
    dataset = ["a", "b", "c", "d"]
    suplier = DataSuplier(dataset)

    assert list(suplier.test_set()) == [(1, "b"), (3, "d")]
    assert suplier.training_indexes == [0, 2]
    assert suplier.test_indexes == [1, 3]


def test_training_and_test_sets_are_consistent_with_each_other(monkeypatch):
    monkeypatch.setattr(ds_module, "partitioner", make_partition_factory([0, 1, 1, 0, 0]))
    dataset = list("abcde")
    suplier = DataSuplier(dataset)

    training = list(suplier.training_set())
    test = list(suplier.test_set())

    assert training == [(0, "a"), (3, "d"), (4, "e")]
    assert test == [(1, "b"), (2, "c")]
    assert sorted([i for i, _ in training + test]) == list(range(len(dataset)))


def test_training_set_lazy_evaluation_until_iteration(monkeypatch):
    monkeypatch.setattr(ds_module, "partitioner", make_partition_factory([0, 1, 0]))
    suplier = DataSuplier(["a", "b", "c"])

    gen = suplier.training_set()
    assert suplier.index == 0
    assert suplier.training_indexes == []
    assert suplier.test_indexes == []

    first = next(gen)
    assert first == (0, "a")
    assert suplier.index == 1
    assert suplier.training_indexes == [0]
    assert suplier.test_indexes == []


def test_test_set_lazy_evaluation_until_iteration(monkeypatch):
    monkeypatch.setattr(ds_module, "partitioner", make_partition_factory([0, 1, 0]))
    suplier = DataSuplier(["a", "b", "c"])

    gen = suplier.test_set()
    assert suplier.index == 0

    first = next(gen)
    assert first == (1, "b")
    assert suplier.index == 2
    assert suplier.training_indexes == [0, 2]
    assert suplier.test_indexes == [1]


def test_generators_can_be_consumed_in_any_order(monkeypatch):
    monkeypatch.setattr(ds_module, "partitioner", make_partition_factory([0, 1, 0, 1]))
    suplier = DataSuplier(["a", "b", "c", "d"])

    test_items = list(suplier.test_set())
    training_items = list(suplier.training_set())

    assert test_items == [(1, "b"), (3, "d")]
    assert training_items == [(0, "a"), (2, "c")]


def test_multiple_calls_to_same_set_return_equivalent_generators(monkeypatch):
    monkeypatch.setattr(ds_module, "partitioner", make_partition_factory([0, 1, 0]))
    suplier = DataSuplier(["a", "b", "c"])

    gen1 = suplier.training_set()
    gen2 = suplier.training_set()

    assert list(gen1) == [(0, "a"), (2, "c")]
    assert list(gen2) == [(0, "a"), (2, "c")]


def test_empty_dataset_yields_no_items(monkeypatch):
    monkeypatch.setattr(ds_module, "partitioner", make_partition_factory([]))
    suplier = DataSuplier([])

    assert list(suplier.training_set()) == []
    assert list(suplier.test_set()) == []
    assert suplier.index == 0
    assert suplier.training_indexes == []
    assert suplier.test_indexes == []


def test_all_items_can_go_to_training(monkeypatch):
    monkeypatch.setattr(ds_module, "partitioner", make_partition_factory([0, 0, 0]))
    suplier = DataSuplier(["a", "b", "c"])

    assert list(suplier.training_set()) == [(0, "a"), (1, "b"), (2, "c")]
    assert list(suplier.test_set()) == []
    assert suplier.training_indexes == [0, 1, 2]
    assert suplier.test_indexes == []


def test_all_items_can_go_to_test(monkeypatch):
    monkeypatch.setattr(ds_module, "partitioner", make_partition_factory([1, 1, 1]))
    suplier = DataSuplier(["a", "b", "c"])

    assert list(suplier.training_set()) == []
    assert list(suplier.test_set()) == [(0, "a"), (1, "b"), (2, "c")]
    assert suplier.training_indexes == []
    assert suplier.test_indexes == [0, 1, 2]


def test_partitioning_function_is_forwarded_without_modification(monkeypatch):
    captured = {}

    def custom_factory(n_of_elements, probabilities, partitioning_function):
        captured["partitioning_function"] = partitioning_function
        return FakePartitioner([0])

    monkeypatch.setattr(ds_module, "partitioner", custom_factory)

    def my_pf():
        return "pf"

    DataSuplier(["a"], partitioning_function=my_pf)

    assert captured["partitioning_function"] is my_pf
