import pytest

import data_partitioner.data_suplier as data_suplier_module
from data_partitioner.data_suplier import DataSuplier


class DummyPartitioner:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def __next__(self):
        if self.calls >= len(self.outputs):
            raise AssertionError("Partitioner was consumed more times than expected")
        value = self.outputs[self.calls]
        self.calls += 1
        return value


def install_fake_partitioner(monkeypatch, outputs, recorder=None):
    """
    Patch data_partitioner.data_suplier.partitioner so that it returns a
    DummyPartitioner with the given outputs and optionally records constructor args.
    """
    def fake_partitioner(n_of_elements, weights, partitioning_function):
        if recorder is not None:
            recorder["n_of_elements"] = n_of_elements
            recorder["weights"] = weights
            recorder["partitioning_function"] = partitioning_function
        return DummyPartitioner(outputs)

    monkeypatch.setattr(data_suplier_module, "partitioner", fake_partitioner)


def test_init_sets_expected_attributes_and_calls_partitioner(monkeypatch):
    recorder = {}
    install_fake_partitioner(monkeypatch, outputs=[0, 1, 0], recorder=recorder)

    dataset = ["a", "b", "c"]
    ds = DataSuplier(dataset, training_percent=0.7, partitioning_function="pf")

    assert ds.dataset is dataset
    assert ds.training_percent == 0.7
    assert ds.n_of_elements == 3
    assert ds.test_indexes == []
    assert ds.training_indexes == []
    assert ds.index == 0
    assert recorder["n_of_elements"] == 3
    assert recorder["weights"] == [0.7, 0.30000000000000004]
    assert recorder["partitioning_function"] == "pf"


def test_training_set_returns_generator(monkeypatch):
    install_fake_partitioner(monkeypatch, outputs=[0, 0, 0])

    ds = DataSuplier([1, 2, 3])
    gen = ds.training_set()

    assert hasattr(gen, "__iter__")
    assert iter(gen) is gen


def test_test_set_returns_generator(monkeypatch):
    install_fake_partitioner(monkeypatch, outputs=[1, 1, 1])

    ds = DataSuplier([1, 2, 3])
    gen = ds.test_set()

    assert hasattr(gen, "__iter__")
    assert iter(gen) is gen


def test_training_set_yields_expected_items(monkeypatch):
    install_fake_partitioner(monkeypatch, outputs=[0, 1, 0, 1, 0])

    dataset = ["a", "b", "c", "d", "e"]
    ds = DataSuplier(dataset)

    assert list(ds.training_set()) == [(0, "a"), (2, "c"), (4, "e")]
    assert ds.training_indexes == [0, 2, 4]
    assert ds.test_indexes == [1, 3]


def test_test_set_yields_expected_items(monkeypatch):
    install_fake_partitioner(monkeypatch, outputs=[0, 1, 0, 1, 0])

    dataset = ["a", "b", "c", "d", "e"]
    ds = DataSuplier(dataset)

    assert list(ds.test_set()) == [(1, "b"), (3, "d")]
    assert ds.training_indexes == [0, 2, 4]
    assert ds.test_indexes == [1, 3]


def test_training_and_test_sets_can_be_consumed_in_one_order(monkeypatch):
    install_fake_partitioner(monkeypatch, outputs=[0, 1, 0, 1, 0])

    dataset = ["a", "b", "c", "d", "e"]
    ds = DataSuplier(dataset)

    assert list(ds.training_set()) == [(0, "a"), (2, "c"), (4, "e")]
    assert list(ds.test_set()) == [(1, "b"), (3, "d")]


def test_training_and_test_sets_can_be_consumed_in_reverse_order(monkeypatch):
    install_fake_partitioner(monkeypatch, outputs=[0, 1, 0, 1, 0])

    dataset = ["a", "b", "c", "d", "e"]
    ds = DataSuplier(dataset)

    assert list(ds.test_set()) == [(1, "b"), (3, "d")]
    assert list(ds.training_set()) == [(0, "a"), (2, "c"), (4, "e")]


def test_empty_dataset_yields_no_items(monkeypatch):
    install_fake_partitioner(monkeypatch, outputs=[])

    ds = DataSuplier([])
    assert list(ds.training_set()) == []
    assert list(ds.test_set()) == []
    assert ds.index == 0


def test_all_items_assigned_to_training(monkeypatch):
    install_fake_partitioner(monkeypatch, outputs=[0, 0, 0, 0])

    dataset = ["a", "b", "c", "d"]
    ds = DataSuplier(dataset)

    assert list(ds.training_set()) == [(0, "a"), (1, "b"), (2, "c"), (3, "d")]
    assert list(ds.test_set()) == []
    assert ds.training_indexes == [0, 1, 2, 3]
    assert ds.test_indexes == []


def test_all_items_assigned_to_test(monkeypatch):
    install_fake_partitioner(monkeypatch, outputs=[1, 1, 1, 1])

    dataset = ["a", "b", "c", "d"]
    ds = DataSuplier(dataset)

    assert list(ds.training_set()) == []
    assert list(ds.test_set()) == [(0, "a"), (1, "b"), (2, "c"), (3, "d")]
    assert ds.training_indexes == []
    assert ds.test_indexes == [0, 1, 2, 3]


def test_generators_are_repeatable_after_partitioning_is_complete(monkeypatch):
    install_fake_partitioner(monkeypatch, outputs=[0, 1, 0, 1])

    dataset = ["a", "b", "c", "d"]
    ds = DataSuplier(dataset)

    first = list(ds.training_set())
    second = list(ds.training_set())

    assert first == [(0, "a"), (2, "c")]
    assert second == [(0, "a"), (2, "c")]


def test_interleaved_iteration_preserves_order(monkeypatch):
    install_fake_partitioner(monkeypatch, outputs=[0, 1, 0, 1, 0, 1])

    dataset = ["a", "b", "c", "d", "e", "f"]
    ds = DataSuplier(dataset)

    train_gen = ds.training_set()
    test_gen = ds.test_set()

    assert next(train_gen) == (0, "a")
    assert next(test_gen) == (1, "b")
    assert next(train_gen) == (2, "c")
    assert next(test_gen) == (3, "d")
    assert next(train_gen) == (4, "e")
    assert next(test_gen) == (5, "f")

    with pytest.raises(StopIteration):
        next(train_gen)
    with pytest.raises(StopIteration):
        next(test_gen)


def test_custom_training_percent_is_respected_in_partition_weights(monkeypatch):
    recorder = {}
    install_fake_partitioner(monkeypatch, outputs=[0], recorder=recorder)

    DataSuplier([1], training_percent=0.25, partitioning_function="custom")

    assert recorder["weights"] == [0.25, 0.75]
    assert recorder["partitioning_function"] == "custom"
