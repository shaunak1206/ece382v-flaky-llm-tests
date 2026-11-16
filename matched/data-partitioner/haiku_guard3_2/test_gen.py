import random

import pytest

import data_partitioner.data_suplier as suplier_module
from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import pseudorandom_function


class FakePartitioner(object):
    """Deterministic stand-in for the partitioner: yields a fixed 0/1 sequence."""

    def __init__(self, sequence):
        self._values = list(sequence)
        self._pos = 0
        self.calls = 0

    def __next__(self):
        value = self._values[self._pos]
        self._pos += 1
        self.calls += 1
        return value


def install_fake(monkeypatch, sequence):
    """Replace the module-level `partitioner` name with a recording factory.

    Returns a list that receives one record per constructed partitioner.
    monkeypatch restores the original attribute after the test, even on failure.
    """
    created = []

    def factory(n, probabilities, function):
        fake = FakePartitioner(sequence)
        created.append(
            {"n": n, "probabilities": probabilities, "function": function, "fake": fake}
        )
        return fake

    monkeypatch.setattr(suplier_module, "partitioner", factory)
    return created


DATASET = ["a", "b", "c", "d", "e"]


def test_init_stores_dataset_and_training_percent(monkeypatch):
    install_fake(monkeypatch, [0, 1, 0, 1, 0])
    suplier = DataSuplier(DATASET, training_percent=0.6)
    assert suplier.dataset is DATASET
    assert suplier.training_percent == 0.6
    assert suplier.n_of_elements == len(DATASET)


def test_init_passes_size_and_probabilities_to_partitioner(monkeypatch):
    created = install_fake(monkeypatch, [0, 1, 0, 1, 0])
    DataSuplier(DATASET, training_percent=0.6)
    assert len(created) == 1
    assert created[0]["n"] == 5
    assert created[0]["probabilities"] == pytest.approx([0.6, 0.4])


def test_init_default_training_percent_is_0_8(monkeypatch):
    created = install_fake(monkeypatch, [0, 1, 0, 1, 0])
    DataSuplier(DATASET)
    assert created[0]["probabilities"] == pytest.approx([0.8, 0.2])


def test_init_default_partitioning_function_is_pseudorandom_function(monkeypatch):
    created = install_fake(monkeypatch, [0, 1, 0, 1, 0])
    DataSuplier(DATASET)
    assert created[0]["function"] is pseudorandom_function


def test_init_custom_partitioning_function_is_forwarded(monkeypatch):
    created = install_fake(monkeypatch, [0, 1, 0, 1, 0])

    def custom(*args, **kwargs):
        return 0

    DataSuplier(DATASET, partitioning_function=custom)
    assert created[0]["function"] is custom


def test_init_starts_with_empty_state(monkeypatch):
    fake_created = install_fake(monkeypatch, [0, 1, 0, 1, 0])
    suplier = DataSuplier(DATASET)
    assert suplier.training_indexes == []
    assert suplier.test_indexes == []
    assert suplier.index == 0
    assert fake_created[0]["fake"].calls == 0


def test_training_set_yields_index_and_value_pairs(monkeypatch):
    install_fake(monkeypatch, [0, 1, 0, 0, 1])
    suplier = DataSuplier(DATASET)
    assert list(suplier.training_set()) == [(0, "a"), (2, "c"), (3, "d")]


def test_test_set_yields_index_and_value_pairs(monkeypatch):
    install_fake(monkeypatch, [0, 1, 0, 0, 1])
    suplier = DataSuplier(DATASET)
    assert list(suplier.test_set()) == [(1, "b"), (4, "e")]


def test_training_and_test_sets_partition_the_dataset(monkeypatch):
    install_fake(monkeypatch, [1, 0, 0, 1, 0])
    suplier = DataSuplier(DATASET)
    training = list(suplier.training_set())
    test = list(suplier.test_set())
    training_idx = [i for i, _ in training]
    test_idx = [i for i, _ in test]
    assert sorted(training_idx + test_idx) == list(range(len(DATASET)))
    assert set(training_idx).isdisjoint(test_idx)
    assert all(DATASET[i] == v for i, v in training + test)


def test_generator_is_lazy_and_does_not_consume_partitioner(monkeypatch):
    created = install_fake(monkeypatch, [0, 1, 0, 1, 0])
    suplier = DataSuplier(DATASET)
    gen = suplier.training_set()
    assert created[0]["fake"].calls == 0
    assert suplier.index == 0
    next(gen)
    assert suplier.index >= 1


def test_first_training_element_requires_only_needed_partitioner_calls(monkeypatch):
    created = install_fake(monkeypatch, [1, 1, 0, 1, 0])
    suplier = DataSuplier(DATASET)
    gen = suplier.training_set()
    assert next(gen) == (2, "c")
    assert suplier.index == 3
    assert created[0]["fake"].calls == 3


def test_partitioner_called_exactly_once_per_element(monkeypatch):
    created = install_fake(monkeypatch, [0, 1, 0, 1, 0])
    suplier = DataSuplier(DATASET)
    list(suplier.training_set())
    list(suplier.test_set())
    assert created[0]["fake"].calls == len(DATASET)
    assert suplier.index == len(DATASET)


def test_all_training_gives_empty_test_set(monkeypatch):
    install_fake(monkeypatch, [0, 0, 0, 0, 0])
    suplier = DataSuplier(DATASET)
    assert list(suplier.test_set()) == []
    assert list(suplier.training_set()) == [(i, DATASET[i]) for i in range(5)]


def test_all_test_gives_empty_training_set(monkeypatch):
    install_fake(monkeypatch, [1, 1, 1, 1, 1])
    suplier = DataSuplier(DATASET)
    assert list(suplier.training_set()) == []
    assert list(suplier.test_set()) == [(i, DATASET[i]) for i in range(5)]


def test_empty_dataset_yields_nothing(monkeypatch):
    install_fake(monkeypatch, [])
    suplier = DataSuplier([])
    assert suplier.n_of_elements == 0
    assert list(suplier.training_set()) == []
    assert list(suplier.test_set()) == []


def test_repeated_iteration_yields_same_results(monkeypatch):
    install_fake(monkeypatch, [0, 1, 0, 1, 0])
    suplier = DataSuplier(DATASET)
    first = list(suplier.training_set())
    second = list(suplier.training_set())
    assert first == second == [(0, "a"), (2, "c"), (4, "e")]


def test_interleaved_iteration_yields_consistent_results(monkeypatch):
    install_fake(monkeypatch, [0, 1, 0, 1])
    suplier = DataSuplier(["a", "b", "c", "d"])
    train_gen = suplier.training_set()
    test_gen = suplier.test_set()
    got_train = [next(train_gen)]
    got_test = [next(test_gen)]
    got_train += list(train_gen)
    got_test += list(test_gen)
    assert got_train == [(0, "a"), (2, "c")]
    assert got_test == [(1, "b"), (3, "d")]


def test_real_partitioner_yields_disjoint_cover_of_dataset():
    random.seed(12345)
    dataset = list(range(50))
    suplier = DataSuplier(dataset, training_percent=0.8)
    training = list(suplier.training_set())
    test = list(suplier.test_set())
    training_idx = [i for i, _ in training]
    test_idx = [i for i, _ in test]
    assert sorted(training_idx + test_idx) == list(range(50))
    assert set(training_idx).isdisjoint(test_idx)
    assert all(dataset[i] == v for i, v in training + test)
    assert training_idx == sorted(training_idx)
    assert test_idx == sorted(test_idx)


def test_real_partitioner_empty_dataset():
    random.seed(12345)
    suplier = DataSuplier([])
    assert list(suplier.training_set()) == []
    assert list(suplier.test_set()) == []
