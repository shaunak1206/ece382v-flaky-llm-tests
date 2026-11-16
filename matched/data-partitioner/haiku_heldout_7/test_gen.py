"""Unit tests for data_partitioner.data_suplier.DataSuplier."""

import data_partitioner.data_suplier as data_suplier_module
from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import pseudorandom_function


# Assignment codes produced by the partitioner: 0 -> training, anything else -> test.


def install_fake_partitioner(monkeypatch, assignments):
    """Replace the partitioner used by data_suplier with a deterministic fake.

    Returns a list that records the arguments each constructed fake was given.
    """
    calls = []

    class FakePartitioner(object):
        def __init__(self, n, proportions, partitioning_function):
            calls.append((n, list(proportions), partitioning_function))
            self._iter = iter(list(assignments))

        def __next__(self):
            return next(self._iter)

    monkeypatch.setattr(data_suplier_module, "partitioner", FakePartitioner)
    return calls


def test_init_stores_dataset_and_training_percent(monkeypatch):
    install_fake_partitioner(monkeypatch, [0, 1, 0])
    dataset = ["a", "b", "c"]
    suplier = DataSuplier(dataset, training_percent=0.5)

    assert suplier.dataset is dataset
    assert suplier.training_percent == 0.5
    assert suplier.n_of_elements == 3


def test_init_default_training_percent_is_point_eight(monkeypatch):
    install_fake_partitioner(monkeypatch, [])
    suplier = DataSuplier([1, 2, 3])

    assert suplier.training_percent == 0.8


def test_init_passes_size_and_proportions_to_partitioner(monkeypatch):
    calls = install_fake_partitioner(monkeypatch, [0, 1, 0, 0])
    DataSuplier([10, 20, 30, 40], training_percent=0.75)

    assert len(calls) == 1
    n, proportions, _ = calls[0]
    assert n == 4
    assert proportions == [0.75, 1 - 0.75]


def test_init_passes_custom_partitioning_function_to_partitioner(monkeypatch):
    calls = install_fake_partitioner(monkeypatch, [0])

    def custom(*args, **kwargs):
        return 0

    DataSuplier([1], partitioning_function=custom)

    assert calls[0][2] is custom


def test_init_default_partitioning_function_is_pseudorandom(monkeypatch):
    calls = install_fake_partitioner(monkeypatch, [0])
    DataSuplier([1])

    assert calls[0][2] is pseudorandom_function


def test_init_starts_with_empty_state(monkeypatch):
    install_fake_partitioner(monkeypatch, [0, 1])
    suplier = DataSuplier(["x", "y"])

    assert suplier.training_indexes == []
    assert suplier.test_indexes == []
    assert suplier.index == 0


def test_init_empty_dataset_has_zero_elements(monkeypatch):
    calls = install_fake_partitioner(monkeypatch, [])
    suplier = DataSuplier([])

    assert suplier.n_of_elements == 0
    assert calls[0][0] == 0


def test_training_set_yields_training_items_in_order(monkeypatch):
    install_fake_partitioner(monkeypatch, [0, 1, 0, 1])
    suplier = DataSuplier(["a", "b", "c", "d"])

    assert list(suplier.training_set()) == [(0, "a"), (2, "c")]


def test_test_set_yields_test_items_in_order(monkeypatch):
    install_fake_partitioner(monkeypatch, [0, 1, 0, 1])
    suplier = DataSuplier(["a", "b", "c", "d"])

    assert list(suplier.test_set()) == [(1, "b"), (3, "d")]


def test_training_and_test_sets_partition_all_indexes(monkeypatch):
    install_fake_partitioner(monkeypatch, [1, 0, 0, 1, 0, 1, 1])
    dataset = list(range(7))
    suplier = DataSuplier(dataset)

    train_indexes = [i for i, _ in suplier.training_set()]
    test_indexes = [i for i, _ in suplier.test_set()]

    assert sorted(train_indexes + test_indexes) == list(range(7))
    assert set(train_indexes).isdisjoint(test_indexes)


def test_yielded_values_match_dataset_entries(monkeypatch):
    install_fake_partitioner(monkeypatch, [0, 1, 1, 0])
    dataset = ["zero", "one", "two", "three"]
    suplier = DataSuplier(dataset)

    for index, value in suplier.training_set():
        assert value == dataset[index]
    for index, value in suplier.test_set():
        assert value == dataset[index]


def test_all_training_assignments_leave_test_set_empty(monkeypatch):
    install_fake_partitioner(monkeypatch, [0, 0, 0])
    suplier = DataSuplier(["a", "b", "c"])

    assert list(suplier.training_set()) == [(0, "a"), (1, "b"), (2, "c")]
    assert list(suplier.test_set()) == []


def test_all_test_assignments_leave_training_set_empty(monkeypatch):
    install_fake_partitioner(monkeypatch, [1, 1, 1])
    suplier = DataSuplier(["a", "b", "c"])

    assert list(suplier.training_set()) == []
    assert list(suplier.test_set()) == [(0, "a"), (1, "b"), (2, "c")]


def test_empty_dataset_yields_nothing_from_either_set(monkeypatch):
    install_fake_partitioner(monkeypatch, [])
    suplier = DataSuplier([])

    assert list(suplier.training_set()) == []
    assert list(suplier.test_set()) == []


def test_sets_are_lazy_until_iterated(monkeypatch):
    install_fake_partitioner(monkeypatch, [0, 1])
    suplier = DataSuplier(["a", "b"])

    generator = suplier.training_set()

    assert suplier.index == 0
    assert suplier.training_indexes == []
    assert suplier.test_indexes == []

    assert next(generator) == (0, "a")
    assert suplier.index >= 1


def test_repeated_iteration_is_consistent(monkeypatch):
    install_fake_partitioner(monkeypatch, [0, 1, 0, 1, 0])
    suplier = DataSuplier(list("abcde"))

    first = list(suplier.training_set())
    second = list(suplier.training_set())

    assert first == second == [(0, "a"), (2, "c"), (4, "e")]


def test_test_set_first_then_training_set_yields_full_results(monkeypatch):
    install_fake_partitioner(monkeypatch, [0, 1, 0, 1])
    suplier = DataSuplier(["a", "b", "c", "d"])

    assert list(suplier.test_set()) == [(1, "b"), (3, "d")]
    assert list(suplier.training_set()) == [(0, "a"), (2, "c")]


def test_partial_consumption_then_other_set_still_correct(monkeypatch):
    install_fake_partitioner(monkeypatch, [0, 1, 1, 0])
    suplier = DataSuplier(["a", "b", "c", "d"])

    training = suplier.training_set()
    assert next(training) == (0, "a")

    assert list(suplier.test_set()) == [(1, "b"), (2, "c")]
    assert list(training) == [(3, "d")]


def test_training_set_returns_generator(monkeypatch):
    install_fake_partitioner(monkeypatch, [0])
    suplier = DataSuplier(["a"])

    generator = suplier.training_set()

    assert hasattr(generator, "__next__")
    assert iter(generator) is generator


def test_real_partitioner_splits_dataset_disjointly():
    dataset = list(range(200))
    suplier = DataSuplier(dataset)

    train_indexes = [i for i, _ in suplier.training_set()]
    test_indexes = [i for i, _ in suplier.test_set()]

    assert sorted(train_indexes + test_indexes) == list(range(200))
    assert set(train_indexes).isdisjoint(test_indexes)
    assert all(value == index for index, value in suplier.training_set())


def test_real_partitioner_is_consistent_across_instances():
    dataset = list(range(100))

    first = DataSuplier(dataset)
    second = DataSuplier(dataset)

    assert list(first.training_set()) == list(second.training_set())
    assert list(first.test_set()) == list(second.test_set())
