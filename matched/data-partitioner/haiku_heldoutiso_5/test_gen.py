"""Unit tests for data_partitioner.data_suplier.DataSuplier.

The partitioner is replaced with a scripted fake in most tests so that the
behaviour of DataSuplier can be checked independently of the partitioning
algorithm. A few tests use the real partitioner to check the overall
guarantees (disjoint, complete, consistent splits).
"""

import inspect

import pytest

from data_partitioner import data_suplier
from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import pseudorandom_function


class FakePartitioner(object):
    """Stands in for partitioner: yields a scripted sequence of 0/1 values."""

    def __init__(self, sequence, record):
        self._sequence = list(sequence)
        self.next_calls = 0
        self._record = record

    def __call__(self, n, probabilities, partitioning_function):
        # Used when the class itself is called as the partitioner factory.
        self._record.append((n, probabilities, partitioning_function))
        return self

    def __next__(self):
        self.next_calls += 1
        if not self._sequence:
            raise StopIteration
        return self._sequence.pop(0)


def use_scripted_partitioner(monkeypatch, sequence):
    """Patch data_suplier.partitioner so that it yields `sequence`.

    Returns (constructor_calls, instances). constructor_calls collects the
    (n, probabilities, partitioning_function) arguments passed on creation,
    and instances collects every FakePartitioner created.
    """
    constructor_calls = []
    instances = []

    def factory(n, probabilities, partitioning_function):
        constructor_calls.append((n, probabilities, partitioning_function))
        fake = FakePartitioner(sequence, [])
        instances.append(fake)
        return fake

    monkeypatch.setattr(data_suplier, "partitioner", factory)
    return constructor_calls, instances


DATASET = ["a", "b", "c", "d", "e"]
# Positions 0, 2 -> training; positions 1, 3, 4 -> test.
SEQUENCE = [0, 1, 0, 1, 1]


def test_init_stores_dataset_and_training_percent(monkeypatch):
    use_scripted_partitioner(monkeypatch, SEQUENCE)
    ds = DataSuplier(DATASET, training_percent=0.6)
    assert ds.dataset is DATASET
    assert ds.training_percent == 0.6


def test_init_records_number_of_elements(monkeypatch):
    use_scripted_partitioner(monkeypatch, SEQUENCE)
    ds = DataSuplier(DATASET)
    assert ds.n_of_elements == len(DATASET)


def test_init_builds_partitioner_with_size_and_split(monkeypatch):
    calls, _ = use_scripted_partitioner(monkeypatch, SEQUENCE)
    DataSuplier(DATASET, training_percent=0.7)
    assert len(calls) == 1
    n, probabilities, _ = calls[0]
    assert n == len(DATASET)
    assert probabilities[0] == pytest.approx(0.7)
    assert probabilities[1] == pytest.approx(0.3)


def test_init_passes_custom_partitioning_function(monkeypatch):
    calls, _ = use_scripted_partitioner(monkeypatch, SEQUENCE)

    def custom(*args, **kwargs):
        return 0

    DataSuplier(DATASET, partitioning_function=custom)
    assert calls[0][2] is custom


def test_default_training_percent_is_point_eight():
    param = inspect.signature(DataSuplier.__init__).parameters["training_percent"]
    assert param.default == 0.8


def test_default_partitioning_function_is_pseudorandom_function():
    param = inspect.signature(DataSuplier.__init__).parameters["partitioning_function"]
    assert param.default is pseudorandom_function


def test_init_starts_with_empty_index_lists_and_zero_index(monkeypatch):
    use_scripted_partitioner(monkeypatch, SEQUENCE)
    ds = DataSuplier(DATASET)
    assert ds.training_indexes == []
    assert ds.test_indexes == []
    assert ds.index == 0


def test_training_set_returns_generator(monkeypatch):
    use_scripted_partitioner(monkeypatch, SEQUENCE)
    ds = DataSuplier(DATASET)
    assert inspect.isgenerator(ds.training_set())


def test_test_set_returns_generator(monkeypatch):
    use_scripted_partitioner(monkeypatch, SEQUENCE)
    ds = DataSuplier(DATASET)
    assert inspect.isgenerator(ds.test_set())


def test_training_set_yields_index_and_item_pairs(monkeypatch):
    use_scripted_partitioner(monkeypatch, SEQUENCE)
    ds = DataSuplier(DATASET)
    assert list(ds.training_set()) == [(0, "a"), (2, "c")]


def test_test_set_yields_index_and_item_pairs(monkeypatch):
    use_scripted_partitioner(monkeypatch, SEQUENCE)
    ds = DataSuplier(DATASET)
    assert list(ds.test_set()) == [(1, "b"), (3, "d"), (4, "e")]


def test_sets_preserve_ascending_index_order(monkeypatch):
    use_scripted_partitioner(monkeypatch, [1, 0, 1, 0, 0, 1, 0])
    ds = DataSuplier(list(range(7)))
    train_indexes = [i for i, _ in ds.training_set()]
    test_indexes = [i for i, _ in ds.test_set()]
    assert train_indexes == sorted(train_indexes)
    assert test_indexes == sorted(test_indexes)


def test_sets_are_disjoint_and_cover_dataset_with_fake(monkeypatch):
    use_scripted_partitioner(monkeypatch, [0, 1, 1, 0, 0, 1, 0, 1])
    ds = DataSuplier(list(range(8)))
    train_indexes = [i for i, _ in ds.training_set()]
    test_indexes = [i for i, _ in ds.test_set()]
    assert set(train_indexes).isdisjoint(test_indexes)
    assert sorted(train_indexes + test_indexes) == list(range(8))


def test_each_item_matches_its_index_in_dataset(monkeypatch):
    use_scripted_partitioner(monkeypatch, SEQUENCE)
    ds = DataSuplier(DATASET)
    for index, item in list(ds.training_set()) + list(ds.test_set()):
        assert item == DATASET[index]


def test_empty_dataset_yields_nothing(monkeypatch):
    use_scripted_partitioner(monkeypatch, [])
    ds = DataSuplier([])
    assert list(ds.training_set()) == []
    assert list(ds.test_set()) == []


def test_all_training_partition_leaves_test_set_empty(monkeypatch):
    use_scripted_partitioner(monkeypatch, [0, 0, 0])
    ds = DataSuplier(["x", "y", "z"])
    assert list(ds.training_set()) == [(0, "x"), (1, "y"), (2, "z")]
    assert list(ds.test_set()) == []


def test_all_test_partition_leaves_training_set_empty(monkeypatch):
    use_scripted_partitioner(monkeypatch, [1, 1, 1])
    ds = DataSuplier(["x", "y", "z"])
    assert list(ds.training_set()) == []
    assert list(ds.test_set()) == [(0, "x"), (1, "y"), (2, "z")]


def test_test_set_consumed_first_still_yields_full_training_set(monkeypatch):
    use_scripted_partitioner(monkeypatch, SEQUENCE)
    ds = DataSuplier(DATASET)
    assert list(ds.test_set()) == [(1, "b"), (3, "d"), (4, "e")]
    assert list(ds.training_set()) == [(0, "a"), (2, "c")]


def test_iterating_a_set_twice_yields_same_results(monkeypatch):
    use_scripted_partitioner(monkeypatch, SEQUENCE)
    ds = DataSuplier(DATASET)
    first = list(ds.training_set())
    second = list(ds.training_set())
    assert first == second == [(0, "a"), (2, "c")]


def test_training_set_is_lazy_and_advances_partitioner_on_demand(monkeypatch):
    _, instances = use_scripted_partitioner(monkeypatch, SEQUENCE)
    ds = DataSuplier(DATASET)
    gen = ds.training_set()
    assert next(gen) == (0, "a")
    assert ds.index == 1
    assert instances[0].next_calls == 1


def test_partitioner_advanced_exactly_once_per_element(monkeypatch):
    _, instances = use_scripted_partitioner(monkeypatch, SEQUENCE)
    ds = DataSuplier(DATASET)
    list(ds.training_set())
    list(ds.test_set())
    assert instances[0].next_calls == len(DATASET)
    assert ds.index == len(DATASET)


def test_index_lists_are_populated_as_partitioner_advances(monkeypatch):
    use_scripted_partitioner(monkeypatch, SEQUENCE)
    ds = DataSuplier(DATASET)
    list(ds.training_set())
    assert ds.training_indexes == [0, 2]
    assert ds.test_indexes == [1, 3, 4]


def test_real_partitioner_gives_disjoint_complete_split():
    dataset = list(range(100))
    ds = DataSuplier(dataset)
    train_indexes = [i for i, _ in ds.training_set()]
    test_indexes = [i for i, _ in ds.test_set()]
    assert set(train_indexes).isdisjoint(test_indexes)
    assert sorted(train_indexes + test_indexes) == list(range(100))


def test_real_partitioner_split_is_consistent_across_instances():
    dataset = list(range(50))
    first = DataSuplier(dataset)
    second = DataSuplier(dataset)
    assert [i for i, _ in first.training_set()] == [i for i, _ in second.training_set()]
    assert [i for i, _ in first.test_set()] == [i for i, _ in second.test_set()]
