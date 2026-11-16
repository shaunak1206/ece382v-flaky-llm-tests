import pytest

import data_partitioner.data_suplier as ds_module
from data_partitioner.data_suplier import DataSuplier


class FakePartitionerFactory(object):
    """Replaces `partitioner` in the module; yields a scripted sequence of partition ids."""

    def __init__(self, pattern):
        self.pattern = list(pattern)
        self.calls = []

    def __call__(self, n, percents, function):
        self.calls.append((n, percents, function))
        return iter(self.pattern)


@pytest.fixture
def patch_partitioner(monkeypatch):
    def _patch(pattern):
        factory = FakePartitionerFactory(pattern)
        monkeypatch.setattr(ds_module, "partitioner", factory)
        return factory
    return _patch


def test_init_stores_attributes(patch_partitioner):
    patch_partitioner([0, 1, 0])
    data = ["a", "b", "c"]
    supplier = DataSuplier(data, training_percent=0.7)
    assert supplier.dataset is data
    assert supplier.training_percent == 0.7
    assert supplier.n_of_elements == 3
    assert supplier.index == 0
    assert supplier.training_indexes == []
    assert supplier.test_indexes == []


def test_init_passes_arguments_to_partitioner(patch_partitioner):
    factory = patch_partitioner([0, 0, 1, 0])

    def func(x):
        return x

    DataSuplier([1, 2, 3, 4], training_percent=0.75, partitioning_function=func)
    assert len(factory.calls) == 1
    n, percents, function = factory.calls[0]
    assert n == 4
    assert percents == [0.75, 1 - 0.75]
    assert function is func


def test_default_training_percent_and_function(patch_partitioner):
    factory = patch_partitioner([0, 1])
    supplier = DataSuplier([1, 2])
    assert supplier.training_percent == 0.8
    n, percents, function = factory.calls[0]
    assert percents == [0.8, 1 - 0.8]
    assert function is ds_module.pseudorandom_function


def test_training_set_yields_index_and_element(patch_partitioner):
    patch_partitioner([0, 1, 0, 1, 1])
    supplier = DataSuplier(["a", "b", "c", "d", "e"])
    assert list(supplier.training_set()) == [(0, "a"), (2, "c")]


def test_test_set_yields_index_and_element(patch_partitioner):
    patch_partitioner([0, 1, 0, 1, 1])
    supplier = DataSuplier(["a", "b", "c", "d", "e"])
    assert list(supplier.test_set()) == [(1, "b"), (3, "d"), (4, "e")]


def test_training_then_test_cover_dataset_without_overlap(patch_partitioner):
    patch_partitioner([0, 1, 1, 0, 0, 1, 0, 0])
    data = list(range(10, 18))
    supplier = DataSuplier(data)
    training = list(supplier.training_set())
    test = list(supplier.test_set())
    train_idx = [i for i, _ in training]
    test_idx = [i for i, _ in test]
    assert set(train_idx).isdisjoint(test_idx)
    assert sorted(train_idx + test_idx) == list(range(8))
    for i, value in training + test:
        assert value == data[i]


def test_test_then_training_order_gives_same_result(patch_partitioner):
    patch_partitioner([1, 0, 0, 1, 0])
    supplier = DataSuplier("vwxyz")
    test = list(supplier.test_set())
    training = list(supplier.training_set())
    assert test == [(0, "v"), (3, "y")]
    assert training == [(1, "w"), (2, "x"), (4, "z")]


def test_generators_are_lazy(patch_partitioner):
    patch_partitioner([1, 1, 0, 0, 1])
    supplier = DataSuplier([10, 20, 30, 40, 50])
    gen = supplier.training_set()
    assert supplier.index == 0
    assert next(gen) == (2, 30)
    assert supplier.index == 3
    assert supplier.test_indexes == [0, 1]
    assert supplier.training_indexes == [2]


def test_interleaved_generators(patch_partitioner):
    patch_partitioner([0, 1, 0, 1, 0, 1])
    supplier = DataSuplier(list("abcdef"))
    train_gen = supplier.training_set()
    test_gen = supplier.test_set()
    results_train = []
    results_test = []
    results_train.append(next(train_gen))
    results_test.append(next(test_gen))
    results_train.append(next(train_gen))
    results_test.extend(list(test_gen))
    results_train.extend(list(train_gen))
    assert results_train == [(0, "a"), (2, "c"), (4, "e")]
    assert results_test == [(1, "b"), (3, "d"), (5, "f")]


def test_set_can_be_iterated_repeatedly(patch_partitioner):
    patch_partitioner([0, 1, 0, 1])
    supplier = DataSuplier([1, 2, 3, 4])
    first = list(supplier.training_set())
    second = list(supplier.training_set())
    assert first == second == [(0, 1), (2, 3)]
    assert list(supplier.test_set()) == list(supplier.test_set())


def test_all_training(patch_partitioner):
    patch_partitioner([0, 0, 0])
    supplier = DataSuplier([1, 2, 3])
    assert list(supplier.test_set()) == []
    assert list(supplier.training_set()) == [(0, 1), (1, 2), (2, 3)]


def test_all_test(patch_partitioner):
    patch_partitioner([1, 1, 1])
    supplier = DataSuplier([1, 2, 3])
    assert list(supplier.training_set()) == []
    assert list(supplier.test_set()) == [(0, 1), (1, 2), (2, 3)]


def test_empty_dataset_yields_nothing(patch_partitioner):
    patch_partitioner([])
    supplier = DataSuplier([])
    assert supplier.n_of_elements == 0
    assert list(supplier.training_set()) == []
    assert list(supplier.test_set()) == []


def test_nonzero_partition_ids_count_as_test(patch_partitioner):
    patch_partitioner([0, 2, 1])
    supplier = DataSuplier(["x", "y", "z"])
    assert list(supplier.test_set()) == [(1, "y"), (2, "z")]
    assert list(supplier.training_set()) == [(0, "x")]


def test_index_reaches_dataset_size_after_full_iteration(patch_partitioner):
    patch_partitioner([0, 1, 0, 1])
    supplier = DataSuplier([1, 2, 3, 4])
    list(supplier.training_set())
    assert supplier.index == 4
    assert len(supplier.training_indexes) + len(supplier.test_indexes) == 4


def test_real_partitioner_partitions_dataset_completely():
    data = list(range(200))
    supplier = DataSuplier(data)
    training = list(supplier.training_set())
    test = list(supplier.test_set())
    train_idx = [i for i, _ in training]
    test_idx = [i for i, _ in test]
    assert set(train_idx).isdisjoint(test_idx)
    assert sorted(train_idx + test_idx) == data
    assert all(data[i] == v for i, v in training + test)


def test_real_partitioner_indexes_are_increasing():
    supplier = DataSuplier(list(range(100)), training_percent=0.5)
    train_idx = [i for i, _ in supplier.training_set()]
    test_idx = [i for i, _ in supplier.test_set()]
    assert train_idx == sorted(train_idx)
    assert test_idx == sorted(test_idx)
