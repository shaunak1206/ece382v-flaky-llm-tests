import pytest

import data_partitioner.data_suplier as ds_module
from data_partitioner.data_suplier import DataSuplier


def _patch_partitioner(monkeypatch, pattern, calls=None):
    """Replace the partitioner factory with one that yields a fixed pattern."""
    def fake(n, percents, function):
        if calls is not None:
            calls.append((n, percents, function))
        return iter(pattern)
    monkeypatch.setattr(ds_module, "partitioner", fake)


def test_init_stores_attributes_with_defaults():
    data = list(range(10))
    supplier = DataSuplier(data)
    assert supplier.dataset is data
    assert supplier.training_percent == 0.8
    assert supplier.n_of_elements == 10
    assert supplier.test_indexes == []
    assert supplier.training_indexes == []
    assert supplier.index == 0


def test_init_passes_arguments_to_partitioner(monkeypatch):
    calls = []
    _patch_partitioner(monkeypatch, [], calls)
    func = object()
    DataSuplier([1, 2, 3, 4], training_percent=0.7, partitioning_function=func)
    assert len(calls) == 1
    n, percents, function = calls[0]
    assert n == 4
    assert percents[0] == pytest.approx(0.7)
    assert percents[1] == pytest.approx(0.3)
    assert function is func


def test_training_set_follows_partition_pattern(monkeypatch):
    _patch_partitioner(monkeypatch, [0, 1, 0, 0, 1])
    data = ["a", "b", "c", "d", "e"]
    supplier = DataSuplier(data)
    assert list(supplier.training_set()) == [(0, "a"), (2, "c"), (3, "d")]


def test_test_set_follows_partition_pattern(monkeypatch):
    _patch_partitioner(monkeypatch, [0, 1, 0, 0, 1])
    data = ["a", "b", "c", "d", "e"]
    supplier = DataSuplier(data)
    assert list(supplier.test_set()) == [(1, "b"), (4, "e")]


def test_generators_are_lazy(monkeypatch):
    _patch_partitioner(monkeypatch, [0, 0, 1, 0])
    supplier = DataSuplier([10, 20, 30, 40])
    gen = supplier.test_set()
    assert supplier.index == 0
    assert next(gen) == (2, 30)
    assert supplier.index == 3
    assert supplier.training_indexes == [0, 1]
    assert supplier.test_indexes == [2]


def test_interleaved_consumption_yields_complete_sets(monkeypatch):
    _patch_partitioner(monkeypatch, [0, 1, 1, 0, 0, 1])
    data = list("abcdef")
    supplier = DataSuplier(data)
    train_gen = supplier.training_set()
    test_gen = supplier.test_set()
    train, test = [], []
    for _ in range(3):
        train.append(next(train_gen))
        test.append(next(test_gen))
    assert train == [(0, "a"), (3, "d"), (4, "e")]
    assert test == [(1, "b"), (2, "c"), (5, "f")]
    assert list(train_gen) == []
    assert list(test_gen) == []


def test_test_set_consumed_before_training_set(monkeypatch):
    _patch_partitioner(monkeypatch, [1, 0, 0, 1])
    supplier = DataSuplier([1, 2, 3, 4])
    assert list(supplier.test_set()) == [(0, 1), (3, 4)]
    assert list(supplier.training_set()) == [(1, 2), (2, 3)]


def test_sets_can_be_requested_again_after_exhaustion(monkeypatch):
    _patch_partitioner(monkeypatch, [0, 1, 0])
    supplier = DataSuplier(["x", "y", "z"])
    first = list(supplier.training_set())
    second = list(supplier.training_set())
    assert first == second == [(0, "x"), (2, "z")]
    assert list(supplier.test_set()) == [(1, "y")]
    assert list(supplier.test_set()) == [(1, "y")]


def test_all_training_pattern_gives_empty_test_set(monkeypatch):
    _patch_partitioner(monkeypatch, [0, 0, 0])
    supplier = DataSuplier([1, 2, 3])
    assert list(supplier.test_set()) == []
    assert list(supplier.training_set()) == [(0, 1), (1, 2), (2, 3)]


def test_all_test_pattern_gives_empty_training_set(monkeypatch):
    _patch_partitioner(monkeypatch, [1, 1, 1])
    supplier = DataSuplier([1, 2, 3])
    assert list(supplier.training_set()) == []
    assert list(supplier.test_set()) == [(0, 1), (1, 2), (2, 3)]


def test_empty_dataset_yields_nothing(monkeypatch):
    _patch_partitioner(monkeypatch, [])
    supplier = DataSuplier([])
    assert supplier.n_of_elements == 0
    assert list(supplier.training_set()) == []
    assert list(supplier.test_set()) == []


def test_works_with_non_list_sequences(monkeypatch):
    _patch_partitioner(monkeypatch, [0, 1, 0])
    supplier = DataSuplier(("a", "b", "c"))
    assert list(supplier.training_set()) == [(0, "a"), (2, "c")]
    assert list(supplier.test_set()) == [(1, "b")]


def test_default_partitioner_covers_dataset_without_overlap():
    data = list(range(200))
    supplier = DataSuplier(data)
    train = list(supplier.training_set())
    test = list(supplier.test_set())
    train_idx = [i for i, _ in train]
    test_idx = [i for i, _ in test]
    assert set(train_idx).isdisjoint(test_idx)
    assert sorted(train_idx + test_idx) == data
    assert len(train) + len(test) == 200


def test_default_partitioner_yields_index_element_pairs():
    data = ["item%d" % i for i in range(50)]
    supplier = DataSuplier(data)
    for index, element in list(supplier.training_set()) + list(supplier.test_set()):
        assert element == data[index]


def test_default_partitioner_roughly_respects_training_percent():
    supplier = DataSuplier(list(range(1000)), training_percent=0.8)
    train = list(supplier.training_set())
    fraction = len(train) / 1000.0
    assert 0.6 < fraction < 0.95


def test_repeated_iteration_with_default_partitioner_is_consistent():
    supplier = DataSuplier(list(range(100)))
    assert list(supplier.training_set()) == list(supplier.training_set())
    assert list(supplier.test_set()) == list(supplier.test_set())


def test_indexes_are_in_increasing_order(monkeypatch):
    _patch_partitioner(monkeypatch, [1, 0, 0, 1, 0, 1, 1, 0])
    supplier = DataSuplier(list(range(8)))
    train_idx = [i for i, _ in supplier.training_set()]
    test_idx = [i for i, _ in supplier.test_set()]
    assert train_idx == sorted(train_idx) == [1, 2, 4, 7]
    assert test_idx == sorted(test_idx) == [0, 3, 5, 6]
