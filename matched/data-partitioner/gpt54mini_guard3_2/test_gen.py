import pytest

import data_partitioner.data_suplier as dsmod
from data_partitioner.data_suplier import DataSuplier


class FakePartitioner:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self._pos = 0

    def __next__(self):
        if self._pos >= len(self._outcomes):
            raise StopIteration
        value = self._outcomes[self._pos]
        self._pos += 1
        return value


def make_partition_factory(outcomes, recorder=None):
    def factory(n_of_elements, percentages, partitioning_function):
        if recorder is not None:
            recorder["n_of_elements"] = n_of_elements
            recorder["percentages"] = list(percentages)
            recorder["partitioning_function"] = partitioning_function
        return FakePartitioner(outcomes)

    return factory


def test_init_sets_basic_attributes_and_partitioner_arguments(monkeypatch):
    recorder = {}
    monkeypatch.setattr(dsmod, "partitioner", make_partition_factory([0, 1], recorder))
    dataset = ["a", "b"]
    supplied = DataSuplier(dataset, training_percent=0.7, partitioning_function=object())

    assert supplied.dataset is dataset
    assert supplied.training_percent == 0.7
    assert supplied.n_of_elements == 2
    assert supplied.test_indexes == []
    assert supplied.training_indexes == []
    assert supplied.index == 0
    assert recorder["n_of_elements"] == 2
    assert recorder["percentages"] == [0.7, 0.30000000000000004]
    assert recorder["partitioning_function"] is not None


def test_training_set_yields_only_training_items_in_index_order(monkeypatch):
    monkeypatch.setattr(dsmod, "partitioner", make_partition_factory([0, 1, 0, 1]))
    dataset = ["a", "b", "c", "d"]
    supplied = DataSuplier(dataset, training_percent=0.5, partitioning_function=object())

    assert list(supplied.training_set()) == [(0, "a"), (2, "c")]
    assert supplied.training_indexes == [0, 2]
    assert supplied.test_indexes == [1, 3]
    assert supplied.index == 4


def test_test_set_yields_only_test_items_in_index_order(monkeypatch):
    monkeypatch.setattr(dsmod, "partitioner", make_partition_factory([0, 1, 0, 1]))
    dataset = ["a", "b", "c", "d"]
    supplied = DataSuplier(dataset, training_percent=0.5, partitioning_function=object())

    assert list(supplied.test_set()) == [(1, "b"), (3, "d")]
    assert supplied.training_indexes == [0, 2]
    assert supplied.test_indexes == [1, 3]
    assert supplied.index == 4


def test_generators_are_lazy_until_iterated(monkeypatch):
    monkeypatch.setattr(dsmod, "partitioner", make_partition_factory([0, 1, 0]))
    supplied = DataSuplier([10, 20, 30], training_percent=0.5, partitioning_function=object())

    gen = supplied.training_set()
    assert supplied.index == 0
    assert supplied.training_indexes == []
    assert supplied.test_indexes == []

    assert next(gen) == (0, 10)
    assert supplied.index == 1
    assert supplied.training_indexes == [0]
    assert supplied.test_indexes == []

    assert next(gen) == (2, 30)
    assert supplied.index == 3
    with pytest.raises(StopIteration):
        next(gen)


def test_test_and_training_generators_share_cached_partitioning_state(monkeypatch):
    monkeypatch.setattr(dsmod, "partitioner", make_partition_factory([0, 1, 0, 1]))
    supplied = DataSuplier(["a", "b", "c", "d"], training_percent=0.5, partitioning_function=object())

    assert next(supplied.training_set()) == (0, "a")
    assert supplied.index == 1

    assert list(supplied.test_set()) == [(1, "b"), (3, "d")]
    assert supplied.index == 4
    assert supplied.training_indexes == [0, 2]
    assert supplied.test_indexes == [1, 3]


def test_consuming_test_set_first_still_allows_training_set_later(monkeypatch):
    monkeypatch.setattr(dsmod, "partitioner", make_partition_factory([1, 0, 1, 0]))
    supplied = DataSuplier(["a", "b", "c", "d"], training_percent=0.5, partitioning_function=object())

    assert list(supplied.test_set()) == [(0, "a"), (2, "c")]
    assert supplied.index == 4
    assert list(supplied.training_set()) == [(1, "b"), (3, "d")]


def test_empty_dataset_yields_empty_generators(monkeypatch):
    monkeypatch.setattr(dsmod, "partitioner", make_partition_factory([]))
    supplied = DataSuplier([], training_percent=0.8, partitioning_function=object())

    assert list(supplied.training_set()) == []
    assert list(supplied.test_set()) == []
    assert supplied.index == 0
    assert supplied.training_indexes == []
    assert supplied.test_indexes == []


def test_all_training_partition(monkeypatch):
    monkeypatch.setattr(dsmod, "partitioner", make_partition_factory([0, 0, 0]))
    supplied = DataSuplier([1, 2, 3], training_percent=0.9, partitioning_function=object())

    assert list(supplied.training_set()) == [(0, 1), (1, 2), (2, 3)]
    assert list(supplied.test_set()) == []
    assert supplied.training_indexes == [0, 1, 2]
    assert supplied.test_indexes == []


def test_all_test_partition(monkeypatch):
    monkeypatch.setattr(dsmod, "partitioner", make_partition_factory([1, 1, 1]))
    supplied = DataSuplier([1, 2, 3], training_percent=0.1, partitioning_function=object())

    assert list(supplied.training_set()) == []
    assert list(supplied.test_set()) == [(0, 1), (1, 2), (2, 3)]
    assert supplied.training_indexes == []
    assert supplied.test_indexes == [0, 1, 2]


def test_custom_partitioning_function_is_forwarded_to_partitioner(monkeypatch):
    recorder = {}
    sentinel = object()
    monkeypatch.setattr(dsmod, "partitioner", make_partition_factory([0], recorder))

    DataSuplier([1], training_percent=0.25, partitioning_function=sentinel)

    assert recorder["partitioning_function"] is sentinel


def test_default_training_percent_creates_two_complementary_buckets(monkeypatch):
    recorder = {}
    monkeypatch.setattr(dsmod, "partitioner", make_partition_factory([0], recorder))

    DataSuplier([1, 2, 3], partitioning_function=object())

    assert recorder["percentages"] == [0.8, 0.19999999999999996]


def test_dataset_is_read_by_index_not_by_iteration(monkeypatch):
    class IndexedOnlyDataset:
        def __init__(self):
            self.values = {0: "zero", 1: "one", 2: "two"}

        def __len__(self):
            return 3

        def __getitem__(self, index):
            return self.values[index]

    monkeypatch.setattr(dsmod, "partitioner", make_partition_factory([0, 1, 0]))
    dataset = IndexedOnlyDataset()
    supplied = DataSuplier(dataset, training_percent=0.5, partitioning_function=object())

    assert list(supplied.training_set()) == [(0, "zero"), (2, "two")]
    assert list(supplied.test_set()) == [(1, "one")]


def test_multiple_partial_iterations_resume_from_cached_state(monkeypatch):
    monkeypatch.setattr(dsmod, "partitioner", make_partition_factory([0, 1, 0, 1]))
    supplied = DataSuplier(["a", "b", "c", "d"], training_percent=0.5, partitioning_function=object())

    train_gen = supplied.training_set()
    assert next(train_gen) == (0, "a")
    assert next(train_gen) == (2, "c")
    with pytest.raises(StopIteration):
        next(train_gen)

    test_gen = supplied.test_set()
    assert next(test_gen) == (1, "b")
    assert next(test_gen) == (3, "d")
    with pytest.raises(StopIteration):
        next(test_gen)

    assert supplied.index == 4
