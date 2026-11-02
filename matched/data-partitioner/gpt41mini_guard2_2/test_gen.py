import pytest
import random
import numpy as np
import torch

from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import pseudorandom_function

@pytest.fixture
def simple_dataset():
    return list(range(10))

def _consume_all(ds):
    # fully consume training_set and test_set generators to force full partitioning
    list(ds.training_set())
    list(ds.test_set())

def test_init_attributes(simple_dataset):
    ds = DataSuplier(simple_dataset)
    assert ds.dataset == simple_dataset
    assert ds.training_percent == 0.8
    assert ds.n_of_elements == len(simple_dataset)
    assert hasattr(ds.partitioner, "__next__")
    assert ds.test_indexes == []
    assert ds.training_indexes == []
    assert ds.index == 0

def test_partitioning_proportions_simple():
    # seed all random sources used by pseudorandom_function or partitioner
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    dataset = list(range(100))
    ds = DataSuplier(dataset, training_percent=0.7)
    # Force complete partitioning by consuming generators
    train = list(ds.training_set())
    test = list(ds.test_set())

    assert len(train) + len(test) == len(dataset)
    # approx 70% training
    training_ratio = len(train) / len(dataset)
    assert abs(training_ratio - 0.7) < 0.1
    # approx 30% test
    test_ratio = len(test) / len(dataset)
    assert abs(test_ratio - 0.3) < 0.1

def test_training_test_sets_are_disjoint_and_cover_all(simple_dataset):
    ds = DataSuplier(simple_dataset)
    train = list(ds.training_set())
    test = list(ds.test_set())

    train_indices = {i for i, _ in train}
    test_indices = {i for i, _ in test}

    assert train_indices.isdisjoint(test_indices)
    combined = train_indices | test_indices
    assert combined == set(range(len(simple_dataset)))

def test_index_yields_correct_data(simple_dataset):
    ds = DataSuplier(simple_dataset)
    train = list(ds.training_set())
    test = list(ds.test_set())

    for idx, value in train + test:
        assert value == simple_dataset[idx]

def test_iteration_state_progresses_correctly(simple_dataset):
    ds = DataSuplier(simple_dataset)
    assert ds.index == 0
    # get a few items from training_set generator
    train_gen = ds.training_set()
    for _ in range(3):
        next(train_gen)
    # internal index should have advanced
    assert ds.index >= 3
    # get some from test_set too
    test_gen = ds.test_set()
    for _ in range(3):
        next(test_gen)
    # index should be at least 6 now (some overlap possible)
    assert ds.index >= 6

def test_yield_all_items_if_training_percent_1(simple_dataset):
    ds = DataSuplier(simple_dataset, training_percent=1.0)
    train = list(ds.training_set())
    test = list(ds.test_set())
    # all items in training, none in test
    assert len(train) == len(simple_dataset)
    assert len(test) == 0
    indices_train = [i for i, _ in train]
    assert sorted(indices_train) == list(range(len(simple_dataset)))

def test_yield_all_items_if_training_percent_0(simple_dataset):
    ds = DataSuplier(simple_dataset, training_percent=0.0)
    train = list(ds.training_set())
    test = list(ds.test_set())
    # all items in test, none in training
    assert len(test) == len(simple_dataset)
    assert len(train) == 0
    indices_test = [i for i, _ in test]
    assert sorted(indices_test) == list(range(len(simple_dataset)))

def test_empty_dataset():
    ds = DataSuplier([])
    train = list(ds.training_set())
    test = list(ds.test_set())
    assert train == []
    assert test == []

def test_training_and_test_index_lists_consistent_after_consumption(simple_dataset):
    ds = DataSuplier(simple_dataset)
    # Must call both generators to populate indexes fully
    train = list(ds.training_set())
    test = list(ds.test_set())
    # The internal index lists should be sorted and contain exact indices
    assert sorted(ds.training_indexes) == sorted(i for i, _ in train)
    assert sorted(ds.test_indexes) == sorted(i for i, _ in test)
    combined = sorted(ds.training_indexes + ds.test_indexes)
    assert combined == list(range(len(simple_dataset)))

def test_custom_partitioning_function_effect(monkeypatch):
    # A trivial partitioning_function: alternate partitions 0,1,0,1...
    def alternating_partition_generator(n_elements, proportions, part_fn):
        # yields 0,1,0,1,... ignoring part_fn
        for i in range(n_elements):
            yield i % 2

    # Mock partitioner to use the above generator
    class DummyPartitioner:
        def __init__(self, n, props, part_fn):
            self._gen = alternating_partition_generator(n, props, part_fn)
        def __next__(self):
            return next(self._gen)

    monkeypatch.setattr("data_partitioner.data_suplier.partitioner", DummyPartitioner)

    dataset = list(range(6))
    ds = DataSuplier(dataset)

    train = list(ds.training_set())
    test = list(ds.test_set())

    # Since partitions alternate, training_indexes -> indices with 0 (even),
    # test_indexes -> indices with 1 (odd)
    assert set(ds.training_indexes) == {0, 2, 4}
    assert set(ds.test_indexes) == {1, 3, 5}

    # Corresponding yielded data matches
    for idx, val in train:
        assert val == dataset[idx]
        assert idx % 2 == 0
    for idx, val in test:
        assert val == dataset[idx]
        assert idx % 2 == 1

def test___next_returns_correct_partition_value(simple_dataset):
    ds = DataSuplier(simple_dataset)
    # Advance using internal __next__ and compare output with recorded indexes
    # __next__ is called by training_set/test_set internal logic but we can call directly as well
    # It returns 0 for training, 1 for test
    results = []
    for _ in range(len(simple_dataset)):
        res = ds.__next__()
        results.append(res)
    # Number of 0s should approx equal number in training_indexes
    assert results.count(0) == len(ds.training_indexes)
    assert results.count(1) == len(ds.test_indexes)
    # Counts sum to the length of dataset
    assert len(ds.training_indexes) + len(ds.test_indexes) == len(simple_dataset)

def test_yield_set_generator_yields_all_indices(simple_dataset):
    ds = DataSuplier(simple_dataset)
    # Manually fill indexes using internal __next__
    while ds.index < ds.n_of_elements:
        ds.__next__()
    # Now we have both indexes filled
    train_items = list(ds._yield_set(ds.training_indexes))
    test_items = list(ds._yield_set(ds.test_indexes))
    # The yielded indices match those in the index lists
    assert sorted(i for i, _ in train_items) == sorted(ds.training_indexes)
    assert sorted(i for i, _ in test_items) == sorted(ds.test_indexes)
    # Items value matches dataset
    for idx, val in train_items + test_items:
        assert val == simple_dataset[idx]

def test_yield_set_generator_empty_indexes_yields_nothing(simple_dataset):
    ds = DataSuplier(simple_dataset)
    # Using yield_set on empty lists yields no items
    assert list(ds._yield_set([])) == []

def test_training_and_test_sets_do_not_repeat_items(simple_dataset):
    ds = DataSuplier(simple_dataset)
    train_items = list(ds.training_set())
    test_items = list(ds.test_set())
    # No duplicates within training or test sets
    train_indices = [idx for idx, _ in train_items]
    test_indices = [idx for idx, _ in test_items]
    assert len(train_indices) == len(set(train_indices))
    assert len(test_indices) == len(set(test_indices))

def test_index_does_not_exceed_n_of_elements(simple_dataset):
    ds = DataSuplier(simple_dataset)
    # Exhaust generators, index max is len(dataset)
    list(ds.training_set())
    list(ds.test_set())
    assert ds.index == ds.n_of_elements

def test_training_set_generator_progressive_yield(monkeypatch):
    # Patch partitioner to always return 0 (all train)
    class AlwaysTrainPartitioner:
        def __init__(self, n, ps, fn):
            self.calls = 0
            self.n = n
        def __next__(self):
            if self.calls >= self.n:
                raise StopIteration
            self.calls += 1
            return 0

    monkeypatch.setattr("data_partitioner.data_suplier.partitioner", AlwaysTrainPartitioner)

    dataset = list(range(5))
    ds = DataSuplier(dataset)
    gen = ds.training_set()
    # Consume elements progressively and check index increments
    for i, (idx, val) in enumerate(gen):
        assert idx == i
        assert val == dataset[idx]
        assert ds.index >= i+1

def test_test_set_generator_progressive_yield(monkeypatch):
    # Patch partitioner to always return 1 (all test)
    class AlwaysTestPartitioner:
        def __init__(self, n, ps, fn):
            self.calls = 0
            self.n = n
        def __next__(self):
            if self.calls >= self.n:
                raise StopIteration
            self.calls += 1
            return 1

    monkeypatch.setattr("data_partitioner.data_suplier.partitioner", AlwaysTestPartitioner)

    dataset = list(range(5))
    ds = DataSuplier(dataset)
    gen = ds.test_set()
    # Consume elements progressively and check index increments
    for i, (idx, val) in enumerate(gen):
        assert idx == i
        assert val == dataset[idx]
        assert ds.index >= i+1
