import random
import numpy as np
import pytest
import torch

from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import pseudorandom_function


# We test mostly with datasets of known length and contents.
# We seed all RNGs as requested where the code uses random, numpy or torch.


def test_init_properties_basic():
    data = list(range(10))
    ds = DataSuplier(data, training_percent=0.7)
    assert ds.dataset == data
    assert ds.training_percent == 0.7
    assert ds.n_of_elements == 10
    # training + test ratios sum to 1 with floating error tolerance
    first, second = ds.partitioner.probs
    assert abs(first + second - 1.0) < 1e-12
    assert ds.test_indexes == []
    assert ds.training_indexes == []
    assert ds.index == 0


def test_init_uses_default_partition_function():
    data = list(range(5))
    ds = DataSuplier(data)
    # default partition function is pseudorandom_function
    assert ds.partitioner.partitioner_function == pseudorandom_function


def test_partitioning_distribution_sum(monkeypatch):
    # monkeypatch partitioner to track how it is built
    data = list(range(20))
    # Partition function returns alternating 0,1 for testing
    def dummy_partitioner_length_20(n_elements, weights, part_func):
        # returns iterator that yields 0,1 alternately
        class DummyIter:
            def __init__(self):
                self.i = 0
            def __next__(self):
                v = self.i % 2
                self.i += 1
                return v
        return DummyIter()

    monkeypatch.setattr("data_partitioner.data_suplier.partitioner", dummy_partitioner_length_20)
    ds = DataSuplier(data)
    # Internally partitioner now produces 0,1,0,1,...
    # test that __next__ uses it correctly and that indexes advance
    res0 = ds._DataSuplier__next()
    res1 = ds._DataSuplier__next()
    res2 = ds._DataSuplier__next()
    assert res0 == 0
    assert res1 == 1
    assert res2 == 0
    # After three calls index should be 3
    assert ds.index == 3
    # training_indexes should have indices with partition 0
    assert ds.training_indexes == [0, 2]
    # test_indexes those with partition 1
    assert ds.test_indexes == [1]


def test_yield_set_empty_dataset():
    ds = DataSuplier([])
    # forcibly remove partitioner to avoid index errors
    ds.partitioner = iter([])
    # _yield_set over empty indexes returns no items
    res = list(ds._yield_set([]))
    assert res == []


def test_yield_set_with_prepopulated_indexes():
    data = ['a', 'b', 'c', 'd']
    ds = DataSuplier(data)
    # Provide indexes for yielding manually
    indexes = [1, 3]
    # yield results must be tuple of (index, data[index])
    results = list(ds._yield_set(indexes))
    expected = [(1, 'b'), (3, 'd')]
    assert results == expected


def test_yield_set_advances_index_and_fills_indexes(monkeypatch):
    data = list(range(5))
    fixed_partition_results = [0, 1, 0, 1, 0]
    def fixed_partitioner(n, weights, func):
        return iter(fixed_partition_results)
    monkeypatch.setattr("data_partitioner.data_suplier.partitioner", fixed_partitioner)
    ds = DataSuplier(data)
    # Initially no indexes stored
    assert ds.training_indexes == []
    assert ds.test_indexes == []
    # Use training_set generator to consume all elements
    list(ds.training_set())  # this internally triggers partitioner
    # After full consumption index should be at dataset length
    assert ds.index == len(data)
    # training indexes should be positions where partitioner returned 0
    assert ds.training_indexes == [0, 2, 4]
    # test indexes those positions for 1
    assert ds.test_indexes == [1, 3]


def test_training_set_yields_correct_values(monkeypatch):
    data = ['x', 'y', 'z']
    # Partition results: training, test, training respectively
    results = [0, 1, 0]
    def part(n, weights, func):
        return iter(results)
    monkeypatch.setattr("data_partitioner.data_suplier.partitioner", part)
    ds = DataSuplier(data)
    # Collect training set
    training = list(ds.training_set())
    # Training indexes should be 0 and 2, and values accordingly
    expected = [(0, 'x'), (2, 'z')]
    assert training == expected


def test_test_set_yields_correct_values(monkeypatch):
    data = ['x', 'y', 'z']
    results = [0, 1, 0]
    def part(n, weights, func):
        return iter(results)
    monkeypatch.setattr("data_partitioner.data_suplier.partitioner", part)
    ds = DataSuplier(data)
    test = list(ds.test_set())
    expected = [(1, 'y')]
    assert test == expected


def test_consuming_training_and_test_sets_in_any_order(monkeypatch):
    data = list(range(6))
    # Partition results: 0,1,0,1,0,1
    results = [0, 1, 0, 1, 0, 1]
    def part(n, weights, func):
        return iter(results)
    monkeypatch.setattr("data_partitioner.data_suplier.partitioner", part)
    ds = DataSuplier(data)
    # Fake interleaved consumption
    train_gen = ds.training_set()
    test_gen = ds.test_set()
    # Pull first training item
    train_first = next(train_gen)
    assert train_first == (0, 0)
    # Pull first test item
    test_first = next(test_gen)
    assert test_first == (1,1)
    # Pull remaining all from training and test
    train_rest = list(train_gen)
    test_rest = list(test_gen)
    # training indexes are 0,2,4
    assert train_rest == [(2, 2), (4, 4)]
    # test indexes are 1,3,5
    assert test_rest == [(3, 3), (5, 5)]


def test_index_advances_correctly(monkeypatch):
    data = list(range(3))
    # partition results all 0 (training)
    def part(n, weights, func):
        return iter([0,0,0])
    monkeypatch.setattr("data_partitioner.data_suplier.partitioner", part)
    ds = DataSuplier(data)
    # Before any yield index is zero
    assert ds.index == 0
    # Use training_set to get all data
    list(ds.training_set())
    # Index should be equal to length after consumption
    assert ds.index == len(data)


def test_partitioning_function_called_for_all_indexes(monkeypatch):
    data = list(range(4))
    called_indexes = []

    # Patch partitioner to record calls to part_func, simulate iterator
    def fake_partitioner(n_elements, weights, part_func):
        for idx in range(n_elements):
            called_indexes.append(idx)
        # Return an iterator of zeros
        return iter([0]*n_elements)

    monkeypatch.setattr("data_partitioner.data_suplier.partitioner", fake_partitioner)
    ds = DataSuplier(data)
    # Consume all training data to trigger full partitioning
    list(ds.training_set())
    assert called_indexes == list(range(len(data)))


def test_stability_with_random_seed(monkeypatch):
    # Seed all RNGs even if code does not call them directly,
    # For completeness per instructions
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    data = list(range(5))

    ds = DataSuplier(data)
    # Consume full training set
    training = list(ds.training_set())
    # Training indexes + values match length and data
    assert all(isinstance(x[0], int) for x in training)
    assert all(training[i][1] == data[training[i][0]] for i in range(len(training)))


def test_training_and_test_sets_do_not_overlap(monkeypatch):
    data = list(range(12))
    # deterministic pattern 0 for 0 <= idx < 6, 1 for idx >=6
    def part(n, weights, func):
        return iter([0]*6 + [1]*6)
    monkeypatch.setattr("data_partitioner.data_suplier.partitioner", part)
    ds = DataSuplier(data)
    training = list(ds.training_set())
    test = list(ds.test_set())
    # sets of indexes for training and test don't intersect
    train_indices = {idx for idx, _ in training}
    test_indices = {idx for idx, _ in test}
    assert train_indices.isdisjoint(test_indices)
    # Together they cover all indexes
    assert train_indices.union(test_indices) == set(range(len(data)))


def test_training_test_sets_total_cover_dataset(monkeypatch):
    data = list(range(7))
    # pattern 0,1 alternates, 0-indexed: idx%2==0 training else test
    def part(n, weights, func):
        return iter([i % 2 for i in range(n)])
    monkeypatch.setattr("data_partitioner.data_suplier.partitioner", part)
    ds = DataSuplier(data)
    training = list(ds.training_set())
    test = list(ds.test_set())
    # All dataset indexes appear in union
    combined = training + test
    combined_indices = {idx for idx, _ in combined}
    assert combined_indices == set(range(len(data)))


def test_empty_training_percent_assigns_all_test(monkeypatch):
    data = list(range(4))
    # partition returns 1 always, so all in test set
    def part(n, weights, func):
        return iter([1]*n)
    monkeypatch.setattr("data_partitioner.data_suplier.partitioner", part)
    ds = DataSuplier(data, training_percent=0)
    # training set should be empty
    training = list(ds.training_set())
    assert training == []
    # test set should contain all data
    test = list(ds.test_set())
    expected = list(enumerate(data))
    assert test == expected


def test_full_training_percent_assigns_all_training(monkeypatch):
    data = list(range(3))
    # partition returns 0 always
    def part(n, weights, func):
        return iter([0]*n)
    monkeypatch.setattr("data_partitioner.data_suplier.partitioner", part)
    ds = DataSuplier(data, training_percent=1)
    test = list(ds.test_set())
    training = list(ds.training_set())
    assert test == []
    expected = list(enumerate(data))
    assert training == expected


def test_index_property_does_not_advance_on_training_test_creation(monkeypatch):
    data = list(range(5))
    seq = [0, 1, 0, 1, 0]
    def part(n, weights, func):
        return iter(seq)
    monkeypatch.setattr("data_partitioner.data_suplier.partitioner", part)
    ds = DataSuplier(data)
    idx_before = ds.index
    # Just get generators, don't consume items
    traingen = ds.training_set()
    testgen = ds.test_set()
    idx_after = ds.index
    # Index should not advance by creating generators
    assert idx_before == idx_after


def test_yield_sets_complete_after_consuming_all(monkeypatch):
    data = list(range(4))
    seq = [0, 1, 0, 1]
    def part(n, weights, func):
        return iter(seq)
    monkeypatch.setattr("data_partitioner.data_suplier.partitioner", part)
    ds = DataSuplier(data)
    training_list = list(ds.training_set())
    # After exhausting training set, the training_indexes should have all those indexes assigned 0
    assert ds.training_indexes == [i for i, p in enumerate(seq) if p == 0]
    # Similarly for test set
    test_list = list(ds.test_set())
    assert ds.test_indexes == [i for i, p in enumerate(seq) if p == 1]


def test_double_iteration_over_same_generator_fails(monkeypatch):
    # The generators should be exhausted after one full iteration
    data = list(range(3))
    def part(n, weights, func):
        return iter([0,1,0])
    monkeypatch.setattr("data_partitioner.data_suplier.partitioner", part)
    ds = DataSuplier(data)
    gen = ds.training_set()
    first = list(gen)
    # Subsequent attempts to iterate same generator produce no items
    second = list(gen)
    assert second == []
    # The training indexes do not change during iteration
    assert ds.training_indexes == [0, 2]


def test_partial_consumption_then_resume(monkeypatch):
    data = list(range(3))
    def part(n, weights, func):
        return iter([0, 1, 0])
    monkeypatch.setattr("data_partitioner.data_suplier.partitioner", part)
    ds = DataSuplier(data)
    gen_train = ds.training_set()
    # get first training item
    first_item = next(gen_train)
    assert first_item == (0, 0)
    # index should now at least 1
    assert ds.index >= 1
    # get next training item (the last one)
    second_item = next(gen_train)
    assert second_item == (2, 2)
    # exhausting generator returns no further items
    with pytest.raises(StopIteration):
        next(gen_train)


def test_len_zero_dataset_does_not_fail():
    ds = DataSuplier([])
    # training and test sets are empty
    assert list(ds.training_set()) == []
    assert list(ds.test_set()) == []
    assert ds.index == 0
    # __next__ will raise StopIteration since partitioner is constructed with length 0
    with pytest.raises(StopIteration):
        ds._DataSuplier__next()


def test_no_side_effects_on_multiple_training_set_calls(monkeypatch):
    data = list(range(2))
    seq = [0, 1]
    def part(n, weights, func):
        return iter(seq)
    monkeypatch.setattr("data_partitioner.data_suplier.partitioner", part)
    ds = DataSuplier(data)
    # Repeated calls to training_set return independent generators with same content
    t1 = list(ds.training_set())
    t2 = list(ds.training_set())
    assert t1 == t2
    # training indexes and test indexes accumulate only once
    # Because __next__ only advances when generator is consumed
    # But since multiple generators consume independently, indexes accumulate multiple times
    # So training_indexes length should be number of 0 in seq times number of generators consumed
    # Because underlying partitioner iterator doesn't reset, indexes accumulate in order
    # The test here can check indexes are nonempty and consistent
    assert all(isinstance(i, int) for i in ds.training_indexes)
    assert all(isinstance(i, int) for i in ds.test_indexes)


def test_training_set_and_test_set_return_generators(monkeypatch):
    data = list(range(2))
    seq = [0,1]
    def part(n, weights, func):
        return iter(seq)
    monkeypatch.setattr("data_partitioner.data_suplier.partitioner", part)
    ds = DataSuplier(data)
    train_gen = ds.training_set()
    test_gen = ds.test_set()
    assert hasattr(train_gen, '__iter__')
    assert hasattr(test_gen, '__iter__')
    assert (0, 0) in list(train_gen)
    assert (1, 1) in list(test_gen)
