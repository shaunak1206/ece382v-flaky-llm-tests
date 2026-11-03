import random
import numpy as np
import pytest

from data_partitioner.data_suplier import DataSuplier
from data_partitioner import partitioner, pseudorandom_function

# We know DataSuplier uses partitioner and pseudorandom_function from data_partitioner.partitioner.
# We will seed only the random (and np.random if needed) at start of each test.
# The given code does NOT use torch, so we only seed random and numpy as precaution (if any).
# We will write tests covering the initialization, training_set, test_set, _yield_set, and __next behavior indirectly.

def make_dataset(size=10):
    """Helper to make a list dataset of given size with values equal to their index strings."""
    return [str(i) for i in range(size)]

def get_partitions_indices(data_suplier: DataSuplier):
    """Force iteration over all data, return training and test indexes."""
    train = list(i for i, _ in data_suplier.training_set())
    test = list(i for i, _ in data_suplier.test_set())
    return train, test

def total_elements_in_sets(data_suplier: DataSuplier):
    """Sum number of elements returned by training_set + test_set"""
    train_count = sum(1 for _ in data_suplier.training_set())
    test_count = sum(1 for _ in data_suplier.test_set())
    return train_count + test_count


def test_initialization_sets_indexes_sizes():
    random.seed(1234)
    np.random.seed(1234)
    dataset = make_dataset(20)
    ds = DataSuplier(dataset, training_percent=0.7)
    assert ds.dataset == dataset
    assert ds.training_percent == 0.7
    assert ds.n_of_elements == 20
    # The sum of partition weights is 1
    assert ds.partitioner.weights == [0.7, 0.3]
    # Initially the index lists are empty, index is zero
    assert ds.test_indexes == []
    assert ds.training_indexes == []
    assert ds.index == 0


def test_training_and_test_sets_exhaust_full_dataset():
    random.seed(42)
    np.random.seed(42)
    dataset = make_dataset(100)
    ds = DataSuplier(dataset)
    train_idxs, test_idxs = get_partitions_indices(ds)

    # All indexes covered exactly once between train and test
    combined = set(train_idxs) | set(test_idxs)
    assert combined == set(range(len(dataset)))

    # Training indexes length approx 80% of dataset size, tolerance +-1 due to possibly rounding behavior in partitioner
    n_train = len(train_idxs)
    n_total = len(dataset)
    expected_train = 0.8 * n_total
    assert abs(n_train - expected_train) < 2  # allow a tolerance of 1 or 2

    # Test indexes length is remainder
    n_test = len(test_idxs)
    assert n_train + n_test == n_total

    # Check no duplicates inside each
    assert len(set(train_idxs)) == n_train
    assert len(set(test_idxs)) == n_test


def test_iteration_returns_correct_items_and_order():
    random.seed(7)
    np.random.seed(7)
    dataset = make_dataset(15)
    ds = DataSuplier(dataset, training_percent=0.6)

    ds_train_items = list(ds.training_set())
    ds_test_items = list(ds.test_set())

    # Values yielded are correct and correspond to those indexes in dataset
    for i, val in ds_train_items:
        assert val == dataset[i]
    for i, val in ds_test_items:
        assert val == dataset[i]

    # All indexes returned by training_set are in ascending order (because partitioner.__next__ is called sequentially increasing i)
    train_indexes = [i for i, _ in ds_train_items]
    test_indexes = [i for i, _ in ds_test_items]
    assert train_indexes == sorted(train_indexes)
    assert test_indexes == sorted(test_indexes)


def test_empty_dataset_yields_nothing():
    random.seed(1)
    np.random.seed(1)
    ds = DataSuplier([])
    assert list(ds.training_set()) == []
    assert list(ds.test_set()) == []


def test_training_percent_zero_and_one(monkeypatch):
    random.seed(3)
    np.random.seed(3)
    dataset = make_dataset(10)

    # training_percent=0 means all data in test set
    ds0 = DataSuplier(dataset, training_percent=0.0)
    train0 = list(ds0.training_set())
    test0 = list(ds0.test_set())
    assert train0 == []
    assert {i for i, _ in test0} == set(range(10))

    # training_percent=1 means all data in training set
    ds1 = DataSuplier(dataset, training_percent=1.0)
    train1 = list(ds1.training_set())
    test1 = list(ds1.test_set())
    assert test1 == []
    assert {i for i, _ in train1} == set(range(10))


def test_custom_partitioning_function(monkeypatch):
    # This test verifies DataSuplier uses the given partitioning_function correctly.
    # We will patch partitioner to confirm it receives correct args and return predictable results.
    random.seed(10)
    np.random.seed(10)
    dataset = make_dataset(5)

    calls = []

    def fake_partitioning_function(idx):
        calls.append(idx)
        # alternate between training (0) and test (1)
        return idx % 2

    class FakePartitioner:
        def __init__(self, n_of_elements, weights, partitioning_func):
            assert n_of_elements == len(dataset)
            assert weights == [0.75, 0.25]
            assert callable(partitioning_func)
            self.idx = 0

        def __next__(self):
            if self.idx >= len(dataset):
                raise StopIteration
            res = fake_partitioning_function(self.idx)
            self.idx += 1
            return res

    monkeypatch.setattr('data_partitioner.data_suplier.partitioner', lambda n, weights, fun: FakePartitioner(n, weights, fun))

    ds = DataSuplier(dataset, training_percent=0.75, partitioning_function=fake_partitioning_function)
    train_idxs, test_idxs = get_partitions_indices(ds)

    # Expect even indexes in training (0), odd in test (1)
    assert set(train_idxs) == {0,2,4}
    assert set(test_idxs) == {1,3}
    # Confirm our fake partitioning_function was called for each index in order
    assert calls == list(range(len(dataset)))


def test_yield_set_continues_to_yield_once_called():
    # Calling training_set twice produces no repeated data or errors (generators recreate state)
    random.seed(11)
    np.random.seed(11)
    dataset = make_dataset(20)
    ds = DataSuplier(dataset)
    first_call = list(ds.training_set())
    # Calling again, because index in DataSuplier is preserved, only new yields after index
    second_call = list(ds.training_set())
    # The second call should yield no additional elements because index reached end in first call
    assert second_call == []
    # The combined elements are correct, no duplicates
    indexes = [i for i, _ in first_call]
    assert len(indexes) == len(set(indexes))


def test_partial_iteration_and_further_iteration_resume():
    random.seed(15)
    np.random.seed(15)
    dataset = make_dataset(10)
    ds = DataSuplier(dataset)

    train_gen = ds.training_set()
    # Get first 3 items only
    partial_train = [next(train_gen) for _ in range(3)]

    # Now get rest
    rest_train = list(train_gen)

    # Combined equals full training set items without repeats
    full_train = partial_train + rest_train
    assert len(full_train) == len(set(i for i, _ in full_train))
    # All indexes in training_indexes are reflected in full_train indexes
    assert set(ds.training_indexes) == set(i for i, _ in full_train)


def test_index_advances_correctly_and_partition_lists_grow_consistently():
    random.seed(19)
    np.random.seed(19)
    dataset = make_dataset(12)
    ds = DataSuplier(dataset)

    # Initially index is zero
    assert ds.index == 0
    # Before any iteration, both lists empty
    assert ds.training_indexes == []
    assert ds.test_indexes == []

    # Call next 5 times using internal __next__
    for _ in range(5):
        res = ds._DataSuplier__next()
        assert res in (0,1)
    assert ds.index == 5
    # training + test indexes sums to 5
    assert len(ds.training_indexes) + len(ds.test_indexes) == 5
    # All indexes recorded are distinct and in 0..4
    recorded = ds.training_indexes + ds.test_indexes
    assert set(recorded) == set(range(5))


def test_yield_set_handles_building_index_lists_during_iteration():
    random.seed(23)
    np.random.seed(23)
    dataset = make_dataset(8)
    ds = DataSuplier(dataset)

    # Artificially fill one partition's indexes, then yield from the other
    # but because the indexes are empty, _yield_set will call __next to fill it gradually

    # Yield 1 item from test_set iterator
    test_iter = ds.test_set()
    first_item = next(test_iter)
    assert first_item[0] in range(len(dataset))
    # test_indexes must have at least one index now
    assert len(ds.test_indexes) >= 1

    # Yield rest of test_set
    rest = list(test_iter)
    # Total test indexes = len(ds.test_indexes)
    assert len(rest) + 1 == len(ds.test_indexes)
    # test_indexes values sorted ascending
    assert ds.test_indexes == sorted(ds.test_indexes)

def test_training_percent_invalid_values(monkeypatch):
    # We check behavior on training_percent outside [0,1], expecting no exception but valid partitions sum to 1
    random.seed(31)
    np.random.seed(31)
    dataset = make_dataset(5)

    # Use training_percent less than 0
    ds_neg = DataSuplier(dataset, training_percent=-0.5)
    # Check that weights sum to 1 even for negative (depends on partitioner but mainly test that weights are floats)
    weights = ds_neg.partitioner.weights
    assert pytest.approx(sum(weights), 1.0) == 1.0

    # Use training_percent greater than 1
    ds_over = DataSuplier(dataset, training_percent=1.5)
    weights2 = ds_over.partitioner.weights
    assert pytest.approx(sum(weights2), 1.0) == 1.0

def test_dataset_with_non_string_elements():
    random.seed(45)
    np.random.seed(45)
    dataset = list(range(10))
    ds = DataSuplier(dataset)
    train = list(ds.training_set())
    test = list(ds.test_set())
    # Check values returned are from dataset at given indexes and are integers
    for i, val in train + test:
        assert val == dataset[i]
        assert isinstance(val, int)


def test_multiple_instances_independent_iteration():
    random.seed(50)
    np.random.seed(50)
    dataset = make_dataset(30)
    ds1 = DataSuplier(dataset)
    ds2 = DataSuplier(dataset)

    # Iterate partially on ds1 training set
    gen1 = ds1.training_set()
    partial1 = [next(gen1) for _ in range(5)]

    # ds2 independent full iteration
    full2 = list(ds2.training_set())

    # Continue ds1 first generator
    rest1 = list(gen1)

    # Combined length matches length of training_indexes for ds1
    combined1 = partial1 + rest1
    assert len(combined1) == len(ds1.training_indexes)
    # ds2's training indexes different or equal but independent
    assert ds2.training_indexes == [i for i, _ in full2]

def test_training_and_test_sets_do_not_overlap():
    random.seed(60)
    np.random.seed(60)
    dataset = make_dataset(50)
    ds = DataSuplier(dataset)
    train = list(ds.training_set())
    test = list(ds.test_set())
    train_indexes = {i for i, _ in train}
    test_indexes = {i for i, _ in test}
    # The two sets should be disjoint
    assert train_indexes.isdisjoint(test_indexes)

def test_accessing_generator_after_all_data_consumed_produces_no_yield():
    random.seed(70)
    np.random.seed(70)
    dataset = make_dataset(10)
    ds = DataSuplier(dataset)
    # Exhaust training generator
    list(ds.training_set())
    # Exhaust test generator
    list(ds.test_set())
    # Subsequent calls return empty generators
    assert list(ds.training_set()) == []
    assert list(ds.test_set()) == []


def test_contains_no_modifications_to_global_or_module_state(monkeypatch):
    # The test tries to make sure DataSuplier does not write to global or module level variables unexpectedly
    # Since code does not modify any global variables, we confirm partitioner and pseudorandom_function are called correctly.

    random.seed(80)
    np.random.seed(80)
    dataset = make_dataset(4)

    call_args = {}

    def fake_partitioner(n_elements, weights, func):
        call_args["n_elements"] = n_elements
        call_args["weights"] = weights
        call_args["func"] = func
        class Iter:
            def __init__(self):
                self.i = 0
            def __next__(self):
                if self.i >= n_elements:
                    raise StopIteration
                val = 0
                self.i += 1
                return val
        return Iter()

    # Patch partitioner only during creation of DataSuplier instance
    monkeypatch.setattr('data_partitioner.data_suplier.partitioner', fake_partitioner)

    ds = DataSuplier(dataset, training_percent=0.6)
    # Check partitioner called with correct arguments
    assert call_args["n_elements"] == len(dataset)
    assert call_args["weights"] == [0.6, 0.4]
    assert callable(call_args["func"])

    # Calling training_set yields all items in training partition (all index 0 in our fake_partitioner)
    train = list(ds.training_set())
    assert len(train) == len(dataset)
    for idx, val in train:
        assert val == dataset[idx]

