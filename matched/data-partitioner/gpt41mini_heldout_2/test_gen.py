import pytest
from data_partitioner.data_suplier import DataSuplier
from data_partitioner import data_suplier

# Because DataSuplier depends on 'partitioner' and 'pseudorandom_function' from data_partitioner.partitioner,
# we need to do some basic sanity checks that they exist and behave as expected.
# But since we focus on DataSuplier, the following tests rely on the normal behavior of 'partitioner'.


def test_init_sets_attributes_correctly():
    dataset = list(range(10))
    ds = DataSuplier(dataset, training_percent=0.7)
    assert ds.dataset == dataset
    assert ds.training_percent == 0.7
    assert ds.n_of_elements == 10
    # test_indexes and training_indexes start empty
    assert ds.test_indexes == []
    assert ds.training_indexes == []
    assert ds.index == 0
    # partitioner is initialized with correct lengths and ratios
    assert len(ds.partitioner.proportions) == 2
    assert pytest.approx(sum(ds.partitioner.proportions), 0.001) == 1.0

def test_init_uses_default_partitioning_function():
    dataset = list(range(5))
    ds = DataSuplier(dataset)
    # Default partitioning_function comes from data_partitioner.partitioner.pseudorandom_function
    # Check that ds.partitioner._partition_function is this function
    assert ds.partitioner._partition_function == data_suplier.pseudorandom_function

def test_length_with_empty_dataset():
    ds = DataSuplier([])
    assert ds.n_of_elements == 0
    # Getting generators should work but yield nothing
    assert list(ds.training_set()) == []
    assert list(ds.test_set()) == []

def test_training_and_test_sets_cover_entire_dataset():
    dataset = list("abcdefghij")
    ds = DataSuplier(dataset, training_percent=0.8)
    # exhaust generators
    train_items = list(ds.training_set())
    test_items = list(ds.test_set())
    # Check indexes cover all elements uniquely
    all_indexes = {i for i, _ in train_items} | {i for i, _ in test_items}
    assert all_indexes == set(range(len(dataset)))
    # Check no duplicate indexes
    assert len(all_indexes) == (len(train_items) + len(test_items))
    # The union of datasets from indexes reconstructs original dataset
    reconstructed = [val for _, val in sorted(train_items + test_items, key=lambda x: x[0])]
    assert reconstructed == dataset

def test_training_and_test_sets_are_disjoint():
    ds = DataSuplier(list(range(20)), training_percent=0.6)
    train = list(ds.training_set())
    test = list(ds.test_set())
    train_indexes = {i for i, _ in train}
    test_indexes = {i for i, _ in test}
    assert train_indexes.isdisjoint(test_indexes)

def test_yield_set_generator_returns_expected_tuples():
    dataset = ['a', 'b', 'c', 'd', 'e']
    ds = DataSuplier(dataset, training_percent=0.4)
    # consume training set
    training_gen = ds.training_set()
    for index, value in training_gen:
        assert dataset[index] == value
        assert 0 <= index < len(dataset)
    # consume test set
    ds.index = 0  # Resetting index to simulate a new instance, but better to create fresh object
    ds = DataSuplier(dataset, training_percent=0.4)
    test_gen = ds.test_set()
    for index, value in test_gen:
        assert dataset[index] == value
        assert 0 <= index < len(dataset)

def test_internal_next_updates_training_and_test_indexes():
    dataset = list(range(10))
    ds = DataSuplier(dataset)
    # call __next__ repeatedly, check indexes recorded
    for i in range(len(dataset)):
        val = ds.__next()
        assert val in (0, 1)
        assert ds.index == i + 1
        # One of training_indexes or test_indexes should contain i
        assert i in ds.training_indexes or i in ds.test_indexes
    # All indexes assigned exactly once
    all_assigned = set(ds.training_indexes) | set(ds.test_indexes)
    assert all_assigned == set(range(len(dataset)))
    # No overlap
    assert set(ds.training_indexes).isdisjoint(set(ds.test_indexes))

def test_yield_set_handles_two_calls_interleaved():
    # Check if successive calls to training_set and test_set work correctly
    dataset = list(range(10))
    ds = DataSuplier(dataset, training_percent=0.5)
    train_gen = ds.training_set()
    test_gen = ds.test_set()

    train_results = []
    test_results = []

    # Pull 3 items from train, then 2 from test, then rest from both
    for _ in range(3):
        train_results.append(next(train_gen))
    for _ in range(2):
        test_results.append(next(test_gen))
    train_results.extend(list(train_gen))
    test_results.extend(list(test_gen))

    # Check all yielded are within dataset and indexes are as expected
    for idx, val in train_results + test_results:
        assert 0 <= idx < len(dataset)
        assert val == dataset[idx]

    # The union of all indexes covers the whole dataset without overlap
    train_idxs = {i for i, _ in train_results}
    test_idxs = {i for i, _ in test_results}
    assert train_idxs | test_idxs == set(range(len(dataset)))
    assert train_idxs.isdisjoint(test_idxs)

def test_training_percent_0_train_all():
    dataset = list(range(5))
    ds = DataSuplier(dataset, training_percent=1.0)
    training = list(ds.training_set())
    test = list(ds.test_set())
    # All data should be training, test empty
    assert len(training) == len(dataset)
    assert test == []

def test_training_percent_0_test_all():
    dataset = list(range(5))
    ds = DataSuplier(dataset, training_percent=0.0)
    training = list(ds.training_set())
    test = list(ds.test_set())
    # All data should be test, training empty
    assert len(test) == len(dataset)
    assert training == []

def test_partitioning_function_custom_behavior(monkeypatch):
    dataset = list(range(6))

    # Custom partitioning function: always return 1 (test set)
    def always_test(_):
        return 1

    ds = DataSuplier(dataset, training_percent=0.5, partitioning_function=always_test)
    # All indexes should be in test indexes, training empty
    training = list(ds.training_set())
    test = list(ds.test_set())
    assert training == []
    assert {i for i, _ in test} == set(range(len(dataset)))

def test_yield_set_generator_exhaustion_behavior():
    dataset = list(range(3))
    ds = DataSuplier(dataset, training_percent=0.5)
    # Exhaust the generators
    train_gen = ds.training_set()
    test_gen = ds.test_set()
    train_vals = list(train_gen)
    test_vals = list(test_gen)
    # Getting again from the generators should yield no items (generators exhausted)
    train_gen2 = ds.training_set()
    test_gen2 = ds.test_set()
    assert list(train_gen2) == []
    assert list(test_gen2) == []

def test_yield_set_produces_correct_number_of_elements():
    dataset = list(range(15))
    ds = DataSuplier(dataset, training_percent=0.7)
    train = list(ds.training_set())
    test = list(ds.test_set())
    # The amounts should roughly follow the proportions
    assert abs(len(train) / len(dataset) - 0.7) < 0.3  # margin for pseudorandom split
    assert abs(len(test) / len(dataset) - 0.3) < 0.3
    # All elements accounted for exactly once
    all_indexes = {i for i, _ in train} | {i for i, _ in test}
    assert all_indexes == set(range(len(dataset)))

def test_internal_index_advances_after_next():
    ds = DataSuplier(list("abc"))
    old_index = ds.index
    ds.__next__()
    assert ds.index == old_index + 1
    ds.__next__()
    assert ds.index == old_index + 2

def test_test_set_and_training_set_are_iterators():
    ds = DataSuplier(list(range(5)))
    ts = ds.test_set()
    tr = ds.training_set()
    assert hasattr(ts, "__iter__")
    assert hasattr(tr, "__iter__")
    assert hasattr(ts, "__next__")
    assert hasattr(tr, "__next__")

def test_training_and_test_sets_can_be_called_multiple_times_and_yield_all_data():
    ds = DataSuplier(list(range(10)))
    tr1 = list(ds.training_set())
    te1 = list(ds.test_set())
    tr2 = list(ds.training_set())
    te2 = list(ds.test_set())
    # Results should be consistent
    assert tr1 == tr2
    assert te1 == te2
    # The combined sets cover the entire dataset
    all_indexes = {i for i, _ in tr1} | {i for i, _ in te1}
    assert all_indexes == set(range(len(ds.dataset)))

def test_empty_dataset_yield_set():
    ds = DataSuplier([])
    assert list(ds.training_set()) == []
    assert list(ds.test_set()) == []

def test_index_error_not_raised_for_out_of_range_indexes():
    ds = DataSuplier(list(range(3)))
    # Exhaust generator fully
    t = list(ds.training_set())
    e = list(ds.test_set())
    # The index attribute may equal length after full exhaustion
    assert ds.index == 3

# Edge case: dataset of one item
def test_single_element_dataset():
    ds = DataSuplier([42])
    train = list(ds.training_set())
    test = list(ds.test_set())
    # All indexes either in training or test
    assert len(train) + len(test) == 1
    all_indexes = {i for i, _ in train} | {i for i, _ in test}
    assert all_indexes == {0}
