import random
import numpy as np
import pytest

from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import pseudorandom_function

@pytest.mark.parametrize("dataset_size", [0, 1, 10, 100])
def test_init_sets_attributes_correctly(dataset_size):
    random.seed(1234)
    np.random.seed(1234)
    dataset = list(range(dataset_size))
    ds = DataSuplier(dataset, training_percent=0.75)
    assert ds.dataset == dataset
    assert ds.training_percent == 0.75
    assert ds.n_of_elements == dataset_size
    # check that partitioner is created with correct parameters
    # partitioner is an iterator, check it has __next__
    assert hasattr(ds.partitioner, "__next__")
    # indexes lists and index counter start empty/zero
    assert ds.test_indexes == []
    assert ds.training_indexes == []
    assert ds.index == 0

def test_default_training_percent_is_0_8():
    random.seed(1)
    np.random.seed(1)
    dataset = list(range(10))
    ds = DataSuplier(dataset)
    assert ds.training_percent == 0.8

def test_partitioning_function_can_be_overridden():
    random.seed(42)
    np.random.seed(42)
    def always_zero(_):
        return 0
    dataset = list(range(10))
    ds = DataSuplier(dataset, training_percent=0.5, partitioning_function=always_zero)
    # All elements should go to training set (because partitioning_function returns 0 always)
    # Exhaust the sets by consuming all
    training = list(ds.training_set())
    test = list(ds.test_set())
    # training indexes should cover all dataset indices, test should be empty
    assert set(i for i, _ in training) == set(range(10))
    assert test == []

def test_index_increments_and_indexes_records(monkeypatch):
    random.seed(111)
    np.random.seed(111)
    dataset = list(range(5))
    ds = DataSuplier(dataset)

    # Force the partitioner.__next__ to alternate 0 and 1 deterministically to get control
    calls = [-1]
    def fake_next():
        calls[0] += 1
        return calls[0] % 2
    monkeypatch.setattr(ds.partitioner, "__next__", fake_next)

    # Call __next__ explicitly and check that indexes are recorded properly
    for i in range(len(dataset)):
        res = ds._DataSuplier__next()
        assert res == i % 2
        assert ds.index == i + 1
        if res == 0:
            # training_indexes should have last index appended
            assert ds.training_indexes[-1] == i
        else:
            # test_indexes should have last index appended
            assert ds.test_indexes[-1] == i

def test_yield_set_returns_correct_items(monkeypatch):
    random.seed(123)
    np.random.seed(123)
    dataset = ['a', 'b', 'c', 'd', 'e']
    ds = DataSuplier(dataset)

    # Patch partitioner.__next__ to produce a known pattern: 0,1,0,1,0
    pat = [0, 1, 0, 1, 0]
    calls = [-1]
    def fake_next():
        calls[0] += 1
        return pat[calls[0]]
    monkeypatch.setattr(ds.partitioner, "__next__", fake_next)

    # Exhaust __next__ to fill indexes
    for _ in range(len(dataset)):
        ds._DataSuplier__next()

    # Now test test_set yields correct pairs
    test_items = list(ds.test_set())
    expected_test_indices = [i for i, v in enumerate(pat) if v == 1]
    assert [idx for idx, val in test_items] == expected_test_indices
    assert [val for idx, val in test_items] == [dataset[i] for i in expected_test_indices]

    # And training_set yields correct pairs
    training_items = list(ds.training_set())
    expected_training_indices = [i for i, v in enumerate(pat) if v == 0]
    assert [idx for idx, val in training_items] == expected_training_indices
    assert [val for idx, val in training_items] == [dataset[i] for i in expected_training_indices]

def test_yield_set_lazy_evaluation(monkeypatch):
    random.seed(9)
    np.random.seed(9)
    # Use a dataset with 3 elements
    dataset = ['one', 'two', 'three']
    ds = DataSuplier(dataset)

    # patch partitioner.__next__ to return test or training: 0,0,1
    responses = [0,0,1]
    calls = [-1]
    def fake_next():
        calls[0] += 1
        return responses[calls[0]]
    monkeypatch.setattr(ds.partitioner, "__next__", fake_next)

    # At start, no indexes filled
    # Getting first item from training_set triggers yielding, which triggers __next__
    training_gen = ds.training_set()
    first_training = next(training_gen)
    # This should cause the first index to be assigned to training_indexes
    assert first_training[0] == 0
    assert first_training[1] == 'one'
    # training_indexes should now contain index 0
    assert ds.training_indexes == [0]
    # Next from training_set should yield index 1
    second_training = next(training_gen)
    assert second_training[0] == 1
    assert second_training[1] == 'two'
    assert ds.training_indexes == [0,1]
    # Next from training_set would cause __next__ again, which returns 1, so no append here but test_indexes
    # However training_set's internal while logic will keep going until it yields training indexes only
    # So next training yield should raise StopIteration because only indices left is index 2 is test set
    with pytest.raises(StopIteration):
        next(training_gen)
    # test_indexes should have index 2
    assert ds.test_indexes == [2]

def test_test_set_and_training_set_generators_do_not_overlap(monkeypatch):
    random.seed(77)
    np.random.seed(77)
    dataset = list(range(20))
    ds = DataSuplier(dataset)

    # patch partitioner.__next__ to randomly return 0 or 1, but reproducible because we seed in test
    # We will call real pseudorandom_function via real partitioner
    # So no patch here to test actual partitioning
    # But fix random seeds here to ensure reproducibility

    # Need to exhaust partitioner results fully
    # Force to iterate all
    training_items = list(ds.training_set())
    test_items = list(ds.test_set())

    # Check that the union of training and test indexes is all indices
    train_indices = set(i for i, _ in training_items)
    test_indices = set(i for i, _ in test_items)

    assert train_indices.isdisjoint(test_indices)
    union_indices = train_indices.union(test_indices)
    assert union_indices == set(range(len(dataset)))

def test_empty_dataset_produces_empty_generators():
    random.seed(10)
    np.random.seed(10)
    ds = DataSuplier([])

    training_items = list(ds.training_set())
    test_items = list(ds.test_set())
    assert training_items == []
    assert test_items == []

def test_training_percent_extremes():
    random.seed(44)
    np.random.seed(44)
    dataset = list(range(10))

    # If training_percent=1.0, all go to training set
    ds1 = DataSuplier(dataset, training_percent=1.0)
    training1 = list(ds1.training_set())
    test1 = list(ds1.test_set())
    assert set(i for i, _ in training1) == set(range(10))
    assert test1 == []

    # If training_percent=0.0, all go to test set
    ds2 = DataSuplier(dataset, training_percent=0.0)
    training2 = list(ds2.training_set())
    test2 = list(ds2.test_set())
    assert training2 == []
    assert set(i for i, _ in test2) == set(range(10))

def test_yield_set_returns_generator_type(monkeypatch):
    random.seed(123)
    np.random.seed(123)
    dataset = [0, 1, 2]
    ds = DataSuplier(dataset)

    # patch partitioner.__next__ to produce a fixed partition
    monkeypatch.setattr(ds.partitioner, "__next__", lambda: 0)
    # exhaust partitioner __next__ calls
    for _ in dataset:
        ds._DataSuplier__next()

    gen = ds.training_set()
    # Check it is a generator (has __iter__ and __next__)
    assert hasattr(gen, "__iter__")
    assert hasattr(gen, "__next__")

def test_index_only_in_forms_when_iterated(monkeypatch):
    random.seed(23)
    np.random.seed(23)
    dataset = list(range(5))
    ds = DataSuplier(dataset)

    # patch partitioner.__next__ to produce alternating outputs
    responses = [1,0,1,0,1]
    calls = [-1]
    def fake_next():
        calls[0] += 1
        return responses[calls[0]]
    monkeypatch.setattr(ds.partitioner, "__next__", fake_next)

    # Before iteration, no indexes assigned
    assert ds.training_indexes == []
    assert ds.test_indexes == []
    assert ds.index == 0

    # Create generators but do not iterate yet
    t_gen = ds.training_set()
    s_gen = ds.test_set()
    assert ds.training_indexes == []
    assert ds.test_indexes == []
    assert ds.index == 0

    # Iterate test_set once: should trigger __next__ calls until test index is assigned
    first_test = next(s_gen)
    assert first_test[0] in [0,2,4]
    # index advanced to at least 1
    assert ds.index >= 1
    # test_indexes not empty (contains at least first_test index)
    assert first_test[0] in ds.test_indexes

def test_data_suplier_works_with_non_integer_data_values(monkeypatch):
    random.seed(55)
    np.random.seed(55)
    dataset = ['str', 3.14, None, (1,2), {"a":1}]
    ds = DataSuplier(dataset)

    # patch partitioner.__next__ to simply assign all to test set
    monkeypatch.setattr(ds.partitioner, "__next__", lambda: 1)

    for _ in dataset:
        ds._DataSuplier__next()

    test_items = list(ds.test_set())
    training_items = list(ds.training_set())

    # All should be test set
    assert len(test_items) == len(dataset)
    # training set should be empty
    assert training_items == []
    # Values for test_items should exactly match dataset
    assert [val for _, val in test_items] == dataset

def test_len_dataset_drives_correct_number_of_calls(monkeypatch):
    random.seed(2024)
    np.random.seed(2024)
    dataset = [0]*7
    ds = DataSuplier(dataset)

    call_counter = {"count":0}
    def fake_next():
        call_counter["count"] += 1
        return 0
    monkeypatch.setattr(ds.partitioner, "__next__", fake_next)

    # Exhaust all elements by iterating training set fully
    training_items = list(ds.training_set())

    # The number of calls to partitioner.__next__ should be equal to len(dataset)
    assert call_counter["count"] == len(dataset)
    # training indexes length matches dataset length
    assert len(ds.training_indexes) == len(dataset)
    assert ds.test_indexes == []

def test_partial_iteration_then_again(monkeypatch):
    random.seed(8)
    np.random.seed(8)
    dataset = list(range(4))
    ds = DataSuplier(dataset)

    seq = [0,1,0,1]
    calls = [-1]
    def fake_next():
        calls[0] += 1
        return seq[calls[0]]
    monkeypatch.setattr(ds.partitioner, "__next__", fake_next)

    training_gen = ds.training_set()
    test_gen = ds.test_set()

    first_training_item = next(training_gen)
    assert first_training_item[0] == 0
    first_test_item = next(test_gen)
    assert first_test_item[0] == 1

    second_training_item = next(training_gen)
    assert second_training_item[0] == 2
    second_test_item = next(test_gen)
    assert second_test_item[0] == 3

    # Both exhausted, further calls raises StopIteration
    with pytest.raises(StopIteration):
        next(training_gen)
    with pytest.raises(StopIteration):
        next(test_gen)

def test_training_and_test_set_reproducibility(monkeypatch):
    # Seed actual random generators used by pseudorandom_function and partitioner
    random.seed(42)
    np.random.seed(42)
    dataset = list(range(20))

    ds1 = DataSuplier(dataset)
    ds2 = DataSuplier(dataset)

    # Collect assigned indexes to training and test using full iteration to force partitioning
    t1 = list(ds1.training_set())
    s1 = list(ds1.test_set())

    t2 = list(ds2.training_set())
    s2 = list(ds2.test_set())

    # Both runs should yield same training and test sets by indexes
    assert [i for i, _ in t1] == [i for i, _ in t2]
    assert [i for i, _ in s1] == [i for i, _ in s2]

def test_training_percent_sum_to_one(monkeypatch):
    # training_percent and test_percent sum to 1 internally - test when training_percent >1 or <0
    random.seed(555)
    np.random.seed(555)
    dataset = list(range(5))

    # training_percent > 1 (should still accept)
    ds = DataSuplier(dataset, training_percent=1.5)
    # Evaluate all partition results, should not error
    for _ in dataset:
        ds._DataSuplier__next()
    # lengths should be valid (some can be in training and some in test as partitioner normalizes)
    assert len(ds.training_indexes) + len(ds.test_indexes) == len(dataset)

    # training_percent < 0 (should still accept)
    ds2 = DataSuplier(dataset, training_percent=-0.5)
    for _ in dataset:
        ds2._DataSuplier__next()
    assert len(ds2.training_indexes) + len(ds2.test_indexes) == len(dataset)
