import pytest
from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import partitioner, pseudorandom_function


def test_init_default_values():
    data = list(range(10))
    ds = DataSuplier(data)
    assert ds.dataset == data
    assert ds.training_percent == 0.8
    assert ds.n_of_elements == 10
    # partitioner should be created with correct args
    assert hasattr(ds.partitioner, "__next__")
    assert ds.test_indexes == []
    assert ds.training_indexes == []
    assert ds.index == 0


def test_init_with_custom_training_percent():
    data = list(range(5))
    ds = DataSuplier(data, training_percent=0.6)
    assert ds.training_percent == 0.6
    assert ds.n_of_elements == 5


def test_init_with_custom_partitioning_function():
    data = list(range(3))

    def all_train(i):
        return 0

    ds = DataSuplier(data, partitioning_function=all_train)
    # Simulate partitioning function returns always 0 -> all training
    for _ in range(len(data)):
        next(ds.partitioner)
    assert all(p == 0 for p in ds.partitioner)


def test_training_and_test_sets_yield_correct_elements():
    data = list("abcdef")
    ds = DataSuplier(data)
    # Exhaust the _yield_set by iterating through training_set and test_set
    training_items = list(ds.training_set())
    test_items = list(ds.test_set())

    # The yielded indices are subsets composed from full dataset indices
    all_yielded_indices = {i for i, _ in training_items + test_items}
    assert all_yielded_indices == set(range(len(data)))

    # No index should be both in training and test
    training_indices = {i for i, _ in training_items}
    test_indices = {i for i, _ in test_items}
    assert training_indices.isdisjoint(test_indices)

    # The values yielded correspond to original data
    for i, val in training_items + test_items:
        assert data[i] == val


def test_training_set_and_test_set_are_generators():
    data = list(range(5))
    ds = DataSuplier(data)
    train_gen = ds.training_set()
    test_gen = ds.test_set()
    assert hasattr(train_gen, "__iter__")
    assert hasattr(train_gen, "__next__")
    assert hasattr(test_gen, "__iter__")
    assert hasattr(test_gen, "__next__")


def test_yield_set_empty_dataset():
    ds = DataSuplier([])
    train = list(ds.training_set())
    test = list(ds.test_set())
    assert train == []
    assert test == []


def test_yield_set_correct_order():
    data = list("xyz")
    # Use a fake partitioner that assigns first element to train, second to test, etc.
    seq = [0, 1, 0]

    def fake_partitioning_function(_):
        # Not used directly
        return 0

    class FakePartitioner:
        def __init__(self):
            self.calls = 0

        def __next__(self):
            if self.calls >= len(seq):
                raise StopIteration
            v = seq[self.calls]
            self.calls += 1
            return v

    ds = DataSuplier(data, partitioning_function=fake_partitioning_function)
    ds.partitioner = FakePartitioner()

    train_items = list(ds.training_set())
    test_items = list(ds.test_set())

    # training indices should be [0, 2]
    assert [i for i, _ in train_items] == [0, 2]
    # test indices should be [1]
    assert [i for i, _ in test_items] == [1]


def test_training_percent_zero_means_all_test():
    data = list(range(4))
    ds = DataSuplier(data, training_percent=0.0)
    training = list(ds.training_set())
    test = list(ds.test_set())
    assert training == []
    # All data should be in test set
    test_indices = [idx for idx, _ in test]
    assert sorted(test_indices) == list(range(len(data)))


def test_training_percent_one_means_all_training():
    data = list(range(4))
    ds = DataSuplier(data, training_percent=1.0)
    training = list(ds.training_set())
    test = list(ds.test_set())
    # All data should be in training set
    training_indices = [idx for idx, _ in training]
    assert sorted(training_indices) == list(range(len(data)))
    assert test == []


def test_multiple_calls_to_training_and_test_set_continue_iteration():
    data = list("abcde")
    ds = DataSuplier(data)

    # Get first training item
    train_gen = ds.training_set()
    first_train = next(train_gen)

    # Get first test item from new generator call (new generator)
    test_gen = ds.test_set()
    first_test = next(test_gen)

    # Continue consuming previous training generator
    rest_train = list(train_gen)

    # Continue consuming previous test generator
    rest_test = list(test_gen)

    # Verify combined indices are unique and cover the data
    training_indices = [first_train[0]] + [i for i, _ in rest_train]
    test_indices = [first_test[0]] + [i for i, _ in rest_test]
    assert set(training_indices).isdisjoint(set(test_indices))
    assert set(training_indices + test_indices) == set(range(len(data)))


def test_yield_set_yields_all_elements_before_stop():
    data = list(range(3))

    ds = DataSuplier(data)

    training_items = list(ds.training_set())
    test_items = list(ds.test_set())

    # All indices must appear exactly once in either training or test
    appeared = [idx for idx, _ in training_items + test_items]
    assert sorted(appeared) == list(range(len(data)))


def test_index_moves_forward_correctly():
    data = list(range(4))
    ds = DataSuplier(data)

    # Internal index should increment on calls to __next__
    initial_index = ds.index
    ds._DataSuplier__next()  # call private method
    assert ds.index == initial_index + 1
    ds._DataSuplier__next()
    assert ds.index == initial_index + 2


def test_test_and_training_indexes_filled_correctly():
    data = list(range(10))
    ds = DataSuplier(data)
    # Exhaust the generator to populate indexes
    list(ds.training_set())
    list(ds.test_set())

    # The union should cover all indexes from 0 to n-1
    indexes_union = set(ds.training_indexes) | set(ds.test_indexes)
    assert indexes_union == set(range(len(data)))

    # No overlap between training and test indexes
    assert set(ds.training_indexes).isdisjoint(set(ds.test_indexes))


def test__yield_set_returns_until_all_indexes_returned():
    data = list(range(3))

    # Use a partitioner that always returns 0 (all training)
    def always_train(_):
        return 0

    ds = DataSuplier(data, partitioning_function=always_train)
    training_list = list(ds.training_set())
    test_list = list(ds.test_set())
    # test_list should be empty and training_list should have all elements
    assert len(training_list) == len(data)
    assert test_list == []


def test_yield_set_works_if_indexes_populated_before():
    data = list(range(5))
    ds = DataSuplier(data)
    # Manually populate indexes and index counter
    ds.training_indexes = [0, 2]
    ds.test_indexes = [1, 3]
    ds.index = 4

    # Now yield training_set and test_set, remaining idx=4 should get assigned
    train_items = list(ds.training_set())
    test_items = list(ds.test_set())

    # The last element index=4 should be assigned by __next__
    all_indices = {i for i, _ in train_items + test_items}
    assert all_indices == {0, 1, 2, 3, 4}


def test_empty_index_lists_and_nonzero_index_still_yields():
    data = list(range(3))
    ds = DataSuplier(data)
    # Set index beyond dataset, but empty indexes lists
    ds.index = len(data)
    ds.training_indexes = []
    ds.test_indexes = []

    # Should yield nothing since index at end
    assert list(ds.training_set()) == []
    assert list(ds.test_set()) == []


def test_partitioning_function_affects_sets():
    data = list("abc")

    def partition_func(i):
        # Put only even indices in train, odd in test
        return 0 if i % 2 == 0 else 1

    ds = DataSuplier(data, partitioning_function=partition_func)
    train_indices = [i for i, _ in ds.training_set()]
    test_indices = [i for i, _ in ds.test_set()]

    assert all(i % 2 == 0 for i in train_indices)
    assert all(i % 2 == 1 for i in test_indices)
    assert sorted(train_indices + test_indices) == list(range(len(data)))


def test_yield_set_handles_stop_iteration_from_partitioner(monkeypatch):
    data = list(range(2))
    ds = DataSuplier(data)

    # Patch partitioner's __next__ to raise StopIteration immediately
    def raise_stop():
        raise StopIteration()

    ds.partitioner.__next__ = raise_stop

    # In this case, _yield_set should not hang and yield nothing
    assert list(ds.training_set()) == []
    assert list(ds.test_set()) == []

