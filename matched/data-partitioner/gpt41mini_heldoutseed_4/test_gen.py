import pytest
from data_partitioner.data_suplier import DataSuplier


def test_init_stores_dataset_and_training_percent():
    data = [1, 2, 3]
    ds = DataSuplier(data, training_percent=0.7)
    assert ds.dataset == data
    assert ds.training_percent == 0.7
    assert ds.n_of_elements == len(data)


def test_init_partitioner_produces_correct_length_partitions():
    data = list(range(10))
    ds = DataSuplier(data, training_percent=0.6)
    # The partitioner is created with the length and [training_percent, 1- training_percent]
    assert ds.partitioner is not None
    # The training + test indexes must be empty initially
    assert ds.test_indexes == []
    assert ds.training_indexes == []
    assert ds.index == 0


def test_next_appends_indexes_correctly_and_increments_index():
    data = list(range(5))

    ds = DataSuplier(data, training_percent=0.5)

    # Call __next__ directly several times
    first_res = ds._DataSuplier__next()
    assert ds.index == 1
    if first_res == 0:
        assert ds.training_indexes == [0]
        assert ds.test_indexes == []
    else:
        assert ds.training_indexes == []
        assert ds.test_indexes == [0]

    second_res = ds._DataSuplier__next()
    assert ds.index == 2
    # We can only assert that the length of indexes is correct
    assert len(ds.training_indexes) + len(ds.test_indexes) == 2


def test_training_set_yields_correct_index_and_data():
    data = ['a', 'b', 'c', 'd']
    ds = DataSuplier(data, training_percent=0.5)

    # Exhaust all data by iterating over training_set generator
    # Because training_set adds indexes only when __next__ is called internally,
    # it will call __next__ repeatedly until all indexes are assigned.
    training_items = list(ds.training_set())

    # training_items is a list of tuples (index, data[index])
    for idx, val in training_items:
        assert idx in ds.training_indexes
        assert val == data[idx]

    # Also verify that no test indexes are yielded here
    for idx, _ in training_items:
        assert idx not in ds.test_indexes


def test_test_set_yields_correct_index_and_data():
    data = ['a', 'b', 'c', 'd', 'e']
    ds = DataSuplier(data, training_percent=0.6)
    test_items = list(ds.test_set())

    for idx, val in test_items:
        assert idx in ds.test_indexes
        assert val == data[idx]

    for idx, _ in test_items:
        assert idx not in ds.training_indexes


def test_training_and_test_sets_together_cover_all_indexes():
    data = list(range(20))
    ds = DataSuplier(data, training_percent=0.7)

    training_items = list(ds.training_set())
    test_items = list(ds.test_set())
    all_indexes = {idx for idx, _ in training_items} | {idx for idx, _ in test_items}
    # All indexes from 0 to n-1 are covered
    assert all(i in all_indexes for i in range(len(data)))
    # No index duplicated between train and test
    assert len(set(i for i, _ in training_items).intersection(set(i for i, _ in test_items))) == 0
    # Total number yielded equals dataset length
    assert len(training_items) + len(test_items) == len(data)


def test_training_and_test_sets_are_generators_and_lazy():
    data = list(range(10))
    ds = DataSuplier(data, training_percent=0.8)

    train_gen = ds.training_set()
    test_gen = ds.test_set()
    assert hasattr(train_gen, "__iter__")
    assert hasattr(test_gen, "__iter__")
    # Pulling just one item does not exhaust all data
    first_train = next(train_gen)
    first_test = next(test_gen)
    assert first_train[0] in ds.training_indexes
    assert first_train[1] == data[first_train[0]]
    assert first_test[0] in ds.test_indexes
    assert first_test[1] == data[first_test[0]]


def test_empty_dataset_yields_no_data():
    data = []
    ds = DataSuplier(data, training_percent=0.5)
    training_items = list(ds.training_set())
    test_items = list(ds.test_set())
    assert training_items == []
    assert test_items == []


def test_training_percent_zero_all_to_test():
    data = list(range(5))
    ds = DataSuplier(data, training_percent=0.0)
    training_items = list(ds.training_set())
    test_items = list(ds.test_set())
    # Training set must be empty
    assert training_items == []
    # Test set must have all items
    test_indexes = [idx for idx, _ in test_items]
    assert sorted(test_indexes) == list(range(len(data)))


def test_training_percent_one_all_to_training():
    data = list(range(5))
    ds = DataSuplier(data, training_percent=1.0)
    training_items = list(ds.training_set())
    test_items = list(ds.test_set())
    # Test set must be empty
    assert test_items == []
    # Training set must have all items
    training_indexes = [idx for idx, _ in training_items]
    assert sorted(training_indexes) == list(range(len(data)))


def test_repeated_calls_to_training_set_yield_same_data():
    data = ['x', 'y', 'z']
    ds = DataSuplier(data, training_percent=0.7)

    # Force evaluation once fully.
    first_call = list(ds.training_set())

    # Resetting index to 0 and clearing indexes to test repeated yields is not supported,
    # So calling training_set again continues from previous state, returns no new data
    second_call = list(ds.training_set())
    assert second_call == [] or second_call == first_call


def test_repeated_calls_to_test_set_yield_same_data():
    data = ['x', 'y', 'z']
    ds = DataSuplier(data, training_percent=0.7)

    first_call = list(ds.test_set())
    second_call = list(ds.test_set())
    assert second_call == [] or second_call == first_call


def test_internal_index_increments_till_end():
    data = range(7)
    ds = DataSuplier(data, training_percent=0.5)
    # exhaust training set to increment index to end
    list(ds.training_set())
    # index should equal length of data now
    assert ds.index == len(data)


def test_yield_set_handles_indexes_larger_than_index():
    data = range(5)
    ds = DataSuplier(data, training_percent=0.5)
    # Manually add an index beyond n_of_elements then call _yield_set
    ds.training_indexes.append(10)  # Out of range
    gen = ds._yield_set(ds.training_indexes)
    # Expect IndexError when this index is accessed in dataset
    with pytest.raises(IndexError):
        next(gen)


def test_yield_set_returns_all_indexes_even_if_out_of_order():
    data = ['a', 'b', 'c']
    ds = DataSuplier(data, training_percent=1.0)
    # Manually shuffle training indexes
    ds.training_indexes = [2, 0, 1]
    ds.index = 3  # Pretend iteration done
    results = list(ds._yield_set(ds.training_indexes))
    # Results indexes must be returned in order of indexes list, not sorted
    assert [idx for idx, _ in results] == [2, 0, 1]


def test_strict_partitioning_function_is_used(monkeypatch):
    data = list(range(5))

    called_args = {}

    def fake_partition(n_elements, weights, func):
        called_args['n'] = n_elements
        called_args['weights'] = weights
        called_args['func'] = func
        # return an iterator that yields alternating 0 and 1
        return iter([0, 1] * (n_elements//2 + 1))

    monkeypatch.setattr('data_partitioner.data_suplier.partitioner', fake_partition)

    ds = DataSuplier(data, training_percent=0.4, partitioning_function=lambda *args: 42)
    # Check that fake_partition was called with correct args
    assert called_args['n'] == len(data)
    assert called_args['weights'] == [0.4, 0.6]
    assert callable(called_args['func'])


def test_yield_set_generates_all_indexes_if_index_lags():
    data = list(range(10))
    ds = DataSuplier(data, training_percent=0.8)
    # forcibly remove indexes, reset index to 0
    ds.training_indexes.clear()
    ds.index = 0
    # call _yield_set with empty indexes, it will call __next to fill indexes
    results = list(ds.training_set())
    assert sorted(i for i, _ in results) == sorted(ds.training_indexes)
    # The index passed ends equal n_of_elements
    assert ds.index == ds.n_of_elements


def test_training_set_and_test_set_do_not_infinite_loop_on_complete_data():
    data = [1]
    ds = DataSuplier(data, training_percent=0.5)

    # Exhaust fully
    training = list(ds.training_set())
    test = list(ds.test_set())

    # The training and test indexes and index match expectations
    assert ds.index == ds.n_of_elements
    assert set(ds.training_indexes).union(set(ds.test_indexes)) == {0}

    # Because only one element and training_percent=0.5 either in training or test
    assert len(training) + len(test) == 1

