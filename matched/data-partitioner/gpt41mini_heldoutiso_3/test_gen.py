import pytest
from data_partitioner.data_suplier import DataSuplier


def test_init_sets_attributes_correctly():
    data = [10, 20, 30, 40, 50]
    ds = DataSuplier(data, training_percent=0.7)
    assert ds.dataset == data
    assert ds.training_percent == 0.7
    assert ds.n_of_elements == 5
    # training + test length ratios close to training_percent and 1-training_percent
    total_len = len(data)
    partition_sum = sum(ds.partitioner._ratios)
    assert abs(partition_sum - 1) < 1e-6
    assert ds.index == 0
    assert ds.test_indexes == []
    assert ds.training_indexes == []


def test_iteration_partition_indexes_accumulate_correctly():
    data = list(range(10))
    ds = DataSuplier(data, training_percent=0.5)
    # exhaust all elements via internal __next__
    for _ in range(len(data)):
        res = ds.__next()
        assert res in (0, 1)
    assert set(ds.training_indexes).union(set(ds.test_indexes)) == set(range(len(data)))
    assert len(ds.training_indexes) + len(ds.test_indexes) == len(data)
    assert ds.index == len(data)


def test_training_set_yields_all_training_examples():
    data = [100, 200, 300, 400]
    ds = DataSuplier(data, training_percent=0.75)
    # exhaust all elements to fill training and test indexes
    for _ in range(len(data)):
        ds.__next__()
    training = list(ds.training_set())
    # all indexes yielded are in training_indexes
    assert all(idx in ds.training_indexes for idx, _ in training)
    # values match the dataset values at these indices
    for idx, val in training:
        assert val == data[idx]
    # all training indexes yielded once
    yielded_indexes = [idx for idx, _ in training]
    assert sorted(yielded_indexes) == sorted(ds.training_indexes)


def test_test_set_yields_all_test_examples():
    data = ["a", "b", "c", "d", "e"]
    ds = DataSuplier(data, training_percent=0.6)
    for _ in range(len(data)):
        ds.__next__()
    test_set = list(ds.test_set())
    # all indexes yielded are in test_indexes
    assert all(i in ds.test_indexes for i, _ in test_set)
    # values match dataset at those indices
    for i, val in test_set:
        assert val == data[i]
    # all test indexes yielded exactly once
    yielded = [i for i, _ in test_set]
    assert sorted(yielded) == sorted(ds.test_indexes)


def test_yield_set_yields_items_lazily_and_completely():
    data = list(range(8))
    ds = DataSuplier(data, training_percent=0.5)
    # do not call __next__ first; call training_set generator directly and convert to list
    training_items = list(ds.training_set())
    test_items = list(ds.test_set())

    all_indexes = set(idx for idx, _ in training_items + test_items)
    assert all_indexes == set(range(len(data)))
    # training_items indexes equal ds.training_indexes
    # test_items indexes equal ds.test_indexes
    assert sorted([idx for idx, _ in training_items]) == sorted(ds.training_indexes)
    assert sorted([idx for idx, _ in test_items]) == sorted(ds.test_indexes)
    # values match
    for idx, val in training_items + test_items:
        assert val == data[idx]


def test_empty_dataset_yields_nothing():
    data = []
    ds = DataSuplier(data)
    assert list(ds.training_set()) == []
    assert list(ds.test_set()) == []
    # index should be zero
    assert ds.index == 0
    # training and test indexes empty
    assert ds.training_indexes == []
    assert ds.test_indexes == []


def test_training_percent_zero_all_goes_to_test():
    data = list("abcde")
    ds = DataSuplier(data, training_percent=0)
    # should mark none as training, all as test
    for _ in data:
        ds.__next__()
    assert ds.training_indexes == []
    assert sorted(ds.test_indexes) == list(range(len(data)))
    training = list(ds.training_set())
    test = list(ds.test_set())
    assert training == []
    assert sorted(i for i, _ in test) == list(range(len(data)))


def test_training_percent_one_all_goes_to_training():
    data = [1, 2, 3]
    ds = DataSuplier(data, training_percent=1)
    for _ in data:
        ds.__next__()
    assert sorted(ds.training_indexes) == list(range(len(data)))
    assert ds.test_indexes == []
    training = list(ds.training_set())
    test = list(ds.test_set())
    assert sorted(i for i, _ in training) == list(range(len(data)))
    assert test == []


def test_next_raises_stop_iteration_after_exhaustion():
    data = [0, 1]
    ds = DataSuplier(data)
    # Call __next__ exactly len(data) times - should work
    for _ in range(len(data)):
        res = ds.__next__()
        assert res in (0, 1)
    # Now calling __next__ again should raise StopIteration (coming from partitioner.__next__)
    with pytest.raises(StopIteration):
        ds.__next__()


def test_multiple_calls_to_training_set_resume_iteration_properly():
    data = list(range(5))
    ds = DataSuplier(data, training_percent=0.6)
    # get half items from generator
    gen = ds.training_set()
    first_two = [next(gen) for _ in range(2)]
    # Call next() increments index internally to get more indices appended
    # Getting rest of items
    rest = list(gen)
    # Combined all training indexes yielded
    all_yielded = first_two + rest
    yielded_indexes = [idx for idx, _ in all_yielded]
    # yield set yields only indexes in training_indexes
    assert all(idx in ds.training_indexes for idx in yielded_indexes)
    # all yielded indexes are unique
    assert len(yielded_indexes) == len(set(yielded_indexes))


def test_multiple_calls_to_test_set_resume_iteration_properly():
    data = list(range(10))
    ds = DataSuplier(data, training_percent=0.7)
    gen = ds.test_set()
    first_item = next(gen, None)
    assert first_item is None or first_item[0] in ds.test_indexes or ds.test_indexes == []
    rest = list(gen)
    all_yielded_indexes = [first_item[0]] if first_item else []
    all_yielded_indexes += [idx for idx, _ in rest]
    # all yielded indexes are from test indexes
    assert all(idx in ds.test_indexes for idx in all_yielded_indexes)


def test_yield_set_yields_no_duplicates():
    data = list(range(20))
    ds = DataSuplier(data, training_percent=0.5)
    # Exhaust partitioning
    for _ in range(len(data)):
        ds.__next__()
    training = list(ds.training_set())
    test = list(ds.test_set())
    training_indexes = [idx for idx, _ in training]
    test_indexes = [idx for idx, _ in test]
    # No duplicates in yielded sets
    assert len(training_indexes) == len(set(training_indexes))
    assert len(test_indexes) == len(set(test_indexes))
    # Training and test indexes sets are disjoint
    assert set(training_indexes).isdisjoint(set(test_indexes))


def test_partitioning_function_can_be_customized(monkeypatch):
    data = list(range(4))

    def fake_partition_function(i):
        # Always assign first two to training (0), last two to test (1)
        if i < 2:
            return 0
        else:
            return 1

    from data_partitioner.partitioner import partitioner

    # patch partitioner to use a generator using fake_partition_function
    class DummyPartitioner:
        def __init__(self, n, ratios, func):
            self.n = n
            self.func = func
            self.i = 0

        def __next__(self):
            if self.i >= self.n:
                raise StopIteration
            result = self.func(self.i)
            self.i += 1
            return result

    monkeypatch.setattr("data_partitioner.data_suplier.partitioner", lambda n, ratios, func: DummyPartitioner(n, ratios, func))

    ds = DataSuplier(data, training_percent=0.5, partitioning_function=fake_partition_function)

    # Advance all iterations
    for _ in data:
        ds.__next__()

    # training_indexes should be [0, 1]
    assert ds.training_indexes == [0, 1]
    assert ds.test_indexes == [2, 3]
    training_data = list(ds.training_set())
    test_data = list(ds.test_set())
    assert [idx for idx, _ in training_data] == ds.training_indexes
    assert [idx for idx, _ in test_data] == ds.test_indexes


def test_training_and_test_iterators_on_partial_iteration():
    data = list(range(6))
    ds = DataSuplier(data, training_percent=0.5)

    # Consume only 3 partitions by calling __next__ manually
    for _ in range(3):
        ds.__next__()

    # training and test indexes length sum to 3
    assert len(ds.training_indexes) + len(ds.test_indexes) == 3

    # training_set and test_set generators should still yield all indices eventually
    training_result = list(ds.training_set())
    test_result = list(ds.test_set())

    # total distinct indexes yielded by both sets equal dataset length
    all_yielded_indexes = {idx for idx, _ in training_result + test_result}
    assert all_yielded_indexes == set(range(len(data)))

    # The yielded indexes should match training_indexes and test_indexes respectively
    assert sorted(idx for idx, _ in training_result) == sorted(ds.training_indexes)
    assert sorted(idx for idx, _ in test_result) == sorted(ds.test_indexes)


def test_training_set_generator_without_any_prior_iteration():
    data = ["x", "y", "z"]
    ds = DataSuplier(data, training_percent=0.5)
    # Directly call generator without calling __next__
    training_items = list(ds.training_set())
    # All data should be yielded for training indexes
    assert sorted(idx for idx, _ in training_items) == sorted(ds.training_indexes)
    assert all(ds.dataset[idx] == val for idx, val in training_items)


def test_test_set_generator_without_any_prior_iteration():
    data = ["foo", "bar"]
    ds = DataSuplier(data, training_percent=0.5)
    test_items = list(ds.test_set())
    assert sorted(idx for idx, _ in test_items) == sorted(ds.test_indexes)
    assert all(ds.dataset[idx] == val for idx, val in test_items)
