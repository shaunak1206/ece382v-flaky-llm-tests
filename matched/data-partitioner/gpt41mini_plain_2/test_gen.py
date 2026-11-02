import pytest
from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import pseudorandom_function

def test_init_sets_attributes_correctly():
    data = [10, 20, 30]
    ds = DataSuplier(data, training_percent=0.7)
    assert ds.dataset == data
    assert ds.training_percent == 0.7
    assert ds.n_of_elements == 3
    assert callable(ds.partitioner.__next__)
    assert ds.test_indexes == []
    assert ds.training_indexes == []
    assert ds.index == 0

def test_init_default_partitioning_function():
    data = list(range(5))
    ds = DataSuplier(data)
    # The default partitioning_function is pseudorandom_function
    # We check that partitioner used pseudorandom_function by ensuring returns are 0 or 1
    vals = []
    for _ in range(len(data)):
        vals.append(ds.partitioner.__next__())
    assert all(v in (0,1) for v in vals)

def test___next_adds_indexes_correctly_to_training_and_test():
    data = list(range(10))
    ds = DataSuplier(data, training_percent=0.5)
    # Manually call __next to populate training/test indexes
    for _ in range(len(data)):
        val = ds._DataSuplier__next()
        assert val in (0,1)
    # All indexes should be assigned exactly once
    combined = sorted(ds.training_indexes + ds.test_indexes)
    assert combined == list(range(len(data)))
    # training_indexes and test_indexes should not overlap
    assert set(ds.training_indexes).isdisjoint(set(ds.test_indexes))

def test___next_increments_index():
    data = [1, 2]
    ds = DataSuplier(data)
    assert ds.index == 0
    ds._DataSuplier__next()
    assert ds.index == 1
    ds._DataSuplier__next()
    assert ds.index == 2

def test_training_set_yields_expected_values():
    data = [100, 200, 300, 400]
    ds = DataSuplier(data, training_percent=0.5)
    # Run all __next to partition indexes
    for _ in range(len(data)):
        ds._DataSuplier__next()
    training_gen = ds.training_set()
    training_items = list(training_gen)
    # Each yielded item is tuple(index, dataset[index])
    for idx, val in training_items:
        assert val == data[idx]
    # All training indexes should be present
    assert sorted(idx for idx, _ in training_items) == sorted(ds.training_indexes)

def test_test_set_yields_expected_values():
    data = ['a', 'b', 'c', 'd', 'e']
    ds = DataSuplier(data, training_percent=0.6)
    for _ in range(len(data)):
        ds._DataSuplier__next()
    test_gen = ds.test_set()
    test_items = list(test_gen)
    for idx, val in test_items:
        assert val == data[idx]
    assert sorted(idx for idx, _ in test_items) == sorted(ds.test_indexes)

def test_yield_set_generates_all_indexes_even_if_index_not_at_end():
    data = [1, 2, 3]
    ds = DataSuplier(data, training_percent=0.5)
    # Call __next fewer times than dataset length
    ds._DataSuplier__next()
    # Not all indexes assigned yet
    # Collect from training_set generator - it should internally advance __next until done
    training = list(ds.training_set())
    # training indexes from this should now include newly assigned indexes
    assert all(0 <= idx < len(data) for idx, _ in training)

def test_yield_set_works_when_indexes_empty_but_index_at_end():
    data = [1]
    ds = DataSuplier(data, training_percent=1.0)  # all training, no test
    # Force index to end
    ds.index = len(data)
    # indexes lists empty
    ds.training_indexes = []
    # training_set yields nothing as no indexes
    assert list(ds.training_set()) == []
    # test_set yields nothing as no test indexes
    assert list(ds.test_set()) == []

def test_multiple_calls_to_training_set_return_fresh_generators():
    data = [10, 20, 30]
    ds = DataSuplier(data)
    for _ in range(len(data)):
        ds._DataSuplier__next()
    gen1 = ds.training_set()
    gen2 = ds.training_set()
    list1 = list(gen1)
    list2 = list(gen2)
    assert list1 == list2

def test_multiple_calls_to_test_set_return_fresh_generators():
    data = [7, 8, 9, 10]
    ds = DataSuplier(data)
    for _ in range(len(data)):
        ds._DataSuplier__next()
    gen1 = ds.test_set()
    gen2 = ds.test_set()
    list1 = list(gen1)
    list2 = list(gen2)
    assert list1 == list2

def test_training_set_and_test_set_combine_to_full_dataset():
    data = list(range(20))
    ds = DataSuplier(data, training_percent=0.75)
    for _ in range(len(data)):
        ds._DataSuplier__next()
    training = list(ds.training_set())
    test = list(ds.test_set())
    combined_indexes = sorted(idx for idx,_ in training) + sorted(idx for idx,_ in test)
    combined_indexes_sorted = sorted(combined_indexes)
    assert combined_indexes_sorted == list(range(len(data)))
    combined_values = [v for _,v in training] + [v for _,v in test]
    combined_values_sorted = [data[idx] for idx in combined_indexes_sorted]
    # Values from original data covered without loss
    assert sorted(combined_values) == sorted(data)

def test_training_percent_zero_all_go_to_test():
    data = [1,2,3]
    ds = DataSuplier(data, training_percent=0.0)
    for _ in range(len(data)):
        ds._DataSuplier__next()
    assert ds.training_indexes == []
    assert sorted(ds.test_indexes) == list(range(len(data)))

def test_training_percent_one_all_go_to_training():
    data = [1,2,3,4]
    ds = DataSuplier(data, training_percent=1.0)
    for _ in range(len(data)):
        ds._DataSuplier__next()
    assert sorted(ds.training_indexes) == list(range(len(data)))
    assert ds.test_indexes == []

def test_empty_dataset_results_in_no_indexes():
    data = []
    ds = DataSuplier(data)
    # index should be 0, no elements exist
    assert ds.n_of_elements == 0
    # training_set and test_set yield no elements
    assert list(ds.training_set()) == []
    assert list(ds.test_set()) == []

def test_partitioning_function_can_be_custom():
    data = list(range(4))
    # Define a partitioning function that always returns 0 (training)
    def always_train(x):
        return 0
    ds = DataSuplier(data, partitioning_function=always_train)
    for _ in range(len(data)):
        ds._DataSuplier__next()
    assert ds.test_indexes == []
    assert sorted(ds.training_indexes) == list(range(len(data)))

def test_yield_set_waits_until_indexes_append_if_initially_empty():
    data = [1,2,3]
    ds = DataSuplier(data, training_percent=0.5)
    # Steal partitioner and intercept next calls to control indices returned (simulate all zeros so all training)
    # We patch ds.partitioner to an iterator that yields 0 three times then StopIteration
    class FakePartitioner:
        def __init__(self):
            self.count = 0
        def __next__(self):
            if self.count < 3:
                self.count += 1
                return 0
            raise StopIteration()
    ds.partitioner = FakePartitioner()
    # Call training_set generator and consume all
    results = list(ds.training_set())
    # Should yield 3 elements all in training_indexes
    assert sorted(ds.training_indexes) == [0,1,2]
    assert ds.test_indexes == []
    assert len(results) == 3
    for idx, val in results:
        assert val == data[idx]

def test_yield_set_raises_stopiteration_when_consumed():
    data = [0, 1]
    ds = DataSuplier(data, training_percent=1.0)
    for _ in range(len(data)):
        ds._DataSuplier__next()
    gen = ds.training_set()
    # exhaust generator
    list(gen)
    # calling next again on exhausted generator raises StopIteration
    with pytest.raises(StopIteration):
        next(gen)

def test_test_set_returns_generator():
    ds = DataSuplier([1,2,3])
    gen = ds.test_set()
    assert hasattr(gen, '__iter__')
    assert hasattr(gen, '__next__')

def test_training_set_returns_generator():
    ds = DataSuplier([1,2,3])
    gen = ds.training_set()
    assert hasattr(gen, '__iter__')
    assert hasattr(gen, '__next__')
