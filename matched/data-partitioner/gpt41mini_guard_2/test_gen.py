import pytest
from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import pseudorandom_function, partitioner

# Simple dummy dataset for testing
dataset = list(range(10))

def test_init_stores_parameters():
    ds = DataSuplier(dataset, training_percent=0.7, partitioning_function=pseudorandom_function)
    assert ds.dataset == dataset
    assert ds.training_percent == 0.7
    assert ds.n_of_elements == len(dataset)
    # partitioner should be iterator with length 10
    assert hasattr(ds.partitioner, '__next__')
    assert ds.test_indexes == []
    assert ds.training_indexes == []
    assert ds.index == 0

def test_partition_lengths_sum_to_dataset_len():
    ds = DataSuplier(dataset, training_percent=0.6, partitioning_function=pseudorandom_function)
    # Exhaust the partition by running through all elements
    # Collect training and test via internal variables
    while ds.index < ds.n_of_elements:
        next(ds.partitioner)
        ds.index += 1  # manually increment index so __next__ would do but here direct
    # Actually this test is just on partitioner: sum of length ratios == dataset length
    assert ds.n_of_elements == 10
    # training + test indexes length should add up to dataset length after partitioning each element once
    # But above we manually iterated partitioner with index increment but didn't update training_indexes/test_indexes
    # So just check during normal __next__ calls
    ds = DataSuplier(dataset, training_percent=0.6, partitioning_function=pseudorandom_function)
    # Exhaust __next__ by calling internal method to assign indexes
    for _ in range(len(dataset)):
        ds._DataSuplier__next()
    total = len(ds.training_indexes) + len(ds.test_indexes)
    assert total == ds.n_of_elements

def test_training_and_test_indexes_disjoint_and_cover_all():
    ds = DataSuplier(dataset, training_percent=0.5, partitioning_function=pseudorandom_function)
    for _ in range(len(dataset)):
        ds._DataSuplier__next()
    train_set = set(ds.training_indexes)
    test_set = set(ds.test_indexes)
    assert train_set.isdisjoint(test_set)
    combined = train_set | test_set
    expected = set(range(len(dataset)))
    assert combined == expected

def test_test_set_generator_yields_test_data_only():
    ds = DataSuplier(dataset, training_percent=0.6, partitioning_function=pseudorandom_function)
    # Run all to partition all
    for _ in range(len(dataset)):
        ds._DataSuplier__next()
    # get all test items from generator
    test_items = list(ds.test_set())
    test_indexes = ds.test_indexes
    # test_items yields tuples (index, dataset[index])
    assert len(test_items) == len(test_indexes)
    for (idx, val), idx_expected in zip(test_items, test_indexes):
        assert idx == idx_expected
        assert val == dataset[idx]

def test_training_set_generator_yields_training_data_only():
    ds = DataSuplier(dataset, training_percent=0.6, partitioning_function=pseudorandom_function)
    for _ in range(len(dataset)):
        ds._DataSuplier__next()
    training_items = list(ds.training_set())
    training_indexes = ds.training_indexes
    assert len(training_items) == len(training_indexes)
    for (idx, val), idx_expected in zip(training_items, training_indexes):
        assert idx == idx_expected
        assert val == dataset[idx]

def test_generators_yield_same_elements_each_call():
    ds = DataSuplier(dataset, training_percent=0.5, partitioning_function=pseudorandom_function)
    for _ in range(len(dataset)):
        ds._DataSuplier__next()
    # Consume test set twice, should yield same results both times
    first_run = list(ds.test_set())
    second_run = list(ds.test_set())
    assert first_run == second_run
    # Consume training set twice: same
    first_train = list(ds.training_set())
    second_train = list(ds.training_set())
    assert first_train == second_train

def test_empty_dataset_results_in_empty_sets():
    empty = []
    ds = DataSuplier(empty)
    assert list(ds.test_set()) == []
    assert list(ds.training_set()) == []

def test_training_percent_zero_results_all_in_test():
    ds = DataSuplier(dataset, training_percent=0.0, partitioning_function=pseudorandom_function)
    for _ in range(len(dataset)):
        ds._DataSuplier__next()
    # training indexes empty, all in test
    assert ds.training_indexes == []
    assert set(ds.test_indexes) == set(range(len(dataset)))
    # Generators correspond
    assert list(ds.training_set()) == []
    test_gen_list = list(ds.test_set())
    assert [idx for idx, val in test_gen_list] == ds.test_indexes

def test_training_percent_one_results_all_in_training():
    ds = DataSuplier(dataset, training_percent=1.0, partitioning_function=pseudorandom_function)
    for _ in range(len(dataset)):
        ds._DataSuplier__next()
    assert ds.test_indexes == []
    assert set(ds.training_indexes) == set(range(len(dataset)))
    assert list(ds.test_set()) == []
    train_gen_list = list(ds.training_set())
    assert [idx for idx, val in train_gen_list] == ds.training_indexes

def test_partitioning_function_is_used():
    # Define a partitioning function that always returns 1 (test)
    def always_test(_):
        return 1
    ds = DataSuplier(dataset, training_percent=0.5, partitioning_function=always_test)
    for _ in range(len(dataset)):
        ds._DataSuplier__next()
    # All indexes should be in test
    assert ds.training_indexes == []
    assert set(ds.test_indexes) == set(range(len(dataset)))

def test_index_advances_properly_after_next_calls():
    ds = DataSuplier(dataset)
    for i in range(len(dataset)):
        ds._DataSuplier__next()
        # After ith __next__, index should be i+1
        assert ds.index == i + 1

def test__yield_set_yields_no_elements_when_no_indexes():
    ds = DataSuplier(dataset)
    # Don't call __next(), so no indexes assigned
    gen = ds._yield_set([])
    # Should not yield anything: generator
    with pytest.raises(StopIteration):
        next(gen)
    # Or simply check list is empty
    assert list(gen) == []

def test__yield_set_yields_elements_correctly_after_partial_partition():
    ds = DataSuplier(dataset, training_percent=0.5)
    # Call __next__ only 5 times to partially partition dataset
    for i in range(5):
        ds._DataSuplier__next()
    # yield_set on current training indexes
    training_items = list(ds._yield_set(ds.training_indexes))
    # training_items length matches training_indexes length
    assert len(training_items) == len(ds.training_indexes)
    for (idx, val), idx_expected in zip(training_items, ds.training_indexes):
        assert idx == idx_expected
        assert val == dataset[idx]

def test__yield_set_waits_for_more_indexes_until_complete():
    # Custom partitioner that yields alternating 0 and 1
    class AlternatingPartitioner:
        def __init__(self):
            self.calls = 0
        def __next__(self):
            val = self.calls % 2
            self.calls += 1
            return val
    part = AlternatingPartitioner()
    ds = DataSuplier(dataset, partitioning_function=lambda _: None)
    ds.partitioner = part
    # Use _yield_set on test_indexes (initially empty)
    gen = ds._yield_set(ds.test_indexes)
    # Before yielding anything, the generator will call __next__ until test_indexes fills
    # We advance generator partially:
    first = next(gen)
    # first should be a tuple with an index and value
    assert isinstance(first, tuple)
    # test_indexes must now have one element
    assert len(ds.test_indexes) == 1
    # The index reported should be the last ds.index-1 because __next__ increments index after assigning
    last_index = ds.index - 1
    assert ds.test_indexes[-1] == last_index
    # The yielded index matches the first element of the returned tuple
    assert first[0] == last_index
    # Now exhaust the generator fully
    rest = list(gen)
    # Together, first + rest covers all test indexes generated by the partitioner up to dataset length
    total_test_indexes = ds.test_indexes
    yielded_indexes = [first[0]] + [x[0] for x in rest]
    assert set(total_test_indexes) == set(yielded_indexes)
    # All yielded data matches the dataset values exactly
    for idx, val in zip(yielded_indexes, [x[1] for x in [first] + rest]):
        assert val == ds.dataset[idx]

def test_pseudorandom_partition_consistency_for_fixed_seed():
    # We know pseudorandom_function is based on hash - it's deterministic,
    # but test deterministic behavior by running two DataSuplier instances
    ds1 = DataSuplier(dataset, training_percent=0.7, partitioning_function=pseudorandom_function)
    ds2 = DataSuplier(dataset, training_percent=0.7, partitioning_function=pseudorandom_function)
    for _ in range(len(dataset)):
        ds1._DataSuplier__next()
        ds2._DataSuplier__next()
    # training indexes should match and test indexes should match
    assert sorted(ds1.training_indexes) == sorted(ds2.training_indexes)
    assert sorted(ds1.test_indexes) == sorted(ds2.test_indexes)

def test_raises_stopiteration_if_next_called_after_all_partitions():
    ds = DataSuplier(dataset)
    for _ in range(len(dataset)):
        ds._DataSuplier__next()
    # The partitioner will have yielded n_of_elements partitions
    with pytest.raises(StopIteration):
        _ = next(ds.partitioner)

