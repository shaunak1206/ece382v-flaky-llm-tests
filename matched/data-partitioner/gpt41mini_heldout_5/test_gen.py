import pytest
from data_partitioner.data_suplier import DataSuplier
from data_partitioner import data_suplier
from data_partitioner.partitioner import pseudorandom_function, partitioner


def test_init_with_default_partitioning_function():
    dataset = list(range(10))
    ds = DataSuplier(dataset)
    # Check attributes types and values
    assert ds.dataset == dataset
    assert ds.training_percent == 0.8
    assert ds.n_of_elements == 10
    assert callable(ds.partitioner.__next__)
    # After init no indexes assigned yet
    assert ds.test_indexes == []
    assert ds.training_indexes == []
    assert ds.index == 0


def test_init_with_custom_partitioning_function():
    dataset = list(range(5))
    def custom_pf(x):
        # dummy partitioning function always returning 1 (test set)
        return 1
    ds = DataSuplier(dataset, training_percent=0.6, partitioning_function=custom_pf)
    assert ds.training_percent == 0.6
    # The partitioner uses custom_pf, so all should be test set
    # Advance all elements and check distribution
    for _ in range(len(dataset)):
        ds._DataSuplier__next()
    assert all(i in ds.test_indexes for i in range(len(dataset)))
    assert ds.training_indexes == []


def test_next_and_indexes_assignment():
    dataset = list(range(4))
    ds = DataSuplier(dataset)
    # Call __next__ exactly for n_of_elements
    for i in range(4):
        part_res = ds._DataSuplier__next()
        # part_res should be 0 or 1
        assert part_res in (0, 1)
        # index should increment
        assert ds.index == i + 1
    # The union of training_indexes and test_indexes covers all indices, no repeats
    all_indexes = ds.training_indexes + ds.test_indexes
    assert sorted(all_indexes) == list(range(4))
    assert len(set(all_indexes)) == 4


def test_yield_set_training_and_test():
    dataset = list(range(5))
    ds = DataSuplier(dataset)

    # Advance all partitions by exhausting training_set generator
    training_items = list(ds.training_set())
    # The training indexes in ds should match the keys yielded
    training_indexes_yielded = [idx for idx, _ in training_items]
    assert training_indexes_yielded == ds.training_indexes

    test_items = list(ds.test_set())
    test_indexes_yielded = [idx for idx, _ in test_items]
    assert test_indexes_yielded == ds.test_indexes

    # The yielded data matches dataset values
    for idx, value in training_items:
        assert dataset[idx] == value
    for idx, value in test_items:
        assert dataset[idx] == value

    # All indexes returned among training and test sets are all dataset indices, no duplicates
    yielded_indexes = training_indexes_yielded + test_indexes_yielded
    assert sorted(yielded_indexes) == list(range(len(dataset)))
    assert len(set(yielded_indexes)) == len(dataset)


def test_training_set_generator_yields_nothing_on_empty_dataset():
    ds = DataSuplier([])
    training_items = list(ds.training_set())
    assert training_items == []
    test_items = list(ds.test_set())
    assert test_items == []


def test_yield_set_multiple_generators_progress_separately():
    data = list(range(6))
    ds = DataSuplier(data)
    # Start training set generator and consume 2 items
    gen_train = ds.training_set()
    yielded_train = [next(gen_train) for _ in range(2)]

    # Start test set generator and consume 1 item
    gen_test = ds.test_set()
    yielded_test = [next(gen_test) for _ in range(1)]

    # Continue with training_set generator fully
    yielded_train.extend(list(gen_train))

    # Continue with test set generator fully
    yielded_test.extend(list(gen_test))

    # Check all yielded indexes cover dataset
    all_yielded_indexes = [i for i, _ in yielded_train] + [i for i, _ in yielded_test]
    assert sorted(all_yielded_indexes) == list(range(len(data)))
    # Make sure no duplicates
    assert len(set(all_yielded_indexes)) == len(data)


def test_index_returns_to_zero_when_new_instance_created():
    ds1 = DataSuplier([1, 2, 3])
    for _ in range(3):
        ds1._DataSuplier__next()
    assert ds1.index == 3

    ds2 = DataSuplier([1, 2, 3])
    assert ds2.index == 0
    assert ds2.training_indexes == []
    assert ds2.test_indexes == []

    # ds2 iteration independent of ds1
    ds2._DataSuplier__next()
    assert ds2.index == 1
    assert ds1.index == 3


def test_training_set_and_test_set_exhaustion_order():
    dataset = list(range(10))
    ds = DataSuplier(dataset)

    training_gen = ds.training_set()
    test_gen = ds.test_set()

    # Exhaust training generator fully
    training_results = list(training_gen)
    assert set(i for i, _ in training_results) == set(ds.training_indexes)

    # Exhaust test generator fully
    test_results = list(test_gen)
    assert set(i for i, _ in test_results) == set(ds.test_indexes)

    # Exhausting again yields nothing
    assert list(training_gen) == []
    assert list(test_gen) == []


def test_internal_next_returns_partition_value_consistent_with_indexes():
    dataset = list(range(7))
    ds = DataSuplier(dataset)
    # Collect (index, partition_value)
    results = []
    for _ in range(len(dataset)):
        part_val = ds._DataSuplier__next()
        if ds.index - 1 in ds.training_indexes:
            assert part_val == 0
        else:
            assert part_val == 1
        results.append((ds.index - 1, part_val))
    # Check all indices assigned
    assigned_indices = [idx for idx, _ in results]
    assert sorted(assigned_indices) == list(range(len(dataset)))


def test_partition_ratios_approximate_training_percent():
    # This test is probabilistic due to pseudorandom_function, we accept some tolerance
    dataset = list(range(1000))
    ds = DataSuplier(dataset, training_percent=0.7)
    # Assign all partitions
    for _ in range(len(dataset)):
        ds._DataSuplier__next()
    train_ratio = len(ds.training_indexes) / len(dataset)
    # Check train_ratio near 0.7 (+/- 0.05 tolerance)
    assert 0.65 <= train_ratio <= 0.75
    test_ratio = len(ds.test_indexes) / len(dataset)
    assert 0.25 <= test_ratio <= 0.35


def test_data_suplier_iterating_twice_returns_same_partition():
    # This tests that calling training_set or test_set twice produces same outputs
    dataset = list(range(20))
    ds = DataSuplier(dataset, training_percent=0.8)

    train_list_1 = list(ds.training_set())
    test_list_1 = list(ds.test_set())
    # Reset ds index to zero to simulate new iteration
    ds.index = 0

    train_list_2 = list(ds.training_set())
    test_list_2 = list(ds.test_set())

    # They should be identical (indexes and values)
    assert train_list_1 == train_list_2
    assert test_list_1 == test_list_2


def test_training_and_test_sets_do_not_intersect():
    dataset = list(range(30))
    ds = DataSuplier(dataset)
    # Assign all
    for _ in range(len(dataset)):
        ds._DataSuplier__next()
    # Confirm no index appears in both training and test indexes
    intersection = set(ds.training_indexes).intersection(set(ds.test_indexes))
    assert intersection == set()
    # Combined covers all dataset indices
    combined = set(ds.training_indexes).union(set(ds.test_indexes))
    assert combined == set(range(len(dataset)))


def test_training_percent_zero_all_test():
    dataset = list(range(10))
    ds = DataSuplier(dataset, training_percent=0.0)
    for _ in range(len(dataset)):
        ds._DataSuplier__next()
    # All indexes must be in test_indexes
    assert ds.training_indexes == []
    assert sorted(ds.test_indexes) == list(range(len(dataset)))


def test_training_percent_one_all_training():
    dataset = list(range(10))
    ds = DataSuplier(dataset, training_percent=1.0)
    for _ in range(len(dataset)):
        ds._DataSuplier__next()
    # All indexes in training_indexes
    assert sorted(ds.training_indexes) == list(range(len(dataset)))
    assert ds.test_indexes == []


def test_yield_set_generator_does_not_stop_early():
    # Create dataset that delays yields by consuming both sets in alternating calls
    dataset = list(range(5))
    ds = DataSuplier(dataset)

    training_gen = ds.training_set()
    test_gen = ds.test_set()

    # Consume alternately from both generators until StopIteration
    training_results = []
    test_results = []
    while True:
        try:
            training_results.append(next(training_gen))
        except StopIteration:
            break
        try:
            test_results.append(next(test_gen))
        except StopIteration:
            break

    # The yielded indexes cover all dataset indexes
    all_yielded = [i for i, _ in training_results] + [i for i, _ in test_results]
    assert sorted(all_yielded) == list(range(len(dataset)))


def test_yield_set_generator_with_partial_index_lists():
    # Directly test _yield_set with partial index lists
    dataset = list(range(3))
    ds = DataSuplier(dataset)
    # Manually set training_indexes partially and index accordingly
    ds.training_indexes = [0, 2]
    ds.test_indexes = [1]
    ds.index = 3  # simulate all assigned

    # _yield_set with training_indexes yields 2 items
    training_items = list(ds._yield_set(ds.training_indexes))
    assert training_items == [(0, 0), (2, 2)]

    test_items = list(ds._yield_set(ds.test_indexes))
    assert test_items == [(1, 1)]


def test_yield_set_generator_handles_new_indexes_appended_on_the_fly():
    # This test simulates that _yield_set keeps yielding new indexes appended during iteration
    dataset = list(range(3))
    ds = DataSuplier(dataset)

    # Initially no indexes
    ds.training_indexes = []
    ds.index = 0

    # Patch __next__ to add indexes to training_indexes when called
    orig_next = ds._DataSuplier__next

    def side_effect_next():
        val = orig_next()
        # Assign all indices to training_indexes
        return val

    ds._DataSuplier__next = side_effect_next

    gen = ds._yield_set(ds.training_indexes)

    results = []
    for idx, val in gen:
        results.append((idx, val))
        if len(results) == 3:
            break

    # Check all indexes yielded
    yielded_indices = [i[0] for i in results]
    assert sorted(yielded_indices) == list(range(3))


def test_len_zero_dataset_yielding_generators_produces_nothing():
    ds = DataSuplier([])
    training_gen = ds.training_set()
    test_gen = ds.test_set()
    assert list(training_gen) == []
    assert list(test_gen) == []

