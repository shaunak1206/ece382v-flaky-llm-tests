import pytest
from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import pseudorandom_function


def test_initialization_with_default_partitioning_function():
    dataset = list(range(10))
    ds = DataSuplier(dataset)
    assert ds.dataset == dataset
    assert ds.training_percent == 0.8
    assert ds.n_of_elements == 10
    # training + test proportions should sum to 1
    proportions = [ds.training_percent, 1 - ds.training_percent]
    assert abs(sum(proportions) - 1) < 1e-8
    # training and test indexes are empty, index is 0
    assert ds.training_indexes == []
    assert ds.test_indexes == []
    assert ds.index == 0


def test_initialization_with_custom_training_percent():
    dataset = ["a", "b", "c", "d"]
    ds = DataSuplier(dataset, training_percent=0.5)
    assert ds.training_percent == 0.5
    assert ds.n_of_elements == 4


def test_custom_partitioning_function_is_used():
    calls = []

    def custom_pf(x):
        calls.append(x)
        # Just alternate 0 and 1
        return x % 2

    dataset = list(range(5))
    ds = DataSuplier(dataset, training_percent=0.6, partitioning_function=custom_pf)

    # We expect that calls equal n_of_elements as partitioner calls function once per element
    # The partitioner is an iterator, so trigger entire sequence by exhausting sets
    list(ds.training_set())
    list(ds.test_set())

    # The calls list should contain all indices from 0 to 4
    assert calls == list(range(5))


def test_yield_training_and_test_sets_non_empty_dataset():
    dataset = list("abcdefghij")  # length 10
    ds = DataSuplier(dataset, training_percent=0.7)

    training = list(ds.training_set())
    test = list(ds.test_set())

    # All indices should be accounted for exactly once across training and test sets
    all_indices = {i for i, _ in training} | {i for i, _ in test}
    assert all_indices == set(range(len(dataset)))

    # No overlap in indices
    training_indices = {i for i, _ in training}
    test_indices = {i for i, _ in test}
    assert training_indices.isdisjoint(test_indices)

    # Items should correspond to dataset content
    for i, v in training:
        assert dataset[i] == v
    for i, v in test:
        assert dataset[i] == v


def test_training_set_exhaustion_before_test_set_starts():
    dataset = list(range(5))

    ds = DataSuplier(dataset)
    # Exhaust training set generator
    train = list(ds.training_set())
    # Now get test set
    test = list(ds.test_set())

    # Combined indices cover all elements
    combined = {i for i, _ in train} | {i for i, _ in test}
    assert combined == set(range(len(dataset)))


def test_test_set_exhaustion_before_training_set():
    dataset = list(range(5))

    ds = DataSuplier(dataset)
    # Exhaust test set first
    test = list(ds.test_set())
    # Then training set
    train = list(ds.training_set())

    combined = {i for i, _ in train} | {i for i, _ in test}
    assert combined == set(range(len(dataset)))


def test_empty_dataset_yields_nothing():
    ds = DataSuplier([])
    training = list(ds.training_set())
    test = list(ds.test_set())
    assert training == []
    assert test == []


def test_yield_set_yields_correct_tuples():
    dataset = ["x", "y"]
    ds = DataSuplier(dataset, training_percent=1.0)
    # We have 2 elements, all should be training because partition is (1.0, 0.0)
    results = list(ds.training_set())
    assert results == [(0, "x"), (1, "y")]

    test_results = list(ds.test_set())
    # There should be none in test set
    assert test_results == []


def test_yield_set_with_all_test_data():
    dataset = ["x", "y", "z"]
    ds = DataSuplier(dataset, training_percent=0.0)
    # All test, none training
    test_results = list(ds.test_set())
    assert {idx for idx, _ in test_results} == set(range(3))

    training_results = list(ds.training_set())
    assert training_results == []


def test_internal_state_advance_after_yielding():
    dataset = list(range(4))
    ds = DataSuplier(dataset, training_percent=0.5)

    # Initially index is 0
    assert ds.index == 0
    # When we get one element from training_set or test_set, index should advance
    gen = ds.training_set()
    first_item = next(gen)
    assert ds.index > 0
    # Subsequent calls increase index
    second_item = next(gen)
    assert ds.index > 1


def test_training_and_test_indexes_populated_correctly():
    dataset = list(range(10))
    ds = DataSuplier(dataset, training_percent=0.5)

    # Exhaust both sets to trigger all __next__ calls
    list(ds.training_set())
    list(ds.test_set())

    total_indexes = set(ds.training_indexes) | set(ds.test_indexes)
    # Should contain all indices exactly once
    assert total_indexes == set(range(len(dataset)))

    # No overlaps
    assert set(ds.training_indexes).isdisjoint(set(ds.test_indexes))


def test__yield_set_returns_all_indices_even_if_index_advances_during_iteration():
    dataset = list(range(3))
    ds = DataSuplier(dataset, training_percent=0.5)

    result = list(ds._yield_set(ds.training_indexes))
    # Should yield all training indexes and their values
    for idx, val in result:
        assert dataset[idx] == val


def test__yield_set_yields_no_elements_if_indexes_empty():
    dataset = list(range(3))
    ds = DataSuplier(dataset)
    result = list(ds._yield_set([]))
    assert result == []


def test_test_set_generator_next_and_stopiteration_behavior():
    dataset = list(range(3))
    ds = DataSuplier(dataset)
    gen = ds.test_set()
    collected = []
    try:
        while True:
            collected.append(next(gen))
    except StopIteration:
        pass
    # All test indexes yielded with data
    test_idxs = set(idx for idx, _ in collected)
    assert test_idxs == set(ds.test_indexes)


def test_training_set_generator_next_and_stopiteration_behavior():
    dataset = list(range(3))
    ds = DataSuplier(dataset)
    gen = ds.training_set()
    collected = []
    try:
        while True:
            collected.append(next(gen))
    except StopIteration:
        pass
    train_idxs = set(idx for idx, _ in collected)
    assert train_idxs == set(ds.training_indexes)


def test_index_advances_to_n_of_elements_after_iteration():
    dataset = list(range(5))
    ds = DataSuplier(dataset)
    # Exhaust all sets
    list(ds.training_set())
    list(ds.test_set())
    # After exhaustion, index should equal n_of_elements
    assert ds.index == ds.n_of_elements


def test_partitioning_stability_with_same_training_percent():
    dataset = list(range(10))
    ds1 = DataSuplier(dataset, training_percent=0.6)
    ds2 = DataSuplier(dataset, training_percent=0.6)

    # Exhaust both data supliers
    train1 = list(ds1.training_set())
    test1 = list(ds1.test_set())

    train2 = list(ds2.training_set())
    test2 = list(ds2.test_set())

    # Assuming pseudorandom_function is stable, indexes assigned to sets should be equal between runs
    assert {idx for idx, _ in train1} == {idx for idx, _ in train2}
    assert {idx for idx, _ in test1} == {idx for idx, _ in test2}


def test_partitioning_function_override_changes_results():
    dataset = list(range(10))

    # Define a custom partitioner function which returns 0 for even index, 1 for odd index
    def even_odd_pf(x):
        return 0 if x % 2 == 0 else 1

    ds = DataSuplier(dataset, training_percent=0.5, partitioning_function=even_odd_pf)

    training_result = list(ds.training_set())
    test_result = list(ds.test_set())

    training_indices = {idx for idx, _ in training_result}
    test_indices = {idx for idx, _ in test_result}

    # Check that evens are in training and odds are in test
    expected_training = {i for i in range(len(dataset)) if i % 2 == 0}
    expected_test = {i for i in range(len(dataset)) if i % 2 == 1}

    assert training_indices == expected_training
    assert test_indices == expected_test
