import pytest
from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import pseudorandom_function


def test_init_basic_behavior():
    data = list(range(10))
    suplier = DataSuplier(data)
    assert suplier.dataset == data
    assert suplier.training_percent == 0.8
    assert suplier.n_of_elements == 10
    # partitioner yields 0 or 1 only, but no error means ok
    training_len = 0
    test_len = 0
    for _ in range(10):
        r = suplier.partitioner.__next__()
        assert r in (0, 1)
        if r == 0:
            training_len += 1
        else:
            test_len += 1
    assert training_len + test_len == 10


def test_training_and_test_indexes_populated_after_iterating():
    data = list(range(5))
    suplier = DataSuplier(data)
    # Before any iteration
    assert suplier.training_indexes == []
    assert suplier.test_indexes == []

    # Force some iteration by calling training_set generator partially
    gen = suplier.training_set()
    # Get all training items
    training_items = list(gen)
    # Likewise test set
    test_items = list(suplier.test_set())

    # training_indexes and test_indexes should be complementary and cover all indices
    combined = set(suplier.training_indexes + suplier.test_indexes)
    assert set(range(5)) == combined
    assert len(suplier.training_indexes) + len(suplier.test_indexes) == 5
    # All yielded indices in training_items are from training_indexes
    training_indices = [idx for idx, _ in training_items]
    assert set(training_indices) == set(suplier.training_indexes)
    # Same for test_items
    test_indices = [idx for idx, _ in test_items]
    assert set(test_indices) == set(suplier.test_indexes)


def test_training_set_yields_correct_data_and_indexes():
    data = ["a", "b", "c", "d"]
    suplier = DataSuplier(data)
    gen = suplier.training_set()
    for index, val in gen:
        assert val == data[index]
        assert index in suplier.training_indexes


def test_test_set_yields_correct_data_and_indexes():
    data = [100, 200, 300, 400]
    suplier = DataSuplier(data)
    gen = suplier.test_set()
    for index, val in gen:
        assert val == data[index]
        assert index in suplier.test_indexes


def test_generator_yields_all_training_items():
    data = list(range(20))
    suplier = DataSuplier(data)
    training_items = list(suplier.training_set())
    indices_from_gen = [idx for idx, _ in training_items]
    assert set(indices_from_gen) == set(suplier.training_indexes)
    # All indices unique and valid
    assert all(0 <= idx < 20 for idx in indices_from_gen)
    assert len(indices_from_gen) == len(suplier.training_indexes)


def test_generator_yields_all_test_items():
    data = list("abcdefghij")
    suplier = DataSuplier(data)
    test_items = list(suplier.test_set())
    indices_from_gen = [idx for idx, _ in test_items]
    assert set(indices_from_gen) == set(suplier.test_indexes)
    assert all(0 <= idx < 10 for idx in indices_from_gen)
    assert len(indices_from_gen) == len(suplier.test_indexes)


def test_training_percent_0_results_in_all_test_set():
    data = list(range(5))
    suplier = DataSuplier(data, training_percent=0)
    training = list(suplier.training_set())
    test = list(suplier.test_set())
    # training set empty
    assert training == []
    # test set contains all data
    test_indices = [idx for idx, _ in test]
    assert set(test_indices) == set(range(5))


def test_training_percent_1_results_in_all_training_set():
    data = ["x", "y", "z"]
    suplier = DataSuplier(data, training_percent=1)
    training = list(suplier.training_set())
    test = list(suplier.test_set())
    # test set empty
    assert test == []
    training_indices = [idx for idx, _ in training]
    assert set(training_indices) == set(range(3))


def test_partitioning_function_override_results_in_different_partitioning():
    # Define deterministic partitioning function: assign even indices to training (0), odd to test (1)
    def custom_partition(n, weights, _):
        # Return a generator that cycles [0,1,0,1,...]
        i = 0
        while True:
            yield i % 2
            i += 1
    gen = custom_partition(10, [0.5, 0.5], None)

    class DummyPartitioner:
        def __init__(self, n, weights, func):
            self.generator = gen
        def __next__(self):
            return next(self.generator)

    # Patch DataSuplier to use DummyPartitioner to test partitioning function override effect
    # We monkeypatch DataSuplier.partitioner attribute after instantiation
    data = list(range(6))
    suplier = DataSuplier(data, partitioning_function=None)
    suplier.partitioner = DummyPartitioner(len(data), [0.5,0.5], None)

    training = list(suplier.training_set())
    test = list(suplier.test_set())

    for idx, _ in training:
        assert idx % 2 == 0
    for idx, _ in test:
        assert idx % 2 == 1


def test__next_method_increments_index_and_appends_to_appropriate_list():
    data = [10, 20]
    suplier = DataSuplier(data)

    # Store original lists
    assert suplier.index == 0
    result0 = suplier.__next()
    # index incremented
    assert suplier.index == 1
    # result0 is either 0 or 1
    assert result0 in (0,1)
    if result0 == 0:
        assert suplier.training_indexes == [0]
        assert suplier.test_indexes == []
    else:
        assert suplier.test_indexes == [0]
        assert suplier.training_indexes == []

    result1 = suplier.__next__()
    assert suplier.index == 2
    # Now second index 1 should be in one of the lists
    all_indexes = suplier.training_indexes + suplier.test_indexes
    assert set(all_indexes) == {0,1}
    assert result1 in (0,1)


def test__yield_set_generator_lazy_consumption_behavior():
    data = list(range(5))
    suplier = DataSuplier(data)

    # Get generator for training_set but consume only first item
    gen = suplier.training_set()
    first = next(gen, None)
    assert first is not None
    # Now training and test indexes should be partially populated but index count is >= 1
    assert suplier.index >= 1
    # The generator can still proceed to get remaining items without error
    rest = list(gen)
    # Combined lengths should equal length of training_indexes
    total_training_items = 1 + len(rest)
    assert total_training_items == len(suplier.training_indexes)


def test__yield_set_produces_no_items_when_indexes_empty(monkeypatch):
    data = [1,2,3]
    suplier = DataSuplier(data)

    # Patch training_indexes to empty list and n_of_elements also zero to end loop quickly
    suplier.training_indexes = []
    suplier.index = suplier.n_of_elements  # index at end

    gen = suplier._yield_set(suplier.training_indexes)
    # Should yield nothing
    assert list(gen) == []


def test_indices_are_not_duplicated_in_training_and_test_sets():
    data = list(range(15))
    suplier = DataSuplier(data)
    # Consume all training and test sets fully
    training = list(suplier.training_set())
    test = list(suplier.test_set())
    # Check no overlap between training and test indexes
    train_set = set(idx for idx, _ in training)
    test_set = set(idx for idx, _ in test)
    assert train_set.isdisjoint(test_set)


def test_training_and_test_sets_exhaustion_for_empty_dataset():
    data = []
    suplier = DataSuplier(data)
    assert list(suplier.training_set()) == []
    assert list(suplier.test_set()) == []
    assert suplier.training_indexes == []
    assert suplier.test_indexes == []
    assert suplier.index == 0
    assert suplier.n_of_elements == 0


def test_training_and_test_sets_exhaustion_for_single_element_dataset():
    data = [99]
    suplier = DataSuplier(data)
    training = list(suplier.training_set())
    test = list(suplier.test_set())
    # One of lists must contain index 0, the other empty
    combined = suplier.training_indexes + suplier.test_indexes
    assert combined == [0]
    # The yielded data match data[0]
    all_yielded = training + test
    assert all_yielded[0][1] == 99


def test_multiple_iterations_of_training_and_test_sets_resume_correctly():
    data = list("abcd")
    suplier = DataSuplier(data)
    ts1 = list(suplier.training_set())
    ts2 = list(suplier.training_set())
    # Both yields should return the same results if called after the first finished
    assert ts1 == ts2

    test1 = list(suplier.test_set())
    test2 = list(suplier.test_set())
    assert test1 == test2


def test_training_set_and_test_set_yield_all_elements_between_them():
    data = list(range(7))
    suplier = DataSuplier(data)
    training = list(suplier.training_set())
    test = list(suplier.test_set())
    indexes_yielded = {idx for idx, _ in training + test}
    assert set(range(7)) == indexes_yielded


def test_training_percent_near_zero_splits_data_almost_all_test():
    data = list(range(100))
    suplier = DataSuplier(data, training_percent=0.01)
    training = list(suplier.training_set())
    test = list(suplier.test_set())
    # Training should be very small or 0, test majority
    assert len(training) <= 5
    assert len(test) + len(training) == 100


def test_training_percent_near_one_splits_data_almost_all_training():
    data = list(range(100))
    suplier = DataSuplier(data, training_percent=0.99)
    training = list(suplier.training_set())
    test = list(suplier.test_set())
    assert len(test) <= 5
    assert len(test) + len(training) == 100

