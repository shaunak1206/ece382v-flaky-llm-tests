import random

import pytest

from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import LinearFakeRandomFunction


def make_linear_suplier(n=10, training_percent=0.5, dataset=None):
    if dataset is None:
        dataset = ["item{}".format(i) for i in range(n)]
    return DataSuplier(
        dataset,
        training_percent=training_percent,
        partitioning_function=LinearFakeRandomFunction(len(dataset)),
    )


def test_init_stores_attributes():
    dataset = [1, 2, 3, 4]
    suplier = DataSuplier(dataset, training_percent=0.7)
    assert suplier.dataset is dataset
    assert suplier.training_percent == 0.7
    assert suplier.n_of_elements == 4
    assert suplier.index == 0
    assert suplier.test_indexes == []
    assert suplier.training_indexes == []


def test_default_training_percent():
    suplier = DataSuplier([1, 2, 3])
    assert suplier.training_percent == 0.8


def test_init_is_lazy_and_consumes_nothing():
    suplier = make_linear_suplier(10)
    assert suplier.index == 0
    assert suplier.training_indexes == []
    assert suplier.test_indexes == []


def test_training_set_with_linear_function():
    suplier = make_linear_suplier(10, 0.5)
    result = list(suplier.training_set())
    assert [i for i, _ in result] == [0, 1, 2, 3, 4]


def test_test_set_with_linear_function():
    suplier = make_linear_suplier(10, 0.5)
    result = list(suplier.test_set())
    assert [i for i, _ in result] == [5, 6, 7, 8, 9]


def test_yields_index_and_element_pairs():
    dataset = ["a", "b", "c", "d", "e", "f"]
    suplier = make_linear_suplier(dataset=dataset, training_percent=0.5)
    assert list(suplier.training_set()) == [(0, "a"), (1, "b"), (2, "c")]
    assert list(suplier.test_set()) == [(3, "d"), (4, "e"), (5, "f")]


def test_training_and_test_cover_dataset_without_overlap():
    dataset = list(range(20))
    suplier = make_linear_suplier(dataset=dataset, training_percent=0.5)
    train = [i for i, _ in suplier.training_set()]
    test = [i for i, _ in suplier.test_set()]
    assert set(train).isdisjoint(set(test))
    assert sorted(train + test) == dataset


def test_custom_partitioning_function_alternating():
    dataset = list(range(10))

    def alternate(index):
        return 0.0 if index % 2 == 0 else 0.99

    suplier = DataSuplier(dataset, training_percent=0.8, partitioning_function=alternate)
    assert [i for i, _ in suplier.training_set()] == [0, 2, 4, 6, 8]
    assert [i for i, _ in suplier.test_set()] == [1, 3, 5, 7, 9]


def test_all_training_when_percent_is_one():
    suplier = make_linear_suplier(8, 1.0)
    assert [i for i, _ in suplier.training_set()] == list(range(8))
    assert list(suplier.test_set()) == []


def test_all_test_when_percent_is_zero():
    suplier = make_linear_suplier(8, 0)
    assert list(suplier.training_set()) == []
    assert [i for i, _ in suplier.test_set()] == list(range(8))


def test_generators_can_be_interleaved():
    suplier = make_linear_suplier(10, 0.5)
    train_gen = suplier.training_set()
    test_gen = suplier.test_set()
    assert next(test_gen)[0] == 5
    assert next(train_gen)[0] == 0
    assert next(test_gen)[0] == 6
    assert next(train_gen)[0] == 1
    assert [i for i, _ in train_gen] == [2, 3, 4]
    assert [i for i, _ in test_gen] == [7, 8, 9]


def test_repeated_calls_return_same_results():
    suplier = make_linear_suplier(10, 0.5)
    first = list(suplier.training_set())
    second = list(suplier.training_set())
    assert first == second
    assert list(suplier.test_set()) == list(suplier.test_set())


def test_index_advances_only_as_needed():
    suplier = make_linear_suplier(10, 0.5)
    gen = suplier.training_set()
    next(gen)
    assert suplier.index == 1
    next(gen)
    assert suplier.index == 2
    list(gen)
    # Training elements 0..4 are consumed, and the generator stops once
    # the training indexes are exhausted or the data runs out.
    assert suplier.index == 10
    assert suplier.training_indexes == [0, 1, 2, 3, 4]
    assert suplier.test_indexes == [5, 6, 7, 8, 9]


def test_empty_dataset_yields_nothing():
    suplier = DataSuplier([])
    assert suplier.n_of_elements == 0
    assert list(suplier.training_set()) == []
    assert list(suplier.test_set()) == []


def test_single_element_dataset_goes_to_training_with_default_percent():
    suplier = DataSuplier(
        ["only"],
        training_percent=0.8,
        partitioning_function=LinearFakeRandomFunction(1),
    )
    assert list(suplier.training_set()) == [(0, "only")]
    assert list(suplier.test_set()) == []


def test_pseudorandom_partition_is_complete_and_disjoint():
    dataset = list(range(100))
    suplier = DataSuplier(dataset, training_percent=0.8)
    train = [i for i, _ in suplier.training_set()]
    test = [i for i, _ in suplier.test_set()]
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == dataset
    assert len(train) + len(test) == 100


def test_pseudorandom_partition_reproducible_with_seed():
    dataset = list(range(50))

    random.seed(1234)
    first = DataSuplier(dataset, training_percent=0.6)
    first_train = list(first.training_set())
    first_test = list(first.test_set())

    random.seed(1234)
    second = DataSuplier(dataset, training_percent=0.6)
    second_train = list(second.training_set())
    second_test = list(second.test_set())

    assert first_train == second_train
    assert first_test == second_test


def test_works_with_string_dataset():
    suplier = make_linear_suplier(dataset="abcdef", training_percent=0.5)
    assert list(suplier.training_set()) == [(0, "a"), (1, "b"), (2, "c")]
    assert list(suplier.test_set()) == [(3, "d"), (4, "e"), (5, "f")]


def test_invalid_negative_percent_raises_on_iteration():
    suplier = DataSuplier(
        list(range(5)),
        training_percent=-0.5,
        partitioning_function=LinearFakeRandomFunction(5),
    )
    with pytest.raises(AssertionError):
        list(suplier.training_set())


def test_partitioning_function_receives_sequential_indexes():
    seen = []

    def recorder(index):
        seen.append(index)
        return 0.0

    suplier = DataSuplier(list(range(5)), training_percent=0.5, partitioning_function=recorder)
    list(suplier.training_set())
    assert seen == [0, 1, 2, 3, 4]
