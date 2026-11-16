import inspect

import pytest

from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import pseudorandom_function


@pytest.fixture
def dataset():
    return ["item_%d" % i for i in range(200)]


def test_init_stores_attributes(dataset):
    supplier = DataSuplier(dataset, training_percent=0.7)
    assert supplier.dataset is dataset
    assert supplier.training_percent == 0.7
    assert supplier.n_of_elements == len(dataset)
    assert supplier.index == 0
    assert supplier.test_indexes == []
    assert supplier.training_indexes == []


def test_default_training_percent(dataset):
    supplier = DataSuplier(dataset)
    assert supplier.training_percent == 0.8


def test_sets_are_generators(dataset):
    supplier = DataSuplier(dataset)
    assert inspect.isgenerator(supplier.training_set())
    assert inspect.isgenerator(supplier.test_set())


def test_creating_generators_is_lazy(dataset):
    supplier = DataSuplier(dataset)
    supplier.training_set()
    supplier.test_set()
    assert supplier.index == 0
    assert supplier.training_indexes == []
    assert supplier.test_indexes == []


def test_items_are_index_value_pairs(dataset):
    supplier = DataSuplier(dataset)
    items = list(supplier.training_set())
    assert len(items) > 0
    for index, value in items:
        assert value == dataset[index]


def test_test_set_items_are_index_value_pairs(dataset):
    supplier = DataSuplier(dataset)
    items = list(supplier.test_set())
    assert len(items) > 0
    for index, value in items:
        assert value == dataset[index]


def test_sets_are_disjoint_and_cover_dataset(dataset):
    supplier = DataSuplier(dataset)
    training = [i for i, _ in supplier.training_set()]
    test = [i for i, _ in supplier.test_set()]
    assert set(training).isdisjoint(test)
    assert sorted(training + test) == list(range(len(dataset)))


def test_indexes_are_yielded_in_ascending_order(dataset):
    supplier = DataSuplier(dataset)
    training = [i for i, _ in supplier.training_set()]
    test = [i for i, _ in supplier.test_set()]
    assert training == sorted(training)
    assert test == sorted(test)


def test_partition_is_consistent_across_instances(dataset):
    first = DataSuplier(dataset)
    second = DataSuplier(dataset)
    assert list(first.training_set()) == list(second.training_set())
    assert list(first.test_set()) == list(second.test_set())


def test_explicit_default_partitioning_function_matches_default(dataset):
    default = DataSuplier(dataset)
    explicit = DataSuplier(dataset, partitioning_function=pseudorandom_function)
    assert list(default.training_set()) == list(explicit.training_set())
    assert list(default.test_set()) == list(explicit.test_set())


def test_training_set_can_be_iterated_repeatedly(dataset):
    supplier = DataSuplier(dataset)
    first = list(supplier.training_set())
    second = list(supplier.training_set())
    assert first == second


def test_consuming_one_item_advances_index(dataset):
    supplier = DataSuplier(dataset)
    gen = supplier.training_set()
    next(gen)
    assert supplier.index >= 1
    assert len(supplier.training_indexes) >= 1


def test_full_consumption_sets_index_to_n_elements(dataset):
    supplier = DataSuplier(dataset)
    list(supplier.training_set())
    assert supplier.index == len(dataset)
    assert (len(supplier.training_indexes) + len(supplier.test_indexes)
            == len(dataset))


def test_recorded_indexes_match_yielded_indexes(dataset):
    supplier = DataSuplier(dataset)
    training = [i for i, _ in supplier.training_set()]
    test = [i for i, _ in supplier.test_set()]
    assert training == supplier.training_indexes
    assert test == supplier.test_indexes


def test_interleaved_consumption_matches_sequential(dataset):
    sequential = DataSuplier(dataset)
    expected_training = list(sequential.training_set())
    expected_test = list(sequential.test_set())

    interleaved = DataSuplier(dataset)
    train_gen = interleaved.training_set()
    test_gen = interleaved.test_set()
    training, test = [], []
    for _ in range(5):
        training.append(next(train_gen))
        test.append(next(test_gen))
    training.extend(train_gen)
    test.extend(test_gen)

    assert training == expected_training
    assert test == expected_test


def test_duplicate_values_are_distinguished_by_index():
    data = ["same"] * 100
    supplier = DataSuplier(data)
    training = [i for i, _ in supplier.training_set()]
    test = [i for i, _ in supplier.test_set()]
    assert sorted(training + test) == list(range(100))


def test_works_with_tuple_dataset():
    data = tuple(range(100, 300))
    supplier = DataSuplier(data)
    items = list(supplier.training_set()) + list(supplier.test_set())
    assert sorted(items) == [(i, data[i]) for i in range(len(data))]


def test_default_split_is_roughly_80_20():
    data = list(range(2000))
    supplier = DataSuplier(data)
    training = list(supplier.training_set())
    fraction = len(training) / float(len(data))
    assert 0.65 < fraction < 0.95


def test_half_split_is_roughly_balanced():
    data = list(range(2000))
    supplier = DataSuplier(data, training_percent=0.5)
    training = list(supplier.training_set())
    fraction = len(training) / float(len(data))
    assert 0.35 < fraction < 0.65
