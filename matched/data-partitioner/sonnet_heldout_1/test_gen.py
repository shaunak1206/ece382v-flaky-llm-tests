import pytest

from data_partitioner.data_suplier import DataSuplier


def _collect(gen):
    return list(gen)


@pytest.fixture
def dataset():
    return [i * 10 for i in range(200)]


def test_init_sets_attributes(dataset):
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


def test_sets_yield_index_and_element_pairs(dataset):
    supplier = DataSuplier(dataset)
    for index, element in _collect(supplier.training_set()):
        assert dataset[index] == element
    for index, element in _collect(supplier.test_set()):
        assert dataset[index] == element


def test_training_and_test_are_disjoint_and_cover_dataset(dataset):
    supplier = DataSuplier(dataset)
    training = [i for i, _ in supplier.training_set()]
    test = [i for i, _ in supplier.test_set()]
    assert set(training).isdisjoint(test)
    assert sorted(training + test) == list(range(len(dataset)))


def test_indexes_are_ascending_within_each_set(dataset):
    supplier = DataSuplier(dataset)
    training = [i for i, _ in supplier.training_set()]
    test = [i for i, _ in supplier.test_set()]
    assert training == sorted(training)
    assert test == sorted(test)
    assert len(set(training)) == len(training)
    assert len(set(test)) == len(test)


def test_test_set_first_then_training_set_still_complete(dataset):
    supplier = DataSuplier(dataset)
    test = [i for i, _ in supplier.test_set()]
    training = [i for i, _ in supplier.training_set()]
    assert sorted(test + training) == list(range(len(dataset)))
    assert set(test).isdisjoint(training)


def test_interleaved_consumption_gives_same_result_as_sequential(dataset):
    sequential = DataSuplier(dataset)
    expected_training = _collect(sequential.training_set())
    expected_test = _collect(sequential.test_set())

    interleaved = DataSuplier(dataset)
    training_gen = interleaved.training_set()
    test_gen = interleaved.test_set()
    training, test = [], []
    done_training = done_test = False
    while not (done_training and done_test):
        if not done_training:
            try:
                training.append(next(training_gen))
            except StopIteration:
                done_training = True
        if not done_test:
            try:
                test.append(next(test_gen))
            except StopIteration:
                done_test = True

    assert training == expected_training
    assert test == expected_test


def test_calling_training_set_twice_yields_same_elements(dataset):
    supplier = DataSuplier(dataset)
    first = _collect(supplier.training_set())
    second = _collect(supplier.training_set())
    assert first == second
    assert len(first) > 0


def test_calling_test_set_twice_yields_same_elements(dataset):
    supplier = DataSuplier(dataset)
    first = _collect(supplier.test_set())
    second = _collect(supplier.test_set())
    assert first == second
    assert len(first) > 0


def test_generators_are_lazy(dataset):
    supplier = DataSuplier(dataset)
    gen = supplier.training_set()
    assert supplier.index == 0
    assert supplier.training_indexes == []
    next(gen)
    assert 1 <= supplier.index <= len(dataset)


def test_partial_consumption_does_not_process_everything(dataset):
    supplier = DataSuplier(dataset)
    gen = supplier.training_set()
    next(gen)
    # With an 80% training split, the first training element should be found
    # long before the whole dataset has been partitioned.
    assert supplier.index < len(dataset)


def test_full_consumption_processes_all_elements(dataset):
    supplier = DataSuplier(dataset)
    _collect(supplier.training_set())
    _collect(supplier.test_set())
    assert supplier.index == len(dataset)
    assert len(supplier.training_indexes) + len(supplier.test_indexes) == len(dataset)


def test_partition_is_consistent_across_instances(dataset):
    first = DataSuplier(dataset)
    second = DataSuplier(dataset)
    assert _collect(first.training_set()) == _collect(second.training_set())
    assert _collect(first.test_set()) == _collect(second.test_set())


def test_training_proportion_is_approximately_requested():
    data = list(range(2000))
    supplier = DataSuplier(data, training_percent=0.8)
    training = _collect(supplier.training_set())
    ratio = len(training) / len(data)
    assert 0.7 < ratio < 0.9


def test_higher_training_percent_gives_larger_training_set():
    data = list(range(1000))
    small = DataSuplier(data, training_percent=0.2)
    large = DataSuplier(data, training_percent=0.9)
    assert len(_collect(small.training_set())) < len(_collect(large.training_set()))


def test_works_with_string_and_tuple_datasets():
    text = "abcdefghijklmnopqrstuvwxyz" * 4
    supplier = DataSuplier(text)
    pairs = _collect(supplier.training_set()) + _collect(supplier.test_set())
    assert len(pairs) == len(text)
    for index, char in pairs:
        assert text[index] == char

    tup = tuple(range(50))
    supplier = DataSuplier(tup)
    pairs = _collect(supplier.training_set()) + _collect(supplier.test_set())
    assert sorted(i for i, _ in pairs) == list(range(50))
