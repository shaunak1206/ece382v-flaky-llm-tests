import pytest

from data_partitioner.data_suplier import DataSuplier


DATA = ["item-%d" % i for i in range(50)]


def make(data=None, **kwargs):
    return DataSuplier(list(DATA) if data is None else data, **kwargs)


def test_init_stores_attributes():
    data = list(DATA)
    supplier = DataSuplier(data, training_percent=0.7)
    assert supplier.dataset is data
    assert supplier.training_percent == 0.7
    assert supplier.n_of_elements == len(data)


def test_init_default_training_percent():
    supplier = make()
    assert supplier.training_percent == 0.8


def test_init_state_is_empty_and_lazy():
    supplier = make()
    assert supplier.training_indexes == []
    assert supplier.test_indexes == []
    assert supplier.index == 0


def test_creating_generators_does_not_consume_data():
    supplier = make()
    supplier.training_set()
    supplier.test_set()
    assert supplier.index == 0
    assert supplier.training_indexes == []
    assert supplier.test_indexes == []


def test_items_are_index_value_pairs():
    supplier = make()
    items = list(supplier.training_set()) + list(supplier.test_set())
    assert items
    for item in items:
        assert isinstance(item, tuple)
        assert len(item) == 2
        index, value = item
        assert value == DATA[index]


def test_training_and_test_sets_are_disjoint_and_cover_dataset():
    supplier = make()
    training = [i for i, _ in supplier.training_set()]
    test = [i for i, _ in supplier.test_set()]
    assert set(training).isdisjoint(test)
    assert sorted(training + test) == list(range(len(DATA)))


def test_consuming_test_first_gives_same_cover():
    supplier = make()
    test = [i for i, _ in supplier.test_set()]
    training = [i for i, _ in supplier.training_set()]
    assert set(training).isdisjoint(test)
    assert sorted(training + test) == list(range(len(DATA)))


def test_indexes_are_yielded_in_ascending_order():
    supplier = make()
    training = [i for i, _ in supplier.training_set()]
    test = [i for i, _ in supplier.test_set()]
    assert training == sorted(training)
    assert test == sorted(test)


def test_repeated_calls_return_same_training_set():
    supplier = make()
    first = list(supplier.training_set())
    second = list(supplier.training_set())
    assert first == second


def test_repeated_calls_return_same_test_set():
    supplier = make()
    first = list(supplier.test_set())
    second = list(supplier.test_set())
    assert first == second


def test_full_consumption_advances_index_to_end():
    supplier = make()
    list(supplier.training_set())
    assert supplier.index == len(DATA)
    assert len(supplier.training_indexes) + len(supplier.test_indexes) == len(DATA)


def test_partial_consumption_is_incremental():
    supplier = make(list(range(1000)))
    gen = supplier.training_set()
    first = next(gen)
    assert first[0] == first[1]
    assert len(supplier.training_indexes) >= 1
    assert supplier.index < supplier.n_of_elements
    assert supplier.index == len(supplier.training_indexes) + len(supplier.test_indexes)


def test_interleaved_generators_produce_consistent_partition():
    supplier = make()
    train_gen = supplier.training_set()
    test_gen = supplier.test_set()
    train, test = [], []
    train_done = test_done = False
    while not (train_done and test_done):
        if not train_done:
            try:
                train.append(next(train_gen)[0])
            except StopIteration:
                train_done = True
        if not test_done:
            try:
                test.append(next(test_gen)[0])
            except StopIteration:
                test_done = True
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == list(range(len(DATA)))
    # Must match a fresh supplier consumed sequentially
    fresh = make()
    assert train == [i for i, _ in fresh.training_set()]
    assert test == [i for i, _ in fresh.test_set()]


def test_default_split_is_roughly_eighty_percent_training():
    data = list(range(1000))
    supplier = DataSuplier(data)
    training = list(supplier.training_set())
    test = list(supplier.test_set())
    assert len(training) + len(test) == 1000
    fraction = len(training) / 1000.0
    assert 0.6 < fraction < 0.95


def test_lower_training_percent_gives_smaller_training_set():
    data = list(range(1000))
    big = DataSuplier(data, training_percent=0.9)
    small = DataSuplier(data, training_percent=0.3)
    assert len(list(big.training_set())) > len(list(small.training_set()))


def test_works_with_string_dataset():
    text = "abcdefghijklmnopqrstuvwxyz"
    supplier = DataSuplier(text)
    items = list(supplier.training_set()) + list(supplier.test_set())
    assert sorted(i for i, _ in items) == list(range(len(text)))
    for index, value in items:
        assert value == text[index]


def test_single_element_dataset_lands_in_exactly_one_set():
    supplier = DataSuplier(["only"])
    training = list(supplier.training_set())
    test = list(supplier.test_set())
    assert len(training) + len(test) == 1
    assert (training or test)[0] == (0, "only")
