import pytest

from data_partitioner.data_suplier import DataSuplier


DATASET = ["item-{}".format(i) for i in range(50)]


def _consume_both(suplier):
    """Fully consume the training and test generators, in that order."""
    training = list(suplier.training_set())
    test = list(suplier.test_set())
    return training, test


def test_init_stores_dataset_and_training_percent():
    suplier = DataSuplier(DATASET, training_percent=0.7)
    assert suplier.dataset is DATASET
    assert suplier.training_percent == 0.7


def test_init_default_training_percent_is_point_eight():
    suplier = DataSuplier(DATASET)
    assert suplier.training_percent == 0.8


def test_init_sets_n_of_elements_to_dataset_length():
    suplier = DataSuplier(DATASET)
    assert suplier.n_of_elements == len(DATASET)


def test_init_starts_with_empty_index_lists_and_zero_index():
    suplier = DataSuplier(DATASET)
    assert suplier.training_indexes == []
    assert suplier.test_indexes == []
    assert suplier.index == 0


def test_init_creates_partitioner_object():
    suplier = DataSuplier(DATASET)
    assert suplier.partitioner is not None


def test_generators_are_lazy_and_do_not_partition_on_creation():
    suplier = DataSuplier(DATASET)
    training_gen = suplier.training_set()
    test_gen = suplier.test_set()
    assert suplier.index == 0
    assert suplier.training_indexes == []
    assert suplier.test_indexes == []
    # Generators are only created here; close them so nothing lingers.
    training_gen.close()
    test_gen.close()


def test_training_set_yields_index_and_value_tuples():
    suplier = DataSuplier(DATASET)
    items = list(suplier.training_set())
    assert items, "expected at least one training item for a 50-element dataset"
    for item in items:
        assert isinstance(item, tuple)
        assert len(item) == 2
        index, value = item
        assert value == DATASET[index]


def test_test_set_yields_index_and_value_tuples():
    suplier = DataSuplier(DATASET)
    items = list(suplier.test_set())
    assert items, "expected at least one test item for a 50-element dataset"
    for index, value in items:
        assert value == DATASET[index]


def test_full_consumption_covers_every_index_exactly_once():
    suplier = DataSuplier(DATASET)
    training, test = _consume_both(suplier)
    all_indexes = sorted([i for i, _ in training] + [i for i, _ in test])
    assert all_indexes == list(range(len(DATASET)))


def test_training_and_test_indexes_are_disjoint():
    suplier = DataSuplier(DATASET)
    _consume_both(suplier)
    assert set(suplier.training_indexes).isdisjoint(set(suplier.test_indexes))
    assert len(suplier.training_indexes) + len(suplier.test_indexes) == len(DATASET)


def test_index_lists_are_in_ascending_order():
    suplier = DataSuplier(DATASET)
    _consume_both(suplier)
    assert suplier.training_indexes == sorted(suplier.training_indexes)
    assert suplier.test_indexes == sorted(suplier.test_indexes)


def test_consuming_training_set_advances_index_to_end():
    suplier = DataSuplier(DATASET)
    list(suplier.training_set())
    assert suplier.index == len(DATASET)


def test_training_set_yields_exactly_the_training_indexes():
    suplier = DataSuplier(DATASET)
    yielded = [index for index, _ in suplier.training_set()]
    assert yielded == suplier.training_indexes


def test_test_set_yields_exactly_the_test_indexes():
    suplier = DataSuplier(DATASET)
    yielded = [index for index, _ in suplier.test_set()]
    assert yielded == suplier.test_indexes


def test_repeated_iteration_of_training_set_yields_same_items():
    suplier = DataSuplier(DATASET)
    first = list(suplier.training_set())
    second = list(suplier.training_set())
    assert first == second


def test_test_set_consumed_first_still_covers_all_indexes():
    suplier = DataSuplier(DATASET)
    test = list(suplier.test_set())
    training = list(suplier.training_set())
    all_indexes = sorted([i for i, _ in training] + [i for i, _ in test])
    assert all_indexes == list(range(len(DATASET)))


def test_interleaved_iteration_produces_complete_partition():
    suplier = DataSuplier(DATASET)
    training_gen = suplier.training_set()
    test_gen = suplier.test_set()
    training = []
    test = []
    training_done = False
    test_done = False
    while not (training_done and test_done):
        if not training_done:
            try:
                training.append(next(training_gen))
            except StopIteration:
                training_done = True
        if not test_done:
            try:
                test.append(next(test_gen))
            except StopIteration:
                test_done = True
    all_indexes = sorted([i for i, _ in training] + [i for i, _ in test])
    assert all_indexes == list(range(len(DATASET)))
    for index, value in training + test:
        assert value == DATASET[index]


def test_same_inputs_produce_same_partition():
    first = DataSuplier(DATASET, training_percent=0.8)
    second = DataSuplier(DATASET, training_percent=0.8)
    _consume_both(first)
    _consume_both(second)
    assert first.training_indexes == second.training_indexes
    assert first.test_indexes == second.test_indexes


def test_single_element_dataset_is_assigned_to_one_set():
    suplier = DataSuplier(["only"])
    training, test = _consume_both(suplier)
    assert len(training) + len(test) == 1
    yielded = training + test
    assert yielded == [(0, "only")]


def test_tuple_dataset_is_supported():
    dataset = tuple(range(20))
    suplier = DataSuplier(dataset)
    training, test = _consume_both(suplier)
    for index, value in training + test:
        assert value == dataset[index]
    assert len(training) + len(test) == len(dataset)


def test_training_proportion_is_roughly_respected_on_large_dataset():
    dataset = list(range(1000))
    suplier = DataSuplier(dataset, training_percent=0.8)
    _consume_both(suplier)
    ratio = len(suplier.training_indexes) / float(len(dataset))
    assert 0.6 <= ratio <= 0.95
