import inspect

import pytest

from data_partitioner.data_suplier import DataSuplier


def _collect(generator):
    return list(generator)


def _indexes(pairs):
    return [index for index, _ in pairs]


def test_init_stores_dataset_and_training_percent():
    dataset = ["a", "b", "c"]
    supplier = DataSuplier(dataset, training_percent=0.5)
    assert supplier.dataset is dataset
    assert supplier.training_percent == 0.5


def test_init_default_training_percent_is_0_8():
    supplier = DataSuplier(list(range(10)))
    assert supplier.training_percent == 0.8


def test_init_sets_n_of_elements_from_dataset_length():
    supplier = DataSuplier(list(range(37)))
    assert supplier.n_of_elements == 37


def test_init_starts_with_empty_index_lists_and_zero_index():
    supplier = DataSuplier(list(range(5)))
    assert supplier.training_indexes == []
    assert supplier.test_indexes == []
    assert supplier.index == 0


def test_init_rejects_dataset_without_len():
    with pytest.raises(TypeError):
        DataSuplier(iter([1, 2, 3]))


def test_training_set_returns_generator_without_advancing():
    supplier = DataSuplier(list(range(20)))
    gen = supplier.training_set()
    assert inspect.isgenerator(gen)
    assert supplier.index == 0
    assert supplier.training_indexes == []


def test_test_set_returns_generator_without_advancing():
    supplier = DataSuplier(list(range(20)))
    gen = supplier.test_set()
    assert inspect.isgenerator(gen)
    assert supplier.index == 0
    assert supplier.test_indexes == []


def test_training_set_yields_index_and_item_pairs():
    dataset = [f"item-{i}" for i in range(30)]
    supplier = DataSuplier(dataset)
    for index, item in supplier.training_set():
        assert isinstance(index, int)
        assert item is dataset[index]


def test_test_set_yields_index_and_item_pairs():
    dataset = [f"item-{i}" for i in range(30)]
    supplier = DataSuplier(dataset)
    for index, item in supplier.test_set():
        assert isinstance(index, int)
        assert item is dataset[index]


def test_training_and_test_sets_cover_every_index_exactly_once():
    n = 50
    supplier = DataSuplier(list(range(n)))
    training = _indexes(_collect(supplier.training_set()))
    test = _indexes(_collect(supplier.test_set()))
    assert sorted(training + test) == list(range(n))


def test_training_and_test_sets_are_disjoint():
    supplier = DataSuplier(list(range(50)))
    training = set(_indexes(_collect(supplier.training_set())))
    test = set(_indexes(_collect(supplier.test_set())))
    assert training.isdisjoint(test)


def test_yielded_indexes_are_in_ascending_order():
    supplier = DataSuplier(list(range(60)))
    training = _indexes(_collect(supplier.training_set()))
    test = _indexes(_collect(supplier.test_set()))
    assert training == sorted(training)
    assert test == sorted(test)


def test_split_is_consistent_across_instances():
    dataset = list(range(100))
    first = DataSuplier(dataset)
    second = DataSuplier(dataset)
    assert _indexes(_collect(first.training_set())) == _indexes(
        _collect(second.training_set())
    )
    assert _indexes(_collect(first.test_set())) == _indexes(
        _collect(second.test_set())
    )


def test_repeated_iteration_yields_same_results():
    supplier = DataSuplier(list(range(40)))
    first_pass = _collect(supplier.training_set())
    second_pass = _collect(supplier.training_set())
    assert first_pass == second_pass


def test_consuming_test_set_first_gives_same_split_as_training_first():
    dataset = list(range(45))
    training_first = DataSuplier(dataset)
    test_first = DataSuplier(dataset)

    training_a = _indexes(_collect(training_first.training_set()))
    test_a = _indexes(_collect(training_first.test_set()))

    test_b = _indexes(_collect(test_first.test_set()))
    training_b = _indexes(_collect(test_first.training_set()))

    assert training_a == training_b
    assert test_a == test_b


def test_full_consumption_advances_index_to_length():
    n = 25
    supplier = DataSuplier(list(range(n)))
    _collect(supplier.training_set())
    assert supplier.index == n


def test_partial_consumption_advances_index_lazily():
    supplier = DataSuplier(list(range(100)))
    gen = supplier.training_set()
    next(gen)
    assert 1 <= supplier.index <= 100
    assert len(supplier.training_indexes) >= 1


def test_interleaved_iteration_covers_all_elements():
    n = 30
    supplier = DataSuplier(list(range(n)))
    training_gen = supplier.training_set()
    test_gen = supplier.test_set()
    seen = []
    training_done = False
    test_done = False
    while not (training_done and test_done):
        if not training_done:
            try:
                seen.append(next(training_gen)[0])
            except StopIteration:
                training_done = True
        if not test_done:
            try:
                seen.append(next(test_gen)[0])
            except StopIteration:
                test_done = True
    assert sorted(seen) == list(range(n))


def test_empty_dataset_produces_empty_sets():
    supplier = DataSuplier([])
    assert supplier.n_of_elements == 0
    assert _collect(supplier.training_set()) == []
    assert _collect(supplier.test_set()) == []
    assert supplier.index == 0


def test_single_element_goes_to_exactly_one_set():
    supplier = DataSuplier(["only"])
    training = _collect(supplier.training_set())
    test = _collect(supplier.test_set())
    assert len(training) + len(test) == 1
    assert (training + test)[0] == (0, "only")


def test_training_proportion_is_roughly_respected():
    n = 2000
    supplier = DataSuplier(list(range(n)), training_percent=0.8)
    training = _collect(supplier.training_set())
    fraction = len(training) / n
    assert 0.7 <= fraction <= 0.9


def test_dataset_is_not_mutated_by_iteration():
    dataset = list(range(20))
    snapshot = list(dataset)
    supplier = DataSuplier(dataset)
    _collect(supplier.training_set())
    _collect(supplier.test_set())
    assert dataset == snapshot
