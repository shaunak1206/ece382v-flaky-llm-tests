import pytest

import data_partitioner.data_suplier as data_suplier_module
from data_partitioner.data_suplier import DataSuplier


class DummyPartitioner:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self._i = 0

    def __next__(self):
        if self._i >= len(self._outputs):
            raise StopIteration
        value = self._outputs[self._i]
        self._i += 1
        return value


def make_supplier(dataset, outputs, training_percent=0.8):
    def fake_partitioner(n_of_elements, probabilities, partitioning_function):
        assert n_of_elements == len(dataset)
        assert probabilities == [training_percent, 1 - training_percent]
        return DummyPartitioner(outputs)

    original = data_suplier_module.partitioner
    data_suplier_module.partitioner = fake_partitioner
    try:
        supplier = DataSuplier(dataset, training_percent=training_percent)
    finally:
        data_suplier_module.partitioner = original
    return supplier


def test_initialization_stores_basic_attributes():
    dataset = ["a", "b", "c"]
    supplier = make_supplier(dataset, [0, 1, 0])

    assert supplier.dataset is dataset
    assert supplier.training_percent == 0.8
    assert supplier.n_of_elements == 3
    assert supplier.test_indexes == []
    assert supplier.training_indexes == []
    assert supplier.index == 0


def test_initialization_passes_expected_arguments_to_partitioner():
    dataset = [1, 2, 3, 4]
    captured = {}

    def fake_partitioner(n_of_elements, probabilities, partitioning_function):
        captured["n_of_elements"] = n_of_elements
        captured["probabilities"] = probabilities
        captured["partitioning_function"] = partitioning_function
        return DummyPartitioner([0, 1, 0, 1])

    original = data_suplier_module.partitioner
    data_suplier_module.partitioner = fake_partitioner
    try:
        supplier = DataSuplier(dataset, training_percent=0.25)
    finally:
        data_suplier_module.partitioner = original

    assert supplier.n_of_elements == 4
    assert captured["n_of_elements"] == 4
    assert captured["probabilities"] == [0.25, 0.75]
    assert callable(captured["partitioning_function"])


def test_training_set_yields_expected_elements_in_order():
    dataset = ["a", "b", "c", "d"]
    supplier = make_supplier(dataset, [0, 1, 0, 1])

    result = list(supplier.training_set())

    assert result == [(0, "a"), (2, "c")]
    assert supplier.training_indexes == [0, 2]
    assert supplier.test_indexes == [1, 3]
    assert supplier.index == 4


def test_test_set_yields_expected_elements_in_order():
    dataset = ["a", "b", "c", "d"]
    supplier = make_supplier(dataset, [0, 1, 0, 1])

    result = list(supplier.test_set())

    assert result == [(1, "b"), (3, "d")]
    assert supplier.training_indexes == [0, 2]
    assert supplier.test_indexes == [1, 3]
    assert supplier.index == 4


def test_training_and_test_sets_can_be_consumed_independently_and_match_partition():
    dataset = ["a", "b", "c", "d", "e"]
    supplier = make_supplier(dataset, [0, 0, 1, 1, 0])

    training = list(supplier.training_set())
    test = list(supplier.test_set())

    assert training == [(0, "a"), (1, "b"), (4, "e")]
    assert test == [(2, "c"), (3, "d")]
    assert supplier.training_indexes == [0, 1, 4]
    assert supplier.test_indexes == [2, 3]
    assert supplier.index == 5


def test_repeated_calls_return_fresh_generators_with_same_output():
    dataset = ["x", "y", "z"]
    supplier = make_supplier(dataset, [1, 0, 1])

    first = list(supplier.test_set())
    second = list(supplier.test_set())

    assert first == [(0, "x"), (2, "z")]
    assert second == [(0, "x"), (2, "z")]


def test_yield_set_consumes_partition_only_as_needed_for_training():
    dataset = ["a", "b", "c", "d", "e"]
    supplier = make_supplier(dataset, [0, 1, 0, 1, 0])

    gen = supplier.training_set()
    first_item = next(gen)

    assert first_item == (0, "a")
    assert supplier.index == 1
    assert supplier.training_indexes == [0]
    assert supplier.test_indexes == []


def test_yield_set_consumes_partition_only_as_needed_for_test():
    dataset = ["a", "b", "c", "d", "e"]
    supplier = make_supplier(dataset, [0, 1, 0, 1, 0])

    gen = supplier.test_set()
    first_item = next(gen)

    assert first_item == (1, "b")
    assert supplier.index == 2
    assert supplier.training_indexes == [0]
    assert supplier.test_indexes == [1]


def test_empty_dataset_yields_nothing():
    supplier = make_supplier([], [])

    assert list(supplier.training_set()) == []
    assert list(supplier.test_set()) == []
    assert supplier.index == 0
    assert supplier.training_indexes == []
    assert supplier.test_indexes == []


def test_all_training_partition_yields_only_training_elements():
    dataset = [10, 20, 30]
    supplier = make_supplier(dataset, [0, 0, 0])

    assert list(supplier.training_set()) == [(0, 10), (1, 20), (2, 30)]
    assert list(supplier.test_set()) == []
    assert supplier.training_indexes == [0, 1, 2]
    assert supplier.test_indexes == []


def test_all_test_partition_yields_only_test_elements():
    dataset = [10, 20, 30]
    supplier = make_supplier(dataset, [1, 1, 1])

    assert list(supplier.training_set()) == []
    assert list(supplier.test_set()) == [(0, 10), (1, 20), (2, 30)]
    assert supplier.training_indexes == []
    assert supplier.test_indexes == [0, 1, 2]


def test_duplicate_indexes_are_returned_according_to_current_index_lists():
    dataset = ["a", "b"]
    supplier = make_supplier(dataset, [0, 1])

    supplier.training_indexes.extend([0, 0])
    supplier.test_indexes.append(1)

    assert list(supplier.training_set()) == [(0, "a"), (0, "a")]
    assert list(supplier.test_set()) == [(1, "b")]


def test_custom_training_percent_changes_partition_probabilities():
    dataset = [1, 2]
    captured = {}

    def fake_partitioner(n_of_elements, probabilities, partitioning_function):
        captured["probabilities"] = probabilities
        return DummyPartitioner([0, 1])

    original = data_suplier_module.partitioner
    data_suplier_module.partitioner = fake_partitioner
    try:
        supplier = DataSuplier(dataset, training_percent=0.6)
    finally:
        data_suplier_module.partitioner = original

    assert supplier.training_percent == 0.6
    assert captured["probabilities"] == [0.6, 0.4]


def test_consuming_training_then_test_then_training_again_is_stable():
    dataset = ["a", "b", "c", "d"]
    supplier = make_supplier(dataset, [0, 1, 0, 1])

    first_training = list(supplier.training_set())
    test = list(supplier.test_set())
    second_training = list(supplier.training_set())

    assert first_training == [(0, "a"), (2, "c")]
    assert test == [(1, "b"), (3, "d")]
    assert second_training == [(0, "a"), (2, "c")]
    assert supplier.index == 4
