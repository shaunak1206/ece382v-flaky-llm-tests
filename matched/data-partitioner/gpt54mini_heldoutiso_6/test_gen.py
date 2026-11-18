import pytest

import data_partitioner.data_suplier as ds_mod
from data_partitioner.data_suplier import DataSuplier


class FakePartitioner:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = 0

    def __next__(self):
        if self.calls >= len(self._outputs):
            raise StopIteration
        value = self._outputs[self.calls]
        self.calls += 1
        return value


def make_supplier(dataset, outputs, training_percent=0.8):
    fake = FakePartitioner(outputs)

    def fake_partitioner(n_of_elements, proportions, partitioning_function):
        assert n_of_elements == len(dataset)
        assert proportions == [training_percent, 1 - training_percent]
        return fake

    original = ds_mod.partitioner
    ds_mod.partitioner = fake_partitioner
    try:
        supplier = DataSuplier(dataset, training_percent=training_percent)
    finally:
        ds_mod.partitioner = original
    return supplier, fake


def test_initial_state_uses_dataset_length_and_empty_indexes():
    dataset = ["a", "b", "c"]
    supplier, fake = make_supplier(dataset, outputs=[0, 1, 0])

    assert supplier.dataset is dataset
    assert supplier.training_percent == 0.8
    assert supplier.n_of_elements == 3
    assert supplier.index == 0
    assert supplier.training_indexes == []
    assert supplier.test_indexes == []
    assert supplier.partitioner is fake


def test_training_set_yields_expected_items_in_order():
    dataset = ["a", "b", "c", "d"]
    supplier, _ = make_supplier(dataset, outputs=[0, 1, 0, 1])

    assert list(supplier.training_set()) == [(0, "a"), (2, "c")]
    assert supplier.training_indexes == [0, 2]
    assert supplier.test_indexes == [1, 3]
    assert supplier.index == 4


def test_test_set_yields_expected_items_in_order():
    dataset = ["a", "b", "c", "d"]
    supplier, _ = make_supplier(dataset, outputs=[0, 1, 0, 1])

    assert list(supplier.test_set()) == [(1, "b"), (3, "d")]
    assert supplier.training_indexes == [0, 2]
    assert supplier.test_indexes == [1, 3]
    assert supplier.index == 4


def test_generators_can_be_consumed_in_any_order_and_share_state():
    dataset = ["a", "b", "c", "d"]
    supplier, _ = make_supplier(dataset, outputs=[0, 1, 0, 1])

    test_items = list(supplier.test_set())
    training_items = list(supplier.training_set())

    assert test_items == [(1, "b"), (3, "d")]
    assert training_items == [(0, "a"), (2, "c")]
    assert supplier.index == 4
    assert supplier.training_indexes == [0, 2]
    assert supplier.test_indexes == [1, 3]


def test_partial_iteration_over_training_set_does_not_consume_all_data():
    dataset = ["a", "b", "c", "d"]
    supplier, fake = make_supplier(dataset, outputs=[0, 1, 0, 1])

    gen = supplier.training_set()
    assert next(gen) == (0, "a")
    assert supplier.index == 1
    assert fake.calls == 2
    assert supplier.training_indexes == [0]
    assert supplier.test_indexes == [1]

    assert next(gen) == (2, "c")
    assert supplier.index == 4
    assert fake.calls == 4
    with pytest.raises(StopIteration):
        next(gen)


def test_partial_iteration_over_test_set_does_not_consume_all_data():
    dataset = ["a", "b", "c", "d"]
    supplier, fake = make_supplier(dataset, outputs=[0, 1, 0, 1])

    gen = supplier.test_set()
    assert next(gen) == (1, "b")
    assert supplier.index == 2
    assert fake.calls == 3
    assert supplier.training_indexes == [0, 2]
    assert supplier.test_indexes == [1]

    assert next(gen) == (3, "d")
    assert supplier.index == 4
    assert fake.calls == 4
    with pytest.raises(StopIteration):
        next(gen)


def test_empty_dataset_yields_nothing():
    dataset = []
    supplier, _ = make_supplier(dataset, outputs=[])

    assert list(supplier.training_set()) == []
    assert list(supplier.test_set()) == []
    assert supplier.index == 0
    assert supplier.training_indexes == []
    assert supplier.test_indexes == []


def test_all_elements_can_go_to_training_set():
    dataset = [10, 20, 30]
    supplier, _ = make_supplier(dataset, outputs=[0, 0, 0])

    assert list(supplier.training_set()) == [(0, 10), (1, 20), (2, 30)]
    assert list(supplier.test_set()) == []
    assert supplier.training_indexes == [0, 1, 2]
    assert supplier.test_indexes == []


def test_all_elements_can_go_to_test_set():
    dataset = [10, 20, 30]
    supplier, _ = make_supplier(dataset, outputs=[1, 1, 1])

    assert list(supplier.training_set()) == []
    assert list(supplier.test_set()) == [(0, 10), (1, 20), (2, 30)]
    assert supplier.training_indexes == []
    assert supplier.test_indexes == [0, 1, 2]


def test_custom_training_percent_is_passed_to_partitioner():
    dataset = [1, 2, 3, 4]
    captured = {}

    def fake_partitioner(n_of_elements, proportions, partitioning_function):
        captured["n_of_elements"] = n_of_elements
        captured["proportions"] = proportions
        captured["partitioning_function"] = partitioning_function
        return FakePartitioner([0, 1, 0, 1])

    original = ds_mod.partitioner
    ds_mod.partitioner = fake_partitioner
    try:
        supplier = DataSuplier(dataset, training_percent=0.25)
    finally:
        ds_mod.partitioner = original

    assert supplier.training_percent == 0.25
    assert captured["n_of_elements"] == 4
    assert captured["proportions"] == [0.25, 0.75]
    assert callable(captured["partitioning_function"])


def test_multiple_training_iterators_continue_from_same_internal_state():
    dataset = ["a", "b", "c", "d"]
    supplier, _ = make_supplier(dataset, outputs=[0, 1, 0, 1])

    gen1 = supplier.training_set()
    assert next(gen1) == (0, "a")

    gen2 = supplier.training_set()
    assert list(gen2) == [(2, "c")]
    assert list(gen1) == []
    assert supplier.index == 4


def test_multiple_test_iterators_continue_from_same_internal_state():
    dataset = ["a", "b", "c", "d"]
    supplier, _ = make_supplier(dataset, outputs=[0, 1, 0, 1])

    gen1 = supplier.test_set()
    assert next(gen1) == (1, "b")

    gen2 = supplier.test_set()
    assert list(gen2) == [(3, "d")]
    assert list(gen1) == []
    assert supplier.index == 4


def test_name_mangled_next_method_populates_indexes():
    dataset = ["a", "b", "c"]
    supplier, _ = make_supplier(dataset, outputs=[0, 1, 0])

    res1 = supplier._DataSuplier__next()
    res2 = supplier._DataSuplier__next()
    res3 = supplier._DataSuplier__next()

    assert (res1, res2, res3) == (0, 1, 0)
    assert supplier.training_indexes == [0, 2]
    assert supplier.test_indexes == [1]
    assert supplier.index == 3


def test_yield_set_stops_after_requested_indexes_are_returned():
    dataset = ["a", "b", "c", "d"]
    supplier, fake = make_supplier(dataset, outputs=[0, 1, 0, 1])

    gen = supplier.training_set()
    assert next(gen) == (0, "a")
    assert next(gen) == (2, "c")
    with pytest.raises(StopIteration):
        next(gen)
    assert fake.calls == 4
