import pytest

import data_partitioner.data_suplier as ds_mod


class FakePartitioner:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def __next__(self):
        if self.calls >= len(self.values):
            raise StopIteration
        value = self.values[self.calls]
        self.calls += 1
        return value


def make_partition_factory(values, calls=None):
    """
    Returns a partitioner factory that records its construction arguments.
    """
    if calls is None:
        calls = []

    def factory(n_of_elements, probs, partitioning_function):
        calls.append((n_of_elements, probs, partitioning_function))
        return FakePartitioner(values)

    return factory, calls


def consume(gen):
    return list(gen)


def test_init_sets_basic_attributes_and_constructs_partitioner(monkeypatch):
    factory, calls = make_partition_factory([0, 1, 0])
    monkeypatch.setattr(ds_mod, "partitioner", factory)

    dataset = ["a", "b", "c"]
    suplier = ds_mod.DataSuplier(dataset, training_percent=0.7, partitioning_function="pf")

    assert suplier.dataset is dataset
    assert suplier.training_percent == 0.7
    assert suplier.n_of_elements == 3
    assert suplier.index == 0
    assert suplier.training_indexes == []
    assert suplier.test_indexes == []
    assert len(calls) == 1
    assert calls[0][0] == 3
    assert calls[0][1] == [0.7, 0.30000000000000004]
    assert calls[0][2] == "pf"


def test_test_set_returns_generator(monkeypatch):
    factory, _ = make_partition_factory([0])
    monkeypatch.setattr(ds_mod, "partitioner", factory)

    suplier = ds_mod.DataSuplier([1])
    gen = suplier.test_set()

    assert hasattr(gen, "__iter__")
    assert hasattr(gen, "__next__")


def test_training_set_returns_generator(monkeypatch):
    factory, _ = make_partition_factory([0])
    monkeypatch.setattr(ds_mod, "partitioner", factory)

    suplier = ds_mod.DataSuplier([1])
    gen = suplier.training_set()

    assert hasattr(gen, "__iter__")
    assert hasattr(gen, "__next__")


def test_empty_dataset_yields_nothing_for_training_and_test(monkeypatch):
    factory, calls = make_partition_factory([])
    monkeypatch.setattr(ds_mod, "partitioner", factory)

    suplier = ds_mod.DataSuplier([], training_percent=0.5)

    assert consume(suplier.training_set()) == []
    assert consume(suplier.test_set()) == []
    assert calls[0][0] == 0
    assert calls[0][1] == [0.5, 0.5]


def test_training_set_yields_only_training_items_in_original_order(monkeypatch):
    factory, _ = make_partition_factory([0, 1, 0, 1, 0])
    monkeypatch.setattr(ds_mod, "partitioner", factory)

    dataset = ["a", "b", "c", "d", "e"]
    suplier = ds_mod.DataSuplier(dataset)

    assert consume(suplier.training_set()) == [(0, "a"), (2, "c"), (4, "e")]


def test_test_set_yields_only_test_items_in_original_order(monkeypatch):
    factory, _ = make_partition_factory([0, 1, 0, 1, 0])
    monkeypatch.setattr(ds_mod, "partitioner", factory)

    dataset = ["a", "b", "c", "d", "e"]
    suplier = ds_mod.DataSuplier(dataset)

    assert consume(suplier.test_set()) == [(1, "b"), (3, "d")]


def test_training_and_test_sets_partition_complementary_items(monkeypatch):
    factory, _ = make_partition_factory([0, 1, 1, 0, 0, 1])
    monkeypatch.setattr(ds_mod, "partitioner", factory)

    dataset = list("abcdef")
    suplier = ds_mod.DataSuplier(dataset)

    training = consume(suplier.training_set())
    test = consume(suplier.test_set())

    assert training == [(0, "a"), (3, "d"), (4, "e")]
    assert test == [(1, "b"), (2, "c"), (5, "f")]
    assert sorted(training + test) == [(0, "a"), (1, "b"), (2, "c"), (3, "d"), (4, "e"), (5, "f")]


def test_training_set_consumption_advances_internal_state(monkeypatch):
    factory, _ = make_partition_factory([0, 1, 0, 1])
    monkeypatch.setattr(ds_mod, "partitioner", factory)

    suplier = ds_mod.DataSuplier(["a", "b", "c", "d"])

    first = suplier.training_set()
    assert next(first) == (0, "a")
    assert suplier.index == 1
    assert suplier.training_indexes == [0]
    assert suplier.test_indexes == []

    assert list(first) == [(2, "c")]
    assert suplier.index == 3
    assert suplier.training_indexes == [0, 2]
    assert suplier.test_indexes == [1]


def test_test_set_can_be_called_before_training_set(monkeypatch):
    factory, _ = make_partition_factory([1, 0, 1, 0])
    monkeypatch.setattr(ds_mod, "partitioner", factory)

    suplier = ds_mod.DataSuplier(["a", "b", "c", "d"])

    assert consume(suplier.test_set()) == [(0, "a"), (2, "c")]
    assert consume(suplier.training_set()) == [(1, "b"), (3, "d")]


def test_generator_after_partial_consumption_returns_remaining_items(monkeypatch):
    factory, _ = make_partition_factory([0, 1, 0, 1, 1])
    monkeypatch.setattr(ds_mod, "partitioner", factory)

    suplier = ds_mod.DataSuplier(["a", "b", "c", "d", "e"])

    gen = suplier.test_set()
    assert next(gen) == (1, "b")
    assert next(gen) == (3, "d")
    assert list(gen) == [(4, "e")]


def test_dataset_items_are_returned_by_reference_not_copied(monkeypatch):
    factory, _ = make_partition_factory([0])
    monkeypatch.setattr(ds_mod, "partitioner", factory)

    item = {"x": 1}
    dataset = [item]
    suplier = ds_mod.DataSuplier(dataset)

    out = consume(suplier.training_set())
    assert out == [(0, item)]
    assert out[0][1] is item


def test_training_percent_is_passed_through_without_validation(monkeypatch):
    factory, calls = make_partition_factory([0, 1])
    monkeypatch.setattr(ds_mod, "partitioner", factory)

    suplier = ds_mod.DataSuplier([1, 2], training_percent=1.25)

    assert suplier.training_percent == 1.25
    assert calls[0][1] == [1.25, -0.25]


def test_partitioning_function_argument_is_forwarded(monkeypatch):
    factory, calls = make_partition_factory([0, 0, 1])
    monkeypatch.setattr(ds_mod, "partitioner", factory)

    marker = object()
    ds_mod.DataSuplier([1, 2, 3], partitioning_function=marker)

    assert calls[0][2] is marker


def test_multiple_iterators_share_the_same_internal_state(monkeypatch):
    factory, _ = make_partition_factory([0, 1, 0])
    monkeypatch.setattr(ds_mod, "partitioner", factory)

    suplier = ds_mod.DataSuplier(["a", "b", "c"])

    gen1 = suplier.training_set()
    gen2 = suplier.training_set()

    assert next(gen1) == (0, "a")
    # The second generator sees the same internal state and continues from there.
    assert list(gen2) == [(2, "c")]
    assert list(gen1) == []
