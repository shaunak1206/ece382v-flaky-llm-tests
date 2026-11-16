"""Unit tests for data_partitioner.data_suplier.DataSuplier."""

import inspect

import pytest

from data_partitioner import data_suplier
from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import pseudorandom_function


def install_fake_partitioner(monkeypatch, sequence):
    """Replace the partitioner used by data_suplier with a deterministic one.

    Returns a list that records the arguments the constructor passes to it.
    """
    calls = []

    def fake_partitioner(n, percents, partitioning_function):
        calls.append((n, percents, partitioning_function))
        return iter(sequence)

    monkeypatch.setattr(data_suplier, "partitioner", fake_partitioner)
    return calls


def collect_indexes(pairs):
    return [index for index, _ in pairs]


def test_init_stores_dataset_and_training_percent():
    dataset = ["a", "b", "c"]
    supplier = DataSuplier(dataset, training_percent=0.5)
    assert supplier.dataset is dataset
    assert supplier.training_percent == 0.5


def test_init_records_number_of_elements():
    supplier = DataSuplier(list(range(17)))
    assert supplier.n_of_elements == 17


def test_init_starts_with_empty_index_lists_and_zero_index():
    supplier = DataSuplier(list(range(10)))
    assert supplier.training_indexes == []
    assert supplier.test_indexes == []
    assert supplier.index == 0


def test_default_training_percent_is_0_8():
    supplier = DataSuplier([1, 2, 3])
    assert supplier.training_percent == 0.8


def test_default_partitioning_function_is_pseudorandom_function():
    default = inspect.signature(DataSuplier.__init__).parameters[
        "partitioning_function"
    ].default
    assert default is pseudorandom_function


def test_partitioner_is_built_with_size_percentages_and_function(monkeypatch):
    calls = install_fake_partitioner(monkeypatch, [0] * 4)

    def custom_function(*args, **kwargs):
        return 0

    DataSuplier(list(range(4)), training_percent=0.75,
                partitioning_function=custom_function)

    assert len(calls) == 1
    n, percents, func = calls[0]
    assert n == 4
    assert percents[0] == pytest.approx(0.75)
    assert percents[1] == pytest.approx(0.25)
    assert func is custom_function


def test_construction_does_not_consume_partitioner(monkeypatch):
    install_fake_partitioner(monkeypatch, [0, 1, 0])
    supplier = DataSuplier(["x", "y", "z"])
    assert supplier.index == 0
    assert supplier.training_indexes == []
    assert supplier.test_indexes == []


def test_training_set_returns_generator(monkeypatch):
    install_fake_partitioner(monkeypatch, [0, 1])
    supplier = DataSuplier(["a", "b"])
    gen = supplier.training_set()
    assert inspect.isgenerator(gen)


def test_training_set_creation_is_lazy(monkeypatch):
    install_fake_partitioner(monkeypatch, [0, 1, 0, 1])
    supplier = DataSuplier(["a", "b", "c", "d"])
    supplier.training_set()  # generator created but not iterated
    assert supplier.index == 0
    assert supplier.training_indexes == []


def test_training_set_yields_index_and_item_pairs(monkeypatch):
    install_fake_partitioner(monkeypatch, [0, 1, 0, 1])
    supplier = DataSuplier(["a", "b", "c", "d"])
    assert list(supplier.training_set()) == [(0, "a"), (2, "c")]


def test_test_set_yields_index_and_item_pairs_after_training(monkeypatch):
    install_fake_partitioner(monkeypatch, [0, 1, 0, 1])
    supplier = DataSuplier(["a", "b", "c", "d"])
    list(supplier.training_set())
    assert list(supplier.test_set()) == [(1, "b"), (3, "d")]


def test_test_set_before_training_set_gives_same_results(monkeypatch):
    install_fake_partitioner(monkeypatch, [0, 1, 0, 1])
    supplier = DataSuplier(["a", "b", "c", "d"])
    assert list(supplier.test_set()) == [(1, "b"), (3, "d")]
    assert list(supplier.training_set()) == [(0, "a"), (2, "c")]


def test_training_set_consumes_partitioner_lazily(monkeypatch):
    install_fake_partitioner(monkeypatch, [1, 1, 0])
    supplier = DataSuplier(["a", "b", "c"])
    gen = supplier.training_set()
    first = next(gen)
    # Only the elements up to the first training element were partitioned.
    assert first == (2, "c")
    assert supplier.test_indexes == [0, 1]
    assert supplier.training_indexes == [2]
    assert supplier.index == 3


def test_repeated_training_set_calls_yield_same_results(monkeypatch):
    install_fake_partitioner(monkeypatch, [0, 1, 1, 0])
    supplier = DataSuplier(["a", "b", "c", "d"])
    first = list(supplier.training_set())
    second = list(supplier.training_set())
    assert first == second == [(0, "a"), (3, "d")]


def test_yielded_items_are_the_original_objects(monkeypatch):
    install_fake_partitioner(monkeypatch, [0, 1])
    obj_a = {"id": 1}
    obj_b = {"id": 2}
    supplier = DataSuplier([obj_a, obj_b])
    (index, item), = list(supplier.training_set())
    assert index == 0
    assert item is obj_a


def test_empty_dataset_yields_empty_sets(monkeypatch):
    install_fake_partitioner(monkeypatch, [])
    supplier = DataSuplier([])
    assert list(supplier.training_set()) == []
    assert list(supplier.test_set()) == []


def test_all_training_elements_when_partitioner_only_yields_zero(monkeypatch):
    install_fake_partitioner(monkeypatch, [0, 0, 0])
    supplier = DataSuplier(["x", "y", "z"])
    assert list(supplier.training_set()) == [(0, "x"), (1, "y"), (2, "z")]
    assert list(supplier.test_set()) == []


def test_all_test_elements_when_partitioner_only_yields_one(monkeypatch):
    install_fake_partitioner(monkeypatch, [1, 1])
    supplier = DataSuplier(["x", "y"])
    assert list(supplier.training_set()) == []
    assert list(supplier.test_set()) == [(0, "x"), (1, "y")]


def test_real_partitioner_splits_dataset_into_disjoint_complete_sets():
    dataset = list(range(50))
    supplier = DataSuplier(dataset)
    train = list(supplier.training_set())
    test = list(supplier.test_set())

    train_idx = collect_indexes(train)
    test_idx = collect_indexes(test)

    assert set(train_idx).isdisjoint(test_idx)
    assert sorted(train_idx + test_idx) == list(range(50))
    for index, item in train + test:
        assert item == dataset[index]


def test_real_partitioner_is_deterministic_across_instances():
    dataset = list(range(40))
    first = DataSuplier(dataset)
    second = DataSuplier(dataset)
    assert list(first.training_set()) == list(second.training_set())
    assert list(first.test_set()) == list(second.test_set())
