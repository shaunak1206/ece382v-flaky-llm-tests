import pytest

import data_partitioner.data_suplier as ds_module
from data_partitioner.data_suplier import DataSuplier


def _install_fake_partitioner(monkeypatch, values, calls=None):
    """Replace the partitioner used by DataSuplier with one that yields fixed values."""

    def fake(n, percents, func):
        if calls is not None:
            calls.append((n, percents, func))
        return (v for v in values)

    monkeypatch.setattr(ds_module, "partitioner", fake)


DATA = ["a", "b", "c", "d", "e"]
PATTERN = [0, 1, 0, 0, 1]  # train, test, train, train, test


def test_init_attributes():
    data = list(range(10))
    supplier = DataSuplier(data, training_percent=0.7)
    assert supplier.dataset is data
    assert supplier.training_percent == 0.7
    assert supplier.n_of_elements == 10
    assert supplier.test_indexes == []
    assert supplier.training_indexes == []
    assert supplier.index == 0


def test_default_training_percent():
    supplier = DataSuplier(list(range(10)))
    assert supplier.training_percent == 0.8


def test_partitioner_called_with_expected_arguments(monkeypatch):
    calls = []
    _install_fake_partitioner(monkeypatch, PATTERN, calls)

    def custom_function(*args, **kwargs):
        return 0

    DataSuplier(DATA, training_percent=0.8, partitioning_function=custom_function)
    assert len(calls) == 1
    n, percents, func = calls[0]
    assert n == 5
    assert percents[0] == pytest.approx(0.8)
    assert percents[1] == pytest.approx(0.2)
    assert func is custom_function


def test_default_partitioning_function_is_pseudorandom(monkeypatch):
    calls = []
    _install_fake_partitioner(monkeypatch, PATTERN, calls)
    DataSuplier(DATA)
    assert calls[0][2] is ds_module.pseudorandom_function


def test_training_set_yields_index_and_element(monkeypatch):
    _install_fake_partitioner(monkeypatch, PATTERN)
    supplier = DataSuplier(DATA)
    assert list(supplier.training_set()) == [(0, "a"), (2, "c"), (3, "d")]


def test_test_set_yields_index_and_element(monkeypatch):
    _install_fake_partitioner(monkeypatch, PATTERN)
    supplier = DataSuplier(DATA)
    assert list(supplier.test_set()) == [(1, "b"), (4, "e")]


def test_test_set_after_training_set_consumed(monkeypatch):
    _install_fake_partitioner(monkeypatch, PATTERN)
    supplier = DataSuplier(DATA)
    training = list(supplier.training_set())
    test = list(supplier.test_set())
    assert training == [(0, "a"), (2, "c"), (3, "d")]
    assert test == [(1, "b"), (4, "e")]
    assert supplier.training_indexes == [0, 2, 3]
    assert supplier.test_indexes == [1, 4]
    assert supplier.index == 5


def test_interleaved_generators(monkeypatch):
    _install_fake_partitioner(monkeypatch, PATTERN)
    supplier = DataSuplier(DATA)
    training = supplier.training_set()
    test = supplier.test_set()
    assert next(test) == (1, "b")
    assert next(training) == (0, "a")
    assert next(training) == (2, "c")
    assert next(test) == (4, "e")
    assert next(training) == (3, "d")
    with pytest.raises(StopIteration):
        next(training)
    with pytest.raises(StopIteration):
        next(test)


def test_generation_is_lazy(monkeypatch):
    _install_fake_partitioner(monkeypatch, [1, 1, 0, 0, 0])
    supplier = DataSuplier(DATA)
    gen = supplier.training_set()
    assert supplier.index == 0
    assert next(gen) == (2, "c")
    assert supplier.index == 3
    assert supplier.test_indexes == [0, 1]
    assert supplier.training_indexes == [2]


def test_empty_dataset():
    supplier = DataSuplier([])
    assert supplier.n_of_elements == 0
    assert list(supplier.training_set()) == []
    assert list(supplier.test_set()) == []


def test_all_training(monkeypatch):
    _install_fake_partitioner(monkeypatch, [0] * 5)
    supplier = DataSuplier(DATA, training_percent=1.0)
    assert list(supplier.test_set()) == []
    assert list(supplier.training_set()) == list(enumerate(DATA))


def test_all_test(monkeypatch):
    _install_fake_partitioner(monkeypatch, [1] * 5)
    supplier = DataSuplier(DATA, training_percent=0.0)
    assert list(supplier.training_set()) == []
    assert list(supplier.test_set()) == list(enumerate(DATA))


def test_set_can_be_iterated_twice(monkeypatch):
    _install_fake_partitioner(monkeypatch, PATTERN)
    supplier = DataSuplier(DATA)
    first = list(supplier.training_set())
    second = list(supplier.training_set())
    assert first == second == [(0, "a"), (2, "c"), (3, "d")]
    assert supplier.training_indexes == [0, 2, 3]


def test_works_with_string_dataset(monkeypatch):
    _install_fake_partitioner(monkeypatch, [0, 1, 0])
    supplier = DataSuplier("xyz")
    assert supplier.n_of_elements == 3
    assert list(supplier.training_set()) == [(0, "x"), (2, "z")]
    assert list(supplier.test_set()) == [(1, "y")]


def test_default_partition_is_disjoint_and_complete():
    data = list(range(200))
    supplier = DataSuplier(data)
    training = list(supplier.training_set())
    test = list(supplier.test_set())
    training_idx = [i for i, _ in training]
    test_idx = [i for i, _ in test]
    assert set(training_idx).isdisjoint(test_idx)
    assert sorted(training_idx + test_idx) == list(range(200))
    for i, element in training + test:
        assert element == data[i]


def test_default_partition_ratio_is_roughly_respected():
    supplier = DataSuplier(list(range(1000)), training_percent=0.8)
    training = list(supplier.training_set())
    fraction = len(training) / 1000.0
    assert 0.6 <= fraction <= 0.95
