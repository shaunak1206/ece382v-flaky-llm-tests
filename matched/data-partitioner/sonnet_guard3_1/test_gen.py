import random

import pytest

from data_partitioner import data_suplier
from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import pseudorandom_function


class FakePartitioner(object):
    """Deterministic stand-in for the partitioner, driven by a preset sequence."""

    def __init__(self, n, percents, function, sequence):
        self.n = n
        self.percents = percents
        self.function = function
        self.sequence = list(sequence)
        self.calls = 0

    def __next__(self):
        value = self.sequence[self.calls]
        self.calls += 1
        return value


def install_fake(monkeypatch, sequence):
    created = []

    def factory(n, percents, function):
        fake = FakePartitioner(n, percents, function, sequence)
        created.append(fake)
        return fake

    monkeypatch.setattr(data_suplier, "partitioner", factory)
    return created


def test_init_stores_attributes_and_builds_partitioner(monkeypatch):
    random.seed(0)
    created = install_fake(monkeypatch, [0, 1, 0])
    data = ["a", "b", "c"]
    supplier = DataSuplier(data)
    assert supplier.dataset is data
    assert supplier.training_percent == 0.8
    assert supplier.n_of_elements == 3
    assert supplier.test_indexes == []
    assert supplier.training_indexes == []
    assert supplier.index == 0
    assert len(created) == 1
    fake = created[0]
    assert fake.n == 3
    assert fake.percents[0] == pytest.approx(0.8)
    assert fake.percents[1] == pytest.approx(0.2)
    assert fake.function is pseudorandom_function
    assert fake.calls == 0


def test_custom_partitioning_function_and_percent_passed(monkeypatch):
    random.seed(0)
    created = install_fake(monkeypatch, [0, 1])

    def my_function(*args, **kwargs):
        return 0

    DataSuplier([1, 2], training_percent=0.25, partitioning_function=my_function)
    fake = created[0]
    assert fake.function is my_function
    assert fake.percents[0] == pytest.approx(0.25)
    assert fake.percents[1] == pytest.approx(0.75)


def test_training_set_yields_index_and_element(monkeypatch):
    random.seed(0)
    install_fake(monkeypatch, [0, 1, 0, 0, 1])
    data = ["a", "b", "c", "d", "e"]
    supplier = DataSuplier(data)
    assert list(supplier.training_set()) == [(0, "a"), (2, "c"), (3, "d")]


def test_test_set_yields_index_and_element(monkeypatch):
    random.seed(0)
    install_fake(monkeypatch, [0, 1, 0, 0, 1])
    data = ["a", "b", "c", "d", "e"]
    supplier = DataSuplier(data)
    assert list(supplier.test_set()) == [(1, "b"), (4, "e")]


def test_creating_generator_does_not_consume_partitioner(monkeypatch):
    random.seed(0)
    created = install_fake(monkeypatch, [0, 1, 0])
    supplier = DataSuplier([1, 2, 3])
    supplier.training_set()
    supplier.test_set()
    assert created[0].calls == 0
    assert supplier.index == 0


def test_lazy_consumption_stops_at_first_match(monkeypatch):
    random.seed(0)
    created = install_fake(monkeypatch, [1, 1, 0, 1])
    supplier = DataSuplier(["a", "b", "c", "d"])
    gen = supplier.training_set()
    assert next(gen) == (2, "c")
    assert supplier.index == 3
    assert created[0].calls == 3
    assert supplier.test_indexes == [0, 1]
    assert supplier.training_indexes == [2]


def test_second_pass_uses_cache(monkeypatch):
    random.seed(0)
    created = install_fake(monkeypatch, [0, 1, 0, 1])
    supplier = DataSuplier([10, 20, 30, 40])
    first = list(supplier.training_set())
    calls_after_first = created[0].calls
    second = list(supplier.training_set())
    assert first == second == [(0, 10), (2, 30)]
    assert calls_after_first == 4
    assert created[0].calls == 4


def test_test_set_after_training_set_uses_cache(monkeypatch):
    random.seed(0)
    created = install_fake(monkeypatch, [0, 1, 1, 0])
    supplier = DataSuplier([10, 20, 30, 40])
    list(supplier.training_set())
    assert created[0].calls == 4
    assert list(supplier.test_set()) == [(1, 20), (2, 30)]
    assert created[0].calls == 4


def test_interleaved_generators(monkeypatch):
    random.seed(0)
    install_fake(monkeypatch, [0, 1, 0, 1])
    data = ["a", "b", "c", "d"]
    supplier = DataSuplier(data)
    tg = supplier.training_set()
    sg = supplier.test_set()
    assert next(tg) == (0, "a")
    assert next(sg) == (1, "b")
    assert next(tg) == (2, "c")
    assert next(sg) == (3, "d")
    assert list(tg) == []
    assert list(sg) == []


def test_two_generators_of_same_set_are_independent(monkeypatch):
    random.seed(0)
    install_fake(monkeypatch, [0, 0, 1])
    supplier = DataSuplier(["a", "b", "c"])
    g1 = supplier.training_set()
    g2 = supplier.training_set()
    assert next(g1) == (0, "a")
    assert next(g1) == (1, "b")
    assert next(g2) == (0, "a")
    assert list(g1) == []
    assert list(g2) == [(1, "b")]


def test_empty_dataset(monkeypatch):
    random.seed(0)
    created = install_fake(monkeypatch, [])
    supplier = DataSuplier([])
    assert supplier.n_of_elements == 0
    assert list(supplier.training_set()) == []
    assert list(supplier.test_set()) == []
    assert created[0].calls == 0


def test_all_elements_training(monkeypatch):
    random.seed(0)
    install_fake(monkeypatch, [0, 0, 0])
    supplier = DataSuplier(["x", "y", "z"])
    assert list(supplier.test_set()) == []
    assert list(supplier.training_set()) == [(0, "x"), (1, "y"), (2, "z")]


def test_all_elements_test(monkeypatch):
    random.seed(0)
    install_fake(monkeypatch, [1, 1, 1])
    supplier = DataSuplier(["x", "y", "z"])
    assert list(supplier.training_set()) == []
    assert list(supplier.test_set()) == [(0, "x"), (1, "y"), (2, "z")]


def test_nonzero_partition_values_go_to_test_set(monkeypatch):
    random.seed(0)
    install_fake(monkeypatch, [2, 0, 5])
    supplier = DataSuplier(["a", "b", "c"])
    assert list(supplier.training_set()) == [(1, "b")]
    assert list(supplier.test_set()) == [(0, "a"), (2, "c")]


def test_works_with_tuple_dataset(monkeypatch):
    random.seed(0)
    install_fake(monkeypatch, [1, 0])
    supplier = DataSuplier(("p", "q"))
    assert list(supplier.training_set()) == [(1, "q")]
    assert list(supplier.test_set()) == [(0, "p")]


def test_real_partitioner_covers_dataset_disjointly():
    random.seed(0)
    data = list(range(100, 150))
    supplier = DataSuplier(data)
    training = list(supplier.training_set())
    test = list(supplier.test_set())
    train_idx = [i for i, _ in training]
    test_idx = [i for i, _ in test]
    assert set(train_idx).isdisjoint(test_idx)
    assert sorted(train_idx + test_idx) == list(range(50))
    assert all(data[i] == v for i, v in training + test)


def test_real_partitioner_repeated_iteration_is_stable():
    random.seed(0)
    supplier = DataSuplier(list(range(30)))
    first_train = list(supplier.training_set())
    first_test = list(supplier.test_set())
    assert list(supplier.training_set()) == first_train
    assert list(supplier.test_set()) == first_test
    assert supplier.index == 30
    assert len(first_train) + len(first_test) == 30


def test_real_partitioner_indexes_are_increasing():
    random.seed(0)
    supplier = DataSuplier(list(range(40)))
    list(supplier.test_set())
    list(supplier.training_set())
    assert supplier.training_indexes == sorted(supplier.training_indexes)
    assert supplier.test_indexes == sorted(supplier.test_indexes)
    assert len(supplier.training_indexes) + len(supplier.test_indexes) == 40
