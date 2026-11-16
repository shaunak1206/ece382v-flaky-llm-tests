import random

import pytest

import data_partitioner.data_suplier as ds_module
from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import pseudorandom_function


class FakePartitioner(object):
    """Deterministic stand-in for the partitioner iterator."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def __next__(self):
        if self.calls >= len(self.results):
            raise StopIteration
        res = self.results[self.calls]
        self.calls += 1
        return res

    def __iter__(self):
        return self


@pytest.fixture
def make_supplier(monkeypatch):
    """Builds a DataSuplier whose partitioner yields a fixed sequence."""

    def _make(dataset, results, **kwargs):
        fake = FakePartitioner(results)
        captured = {}

        def factory(n, percents, func):
            captured["args"] = (n, percents, func)
            return fake

        monkeypatch.setattr(ds_module, "partitioner", factory)
        supplier = DataSuplier(dataset, **kwargs)
        return supplier, fake, captured

    return _make


def test_init_stores_attributes(make_supplier):
    random.seed(0)
    data = ["a", "b", "c"]
    supplier, fake, _ = make_supplier(data, [0, 0, 0])
    assert supplier.dataset is data
    assert supplier.training_percent == 0.8
    assert supplier.n_of_elements == 3
    assert supplier.test_indexes == []
    assert supplier.training_indexes == []
    assert supplier.index == 0
    assert fake.calls == 0


def test_default_partitioner_arguments(make_supplier):
    random.seed(0)
    _, _, captured = make_supplier(list(range(10)), [0] * 10)
    n, percents, func = captured["args"]
    assert n == 10
    assert len(percents) == 2
    assert percents[0] == pytest.approx(0.8)
    assert percents[1] == pytest.approx(0.2)
    assert func is pseudorandom_function


def test_custom_percent_and_function_are_forwarded(make_supplier):
    random.seed(0)

    def my_func(*args, **kwargs):
        return 0

    supplier, _, captured = make_supplier(
        list(range(4)), [0] * 4, training_percent=0.5, partitioning_function=my_func
    )
    n, percents, func = captured["args"]
    assert n == 4
    assert percents[0] == pytest.approx(0.5)
    assert percents[1] == pytest.approx(0.5)
    assert func is my_func
    assert supplier.training_percent == 0.5


def test_training_set_yields_index_and_element(make_supplier):
    random.seed(0)
    supplier, _, _ = make_supplier(["a", "b", "c", "d"], [0, 1, 0, 1])
    assert list(supplier.training_set()) == [(0, "a"), (2, "c")]


def test_test_set_yields_index_and_element(make_supplier):
    random.seed(0)
    supplier, _, _ = make_supplier(["a", "b", "c", "d"], [0, 1, 0, 1])
    assert list(supplier.test_set()) == [(1, "b"), (3, "d")]


def test_generators_are_lazy(make_supplier):
    random.seed(0)
    supplier, fake, _ = make_supplier(["a", "b", "c", "d"], [1, 1, 0, 1])
    gen = supplier.training_set()
    assert fake.calls == 0
    assert next(gen) == (2, "c")
    assert fake.calls == 3
    assert supplier.index == 3
    assert supplier.test_indexes == [0, 1]
    assert supplier.training_indexes == [2]


def test_interleaved_generators_are_consistent(make_supplier):
    random.seed(0)
    data = list("abcde")
    supplier, _, _ = make_supplier(data, [0, 1, 0, 1, 0])
    train_gen = supplier.training_set()
    test_gen = supplier.test_set()
    assert next(train_gen) == (0, "a")
    assert next(test_gen) == (1, "b")
    assert next(train_gen) == (2, "c")
    assert next(test_gen) == (3, "d")
    assert next(train_gen) == (4, "e")
    assert list(train_gen) == []
    assert list(test_gen) == []


def test_full_iteration_consumes_partitioner_exactly_n_times(make_supplier):
    random.seed(0)
    supplier, fake, _ = make_supplier(list(range(6)), [0, 1, 1, 0, 0, 1])
    list(supplier.training_set())
    assert fake.calls == 6
    assert supplier.index == 6
    assert supplier.training_indexes == [0, 3, 4]
    assert supplier.test_indexes == [1, 2, 5]


def test_test_set_after_training_set_does_not_advance_partitioner(make_supplier):
    random.seed(0)
    supplier, fake, _ = make_supplier(list("abcd"), [1, 0, 1, 0])
    list(supplier.training_set())
    calls_before = fake.calls
    assert list(supplier.test_set()) == [(0, "a"), (2, "c")]
    assert fake.calls == calls_before


def test_repeated_calls_give_same_results(make_supplier):
    random.seed(0)
    supplier, _, _ = make_supplier(list("abcde"), [0, 1, 0, 0, 1])
    first = list(supplier.training_set())
    second = list(supplier.training_set())
    assert first == second == [(0, "a"), (2, "c"), (3, "d")]
    assert list(supplier.test_set()) == list(supplier.test_set())


def test_empty_dataset(make_supplier):
    random.seed(0)
    supplier, fake, _ = make_supplier([], [])
    assert list(supplier.training_set()) == []
    assert list(supplier.test_set()) == []
    assert fake.calls == 0
    assert supplier.n_of_elements == 0


def test_all_elements_in_training(make_supplier):
    random.seed(0)
    supplier, _, _ = make_supplier([10, 20, 30], [0, 0, 0])
    assert list(supplier.test_set()) == []
    assert list(supplier.training_set()) == [(0, 10), (1, 20), (2, 30)]


def test_all_elements_in_test(make_supplier):
    random.seed(0)
    supplier, _, _ = make_supplier([10, 20, 30], [1, 1, 1])
    assert list(supplier.training_set()) == []
    assert list(supplier.test_set()) == [(0, 10), (1, 20), (2, 30)]


def test_nonzero_result_goes_to_test_set(make_supplier):
    random.seed(0)
    supplier, _, _ = make_supplier(["a", "b", "c"], [0, 2, 1])
    assert list(supplier.test_set()) == [(1, "b"), (2, "c")]
    assert supplier.training_indexes == [0]


def test_works_with_string_dataset(make_supplier):
    random.seed(0)
    supplier, _, _ = make_supplier("xyz", [1, 0, 1])
    assert list(supplier.training_set()) == [(1, "y")]
    assert list(supplier.test_set()) == [(0, "x"), (2, "z")]


def test_real_partitioner_splits_every_element_exactly_once():
    random.seed(0)
    data = [i * 3 for i in range(30)]
    supplier = DataSuplier(data)
    training = list(supplier.training_set())
    test = list(supplier.test_set())
    train_idx = [i for i, _ in training]
    test_idx = [i for i, _ in test]
    assert sorted(train_idx + test_idx) == list(range(30))
    assert set(train_idx).isdisjoint(test_idx)
    for i, value in training + test:
        assert value == data[i]
    assert train_idx == sorted(train_idx)
    assert test_idx == sorted(test_idx)
