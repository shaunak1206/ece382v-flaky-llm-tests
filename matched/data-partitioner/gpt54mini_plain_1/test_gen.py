import pytest

import data_partitioner.data_suplier as data_suplier_mod
from data_partitioner.data_suplier import DataSuplier


class FakePartitioner:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def __next__(self):
        if self.calls >= len(self.results):
            raise StopIteration
        result = self.results[self.calls]
        self.calls += 1
        return result


def make_ds(monkeypatch, dataset, partition_results, training_percent=0.8):
    fake = FakePartitioner(partition_results)
    monkeypatch.setattr(
        data_suplier_mod,
        "partitioner",
        lambda n, probs, fn: fake,
    )
    ds = DataSuplier(dataset, training_percent=training_percent, partitioning_function=object())
    return ds, fake


def collect(gen):
    return list(gen)


def test_init_stores_dataset_and_percent(monkeypatch):
    dataset = ["a", "b", "c"]
    ds, _ = make_ds(monkeypatch, dataset, [0, 1, 0])
    assert ds.dataset == dataset
    assert ds.training_percent == 0.8
    assert ds.n_of_elements == 3


def test_init_constructs_partitioner_with_expected_arguments(monkeypatch):
    dataset = [1, 2, 3, 4]
    captured = {}

    def fake_partitioner(n, probs, fn):
        captured["n"] = n
        captured["probs"] = probs
        captured["fn"] = fn
        return FakePartitioner([0, 1, 0, 1])

    monkeypatch.setattr(data_suplier_mod, "partitioner", fake_partitioner)
    marker = object()
    DataSuplier(dataset, training_percent=0.25, partitioning_function=marker)

    assert captured["n"] == 4
    assert captured["probs"] == [0.25, 0.75]
    assert captured["fn"] is marker


def test_initial_indexes_are_empty(monkeypatch):
    ds, _ = make_ds(monkeypatch, [10, 20], [0, 1])
    assert ds.training_indexes == []
    assert ds.test_indexes == []
    assert ds.index == 0


def test_test_set_yields_expected_indexes_and_values(monkeypatch):
    dataset = ["x", "y", "z", "w"]
    ds, _ = make_ds(monkeypatch, dataset, [0, 1, 0, 1])
    assert collect(ds.test_set()) == [(1, "y"), (3, "w")]


def test_training_set_yields_expected_indexes_and_values(monkeypatch):
    dataset = ["x", "y", "z", "w"]
    ds, _ = make_ds(monkeypatch, dataset, [0, 1, 0, 1])
    assert collect(ds.training_set()) == [(0, "x"), (2, "z")]


def test_generators_are_lazy_and_do_not_partition_until_iterated(monkeypatch):
    ds, fake = make_ds(monkeypatch, [1, 2, 3], [0, 1, 0])
    gen = ds.training_set()
    assert fake.calls == 0
    first = next(gen)
    assert first == (0, 1)
    assert fake.calls == 1


def test_test_set_only_consumes_needed_partition_results(monkeypatch):
    ds, fake = make_ds(monkeypatch, [10, 20, 30, 40], [0, 1, 0, 1])
    gen = ds.test_set()
    assert next(gen) == (1, 20)
    assert fake.calls == 2  # partitions for indexes 0 and 1 only
    assert next(gen) == (3, 40)
    assert fake.calls == 4


def test_training_set_only_consumes_needed_partition_results(monkeypatch):
    ds, fake = make_ds(monkeypatch, [10, 20, 30, 40], [0, 1, 0, 1])
    gen = ds.training_set()
    assert next(gen) == (0, 10)
    assert fake.calls == 1
    assert next(gen) == (2, 30)
    assert fake.calls == 3


def test_multiple_generators_share_state(monkeypatch):
    ds, _ = make_ds(monkeypatch, ["a", "b", "c", "d"], [0, 1, 0, 1])
    tgen = ds.training_set()
    xgen = ds.test_set()

    assert next(tgen) == (0, "a")
    assert next(xgen) == (1, "b")
    assert next(tgen) == (2, "c")
    assert next(xgen) == (3, "d")


def test_empty_dataset_yields_nothing(monkeypatch):
    ds, fake = make_ds(monkeypatch, [], [])
    assert collect(ds.training_set()) == []
    assert collect(ds.test_set()) == []
    assert fake.calls == 0


def test_all_training_results(monkeypatch):
    ds, _ = make_ds(monkeypatch, ["a", "b", "c"], [0, 0, 0])
    assert collect(ds.training_set()) == [(0, "a"), (1, "b"), (2, "c")]
    assert collect(ds.test_set()) == []


def test_all_test_results(monkeypatch):
    ds, _ = make_ds(monkeypatch, ["a", "b", "c"], [1, 1, 1])
    assert collect(ds.test_set()) == [(0, "a"), (1, "b"), (2, "c")]
    assert collect(ds.training_set()) == []


def test_partitioning_function_argument_is_forwarded_even_if_unused(monkeypatch):
    forwarded = []

    def fake_partitioner(n, probs, fn):
        forwarded.append(fn)
        return FakePartitioner([0, 1])

    monkeypatch.setattr(data_suplier_mod, "partitioner", fake_partitioner)
    marker = object()
    DataSuplier(["a", "b"], partitioning_function=marker)
    assert forwarded == [marker]
