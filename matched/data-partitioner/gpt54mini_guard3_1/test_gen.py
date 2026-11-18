import pytest

import data_partitioner.data_suplier as suplier_mod


class FakePartitioner:
    def __init__(self, outcomes, call_log):
        self._outcomes = list(outcomes)
        self._call_log = call_log
        self._i = 0

    def __next__(self):
        if self._i >= len(self._outcomes):
            raise StopIteration
        value = self._outcomes[self._i]
        self._i += 1
        self._call_log.append(value)
        return value


def make_partitioner_factory(outcomes, call_log, captured_args=None):
    def factory(n_of_elements, probs, partitioning_function):
        if captured_args is not None:
            captured_args["n_of_elements"] = n_of_elements
            captured_args["probs"] = list(probs)
            captured_args["partitioning_function"] = partitioning_function
        return FakePartitioner(outcomes, call_log)

    return factory


def collect(iterator):
    return list(iterator)


def test_init_sets_basic_attributes(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        suplier_mod,
        "partitioner",
        make_partitioner_factory([0, 1], [], captured),
    )

    dataset = ["a", "b", "c"]
    ds = suplier_mod.DataSuplier(dataset, training_percent=0.75, partitioning_function="pf")

    assert ds.dataset is dataset
    assert ds.training_percent == 0.75
    assert ds.n_of_elements == 3
    assert ds.index == 0
    assert ds.training_indexes == []
    assert ds.test_indexes == []
    assert captured["n_of_elements"] == 3
    assert captured["probs"] == [0.75, 0.25]
    assert captured["partitioning_function"] == "pf"


def test_init_uses_default_training_percent(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        suplier_mod,
        "partitioner",
        make_partitioner_factory([], [], captured),
    )

    ds = suplier_mod.DataSuplier([1, 2])

    assert ds.training_percent == 0.8
    assert captured["probs"] == [0.8, 0.19999999999999996]


def test_training_set_all_training(monkeypatch):
    monkeypatch.setattr(suplier_mod, "partitioner", make_partitioner_factory([0, 0, 0], []))

    dataset = ["x", "y", "z"]
    ds = suplier_mod.DataSuplier(dataset)

    assert collect(ds.training_set()) == [(0, "x"), (1, "y"), (2, "z")]
    assert ds.test_indexes == []
    assert ds.training_indexes == [0, 1, 2]
    assert ds.index == 3


def test_test_set_all_test(monkeypatch):
    monkeypatch.setattr(suplier_mod, "partitioner", make_partitioner_factory([1, 1, 1], []))

    dataset = ["x", "y", "z"]
    ds = suplier_mod.DataSuplier(dataset)

    assert collect(ds.test_set()) == [(0, "x"), (1, "y"), (2, "z")]
    assert ds.training_indexes == []
    assert ds.test_indexes == [0, 1, 2]
    assert ds.index == 3


def test_mixed_partitioning_produces_expected_training_and_test_sets(monkeypatch):
    monkeypatch.setattr(suplier_mod, "partitioner", make_partitioner_factory([0, 1, 0, 1], []))

    dataset = ["a", "b", "c", "d"]
    ds = suplier_mod.DataSuplier(dataset)

    assert collect(ds.training_set()) == [(0, "a"), (2, "c")]
    assert collect(ds.test_set()) == [(1, "b"), (3, "d")]
    assert ds.training_indexes == [0, 2]
    assert ds.test_indexes == [1, 3]
    assert ds.index == 4


def test_yield_set_is_lazy_until_consumed(monkeypatch):
    call_log = []
    monkeypatch.setattr(suplier_mod, "partitioner", make_partitioner_factory([1, 0], call_log))

    ds = suplier_mod.DataSuplier(["a", "b"])

    gen = ds.training_set()
    assert call_log == []
    assert next(gen) == (1, "b")
    assert call_log == [1, 0]
    with pytest.raises(StopIteration):
        next(gen)


def test_training_and_test_sets_can_be_consumed_interleaved(monkeypatch):
    monkeypatch.setattr(
        suplier_mod,
        "partitioner",
        make_partitioner_factory([0, 1, 0, 1], []),
    )

    ds = suplier_mod.DataSuplier(["a", "b", "c", "d"])

    train_gen = ds.training_set()
    test_gen = ds.test_set()

    assert next(train_gen) == (0, "a")
    assert next(test_gen) == (1, "b")
    assert next(train_gen) == (2, "c")
    assert next(test_gen) == (3, "d")
    with pytest.raises(StopIteration):
        next(train_gen)
    with pytest.raises(StopIteration):
        next(test_gen)


def test_empty_dataset_returns_empty_generators(monkeypatch):
    monkeypatch.setattr(suplier_mod, "partitioner", make_partitioner_factory([], []))

    ds = suplier_mod.DataSuplier([])

    assert collect(ds.training_set()) == []
    assert collect(ds.test_set()) == []
    assert ds.index == 0
    assert ds.training_indexes == []
    assert ds.test_indexes == []


def test_generators_can_be_called_multiple_times_and_restart_from_beginning_of_index_list(monkeypatch):
    monkeypatch.setattr(suplier_mod, "partitioner", make_partitioner_factory([0, 1, 0], []))

    ds = suplier_mod.DataSuplier(["a", "b", "c"])

    first = collect(ds.training_set())
    second = collect(ds.training_set())

    assert first == [(0, "a"), (2, "c")]
    assert second == [(0, "a"), (2, "c")]


def test_consuming_test_set_first_then_training_set(monkeypatch):
    monkeypatch.setattr(suplier_mod, "partitioner", make_partitioner_factory([1, 0, 1], []))

    ds = suplier_mod.DataSuplier(["a", "b", "c"])

    assert collect(ds.test_set()) == [(0, "a"), (2, "c")]
    assert collect(ds.training_set()) == [(1, "b")]
    assert ds.index == 3


def test_consuming_training_set_first_then_test_set(monkeypatch):
    monkeypatch.setattr(suplier_mod, "partitioner", make_partitioner_factory([0, 1, 1], []))

    ds = suplier_mod.DataSuplier(["a", "b", "c"])

    assert collect(ds.training_set()) == [(0, "a")]
    assert collect(ds.test_set()) == [(1, "b"), (2, "c")]
    assert ds.index == 3


def test_dataset_items_are_returned_by_index_not_by_partitioner_output(monkeypatch):
    monkeypatch.setattr(suplier_mod, "partitioner", make_partitioner_factory([0, 0, 1], []))

    dataset = ["zero", "one", "two"]
    ds = suplier_mod.DataSuplier(dataset)

    assert collect(ds.training_set()) == [(0, "zero"), (1, "one")]
    assert collect(ds.test_set()) == [(2, "two")]


def test_partitioning_function_passed_through_unchanged(monkeypatch):
    captured = {}
    sentinel = object()

    monkeypatch.setattr(
        suplier_mod,
        "partitioner",
        make_partitioner_factory([0], [], captured),
    )

    suplier_mod.DataSuplier([1], partitioning_function=sentinel)

    assert captured["partitioning_function"] is sentinel


def test_more_than_one_generator_reference_shares_underlying_state(monkeypatch):
    monkeypatch.setattr(suplier_mod, "partitioner", make_partitioner_factory([0, 1], []))

    ds = suplier_mod.DataSuplier(["a", "b"])
    g1 = ds.training_set()
    g2 = ds.training_set()

    assert next(g1) == (0, "a")
    assert next(g2) == (0, "a")
    assert collect(g1) == [(0, "a")]
    assert collect(g2) == [(0, "a")]
