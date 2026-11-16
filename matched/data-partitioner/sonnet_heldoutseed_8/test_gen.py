import pytest

import data_partitioner.data_suplier as ds_module
from data_partitioner.data_suplier import DataSuplier


class FakePartitioner(object):
    """Deterministic stand-in for the real partitioner."""

    def __init__(self, sequence):
        self._it = iter(sequence)
        self.calls = 0

    def __next__(self):
        self.calls += 1
        return next(self._it)

    next = __next__

    def __iter__(self):
        return self


@pytest.fixture
def fake(monkeypatch):
    """Install a fake partitioner factory; returns a function to configure it."""
    state = {}

    def configure(sequence):
        def factory(n, percents, function):
            state["args"] = (n, percents, function)
            state["instance"] = FakePartitioner(sequence)
            return state["instance"]

        monkeypatch.setattr(ds_module, "partitioner", factory)
        return state

    return configure


def test_init_stores_attributes_and_initial_state():
    data = list(range(10))
    sup = DataSuplier(data)
    assert sup.dataset is data
    assert sup.training_percent == 0.8
    assert sup.n_of_elements == 10
    assert sup.test_indexes == []
    assert sup.training_indexes == []
    assert sup.index == 0


def test_partitioner_receives_expected_arguments(fake):
    state = fake([0] * 5)
    func = object()
    DataSuplier(list(range(5)), training_percent=0.6, partitioning_function=func)
    n, percents, function = state["args"]
    assert n == 5
    assert percents[0] == 0.6
    assert percents[1] == pytest.approx(0.4)
    assert function is func


def test_default_partitioning_function_is_passed(fake):
    state = fake([0] * 3)
    DataSuplier([1, 2, 3])
    assert state["args"][0] == 3
    assert state["args"][1][0] == 0.8
    assert callable(state["args"][2])


def test_training_set_yields_index_and_element_pairs(fake):
    fake([0, 1, 0, 1, 0])
    sup = DataSuplier(["a", "b", "c", "d", "e"])
    assert list(sup.training_set()) == [(0, "a"), (2, "c"), (4, "e")]


def test_test_set_yields_index_and_element_pairs(fake):
    fake([0, 1, 0, 1, 0])
    sup = DataSuplier(["a", "b", "c", "d", "e"])
    assert list(sup.test_set()) == [(1, "b"), (3, "d")]


def test_all_training_leaves_test_set_empty(fake):
    fake([0, 0, 0])
    sup = DataSuplier([10, 20, 30])
    assert list(sup.test_set()) == []
    assert list(sup.training_set()) == [(0, 10), (1, 20), (2, 30)]


def test_all_test_leaves_training_set_empty(fake):
    fake([1, 1, 1])
    sup = DataSuplier([10, 20, 30])
    assert list(sup.training_set()) == []
    assert list(sup.test_set()) == [(0, 10), (1, 20), (2, 30)]


def test_empty_dataset_yields_nothing(fake):
    fake([])
    sup = DataSuplier([])
    assert list(sup.training_set()) == []
    assert list(sup.test_set()) == []
    assert sup.n_of_elements == 0


def test_generation_is_lazy(fake):
    state = fake([1, 1, 0, 1])
    sup = DataSuplier(["a", "b", "c", "d"])
    gen = sup.training_set()
    assert state["instance"].calls == 0
    assert next(gen) == (2, "c")
    assert sup.index == 3
    assert state["instance"].calls == 3
    assert sup.test_indexes == [0, 1]
    assert sup.training_indexes == [2]


def test_interleaved_generators_cover_everything(fake):
    fake([0, 1, 0, 1, 0])
    data = ["a", "b", "c", "d", "e"]
    sup = DataSuplier(data)
    train_gen = sup.training_set()
    test_gen = sup.test_set()
    train = [next(train_gen)]
    test = [next(test_gen)]
    train.extend(train_gen)
    test.extend(test_gen)
    assert train == [(0, "a"), (2, "c"), (4, "e")]
    assert test == [(1, "b"), (3, "d")]


def test_test_set_consumed_before_training_set(fake):
    fake([0, 1, 1, 0])
    sup = DataSuplier([1, 2, 3, 4])
    assert list(sup.test_set()) == [(1, 2), (2, 3)]
    assert list(sup.training_set()) == [(0, 1), (3, 4)]


def test_repeated_calls_return_same_results(fake):
    state = fake([0, 1, 0, 0, 1, 1])
    sup = DataSuplier(list("abcdef"))
    first = list(sup.training_set())
    calls_after_first = state["instance"].calls
    second = list(sup.training_set())
    assert first == second
    assert state["instance"].calls == calls_after_first
    assert list(sup.test_set()) == list(sup.test_set())


def test_partial_consumption_then_new_generator_restarts_from_beginning(fake):
    fake([0, 0, 0, 0])
    sup = DataSuplier([1, 2, 3, 4])
    gen = sup.training_set()
    assert next(gen) == (0, 1)
    assert next(gen) == (1, 2)
    assert list(sup.training_set()) == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_real_partitioner_splits_disjointly_and_completely():
    data = list(range(100))
    sup = DataSuplier(data)
    train = list(sup.training_set())
    test = list(sup.test_set())
    train_idx = [i for i, _ in train]
    test_idx = [i for i, _ in test]
    assert set(train_idx).isdisjoint(test_idx)
    assert sorted(train_idx + test_idx) == list(range(100))
    assert all(data[i] == v for i, v in train + test)


def test_real_partitioner_results_are_stable_within_instance():
    sup = DataSuplier(list(range(50)))
    first_train = list(sup.training_set())
    first_test = list(sup.test_set())
    assert list(sup.training_set()) == first_train
    assert list(sup.test_set()) == first_test


def test_indexes_are_in_increasing_order(fake):
    fake([1, 0, 0, 1, 0, 1, 1, 0])
    sup = DataSuplier(list(range(8)))
    train_idx = [i for i, _ in sup.training_set()]
    test_idx = [i for i, _ in sup.test_set()]
    assert train_idx == sorted(train_idx)
    assert test_idx == sorted(test_idx)
    assert sorted(train_idx + test_idx) == list(range(8))


def test_works_with_tuple_dataset(fake):
    fake([0, 1, 0])
    sup = DataSuplier(("x", "y", "z"))
    assert list(sup.training_set()) == [(0, "x"), (2, "z")]
    assert list(sup.test_set()) == [(1, "y")]
