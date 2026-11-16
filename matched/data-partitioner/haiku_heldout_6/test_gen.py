import pytest

import data_partitioner.data_suplier as suplier_module
from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import pseudorandom_function


@pytest.fixture
def scripted(monkeypatch):
    """Replace the partitioner used by DataSuplier with a fake that yields a fixed script.

    Returns a function install(script) -> list of recorded constructor calls.
    """
    calls = []

    def install(script):
        class FakePartitioner(object):
            def __init__(self, n, probabilities, partitioning_function):
                calls.append({
                    "n": n,
                    "probabilities": probabilities,
                    "partitioning_function": partitioning_function,
                })
                self._iter = iter(script)

            def __next__(self):
                return next(self._iter)

        monkeypatch.setattr(suplier_module, "partitioner", FakePartitioner)
        return calls

    return install


def test_init_stores_dataset_and_training_percent(scripted):
    scripted([0, 1])
    data = ["a", "b"]
    suplier = DataSuplier(data, training_percent=0.5)
    assert suplier.dataset is data
    assert suplier.training_percent == 0.5


def test_init_records_number_of_elements(scripted):
    scripted([0, 1, 0])
    suplier = DataSuplier(["x", "y", "z"])
    assert suplier.n_of_elements == 3


def test_init_default_training_percent_is_0_8(scripted):
    calls = scripted([0])
    suplier = DataSuplier([1])
    assert suplier.training_percent == 0.8
    assert calls[0]["probabilities"] == [0.8, 1 - 0.8]


def test_init_builds_partitioner_with_length_and_probabilities(scripted):
    calls = scripted([0, 1, 0, 0])
    DataSuplier(list(range(4)), training_percent=0.75)
    assert len(calls) == 1
    assert calls[0]["n"] == 4
    assert calls[0]["probabilities"] == [0.75, 1 - 0.75]


def test_init_uses_default_pseudorandom_partitioning_function(scripted):
    calls = scripted([0])
    DataSuplier([1])
    assert calls[0]["partitioning_function"] is pseudorandom_function


def test_init_passes_custom_partitioning_function(scripted):
    calls = scripted([0])

    def custom(*args, **kwargs):
        return 0

    DataSuplier([1], partitioning_function=custom)
    assert calls[0]["partitioning_function"] is custom


def test_init_initial_state_is_empty(scripted):
    scripted([0, 1])
    suplier = DataSuplier(["a", "b"])
    assert suplier.training_indexes == []
    assert suplier.test_indexes == []
    assert suplier.index == 0


def test_training_set_yields_index_and_value_pairs(scripted):
    scripted([0, 1, 0, 1])
    suplier = DataSuplier(["a", "b", "c", "d"])
    assert list(suplier.training_set()) == [(0, "a"), (2, "c")]


def test_test_set_yields_index_and_value_pairs(scripted):
    scripted([0, 1, 0, 1])
    suplier = DataSuplier(["a", "b", "c", "d"])
    assert list(suplier.test_set()) == [(1, "b"), (3, "d")]


def test_training_and_test_sets_are_disjoint_and_cover_all(scripted):
    scripted([1, 0, 0, 1, 0])
    data = list(range(100, 105))
    suplier = DataSuplier(data)
    train = [i for i, _ in suplier.training_set()]
    test = [i for i, _ in suplier.test_set()]
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == list(range(5))


def test_indexes_are_yielded_in_ascending_order(scripted):
    scripted([0, 1, 1, 0, 1, 0])
    suplier = DataSuplier(list(range(6)))
    train = [i for i, _ in suplier.training_set()]
    test = [i for i, _ in suplier.test_set()]
    assert train == sorted(train)
    assert test == sorted(test)


def test_all_elements_to_training_leaves_test_set_empty(scripted):
    scripted([0, 0, 0])
    suplier = DataSuplier(["a", "b", "c"])
    assert list(suplier.test_set()) == []
    assert list(suplier.training_set()) == [(0, "a"), (1, "b"), (2, "c")]


def test_all_elements_to_test_leaves_training_set_empty(scripted):
    scripted([1, 1])
    suplier = DataSuplier(["a", "b"])
    assert list(suplier.training_set()) == []
    assert list(suplier.test_set()) == [(0, "a"), (1, "b")]


def test_empty_dataset_yields_nothing(scripted):
    scripted([])
    suplier = DataSuplier([])
    assert list(suplier.training_set()) == []
    assert list(suplier.test_set()) == []


def test_generators_are_lazy_and_do_not_advance_partitioner(scripted):
    scripted([0, 1])
    suplier = DataSuplier(["a", "b"])
    gen = suplier.training_set()
    assert suplier.index == 0
    assert suplier.training_indexes == []
    next(gen)
    assert suplier.index >= 1


def test_interleaved_consumption_of_both_sets(scripted):
    scripted([0, 1, 0, 1])
    suplier = DataSuplier(["a", "b", "c", "d"])
    train_gen = suplier.training_set()
    test_gen = suplier.test_set()
    assert next(train_gen) == (0, "a")
    assert next(test_gen) == (1, "b")
    assert next(train_gen) == (2, "c")
    assert next(test_gen) == (3, "d")
    assert list(train_gen) == []
    assert list(test_gen) == []


def test_repeated_iteration_returns_same_results(scripted):
    scripted([0, 1, 0])
    suplier = DataSuplier(["a", "b", "c"])
    first = list(suplier.training_set())
    second = list(suplier.training_set())
    assert first == second == [(0, "a"), (2, "c")]


def test_index_advances_to_length_after_full_consumption(scripted):
    scripted([0, 1, 1, 0])
    suplier = DataSuplier(list(range(4)))
    list(suplier.training_set())
    assert suplier.index == 4
    assert suplier.training_indexes == [0, 3]
    assert suplier.test_indexes == [1, 2]


def test_internal_next_records_training_index(scripted):
    scripted([0])
    suplier = DataSuplier(["a"])
    result = suplier._DataSuplier__next()
    assert result == 0
    assert suplier.training_indexes == [0]
    assert suplier.test_indexes == []
    assert suplier.index == 1


def test_internal_next_records_test_index(scripted):
    scripted([1])
    suplier = DataSuplier(["a"])
    result = suplier._DataSuplier__next()
    assert result == 1
    assert suplier.test_indexes == [0]
    assert suplier.training_indexes == []
    assert suplier.index == 1


def test_real_partitioner_splits_dataset_without_loss(): 
    data = list(range(50))
    suplier = DataSuplier(data, training_percent=0.8)
    train = [i for i, _ in suplier.training_set()]
    test = [i for i, _ in suplier.test_set()]
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == list(range(50))
    assert all(suplier.dataset[i] == i for i in train + test)


def test_real_partitioner_is_consistent_across_instances():
    data = list(range(30))
    first = DataSuplier(data)
    second = DataSuplier(data)
    assert [i for i, _ in first.training_set()] == [i for i, _ in second.training_set()]
