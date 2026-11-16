import pytest

import data_partitioner.data_suplier as ds
from data_partitioner.data_suplier import DataSuplier


@pytest.fixture
def fake_partitioner(monkeypatch):
    """Replace the partitioner with a fake that yields a predetermined pattern."""
    state = {"calls": [], "pattern": []}

    def fake(n_of_elements, percentages, function):
        state["calls"].append((n_of_elements, percentages, function))
        return iter(state["pattern"])

    monkeypatch.setattr(ds, "partitioner", fake)
    return state


def make(fake_state, pattern, dataset=None, **kwargs):
    fake_state["pattern"] = list(pattern)
    if dataset is None:
        dataset = ["e%d" % i for i in range(len(pattern))]
    return DataSuplier(dataset, **kwargs)


def test_init_stores_attributes(fake_partitioner):
    data = ["a", "b", "c"]
    supplier = make(fake_partitioner, [0, 0, 1], dataset=data, training_percent=0.5)
    assert supplier.dataset is data
    assert supplier.training_percent == 0.5
    assert supplier.n_of_elements == 3
    assert supplier.test_indexes == []
    assert supplier.training_indexes == []
    assert supplier.index == 0


def test_init_default_training_percent(fake_partitioner):
    supplier = make(fake_partitioner, [0, 1])
    assert supplier.training_percent == 0.8


def test_init_passes_arguments_to_partitioner(fake_partitioner):
    def func(*args, **kwargs):
        return 0

    make(fake_partitioner, [0, 0, 1, 0], training_percent=0.75,
         partitioning_function=func)
    n, percents, function = fake_partitioner["calls"][0]
    assert n == 4
    assert percents[0] == pytest.approx(0.75)
    assert percents[1] == pytest.approx(0.25)
    assert function is func


def test_init_uses_default_partitioning_function(fake_partitioner):
    make(fake_partitioner, [0, 1])
    assert fake_partitioner["calls"][0][2] is ds.pseudorandom_function


def test_training_set_yields_index_and_element_pairs(fake_partitioner):
    data = ["a", "b", "c", "d", "e"]
    supplier = make(fake_partitioner, [0, 1, 0, 0, 1], dataset=data)
    assert list(supplier.training_set()) == [(0, "a"), (2, "c"), (3, "d")]


def test_test_set_yields_index_and_element_pairs(fake_partitioner):
    data = ["a", "b", "c", "d", "e"]
    supplier = make(fake_partitioner, [0, 1, 0, 0, 1], dataset=data)
    assert list(supplier.test_set()) == [(1, "b"), (4, "e")]


def test_sets_are_disjoint_and_cover_dataset_with_fake(fake_partitioner):
    pattern = [1, 0, 0, 1, 1, 0, 1, 0]
    supplier = make(fake_partitioner, pattern)
    train = [i for i, _ in supplier.training_set()]
    test = [i for i, _ in supplier.test_set()]
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == list(range(len(pattern)))


def test_training_set_is_lazy(fake_partitioner):
    supplier = make(fake_partitioner, [1, 1, 0, 0, 1])
    gen = supplier.training_set()
    assert supplier.index == 0
    assert next(gen) == (2, "e2")
    assert supplier.index == 3
    assert supplier.training_indexes == [2]
    assert supplier.test_indexes == [0, 1]


def test_interleaved_consumption(fake_partitioner):
    data = list("abcdef")
    supplier = make(fake_partitioner, [0, 1, 1, 0, 0, 1], dataset=data)
    train = supplier.training_set()
    test = supplier.test_set()
    assert next(train) == (0, "a")
    assert next(test) == (1, "b")
    assert next(train) == (3, "d")
    assert next(test) == (2, "c")
    assert next(test) == (5, "f")
    assert next(train) == (4, "e")
    with pytest.raises(StopIteration):
        next(train)
    with pytest.raises(StopIteration):
        next(test)


def test_sets_can_be_iterated_repeatedly(fake_partitioner):
    supplier = make(fake_partitioner, [0, 1, 0, 1, 0])
    first = list(supplier.training_set())
    second = list(supplier.training_set())
    assert first == second
    assert len(first) == 3
    assert list(supplier.test_set()) == list(supplier.test_set())


def test_test_set_consumed_after_training_set(fake_partitioner):
    supplier = make(fake_partitioner, [0, 1, 0, 1])
    list(supplier.training_set())
    assert supplier.index == 4
    assert [i for i, _ in supplier.test_set()] == [1, 3]


def test_all_training_pattern(fake_partitioner):
    supplier = make(fake_partitioner, [0, 0, 0])
    assert len(list(supplier.training_set())) == 3
    assert list(supplier.test_set()) == []


def test_all_test_pattern(fake_partitioner):
    supplier = make(fake_partitioner, [1, 1, 1])
    assert list(supplier.training_set()) == []
    assert len(list(supplier.test_set())) == 3


def test_empty_dataset(fake_partitioner):
    supplier = make(fake_partitioner, [], dataset=[])
    assert supplier.n_of_elements == 0
    assert list(supplier.training_set()) == []
    assert list(supplier.test_set()) == []


def test_real_partitioner_covers_every_element_exactly_once():
    data = list(range(200))
    supplier = DataSuplier(data)
    train = list(supplier.training_set())
    test = list(supplier.test_set())
    train_idx = [i for i, _ in train]
    test_idx = [i for i, _ in test]
    assert set(train_idx).isdisjoint(test_idx)
    assert sorted(train_idx + test_idx) == data
    for i, value in train + test:
        assert data[i] == value


def test_real_partitioner_is_not_degenerate_for_default_percent():
    supplier = DataSuplier(list(range(1000)))
    train = list(supplier.training_set())
    test = list(supplier.test_set())
    assert len(train) > len(test) > 0
