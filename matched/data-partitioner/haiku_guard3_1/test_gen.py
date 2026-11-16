"""Unit tests for data_partitioner.data_suplier.DataSuplier.

These tests use a deterministic fake partitioner (installed with pytest's
monkeypatch fixture) wherever exact results are asserted. Tests that exercise
the real pseudorandom partitioner only assert properties that hold for every
possible outcome, and seed the standard-library random generator at the start
of each such test body.
"""

import random

import data_partitioner.data_suplier as data_suplier_module
from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import pseudorandom_function


def _install_fake(monkeypatch, pattern):
    """Replace the partitioner used by data_suplier with a deterministic fake.

    The fake returns pattern[k % len(pattern)] on its k-th call to __next__.
    Every instance it creates is appended to the returned list, so tests can
    inspect the constructor arguments and the number of pulls.
    """
    created = []

    class FakePartitioner(object):
        def __init__(self, n, probabilities, partitioning_function):
            self.n = n
            self.probabilities = list(probabilities)
            self.partitioning_function = partitioning_function
            self.calls = 0
            created.append(self)

        def __next__(self):
            value = pattern[self.calls % len(pattern)]
            self.calls += 1
            return value

    monkeypatch.setattr(data_suplier_module, "partitioner", FakePartitioner)
    return created


# A dataset of 8 items where data[i] == 10 * (i + 1).
# With the pattern [0, 1, 1, 0] the fake assigns:
#   index: 0 1 2 3 4 5 6 7
#   set:   T F F T T F F T   (T = training, F = test)
DATA = [10, 20, 30, 40, 50, 60, 70, 80]
PATTERN = [0, 1, 1, 0]
TRAIN_PAIRS = [(0, 10), (3, 40), (4, 50), (7, 80)]
TEST_PAIRS = [(1, 20), (2, 30), (5, 60), (6, 70)]


def test_init_stores_dataset_length_and_percent(monkeypatch):
    _install_fake(monkeypatch, PATTERN)
    data = list(DATA)
    suplier = DataSuplier(data, training_percent=0.5)
    assert suplier.dataset is data
    assert suplier.training_percent == 0.5
    assert suplier.n_of_elements == len(DATA)


def test_init_default_training_percent_is_0_8(monkeypatch):
    created = _install_fake(monkeypatch, PATTERN)
    suplier = DataSuplier(DATA)
    assert suplier.training_percent == 0.8
    probs = created[0].probabilities
    assert probs[0] == 0.8
    assert probs[1] == pytest_approx(0.2)


def pytest_approx(value):
    # Local tolerance helper so the file only relies on pytest's public API
    # through plain comparisons with an explicit tolerance.
    class _Approx(object):
        def __eq__(self, other):
            return abs(other - value) < 1e-9

        def __repr__(self):
            return "approx(%r)" % (value,)

    return _Approx()


def test_init_passes_length_and_probabilities_to_partitioner(monkeypatch):
    created = _install_fake(monkeypatch, PATTERN)
    DataSuplier(DATA, training_percent=0.25)
    assert len(created) == 1
    assert created[0].n == len(DATA)
    assert created[0].probabilities[0] == 0.25
    assert created[0].probabilities[1] == pytest_approx(0.75)


def test_init_passes_custom_partitioning_function_through(monkeypatch):
    created = _install_fake(monkeypatch, PATTERN)

    def custom(*args, **kwargs):
        return 0

    DataSuplier(DATA, partitioning_function=custom)
    assert created[0].partitioning_function is custom


def test_init_default_partitioning_function_is_pseudorandom_function(monkeypatch):
    created = _install_fake(monkeypatch, PATTERN)
    DataSuplier(DATA)
    assert created[0].partitioning_function is pseudorandom_function


def test_init_starts_with_empty_state_and_no_pulls(monkeypatch):
    created = _install_fake(monkeypatch, PATTERN)
    suplier = DataSuplier(DATA)
    assert suplier.training_indexes == []
    assert suplier.test_indexes == []
    assert suplier.index == 0
    assert created[0].calls == 0


def test_creating_generators_is_lazy(monkeypatch):
    created = _install_fake(monkeypatch, PATTERN)
    suplier = DataSuplier(DATA)
    train_gen = suplier.training_set()
    test_gen = suplier.test_set()
    assert created[0].calls == 0
    assert suplier.index == 0
    assert suplier.training_indexes == []
    assert suplier.test_indexes == []
    # Generators are not exhausted by creation alone.
    assert train_gen is not None and test_gen is not None


def test_training_set_yields_expected_pairs(monkeypatch):
    _install_fake(monkeypatch, PATTERN)
    suplier = DataSuplier(DATA)
    assert list(suplier.training_set()) == TRAIN_PAIRS


def test_test_set_yields_expected_pairs(monkeypatch):
    _install_fake(monkeypatch, PATTERN)
    suplier = DataSuplier(DATA)
    assert list(suplier.test_set()) == TEST_PAIRS


def test_partial_consumption_pulls_only_as_needed(monkeypatch):
    created = _install_fake(monkeypatch, PATTERN)
    suplier = DataSuplier(DATA)
    gen = suplier.training_set()
    first = next(gen)
    assert first == (0, 10)
    assert suplier.index == 1
    assert suplier.training_indexes == [0]
    assert suplier.test_indexes == []
    assert created[0].calls == 1


def test_consuming_training_set_consumes_whole_dataset(monkeypatch):
    created = _install_fake(monkeypatch, PATTERN)
    suplier = DataSuplier(DATA)
    list(suplier.training_set())
    assert suplier.index == len(DATA)
    assert created[0].calls == len(DATA)
    assert suplier.training_indexes == [0, 3, 4, 7]
    assert suplier.test_indexes == [1, 2, 5, 6]


def test_test_set_first_then_training_set_gives_same_partition(monkeypatch):
    _install_fake(monkeypatch, PATTERN)
    suplier = DataSuplier(DATA)
    test_result = list(suplier.test_set())
    train_result = list(suplier.training_set())
    assert test_result == TEST_PAIRS
    assert train_result == TRAIN_PAIRS
    assert suplier.index == len(DATA)


def test_repeated_iteration_gives_same_results(monkeypatch):
    _install_fake(monkeypatch, PATTERN)
    suplier = DataSuplier(DATA)
    first_train = list(suplier.training_set())
    second_train = list(suplier.training_set())
    first_test = list(suplier.test_set())
    second_test = list(suplier.test_set())
    assert first_train == second_train == TRAIN_PAIRS
    assert first_test == second_test == TEST_PAIRS


def test_yielded_indexes_are_in_increasing_order(monkeypatch):
    _install_fake(monkeypatch, PATTERN)
    suplier = DataSuplier(DATA)
    train_idx = [i for i, _ in suplier.training_set()]
    test_idx = [i for i, _ in suplier.test_set()]
    assert train_idx == sorted(train_idx)
    assert test_idx == sorted(test_idx)


def test_training_and_test_indexes_partition_dataset(monkeypatch):
    _install_fake(monkeypatch, PATTERN)
    suplier = DataSuplier(DATA)
    train_idx = [i for i, _ in suplier.training_set()]
    test_idx = [i for i, _ in suplier.test_set()]
    assert set(train_idx).isdisjoint(test_idx)
    assert sorted(train_idx + test_idx) == list(range(len(DATA)))


def test_all_training_pattern_gives_empty_test_set(monkeypatch):
    _install_fake(monkeypatch, [0])
    data = list(range(5))
    suplier = DataSuplier(data)
    assert list(suplier.training_set()) == [(i, i) for i in range(5)]
    assert list(suplier.test_set()) == []


def test_all_test_pattern_gives_empty_training_set(monkeypatch):
    _install_fake(monkeypatch, [1])
    data = list(range(5))
    suplier = DataSuplier(data)
    assert list(suplier.training_set()) == []
    assert list(suplier.test_set()) == [(i, i) for i in range(5)]
    assert suplier.index == 5


def test_empty_dataset_yields_nothing(monkeypatch):
    created = _install_fake(monkeypatch, [0])
    suplier = DataSuplier([])
    assert suplier.n_of_elements == 0
    assert created[0].n == 0
    assert list(suplier.training_set()) == []
    assert list(suplier.test_set()) == []
    assert suplier.index == 0


def test_yields_values_from_original_dataset(monkeypatch):
    _install_fake(monkeypatch, PATTERN)
    data = ["a", "b", "c", "d", "e", "f", "g", "h"]
    suplier = DataSuplier(data)
    for index, value in list(suplier.training_set()) + list(suplier.test_set()):
        assert value is data[index]


def test_real_partitioner_partitions_integer_dataset():
    random.seed(12345)
    data = list(range(50))
    suplier = DataSuplier(data, training_percent=0.7)
    train = list(suplier.training_set())
    test = list(suplier.test_set())
    train_idx = [i for i, _ in train]
    test_idx = [i for i, _ in test]
    assert set(train_idx).isdisjoint(test_idx)
    assert sorted(train_idx + test_idx) == list(range(50))
    for index, value in train + test:
        assert value == data[index]


def test_real_partitioner_pairs_match_string_dataset():
    random.seed(67890)
    data = ["item-%d" % i for i in range(30)]
    suplier = DataSuplier(data)
    train = list(suplier.training_set())
    test = list(suplier.test_set())
    assert len(train) + len(test) == len(data)
    for index, value in train + test:
        assert value == data[index]
        assert value == "item-%d" % index
