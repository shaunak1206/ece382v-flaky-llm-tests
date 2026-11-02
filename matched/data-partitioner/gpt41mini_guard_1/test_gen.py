import pytest
from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import pseudorandom_function

# Helper dataset for testing
dataset = list(range(100))

def test_init_sets_attributes_correctly():
    suplier = DataSuplier(dataset, 0.7)
    assert suplier.dataset == dataset
    assert suplier.training_percent == 0.7
    assert suplier.n_of_elements == len(dataset)
    assert callable(suplier.partitioner.__next__)
    # Initially indexes empty and index=0
    assert suplier.test_indexes == []
    assert suplier.training_indexes == []
    assert suplier.index == 0

def test_training_test_index_partition_counts_sum_to_dataset_length():
    suplier = DataSuplier(dataset)
    # Exhaust the generator to fill indexes
    list(suplier.training_set())
    list(suplier.test_set())
    total_indexes = len(suplier.training_indexes) + len(suplier.test_indexes)
    assert total_indexes == len(dataset)

def test_training_and_test_indexes_are_disjoint():
    suplier = DataSuplier(dataset)
    list(suplier.training_set())
    list(suplier.test_set())
    training_set = set(suplier.training_indexes)
    test_set = set(suplier.test_indexes)
    assert training_set.isdisjoint(test_set)

def test_training_and_test_indexes_cover_all_indexes():
    suplier = DataSuplier(dataset)
    list(suplier.training_set())
    list(suplier.test_set())
    combined = sorted(suplier.training_indexes + suplier.test_indexes)
    assert combined == list(range(len(dataset)))

@pytest.mark.parametrize("training_percent", [0.0, 0.1, 0.5, 0.8, 1.0])
def test_training_test_partition_ratios(training_percent):
    # Since partitioner is pseudorandom, ratios should be close to expected - allow 5% tolerance
    suplier = DataSuplier(dataset, training_percent=training_percent)
    list(suplier.training_set())
    list(suplier.test_set())
    tr_len = len(suplier.training_indexes)
    te_len = len(suplier.test_indexes)
    total = tr_len + te_len
    assert total == len(dataset)
    if training_percent == 0.0:
        assert tr_len == 0
        assert te_len == len(dataset)
    elif training_percent == 1.0:
        assert tr_len == len(dataset)
        assert te_len == 0
    else:
        ratio = tr_len / total
        assert abs(ratio - training_percent) <= 0.05

def test_training_set_generator_yields_correct_elements():
    suplier = DataSuplier(dataset)
    gen = suplier.training_set()
    # Consume first 5 elements from training set
    first_five = [next(gen) for _ in range(5)]
    # The indexes should be present in training_indexes
    for idx, val in first_five:
        assert idx in suplier.training_indexes
        assert val == dataset[idx]

def test_test_set_generator_yields_correct_elements():
    suplier = DataSuplier(dataset)
    gen = suplier.test_set()
    first_five = [next(gen) for _ in range(min(5, len(dataset)))]
    for idx, val in first_five:
        assert idx in suplier.test_indexes
        assert val == dataset[idx]

def test_yield_set_does_not_yield_after_all_data():
    suplier = DataSuplier(dataset)
    list(suplier.training_set())
    list(suplier.test_set())
    # After exhausting all data, generators yield nothing
    with pytest.raises(StopIteration):
        next(suplier.training_set())
    with pytest.raises(StopIteration):
        next(suplier.test_set())

def test_index_increments_correctly_when_iterating():
    suplier = DataSuplier(dataset)
    assert suplier.index == 0
    # Iterating some elements triggers __next__ calls
    for _ in range(10):
        next(suplier.training_set())
    # suplier.index must be at least 10, since partitioner.__next__ is called per element
    assert suplier.index >= 10
    # The sum of indexes selected must be 10
    assert len(suplier.training_indexes) + len(suplier.test_indexes) == suplier.index

def test_partitioning_function_is_used():
    call_log = []
    def test_pf(idx):
        call_log.append(idx)
        return pseudorandom_function(idx)
    suplier = DataSuplier(dataset, partitioning_function=test_pf)
    list(suplier.training_set())
    list(suplier.test_set())
    # The partitioning function was called for each element
    assert len(call_log) == len(dataset)
    # The call_log contains sequential indices from 0 to len(dataset)-1
    assert call_log == list(range(len(dataset)))

def test_training_set_and_test_set_iterators_are_independent():
    suplier = DataSuplier(dataset)
    tr_gen = suplier.training_set()
    te_gen = suplier.test_set()
    # Take some from training
    tr_first = next(tr_gen)
    # Take some from test
    te_first = next(te_gen)
    # Take another from training
    tr_second = next(tr_gen)
    assert tr_first[0] in suplier.training_indexes
    assert te_first[0] in suplier.test_indexes
    assert tr_second[0] in suplier.training_indexes
    # They should not yield same indexes
    assert tr_first[0] != te_first[0]

def test_training_set_iterates_all_training_indexes():
    suplier = DataSuplier(dataset)
    list(suplier.training_set())
    # now yielding again must yield all in order
    gen = suplier.training_set()
    yielded_indexes = []
    for idx, val in gen:
        yielded_indexes.append(idx)
    # All yielded indexes must be exactly training_indexes
    assert yielded_indexes == suplier.training_indexes

def test_test_set_iterates_all_test_indexes():
    suplier = DataSuplier(dataset)
    list(suplier.test_set())
    gen = suplier.test_set()
    yielded_indexes = []
    for idx, val in gen:
        yielded_indexes.append(idx)
    assert yielded_indexes == suplier.test_indexes

def test_yield_set_returns_empty_for_empty_dataset():
    suplier = DataSuplier([])
    assert list(suplier.training_set()) == []
    assert list(suplier.test_set()) == []

def test_training_percent_1_results_all_training():
    suplier = DataSuplier(dataset, training_percent=1.0)
    tr = list(suplier.training_set())
    te = list(suplier.test_set())
    assert len(tr) == len(dataset)
    assert te == []

def test_training_percent_0_results_all_test():
    suplier = DataSuplier(dataset, training_percent=0.0)
    tr = list(suplier.training_set())
    te = list(suplier.test_set())
    assert tr == []
    assert len(te) == len(dataset)

def test_len_dataset_zero_init():
    suplier = DataSuplier([])
    assert suplier.n_of_elements == 0
    assert suplier.test_indexes == []
    assert suplier.training_indexes == []
    assert list(suplier.training_set()) == []
    assert list(suplier.test_set()) == []

def test_consecutive_calls_to_training_set_produce_same_output():
    suplier = DataSuplier(dataset)
    # Exhaust first
    out1 = list(suplier.training_set())
    # Calling again produces same output
    out2 = list(suplier.training_set())
    assert out1 == out2

def test_consecutive_calls_to_test_set_produce_same_output():
    suplier = DataSuplier(dataset)
    _ = list(suplier.training_set())  # force partitioning
    out1 = list(suplier.test_set())
    out2 = list(suplier.test_set())
    assert out1 == out2

def test_partial_iteration_then_full_iteration_training_set():
    suplier = DataSuplier(dataset)
    gen = suplier.training_set()
    partial = [next(gen) for _ in range(5)]
    rest = list(gen)
    full = partial + rest
    assert full == list(zip(suplier.training_indexes, [dataset[i] for i in suplier.training_indexes]))

def test_partial_iteration_then_full_iteration_test_set():
    suplier = DataSuplier(dataset)
    _ = list(suplier.training_set())  # force partitioning
    gen = suplier.test_set()
    partial = [next(gen) for _ in range(3)]
    rest = list(gen)
    full = partial + rest
    assert full == list(zip(suplier.test_indexes, [dataset[i] for i in suplier.test_indexes]))
