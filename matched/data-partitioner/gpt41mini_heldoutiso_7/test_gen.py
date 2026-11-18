import pytest
from data_partitioner.data_suplier import DataSuplier
from data_partitioner.partitioner import pseudorandom_function

def make_dataset(n):
    return list(range(n))

def indexes_from_generator(gen):
    return [i for i, _ in gen]

def test_init_sets_attributes_correctly():
    ds = make_dataset(10)
    suplier = DataSuplier(ds, training_percent=0.7)
    assert suplier.dataset == ds
    assert suplier.training_percent == 0.7
    assert suplier.n_of_elements == 10
    assert hasattr(suplier.partitioner, "__next__")
    assert suplier.test_indexes == []
    assert suplier.training_indexes == []
    assert suplier.index == 0

@pytest.mark.parametrize("training_percent", [0, 0.5, 1])
def test_partitioning_totals_add_up(training_percent):
    ds = make_dataset(20)
    suplier = DataSuplier(ds, training_percent=training_percent)
    # exhaust generators to fill indexes
    list(suplier.training_set())
    list(suplier.test_set())
    total = len(suplier.training_indexes) + len(suplier.test_indexes)
    assert total == suplier.n_of_elements
    # check proportion approximately matches training_percent (only if 0 < training_percent < 1)
    if 0 < training_percent < 1:
        training_ratio = len(suplier.training_indexes) / suplier.n_of_elements
        assert abs(training_ratio - training_percent) < 0.3

def test_training_and_test_sets_yield_all_elements():
    ds = make_dataset(15)
    suplier = DataSuplier(ds, training_percent=0.6)
    train_items = list(suplier.training_set())
    test_items = list(suplier.test_set())
    all_indexes = {i for i, _ in train_items} | {i for i, _ in test_items}
    assert set(range(15)) == all_indexes

def test_training_set_yields_correct_data():
    ds = list("abcdefghij")
    suplier = DataSuplier(ds)
    train_items = list(suplier.training_set())
    for idx, val in train_items:
        assert ds[idx] == val

def test_test_set_yields_correct_data():
    ds = list(range(12))
    suplier = DataSuplier(ds)
    test_items = list(suplier.test_set())
    for idx, val in test_items:
        assert ds[idx] == val

def test_indexing_order_is_strictly_increasing():
    ds = make_dataset(30)
    suplier = DataSuplier(ds, training_percent=0.7)
    train_items = list(suplier.training_set())
    test_items = list(suplier.test_set())
    combined_indexes = sorted([i for i, _ in train_items] + [i for i, _ in test_items])
    assert combined_indexes == list(range(30))

def test_internal_index_advances_to_end_after_iteration():
    ds = make_dataset(10)
    suplier = DataSuplier(ds)
    list(suplier.training_set())
    list(suplier.test_set())
    # internal index matches dataset length after exhausting
    assert suplier.index == len(ds)

def test_repeated_calls_to_training_set_continue_to_yield_empty_after_first_iter():
    ds = make_dataset(8)
    suplier = DataSuplier(ds)
    first_gen = suplier.training_set()
    list(first_gen)
    second_gen = suplier.training_set()
    # Since suplier.index is at end, generator yields nothing new
    assert list(second_gen) == []

def test_repeated_calls_to_test_set_continue_to_yield_empty_after_first_iter():
    ds = make_dataset(8)
    suplier = DataSuplier(ds)
    first_gen = suplier.test_set()
    list(first_gen)
    second_gen = suplier.test_set()
    assert list(second_gen) == []

def test_custom_partition_function_is_used(monkeypatch):
    ds = make_dataset(10)
    calls = []

    def fake_partition_function(n, proportions, pf):
        class FakeIter:
            def __init__(self):
                self.count = 0
            def __next__(self):
                calls.append(self.count)
                self.count += 1
                return 0 if self.count % 2 == 0 else 1
        return FakeIter()

    suplier = DataSuplier(ds, partitioning_function=fake_partition_function)
    list(suplier.training_set())
    list(suplier.test_set())
    # Because fake_partition_function returns alternating 0,1, training and test indexes should be alternating
    combined = sorted([(i,0) for i in suplier.training_indexes] + [(i,1) for i in suplier.test_indexes])
    for idx, (index, label) in enumerate(combined):
        # labels alternate because of fake_partition_function
        expected_label = 0 if idx % 2 == 0 else 1
        assert label == expected_label

def test_yield_set_handles_empty_dataset():
    ds = []
    suplier = DataSuplier(ds)
    assert list(suplier.training_set()) == []
    assert list(suplier.test_set()) == []

def test_training_percent_zero_assigns_all_to_test():
    ds = make_dataset(5)
    suplier = DataSuplier(ds, training_percent=0)
    train = list(suplier.training_set())
    test = list(suplier.test_set())
    assert train == []
    assert sorted([i for i, _ in test]) == list(range(5))

def test_training_percent_one_assigns_all_to_training():
    ds = make_dataset(6)
    suplier = DataSuplier(ds, training_percent=1)
    train = list(suplier.training_set())
    test = list(suplier.test_set())
    assert test == []
    assert sorted([i for i, _ in train]) == list(range(6))

def test__next_appends_correctly_to_training_or_test_indexes():
    ds = make_dataset(4)
    suplier = DataSuplier(ds, training_percent=0.5)
    # manually call __next__ multiple times
    class_suplier = suplier
    indices_before = len(suplier.training_indexes) + len(suplier.test_indexes)
    # call __next__ 4 times
    for _ in range(4):
        res = class_suplier.__next__()
        assert res in (0, 1)
    total_indices = len(suplier.training_indexes) + len(suplier.test_indexes)
    assert total_indices == 4
    # combined indexes must be exactly range(4) in some order
    combined = sorted(suplier.training_indexes + suplier.test_indexes)
    assert combined == list(range(4))

def test__yield_set_returns_all_elements_even_if_indexes_come_late():
    # Patch partitioner to always return 1 first, forcing test index added before training.
    class FakePartitioner:
        def __init__(self, n, props, pf):
            self.count = 0
            self.n = n
        def __next__(self):
            self.count += 1
            return 1 if self.count <= 3 else 0

    ds = make_dataset(6)
    suplier = DataSuplier.__new__(DataSuplier)
    suplier.dataset = ds
    suplier.training_percent = 0.5
    suplier.n_of_elements = len(ds)
    suplier.partitioner = FakePartitioner(suplier.n_of_elements, [0.5, 0.5], pseudorandom_function)
    suplier.test_indexes = []
    suplier.training_indexes = []
    suplier.index = 0

    # Produce generator from test_set - should include indexes even if test indexes are appended late
    test_gen = suplier.test_set()
    test_results = list(test_gen)
    # Test indexes should be first three indexes
    assert sorted(test_results) == [(0,0), (1,1), (2,2)]

def test_training_set_and_test_set_are_generators():
    ds = make_dataset(10)
    suplier = DataSuplier(ds)
    train = suplier.training_set()
    test = suplier.test_set()
    assert iter(train) is train
    assert iter(test) is test

def test_index_reset_across_generators_does_not_occur():
    ds = make_dataset(10)
    suplier = DataSuplier(ds)
    train_1 = list(suplier.training_set())
    idx_after_train = suplier.index
    test_1 = list(suplier.test_set())
    idx_after_test = suplier.index
    # Calling training_set again should not reset index
    train_2 = list(suplier.training_set())
    assert suplier.index == idx_after_test
    assert train_2 == []
    # indexes should be non-decreasing
    assert idx_after_train <= idx_after_test

def test_dataset_elements_are_unchanged_by_iteration():
    ds = ["a", "b", "c", "d"]
    ds_copy = ds.copy()
    suplier = DataSuplier(ds)
    _ = list(suplier.training_set())
    _ = list(suplier.test_set())
    assert ds == ds_copy
