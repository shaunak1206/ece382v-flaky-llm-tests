import pytest
from data_partitioner.data_suplier import DataSuplier

def test_init_stores_dataset_and_defaults():
    dataset = list(range(10))
    ds = DataSuplier(dataset)
    assert ds.dataset == dataset
    assert ds.training_percent == 0.8
    assert ds.n_of_elements == 10
    assert ds.index == 0
    # training and test indexes start empty
    assert ds.training_indexes == []
    assert ds.test_indexes == []

def test_init_custom_training_percent():
    dataset = list(range(10))
    ds = DataSuplier(dataset, training_percent=0.5)
    assert ds.training_percent == 0.5

def test_init_partitioner_called_with_correct_args(monkeypatch):
    dataset = list(range(5))
    called_args = {}
    def fake_partitioner(n_elements, proportions, partitioning_function):
        called_args['n'] = n_elements
        called_args['props'] = proportions
        called_args['pf'] = partitioning_function
        class FakeIter:
            def __init__(self):
                self.calls = 0
            def __next__(self):
                self.calls += 1
                return 0
        return FakeIter()
    monkeypatch.setattr('data_partitioner.data_suplier.partitioner', fake_partitioner)
    ds = DataSuplier(dataset, training_percent=0.6)
    assert called_args['n'] == 5
    assert called_args['props'] == [0.6, 0.4]
    # The partitioning_function callable should be passed along (not None)
    assert callable(called_args['pf'])

def test___next_appends_to_training_indexes(monkeypatch):
    dataset = list(range(3))
    ds = DataSuplier(dataset)
    # Patch partitioner.__next__ to always return 0 (training)
    class FakePartIter:
        def __init__(self):
            self.count = 0
        def __next__(self):
            self.count += 1
            return 0
    ds.partitioner = FakePartIter()
    res = ds.__next__()
    assert res == 0
    assert ds.training_indexes == [0]
    assert ds.test_indexes == []
    assert ds.index == 1

def test___next_appends_to_test_indexes(monkeypatch):
    dataset = list(range(3))
    ds = DataSuplier(dataset)
    # Patch partitioner.__next__ to always return 1 (test)
    class FakePartIter:
        def __init__(self):
            self.count = 0
        def __next__(self):
            self.count += 1
            return 1
    ds.partitioner = FakePartIter()
    res = ds.__next__()
    assert res == 1
    assert ds.test_indexes == [0]
    assert ds.training_indexes == []
    assert ds.index == 1

def test__yield_set_yields_training(monkeypatch):
    dataset = ['a', 'b', 'c']
    ds = DataSuplier(dataset)
    # Create controlled partitioner sequence: first two training, last test
    class FakePartIter:
        def __init__(self):
            self.calls = 0
        def __next__(self):
            self.calls += 1
            return 0 if self.calls <= 2 else 1
    ds.partitioner = FakePartIter()
    # Exhaust training_set generator to populate indexes
    gen = ds.training_set()
    result = list(gen)
    # training_indexes should have 2 elements (indexes 0 and 1)
    assert ds.training_indexes == [0, 1]
    assert ds.test_indexes == [2]
    # Result yields tuples with index and data
    assert result == [(0, 'a'), (1, 'b')]

def test__yield_set_yields_test(monkeypatch):
    dataset = ['x', 'y', 'z', 'w']
    ds = DataSuplier(dataset)
    # Create partition results 1,1,0,1 (test,test,training,test)
    partition_seq = [1,1,0,1]
    class FakePartIter:
        def __init__(self):
            self.calls = -1
        def __next__(self):
            self.calls += 1
            return partition_seq[self.calls]
    ds.partitioner = FakePartIter()
    gen = ds.test_set()
    result = list(gen)
    assert ds.test_indexes == [0,1,3]
    assert ds.training_indexes == [2]
    expected = [(0, 'x'), (1, 'y'), (3, 'w')]
    assert result == expected

def test_training_set_and_test_set_are_generators():
    ds = DataSuplier(['a','b'])
    train_gen = ds.training_set()
    test_gen = ds.test_set()
    assert hasattr(train_gen, '__iter__') and hasattr(train_gen, '__next__')
    assert hasattr(test_gen, '__iter__') and hasattr(test_gen, '__next__')

def test_yield_set_returns_empty_on_empty_dataset():
    ds = DataSuplier([])
    assert list(ds.training_set()) == []
    assert list(ds.test_set()) == []

def test_multiple_calls_to_training_set_continue_correctly(monkeypatch):
    dataset = ['a','b','c','d']
    ds = DataSuplier(dataset)
    # partitioner returns sequence: 0,1,0,1 (train,test,train,test)
    seq = [0,1,0,1]
    class FakePartIter:
        def __init__(self):
            self.idx = -1
        def __next__(self):
            self.idx += 1
            return seq[self.idx]
    ds.partitioner = FakePartIter()
    gen1 = ds.training_set()
    first_batch = [x for x in gen1]
    # Should receive indexes 0 and 2
    assert ds.training_indexes == [0, 2]
    # Calling again returns same results, does not advance more
    gen2 = ds.training_set()
    second_batch = [x for x in gen2]
    assert first_batch == second_batch

def test_multiple_calls_to_test_set_continue_correctly(monkeypatch):
    dataset = ['a','b','c','d']
    ds = DataSuplier(dataset)
    # partitioner returns sequence: 1,0,1,0 (test,train,test,train)
    seq = [1,0,1,0]
    class FakePartIter:
        def __init__(self):
            self.idx = -1
        def __next__(self):
            self.idx += 1
            return seq[self.idx]
    ds.partitioner = FakePartIter()
    gen1 = ds.test_set()
    first_batch = [x for x in gen1]
    assert ds.test_indexes == [0, 2]
    gen2 = ds.test_set()
    second_batch = [x for x in gen2]
    assert first_batch == second_batch

def test_yield_set_repeats_calling___next_until_indexes_populated(monkeypatch):
    # Check that _yield_set triggers __next__ to populate indexes when they are empty
    dataset = ['a','b']
    ds = DataSuplier(dataset)
    call_count = {'cnt':0}
    class FakePartIter:
        def __init__(self):
            self.pos = -1
        def __next__(self):
            self.pos += 1
            call_count['cnt'] += 1
            return 0 if self.pos == 0 else 1
    ds.partitioner = FakePartIter()
    # We call test_set which starts with empty test_indexes
    result = list(ds.test_set())
    # __next__ should have been called until test_indexes is non-empty (one call gives test_indexes=[1])
    assert call_count['cnt'] >= 1
    assert ds.test_indexes == [1]
    assert result == [(1, 'b')]

def test_index_increments_correctly_with_multiple_next_calls(monkeypatch):
    dataset = ['a','b','c']
    ds = DataSuplier(dataset)
    class FakePartIter:
        def __init__(self):
            self.called = 0
        def __next__(self):
            self.called += 1
            # Alternate between training(0) and test(1)
            return 0 if self.called % 2 == 1 else 1
    ds.partitioner = FakePartIter()
    ds.__next__()
    assert ds.index == 1
    ds.__next__()
    assert ds.index == 2
    ds.__next__()
    assert ds.index == 3

def test_training_and_test_indexes_do_not_overlap(monkeypatch):
    dataset = list(range(50))
    ds = DataSuplier(dataset)
    # partitioner returns an equal number of zeros and ones alternating
    class FakePartIter:
        def __init__(self):
            self.idx = -1
        def __next__(self):
            self.idx += 1
            return 0 if self.idx % 2 == 0 else 1
    ds.partitioner = FakePartIter()
    # advance through all elements
    for _ in range(len(dataset)):
        ds.__next__()
    # training_indexes are even indexes, test_indexes odd indexes
    for t in ds.training_indexes:
        assert t % 2 == 0
    for tst in ds.test_indexes:
        assert tst % 2 == 1
    # No overlap
    assert set(ds.training_indexes).isdisjoint(ds.test_indexes)
    # All indexes accounted for
    assert sorted(ds.training_indexes + ds.test_indexes) == list(range(len(dataset)))

def test_yield_set_handles_index_exceeding_n_of_elements(monkeypatch):
    dataset = ['a', 'b']
    ds = DataSuplier(dataset)
    class FakePartIter:
        def __init__(self):
            self.idx = -1
        def __next__(self):
            self.idx += 1
            return 0
    ds.partitioner = FakePartIter()
    # Force index beyond n_of_elements artificially 
    ds.training_indexes = [0]
    ds.test_indexes = []
    ds.index = 3  # beyond dataset length
    result = list(ds.training_set())
    # Should yield items in training_indexes that exist only; index 0 is valid, so it yields one
    assert result == [(0, 'a')]

def test_training_set_and_test_set_yield_only_from_indexes(monkeypatch):
    dataset = ['a', 'b', 'c']
    ds = DataSuplier(dataset)
    # Use a partitioner with controlled sequence
    seq = [0,1,1]
    class FakePartIter:
        def __init__(self):
            self.idx = -1
        def __next__(self):
            self.idx += 1
            return seq[self.idx]
    ds.partitioner = FakePartIter()
    # Exhaust one training_set generator iteration so indexes populate
    list(ds.training_set())
    # Now change dataset values
    ds.dataset[0] = 'x'
    # Training indexes should be [0]
    # The generator yields latest dataset value for index
    results = list(ds.training_set())
    assert results == [(0, 'x')]

def test_yield_set_yields_even_if_index_increments_beyond_length(monkeypatch):
    dataset = ['a']
    ds = DataSuplier(dataset)
    # Make partitioner return 1 (test)
    class FakePartIter:
        def __next__(self):
            return 1
    ds.partitioner = FakePartIter()
    # Manually append test index 0 and set index larger than length
    ds.test_indexes = [0]
    ds.index = 10
    results = list(ds.test_set())
    assert results == [(0, 'a')]

