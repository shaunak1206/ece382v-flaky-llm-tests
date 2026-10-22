import pytest

from boltons.queueutils import (
    HeapPriorityQueue,
    PriorityQueue,
    SortedPriorityQueue,
)


@pytest.mark.parametrize('cls', [HeapPriorityQueue, SortedPriorityQueue])
def test_pop_order_is_priority_then_insertion(cls):
    pq = cls()
    pq.add('low priority task', 0)
    pq.add('high priority task', 2)
    pq.add('medium priority task 1', 1)
    pq.add('medium priority task 2', 1)
    assert len(pq) == 4
    assert pq.peek() == 'high priority task'
    assert len(pq) == 4
    assert pq.pop() == 'high priority task'
    assert pq.peek() == 'medium priority task 1'
    assert len(pq) == 3
    assert [pq.pop() for _ in range(3)] == [
        'medium priority task 1', 'medium priority task 2', 'low priority task']
    assert len(pq) == 0


@pytest.mark.parametrize('cls', [HeapPriorityQueue, SortedPriorityQueue])
def test_readd_changes_priority_and_remove(cls):
    pq = cls()
    pq.add('a', 1)
    pq.add('b', 2)
    pq.add('c', 0)
    pq.add('a', 3)
    assert len(pq) == 3
    assert pq.peek() == 'a'

    pq.remove('a')
    assert len(pq) == 2
    with pytest.raises(KeyError):
        pq.remove('a')
    with pytest.raises(KeyError):
        pq.remove('never added')

    assert pq.pop() == 'b'
    assert pq.pop() == 'c'
    assert len(pq) == 0


def test_empty_queue_and_options():
    assert PriorityQueue is SortedPriorityQueue
    pq = PriorityQueue()
    with pytest.raises(IndexError):
        pq.pop()
    with pytest.raises(IndexError):
        pq.peek()
    assert pq.pop(default='nothing') == 'nothing'
    assert pq.peek(default=None) is None

    # a priority_key that does not negate makes lower numbers pop first
    lowfirst = PriorityQueue(priority_key=lambda p: p)
    lowfirst.add('late', 5)
    lowfirst.add('early', 1)
    assert lowfirst.pop() == 'early'
    assert lowfirst.pop() == 'late'

    with pytest.raises(TypeError):
        PriorityQueue(bogus=1)
