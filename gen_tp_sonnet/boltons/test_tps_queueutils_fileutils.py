import os

import pytest

from boltons.fileutils import FilePerms, atomic_save, iter_find_files, mkdir_p
from boltons.queueutils import HeapPriorityQueue, SortedPriorityQueue

QUEUE_CLASSES = [HeapPriorityQueue, SortedPriorityQueue]


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_queue_pops_highest_priority_first(cls):
    pq = cls()
    pq.add('low', 1)
    pq.add('high', 10)
    pq.add('mid', 5)
    assert [pq.pop(), pq.pop(), pq.pop()] == ['high', 'mid', 'low']
    assert pq.pop(default='empty') == 'empty'
    with pytest.raises(KeyError):
        pq.pop()


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_queue_readd_changes_priority_and_remove(cls):
    pq = cls()
    pq.add('a', 1)
    pq.add('b', 5)
    pq.add('a', 10)
    assert pq.peek() == 'a'
    pq.remove('a')
    assert pq.peek() == 'b'
    with pytest.raises(KeyError):
        pq.remove('never-added')
    assert pq.pop() == 'b'


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_queue_peek_default_priority_and_priority_key(cls):
    pq = cls()
    assert pq.peek(default=None) is None
    with pytest.raises(KeyError):
        pq.peek()
    pq.add('x')
    pq.add('y', 1)
    assert pq.peek() == 'y'
    assert pq.peek() == 'y'

    reversed_pq = cls(priority_key=lambda p: -p)
    reversed_pq.add('a', 1)
    reversed_pq.add('b', 2)
    assert reversed_pq.pop() == 'a'


def test_mkdir_p_and_fileperms(tmp_path):
    target = tmp_path / 'a' / 'b' / 'c'
    mkdir_p(str(target))
    assert target.is_dir()
    mkdir_p(str(target))
    assert target.is_dir()

    perms = FilePerms.from_int(0o644)
    assert perms.user == 'rw'
    assert perms.group == 'r'
    assert perms.other == 'r'


def test_fileperms_from_int_and_from_path(tmp_path):
    perms = FilePerms.from_int(0o644)
    assert perms.user == 'rw'
    assert perms.group == 'r'
    assert perms.other == 'r'

    perms = FilePerms.from_int(0o750)
    assert perms.user == 'rwx'
    assert perms.group == 'rx'
    assert perms.other == ''

    path = tmp_path / 'secret.txt'
    path.write_text('x')
    os.chmod(path, 0o600)
    from_disk = FilePerms.from_path(str(path))
    assert from_disk.user == 'rw'
    assert from_disk.group == ''
    assert from_disk.other == ''


def test_atomic_save_commits_on_success_and_preserves_original_on_error(tmp_path):
    path = tmp_path / 'f.txt'
    with atomic_save(str(path), text_mode=True) as fo:
        fo.write('hello')
    assert path.read_text() == 'hello'

    with pytest.raises(ZeroDivisionError):
        with atomic_save(str(path), text_mode=True) as fo:
            fo.write('partial')
            1 / 0
    assert path.read_text() == 'hello'
    assert os.listdir(tmp_path) == ['f.txt']


def test_iter_find_files_patterns_ignored_and_depth(tmp_path):
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'a.txt').write_text('a')
    (tmp_path / 'b.py').write_text('b')
    (tmp_path / 'sub' / 'c.txt').write_text('c')
    (tmp_path / 'sub' / 'd.md').write_text('d')
    root = str(tmp_path)

    def names(paths):
        return {os.path.basename(p) for p in paths}

    assert names(iter_find_files(root, '*.txt')) == {'a.txt', 'c.txt'}
    assert names(iter_find_files(root, ['*.txt', '*.py'])) == {'a.txt', 'b.py', 'c.txt'}
    assert names(iter_find_files(root, '*.txt', ignored='c.txt')) == {'a.txt'}
    assert names(iter_find_files(root, '*.txt', max_depth=0)) == {'a.txt'}
    assert all(p.startswith(root) for p in iter_find_files(root, '*.txt'))
