import os
import stat

import pytest

from boltons.fileutils import (
    FilePerms,
    atomic_save,
    copy_tree,
    iter_find_files,
    mkdir_p,
    rotate_file,
)


def test_mkdir_p(tmp_path):
    target = tmp_path / 'a' / 'b' / 'c'
    mkdir_p(str(target))
    assert target.is_dir()
    # a second call on an existing directory is not an error
    mkdir_p(str(target))
    assert target.is_dir()

    afile = tmp_path / 'plain.txt'
    afile.write_text('x')
    with pytest.raises(OSError):
        mkdir_p(str(afile))


def test_file_perms(tmp_path):
    full = FilePerms(user='rwx', group='xrw', other='wxr')
    assert repr(full) == "FilePerms(user='rwx', group='rwx', other='rwx')"
    assert int(full) == 0o777
    assert int(FilePerms('r', 'r', '')) == 0o440
    assert int(FilePerms()) == 0

    assert repr(FilePerms.from_int(0o644)) == "FilePerms(user='rw', group='r', other='r')"
    assert int(FilePerms.from_int(0o755)) == 0o755

    with pytest.raises(ValueError):
        FilePerms(user='rq')

    target = tmp_path / 'perm.txt'
    target.write_text('data')
    os.chmod(str(target), 0o640)
    perms = FilePerms.from_path(str(target))
    assert perms.user == 'rw'
    assert perms.group == 'r'
    assert perms.other == ''


def test_atomic_save_success(tmp_path):
    dest = tmp_path / 'out.txt'
    part = tmp_path / 'out.txt.part'
    with atomic_save(str(dest), text_mode=True) as fo:
        fo.write('hello')
        assert not dest.exists()
        assert part.exists()
    assert dest.read_text() == 'hello'
    assert not part.exists()

    # overwriting an existing file in binary mode
    with atomic_save(str(dest)) as fo:
        fo.write(b'new content')
    assert dest.read_bytes() == b'new content'

    secret = tmp_path / 'secret.bin'
    with atomic_save(str(secret), file_perms=0o600) as fo:
        fo.write(b'x')
    assert stat.S_IMODE(os.stat(str(secret)).st_mode) == 0o600


def test_atomic_save_failure_keeps_original(tmp_path):
    dest = tmp_path / 'keep.txt'
    dest.write_text('old')
    part = tmp_path / 'keep.txt.part'

    with pytest.raises(ZeroDivisionError):
        with atomic_save(str(dest), text_mode=True) as fo:
            fo.write('partial')
            1 / 0
    assert dest.read_text() == 'old'
    assert not part.exists()

    with pytest.raises(OSError):
        with atomic_save(str(dest), text_mode=True, overwrite=False) as fo:
            fo.write('never written')
    assert dest.read_text() == 'old'

    with pytest.raises(TypeError):
        atomic_save(str(dest), not_an_option=True)


def _build_tree(root):
    (root / 'sub' / 'deeper').mkdir(parents=True)
    (root / 'a.py').write_text('a')
    (root / 'b.txt').write_text('b')
    (root / 'ignore_me.py').write_text('i')
    (root / 'sub' / 'c.py').write_text('c')
    (root / 'sub' / 'deeper' / 'd.py').write_text('d')


def test_iter_find_files(tmp_path):
    _build_tree(tmp_path)

    def found(*args, **kwargs):
        return {os.path.relpath(p, str(tmp_path))
                for p in iter_find_files(str(tmp_path), *args, **kwargs)}

    assert found('*.py') == {
        'a.py', 'ignore_me.py',
        os.path.join('sub', 'c.py'),
        os.path.join('sub', 'deeper', 'd.py')}
    assert found(['*.txt', 'a*']) == {'a.py', 'b.txt'}
    assert found('*.py', ignored='ignore*') == {
        'a.py', os.path.join('sub', 'c.py'), os.path.join('sub', 'deeper', 'd.py')}
    assert found('*.py', max_depth=0) == {'a.py', 'ignore_me.py'}
    assert found('*.py', max_depth=1) == {
        'a.py', 'ignore_me.py', os.path.join('sub', 'c.py')}
    assert found('sub', include_dirs=True) == {'sub'}
    assert found('sub') == set()


def test_copy_tree_and_rotate_file(tmp_path):
    src = tmp_path / 'src'
    (src / 'nested').mkdir(parents=True)
    (src / 'a.txt').write_text('A')
    (src / 'nested' / 'b.txt').write_text('B')
    (src / 'skip.log').write_text('log')

    dst = tmp_path / 'dst'
    dst.mkdir()
    (dst / 'keep.txt').write_text('K')

    copy_tree(str(src), str(dst),
              ignore=lambda d, names: {n for n in names if n.endswith('.log')})
    assert (dst / 'a.txt').read_text() == 'A'
    assert (dst / 'nested' / 'b.txt').read_text() == 'B'
    assert (dst / 'keep.txt').read_text() == 'K'
    assert not (dst / 'skip.log').exists()

    log = tmp_path / 'log.txt'
    log.write_text('v1')
    rotate_file(str(log))
    assert not log.exists()
    assert (tmp_path / 'log.1.txt').read_text() == 'v1'

    log.write_text('v2')
    rotate_file(str(log))
    assert not log.exists()
    assert (tmp_path / 'log.1.txt').read_text() == 'v2'
    assert (tmp_path / 'log.2.txt').read_text() == 'v1'

    # nothing to rotate is fine; bad arguments are not
    rotate_file(str(tmp_path / 'missing.txt'))
    with pytest.raises(ValueError):
        rotate_file(str(log), keep=0)
    with pytest.raises(ValueError):
        rotate_file(str(src))
