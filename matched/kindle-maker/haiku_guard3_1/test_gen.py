import gc
import tempfile
from types import SimpleNamespace

import pytest

from kindle_maker import ebooklib
from kindle_maker.ebooklib import Ebook, EbookUtil


class _FakeClock:
    """Stands in for the ``time`` module inside ebooklib so each Ebook gets a unique temp dir name."""

    def __init__(self):
        self._n = 0

    def time(self):
        self._n += 1
        return float(self._n)


class _FakeSubprocess:
    """Records calls to subprocess.call instead of running kindlegen or viewers."""

    def __init__(self):
        self.calls = []
        self.on_call = None
        self.returncode = 0

    def call(self, args, *a, **k):
        self.calls.append(list(args))
        if self.on_call is not None:
            self.on_call(args)
        return self.returncode


class _FakePlatform:
    def __init__(self):
        self.name = 'Linux'

    def system(self):
        return self.name


@pytest.fixture
def env(tmp_path, monkeypatch):
    # All replacements go through monkeypatch, so they are undone after each test.
    monkeypatch.setattr(tempfile, 'gettempdir', lambda: str(tmp_path))
    monkeypatch.setattr(ebooklib, 'time', _FakeClock())
    fake_sp = _FakeSubprocess()
    monkeypatch.setattr(ebooklib, 'subprocess', fake_sp)
    fake_pl = _FakePlatform()
    monkeypatch.setattr(ebooklib, 'platform', fake_pl)
    return SimpleNamespace(sp=fake_sp, pl=fake_pl)


def _write(path, data):
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding='utf-8')
    return str(path)


def test_ebook_stores_title_and_author(env):
    ebook = Ebook('My Book', author='Jane')
    assert ebook.title == 'My Book'
    assert ebook.author == 'Jane'


def test_ebook_author_defaults_to_none(env):
    ebook = Ebook('Book')
    assert ebook.author is None
    assert ebook.chapter_list == []


def test_ebook_creates_tmpdir_inside_temp_location(env, tmp_path):
    ebook = Ebook('Book')
    assert ebook.tmpdir.is_dir()
    assert ebook.tmpdir.parent == tmp_path


def test_dest_file_path_strips_illegal_characters(env):
    ebook = Ebook('a/b c+d:e|f"g\\h')
    assert ebook.dest_file_path.name == 'abc-d-e-fgh.mobi'


def test_set_cover_with_path_clears_content_and_chains(env):
    ebook = Ebook('Book')
    ebook.set_cover(cover_content=b'bytes')
    result = ebook.set_cover(cover_path='/some/cover.jpg')
    assert result is ebook
    assert ebook.cover_path == '/some/cover.jpg'
    assert ebook.cover_content is None


def test_set_cover_with_content_clears_path(env):
    ebook = Ebook('Book')
    ebook.set_cover(cover_path='/some/cover.jpg')
    result = ebook.set_cover(cover_content=b'bytes')
    assert result is ebook
    assert ebook.cover_content == b'bytes'
    assert ebook.cover_path is None


def test_set_cover_without_arguments_raises(env):
    ebook = Ebook('Book')
    with pytest.raises(ValueError):
        ebook.set_cover()


def test_add_chapter_rejects_subchapter(env):
    ebook = Ebook('Book')
    parent = ebook.create_chapter('Parent').set_content('p')
    sub = parent.create_subchapter('Sub').set_content('s')
    with pytest.raises(ValueError):
        ebook.add_chapter(sub)
    assert ebook.chapter_list == [parent]


def test_create_chapter_appends_and_returns_chapter(env):
    ebook = Ebook('Book')
    ch = ebook.create_chapter('Intro')
    assert ebook.chapter_list == [ch]
    assert ch.title == 'Intro'
    assert ch.is_top_chapter


def test_create_chapter_with_missing_file_raises(env, tmp_path):
    ebook = Ebook('Book')
    with pytest.raises(ValueError):
        ebook.create_chapter('Missing', str(tmp_path / 'nope.html'))


def test_save_without_chapters_raises_and_does_not_run_kindlegen(env, tmp_path):
    ebook = Ebook('Book')
    with pytest.raises(ValueError):
        ebook.save(str(tmp_path / 'out.mobi'))
    assert env.sp.calls == []


def test_save_copies_kindlegen_output_to_target(env, tmp_path):
    ebook = Ebook('Book')
    ebook.create_chapter('C').set_content('<p>x</p>')
    env.sp.on_call = lambda args: ebook.full_dest_file_path.write_bytes(b'MOBI')
    target = tmp_path / 'out.mobi'
    ebook.save(str(target))
    assert target.read_bytes() == b'MOBI'


def test_show_on_linux_uses_xdg_open(env):
    ebook = Ebook('Book')
    ebook.create_chapter('C').set_content('x')
    env.pl.name = 'Linux'
    rc = ebook.show()
    assert rc == 0
    assert env.sp.calls[-1] == ['xdg-open', str(ebook.full_dest_file_path)]


def test_show_on_darwin_uses_open(env):
    ebook = Ebook('Book')
    ebook.create_chapter('C').set_content('x')
    env.pl.name = 'Darwin'
    rc = ebook.show()
    assert rc == 0
    assert env.sp.calls[-1] == ['open', str(ebook.full_dest_file_path)]


def test_create_renders_toc_and_opf_files(env):
    ebook = Ebook('My Book')
    ebook.create_chapter('C').set_content('x')
    EbookUtil(ebook).create()
    assert (ebook.tmpdir / 'toc.ncx').is_file()
    assert (ebook.tmpdir / 'toc.html').is_file()
    assert (ebook.tmpdir / 'MyBook.opf').is_file()


def test_create_passes_opf_path_to_kindlegen(env):
    ebook = Ebook('My Book')
    ebook.create_chapter('C').set_content('x')
    EbookUtil(ebook).create()
    assert len(env.sp.calls) == 1
    args = env.sp.calls[0]
    assert args[0] == ebooklib.kindlegen
    assert args[1] == '-dont_append_source'
    assert args[2] == str(ebook.tmpdir / 'MyBook.opf')


def test_create_copies_default_cover_when_none_set(env):
    ebook = Ebook('Book')
    ebook.create_chapter('C').set_content('x')
    EbookUtil(ebook).create()
    assert (ebook.tmpdir / 'cover.jpg').is_file()


def test_create_copies_custom_cover_file(env, tmp_path):
    ebook = Ebook('Book')
    ebook.create_chapter('C').set_content('x')
    src = _write(tmp_path / 'mine.png', b'PNGDATA')
    ebook.set_cover(cover_path=src)
    EbookUtil(ebook).create()
    assert (ebook.tmpdir / 'mine.png').read_bytes() == b'PNGDATA'


def test_create_with_cover_content_raises_not_implemented(env):
    ebook = Ebook('Book')
    ebook.create_chapter('C').set_content('x')
    ebook.set_cover(cover_content=b'raw')
    with pytest.raises(NotImplementedError):
        EbookUtil(ebook).create()


def test_create_writes_chapter_content_to_tmpdir(env, tmp_path):
    ebook = Ebook('Book')
    src = _write(tmp_path / 'src.html', 'héllo')
    ebook.create_chapter('Intro', src)
    EbookUtil(ebook).create()
    assert (ebook.tmpdir / 'Intro.html').read_text(encoding='utf-8') == 'héllo'


def test_headings_structure_and_play_order(env):
    ebook = Ebook('Book')
    ch_a = ebook.create_chapter('Ch A').set_content('a')
    ch_a.create_subchapter('Sub A1').set_content('s')
    ebook.create_chapter('Ch B').set_content('b')

    headings = EbookUtil(ebook).headings

    assert headings == [
        {
            'title': 'Ch A',
            'play_order': 2,
            'file_name': 'ChA.html',
            'sub_headings': [
                {'title': 'Sub A1', 'play_order': 3, 'file_name': 'SubA1.html'},
            ],
        },
        {
            'title': 'Ch B',
            'play_order': 4,
            'file_name': 'ChB.html',
            'sub_headings': [],
        },
    ]


def test_headings_empty_when_no_chapters(env):
    ebook = Ebook('Book')
    assert EbookUtil(ebook).headings == []


def test_headings_are_cached_once_populated(env):
    ebook = Ebook('Book')
    util = EbookUtil(ebook)
    ebook.create_chapter('One').set_content('x')
    first = util.headings
    assert len(first) == 1

    ebook.create_chapter('Two').set_content('y')
    second = util.headings
    assert second is first
    assert len(second) == 1


def test_del_removes_tmpdir(env):
    ebook = Ebook('Gone')
    tmp = ebook.tmpdir
    assert tmp.is_dir()
    del ebook
    # EbookUtil holds a reference back to the Ebook, so a cycle must be collected.
    gc.collect()
    assert not tmp.exists()
