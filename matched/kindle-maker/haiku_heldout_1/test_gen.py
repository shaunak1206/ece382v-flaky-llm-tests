import gc
import itertools
import pathlib
import time as real_time

import pytest

import kindle_maker.ebooklib as m
from kindle_maker.ebooklib import Chapter, Ebook, EbookUtil, format_file_name


@pytest.fixture(autouse=True)
def unique_tmpdirs(monkeypatch):
    """Ebook temp dirs are named from time.time(); make them unique per call."""
    counter = itertools.count()
    base = real_time.time()

    class _FakeTime(object):
        @staticmethod
        def time():
            return base + next(counter)

    monkeypatch.setattr(m, 'time', _FakeTime)


@pytest.fixture
def fake_call(monkeypatch):
    """Replace kindlegen invocation: record args and emit a fake .mobi file."""
    calls = []

    def fake(args, *a, **k):
        calls.append(list(args))
        opf = pathlib.Path(args[-1])
        (opf.parent / (opf.stem + '.mobi')).write_bytes(b'MOBI')
        return 0

    monkeypatch.setattr(m.subprocess, 'call', fake)
    return calls


@pytest.fixture
def ebook():
    return Ebook('My Book', author='Tester')


def test_format_file_name_removes_and_replaces_illegal_chars():
    assert format_file_name('a/b c+d"e\\f:g|h') == 'abc-def-g-h'


def test_format_file_name_leaves_plain_names_unchanged():
    assert format_file_name('Chapter1') == 'Chapter1'


def test_ebook_init_stores_metadata_and_creates_tmpdir(ebook):
    assert ebook.title == 'My Book'
    assert ebook.author == 'Tester'
    assert ebook.chapter_list == []
    assert ebook.tmpdir.is_dir()
    assert ebook.tmpdir.parent == pathlib.Path(m.tempfile.gettempdir())


def test_ebook_default_author_is_none():
    assert Ebook('NoAuthor').author is None


def test_ebook_dest_file_path_is_sanitized_mobi_name(ebook):
    assert ebook.dest_file_path == pathlib.Path('MyBook.mobi')
    assert ebook.full_dest_file_path == ebook.tmpdir / 'MyBook.mobi'


def test_set_cover_with_path_clears_content(ebook):
    ebook.set_cover(cover_content=b'abc')
    result = ebook.set_cover(cover_path='/some/cover.jpg')
    assert result is ebook
    assert ebook.cover_path == '/some/cover.jpg'
    assert ebook.cover_content is None


def test_set_cover_with_content_clears_path(ebook):
    ebook.set_cover(cover_path='/some/cover.jpg')
    result = ebook.set_cover(cover_content=b'abc')
    assert result is ebook
    assert ebook.cover_content == b'abc'
    assert ebook.cover_path is None


def test_set_cover_without_arguments_raises(ebook):
    with pytest.raises(ValueError):
        ebook.set_cover()


def test_add_chapter_rejects_non_top_level_chapter(ebook):
    sub = Chapter('Sub', ebook, level=2)
    with pytest.raises(ValueError):
        ebook.add_chapter(sub)
    assert ebook.chapter_list == []


def test_create_chapter_appends_and_returns_chapter(ebook):
    ch = ebook.create_chapter('Intro')
    assert isinstance(ch, Chapter)
    assert ch.title == 'Intro'
    assert ebook.chapter_list == [ch]


def test_create_chapter_with_missing_file_raises(ebook, tmp_path):
    with pytest.raises(ValueError):
        ebook.create_chapter('Ghost', str(tmp_path / 'missing.html'))


def test_save_with_empty_chapter_list_raises(ebook, tmp_path, fake_call):
    with pytest.raises(ValueError):
        ebook.save(str(tmp_path / 'out.mobi'))
    assert fake_call == []


def test_headings_structure_and_play_order(ebook):
    c1 = ebook.create_chapter('One')
    c1.create_subchapter('One-A')
    c1.create_subchapter('One-B')
    ebook.create_chapter('Two')

    hs = EbookUtil(ebook).headings

    assert [h['title'] for h in hs] == ['One', 'Two']
    assert [h['play_order'] for h in hs] == [2, 5]
    assert [s['play_order'] for s in hs[0]['sub_headings']] == [3, 4]
    assert hs[0]['file_name'] == 'One.html'
    assert hs[0]['sub_headings'][0]['file_name'] == 'One-A.html'


def test_headings_are_cached(ebook):
    ebook.create_chapter('Only')
    util = EbookUtil(ebook)
    first = util.headings
    second = util.headings
    assert first is second


def test_create_subchapter_beyond_max_level_raises(ebook):
    parent = ebook.create_chapter('Parent')
    sub = parent.create_subchapter('Sub')
    assert sub.is_top_chapter is False
    assert parent.sub_chapters == [sub]
    with pytest.raises(Exception):
        sub.create_subchapter('SubSub')


def test_chapter_dest_file_path_is_sanitized(ebook):
    ch = Chapter('Intro Part', ebook)
    assert ch.dest_file_path == pathlib.Path('IntroPart.html')
    assert ch.full_dest_file_path == ebook.tmpdir / 'IntroPart.html'


def test_chapter_content_read_from_utf8_file(ebook, tmp_path):
    f = tmp_path / 'ch.html'
    f.write_text(u'h\u00e9llo', encoding='utf-8')
    ch = Chapter('C', ebook, str(f))
    assert ch.content == u'h\u00e9llo'


def test_chapter_set_content_clears_file_path(ebook, tmp_path):
    f = tmp_path / 'ch.html'
    f.write_text('from file', encoding='utf-8')
    ch = Chapter('C', ebook, str(f))
    result = ch.set_content('inline')
    assert result is ch
    assert ch.content == 'inline'


def test_chapter_save_writes_content_and_copies_static_files(ebook, tmp_path):
    pic = tmp_path / 'pic.png'
    pic.write_bytes(b'PNGDATA')
    ch = Chapter('Intro', ebook)
    ch.set_content('hello')
    ch.add_static_files(str(pic))
    ch.save()
    assert (ebook.tmpdir / 'Intro.html').read_text(encoding='utf-8') == 'hello'
    assert (ebook.tmpdir / 'pic.png').read_bytes() == b'PNGDATA'


def test_create_renders_files_and_calls_kindlegen(ebook, fake_call):
    ch = ebook.create_chapter('Intro')
    ch.set_content('<p>text</p>')
    ebook.create_chapter('Next').set_content('<p>more</p>')

    ebook._eu.create()

    assert (ebook.tmpdir / 'toc.ncx').is_file()
    assert (ebook.tmpdir / 'toc.html').is_file()
    opf = ebook.tmpdir / 'MyBook.opf'
    assert opf.is_file()
    assert 'My Book' in opf.read_text(encoding='utf-8')
    assert (ebook.tmpdir / 'Intro.html').is_file()
    assert (ebook.tmpdir / 'cover.jpg').is_file()

    assert len(fake_call) == 1
    assert fake_call[0][0] == m.kindlegen
    assert fake_call[0][1] == '-dont_append_source'
    assert fake_call[0][2] == str(opf)


def test_create_copies_custom_cover(ebook, tmp_path, fake_call):
    cover = tmp_path / 'cover.png'
    cover.write_bytes(b'PNG')
    ebook.set_cover(cover_path=str(cover))
    ebook.create_chapter('Intro').set_content('x')

    ebook._eu.create()

    assert (ebook.tmpdir / 'cover.png').read_bytes() == b'PNG'


def test_create_cover_content_not_implemented(ebook, fake_call):
    ebook.set_cover(cover_content=b'bytes')
    ebook.create_chapter('Intro').set_content('x')
    with pytest.raises(NotImplementedError):
        ebook._eu.create()


def test_save_copies_mobi_to_destination(ebook, tmp_path, fake_call):
    ebook.create_chapter('Intro').set_content('text')
    dest = tmp_path / 'out.mobi'
    ebook.save(str(dest))
    assert dest.read_bytes() == b'MOBI'
    assert len(fake_call) == 1


def test_del_removes_tmpdir():
    book = Ebook('Gone')
    tmp = book.tmpdir
    assert tmp.is_dir()
    del book
    gc.collect()
    assert not tmp.exists()
