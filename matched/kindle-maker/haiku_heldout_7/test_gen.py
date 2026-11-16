import gc
import itertools
import pathlib
from types import SimpleNamespace

import pytest

from kindle_maker import ebooklib
from kindle_maker.ebooklib import Chapter, Ebook, EbookUtil, format_file_name
from kindle_maker.kindlegen import kindlegen


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """
    Isolate temp dirs, time-based directory names and kindlegen calls.
    Returns the list of recorded subprocess.call argument lists.
    """
    counter = itertools.count(1)
    monkeypatch.setattr(
        ebooklib, 'time',
        SimpleNamespace(time=lambda: float(next(counter))))

    tmp_root = tmp_path / 'tmp_root'
    tmp_root.mkdir()
    monkeypatch.setattr(
        ebooklib, 'tempfile',
        SimpleNamespace(gettempdir=lambda: str(tmp_root)))

    calls = []

    def fake_call(args, *a, **kw):
        calls.append(list(args))
        if args[0] == kindlegen:
            # Simulate kindlegen: produce <stem>.mobi next to the opf file.
            opf = pathlib.Path(args[-1])
            (opf.parent / (opf.stem + '.mobi')).write_bytes(b'MOBI')
        return 0

    monkeypatch.setattr(ebooklib, 'subprocess',
                        SimpleNamespace(call=fake_call))
    return calls


@pytest.fixture
def ebook():
    book = Ebook('Book')
    yield book


@pytest.fixture
def ebook_with_chapter(ebook):
    ch = ebook.create_chapter('Intro')
    ch.set_content('<p>hello</p>')
    return ebook


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def test_format_file_name_strips_illegal_characters():
    assert format_file_name('a/b c\\d"e') == 'abcde'


def test_format_file_name_replaces_plus_colon_pipe_with_dash():
    assert format_file_name('A+B:C|D') == 'A-B-C-D'


def test_ebook_creates_existing_tmpdir(ebook):
    assert ebook.tmpdir.is_dir()
    assert ebook.title == 'Book'
    assert ebook.author is None


def test_ebook_dest_file_path_uses_sanitized_title(tmp_path):
    book = Ebook('My Book: A+B')
    assert book.dest_file_path == pathlib.Path('MyBook-A-B.mobi')


def test_set_cover_with_path_clears_content(ebook, tmp_path):
    cover = write_text(tmp_path / 'c.jpg', 'x')
    ebook.set_cover(cover_content=b'bytes')
    result = ebook.set_cover(cover_path=str(cover))
    assert result is ebook
    assert ebook.cover_path == str(cover)
    assert ebook.cover_content is None


def test_set_cover_with_content_clears_path(ebook, tmp_path):
    cover = write_text(tmp_path / 'c.jpg', 'x')
    ebook.set_cover(cover_path=str(cover))
    result = ebook.set_cover(cover_content=b'bytes')
    assert result is ebook
    assert ebook.cover_content == b'bytes'
    assert ebook.cover_path is None


def test_set_cover_without_arguments_raises(ebook):
    with pytest.raises(ValueError):
        ebook.set_cover()


def test_add_chapter_rejects_subchapter_level(ebook):
    sub = Chapter('Sub', ebook, level=2)
    with pytest.raises(ValueError):
        ebook.add_chapter(sub)
    assert ebook.chapter_list == []


def test_create_chapter_appends_and_returns_chapter(ebook):
    ch = ebook.create_chapter('Chapter One')
    assert isinstance(ch, Chapter)
    assert ch.title == 'Chapter One'
    assert ebook.chapter_list == [ch]


def test_chapter_missing_file_raises_value_error(ebook, tmp_path):
    with pytest.raises(ValueError):
        ebook.create_chapter('Missing', str(tmp_path / 'nope.html'))


def test_chapter_content_reads_utf8_file(ebook, tmp_path):
    path = write_text(tmp_path / 'ch.html', 'héllo 中文')
    ch = ebook.create_chapter('Ch', str(path))
    assert ch.content == 'héllo 中文'


def test_set_content_overrides_file_path(ebook, tmp_path):
    path = write_text(tmp_path / 'ch.html', 'from file')
    ch = ebook.create_chapter('Ch', str(path))
    assert ch.set_content('from memory') is ch
    assert ch.content == 'from memory'


def test_create_subchapter_nests_with_increased_level(ebook):
    ch = ebook.create_chapter('Top')
    sub = ch.create_subchapter('Sub')
    assert ch.sub_chapters == [sub]
    assert sub.is_top_chapter is False


def test_create_subchapter_beyond_max_level_raises(ebook):
    ch = ebook.create_chapter('Top')
    sub = ch.create_subchapter('Sub')
    with pytest.raises(Exception):
        sub.create_subchapter('TooDeep')


def test_add_static_files_rejects_missing_file(ebook, tmp_path):
    ch = ebook.create_chapter('Ch')
    with pytest.raises(AssertionError):
        ch.add_static_files(str(tmp_path / 'missing.png'))


def test_headings_compute_play_order_and_nesting(ebook):
    c1 = ebook.create_chapter('Chapter One')
    c1.create_subchapter('Section A')
    ebook.create_chapter('Chapter Two')

    headings = ebook._eu.headings

    assert len(headings) == 2
    assert headings[0] == {
        'title': 'Chapter One',
        'play_order': 2,
        'file_name': 'ChapterOne.html',
        'sub_headings': [{
            'title': 'Section A',
            'play_order': 3,
            'file_name': 'SectionA.html',
        }],
    }
    assert headings[1]['play_order'] == 4
    assert headings[1]['sub_headings'] == []


def test_headings_are_cached_once_computed(ebook):
    ebook.create_chapter('First')
    eu = ebook._eu
    first = eu.headings
    ebook.create_chapter('Second')
    assert eu.headings is first
    assert len(eu.headings) == 1


def test_save_without_chapters_raises_value_error(ebook, tmp_path):
    with pytest.raises(ValueError):
        ebook.save(str(tmp_path / 'out.mobi'))


def test_save_copies_generated_mobi_to_target(ebook_with_chapter, tmp_path):
    out = tmp_path / 'out.mobi'
    ebook_with_chapter.save(str(out))
    assert out.read_bytes() == b'MOBI'


def test_create_invokes_kindlegen_with_opf(ebook_with_chapter, sandbox):
    ebook_with_chapter._eu.create()
    assert len(sandbox) == 1
    args = sandbox[0]
    assert args[0] == kindlegen
    assert args[1] == '-dont_append_source'
    assert args[2].endswith('Book.opf')


def test_create_renders_toc_and_opf_files(ebook_with_chapter):
    ebook_with_chapter._eu.create()
    tmpdir = ebook_with_chapter.tmpdir
    assert (tmpdir / 'toc.ncx').is_file()
    assert (tmpdir / 'toc.html').is_file()
    assert (tmpdir / 'Book.opf').is_file()
    assert (tmpdir / 'Intro.html').is_file()


def test_save_cover_default_copies_bundled_cover(ebook):
    ebook._eu._save_cover()
    assert (ebook.tmpdir / 'cover.jpg').is_file()


def test_save_cover_with_path_copies_given_file(ebook, tmp_path):
    cover = tmp_path / 'mine.jpg'
    cover.write_bytes(b'JPEGDATA')
    ebook.set_cover(cover_path=str(cover))
    ebook._eu._save_cover()
    assert (ebook.tmpdir / 'mine.jpg').read_bytes() == b'JPEGDATA'


def test_save_cover_with_content_is_not_implemented(ebook):
    ebook.set_cover(cover_content=b'raw')
    with pytest.raises(NotImplementedError):
        ebook._eu._save_cover()


def test_render_toc_ncx_defaults_author(ebook):
    captured = {}
    eu = ebook._eu
    eu._render_file = lambda name, ctx, fn: captured.update(
        name=name, ctx=ctx, fn=fn)
    assert eu._render_toc_ncx() == 'toc.ncx'
    assert captured['name'] == 'toc.xml'
    assert captured['ctx']['title'] == 'Book'
    assert captured['ctx']['author'] == 'jachinlin.github.io'


@pytest.mark.parametrize('system, command', [
    ('Linux', 'xdg-open'),
    ('Darwin', 'open'),
])
def test_show_invokes_platform_viewer(ebook_with_chapter, sandbox,
                                      monkeypatch, system, command):
    monkeypatch.setattr(ebooklib, 'platform',
                        SimpleNamespace(system=lambda: system))
    rc = ebook_with_chapter.show()
    assert rc == 0
    assert sandbox[-1] == [command,
                           str(ebook_with_chapter.full_dest_file_path)]


def test_del_removes_tmpdir(tmp_path):
    book = Ebook('Gone')
    tmpdir = book.tmpdir
    assert tmpdir.is_dir()
    del book
    gc.collect()
    assert not tmpdir.exists()
