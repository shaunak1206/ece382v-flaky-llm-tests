# -*- coding: utf-8 -*-
import gc
import itertools
import pathlib
from types import SimpleNamespace

import pytest

from kindle_maker import ebooklib
from kindle_maker.ebooklib import Ebook, EbookUtil


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolate temp dirs, timestamps, platform and subprocess calls."""
    base = tmp_path / 'tmproot'
    base.mkdir()
    counter = itertools.count(1000)
    calls = []

    def fake_call(args):
        calls.append(list(args))
        return 0

    monkeypatch.setattr(
        ebooklib, 'tempfile', SimpleNamespace(gettempdir=lambda: str(base)))
    monkeypatch.setattr(
        ebooklib, 'time', SimpleNamespace(time=lambda: next(counter)))
    monkeypatch.setattr(ebooklib, 'subprocess', SimpleNamespace(call=fake_call))
    return SimpleNamespace(base=base, calls=calls, monkeypatch=monkeypatch,
                           tmp_path=tmp_path)


def make_book(title='My Book', author=None):
    book = Ebook(title, author)
    ch1 = book.create_chapter('Chapter 1')
    ch1.set_content('<html>one</html>')
    sub = ch1.create_subchapter('Section 1.1')
    sub.set_content('<html>one-one</html>')
    ch2 = book.create_chapter('Chapter 2')
    ch2.set_content('<html>two</html>')
    return book


def test_ebook_init_creates_tmpdir_and_defaults(env):
    book = Ebook('T', 'A')
    assert book.title == 'T'
    assert book.author == 'A'
    assert book.tmpdir.is_dir()
    assert book.tmpdir.parent == env.base
    assert book.chapter_list == []
    assert book.cover_path is None
    assert book.cover_content is None


def test_ebook_dest_file_paths(env):
    book = Ebook('A B:C|D+E')
    assert book.dest_file_path == pathlib.Path('AB-C-D-E.mobi')
    assert book.full_dest_file_path == book.tmpdir / 'AB-C-D-E.mobi'


def test_set_cover_with_path_clears_content(env):
    book = Ebook('T')
    book.set_cover(cover_content=b'bytes')
    result = book.set_cover(cover_path='/some/cover.jpg')
    assert result is book
    assert book.cover_path == '/some/cover.jpg'
    assert book.cover_content is None


def test_set_cover_with_content_clears_path(env):
    book = Ebook('T')
    book.set_cover(cover_path='/some/cover.jpg')
    book.set_cover(cover_content=b'bytes')
    assert book.cover_content == b'bytes'
    assert book.cover_path is None


def test_set_cover_without_arguments_raises(env):
    book = Ebook('T')
    with pytest.raises(ValueError):
        book.set_cover()


def test_create_chapter_registers_top_level_chapter(env):
    book = Ebook('T')
    ch = book.create_chapter('First')
    assert book.chapter_list == [ch]
    assert ch.is_top_chapter
    assert ch.title == 'First'


def test_add_chapter_rejects_subchapter(env):
    book = Ebook('T')
    ch = book.create_chapter('First')
    sub = ch.create_subchapter('Sub')
    with pytest.raises(ValueError):
        book.add_chapter(sub)
    assert book.chapter_list == [ch]


def test_create_without_chapters_raises(env):
    book = Ebook('T')
    with pytest.raises(ValueError):
        book._create()


def test_headings_structure_and_play_order(env):
    book = make_book()
    hs = EbookUtil(book).headings
    assert [h['title'] for h in hs] == ['Chapter 1', 'Chapter 2']
    assert [h['play_order'] for h in hs] == [2, 4]
    assert hs[0]['file_name'] == 'Chapter1.html'
    assert hs[0]['sub_headings'] == [
        {'title': 'Section 1.1', 'play_order': 3,
         'file_name': 'Section1.1.html'}]
    assert hs[1]['sub_headings'] == []


def test_headings_are_cached(env):
    book = make_book()
    util = book._eu
    first = util.headings
    book.create_chapter('Late chapter').set_content('x')
    assert util.headings is first
    assert len(util.headings) == 2


def test_render_toc_ncx_default_author(env):
    book = make_book(title='Some Title')
    name = book._eu._render_toc_ncx()
    assert name == 'toc.ncx'
    text = (book.tmpdir / name).read_text(encoding='utf-8')
    assert 'Some Title' in text
    assert 'jachinlin.github.io' in text
    assert 'Chapter 2' in text
    assert 'Section1.1.html' in text


def test_render_toc_ncx_custom_author(env):
    book = make_book(author='Jane Doe')
    name = book._eu._render_toc_ncx()
    text = (book.tmpdir / name).read_text(encoding='utf-8')
    assert 'Jane Doe' in text
    assert 'jachinlin.github.io' not in text


def test_render_toc_html_links_chapters(env):
    book = make_book()
    name = book._eu._render_toc_html()
    assert name == 'toc.html'
    text = (book.tmpdir / name).read_text(encoding='utf-8')
    assert 'href="Chapter1.html"' in text
    assert 'href="Section1.1.html"' in text
    assert 'href="Chapter2.html"' in text


def test_render_opf_file_name_and_content(env):
    book = make_book(title='My Book', author='Jane')
    name = book._eu._render_opf()
    assert name == 'MyBook.opf'
    text = (book.tmpdir / name).read_text(encoding='utf-8')
    assert '<dc:title>My Book</dc:title>' in text
    assert '<dc:creator>Jane</dc:creator>' in text
    assert 'href="Chapter2.html"' in text
    assert 'idref="session_1.1"' in text


def test_save_cover_default_copies_bundled_cover(env):
    book = Ebook('T')
    book._eu._save_cover()
    assert (book.tmpdir / 'cover.jpg').is_file()


def test_save_cover_with_custom_path(env):
    src = env.tmp_path / 'mycover.png'
    src.write_bytes(b'png-data')
    book = Ebook('T')
    book.set_cover(cover_path=str(src))
    book._eu._save_cover()
    copied = book.tmpdir / 'mycover.png'
    assert copied.read_bytes() == b'png-data'


def test_save_cover_with_content_not_implemented(env):
    book = Ebook('T')
    book.set_cover(cover_content=b'bytes')
    with pytest.raises(NotImplementedError):
        book._eu._save_cover()


def test_save_chapters_writes_all_files(env):
    book = make_book()
    book._eu._save_chapters()
    assert (book.tmpdir / 'Chapter1.html').read_text(
        encoding='utf-8') == '<html>one</html>'
    assert (book.tmpdir / 'Section1.1.html').read_text(
        encoding='utf-8') == '<html>one-one</html>'
    assert (book.tmpdir / 'Chapter2.html').read_text(
        encoding='utf-8') == '<html>two</html>'


def test_create_invokes_kindlegen_and_writes_files(env):
    book = make_book(title='My Book')
    book._eu.create()
    assert len(env.calls) == 1
    args = env.calls[0]
    assert args[0] == ebooklib.kindlegen
    assert '-dont_append_source' in args
    assert args[-1] == str(book.tmpdir / 'MyBook.opf')
    for fn in ('toc.ncx', 'toc.html', 'MyBook.opf', 'cover.jpg',
               'Chapter1.html'):
        assert (book.tmpdir / fn).is_file()


def test_save_copies_generated_mobi(env):
    book = make_book(title='My Book')

    def fake_call(args):
        env.calls.append(list(args))
        book.full_dest_file_path.write_bytes(b'MOBI')
        return 0

    env.monkeypatch.setattr(
        ebooklib, 'subprocess', SimpleNamespace(call=fake_call))
    out = env.tmp_path / 'out.mobi'
    book.save(str(out))
    assert out.read_bytes() == b'MOBI'


@pytest.mark.parametrize('system,opener', [('Linux', 'xdg-open'),
                                           ('Darwin', 'open')])
def test_show_uses_platform_opener(env, system, opener):
    book = make_book(title='My Book')
    env.monkeypatch.setattr(
        ebooklib, 'platform', SimpleNamespace(system=lambda: system))
    rc = book.show()
    assert rc == 0
    assert env.calls[-1] == [opener, str(book.full_dest_file_path)]


def test_show_without_chapters_raises(env):
    book = Ebook('T')
    with pytest.raises(ValueError):
        book.show()


def test_del_removes_temp_directory(env):
    book = make_book()
    book._eu._save_chapters()
    tmpdir = book.tmpdir
    assert tmpdir.exists()
    del book
    gc.collect()
    assert not tmpdir.exists()
