# -*- coding: utf-8 -*-
import gc
import itertools
import pathlib
from types import SimpleNamespace

import pytest

from kindle_maker import ebooklib
from kindle_maker.ebooklib import Ebook, EbookUtil, format_file_name


@pytest.fixture(autouse=True)
def isolated_tmp(tmp_path, monkeypatch):
    """Make Ebook temp dirs unique and located under pytest's tmp_path."""
    counter = itertools.count(1)
    monkeypatch.setattr(
        ebooklib, 'tempfile',
        SimpleNamespace(gettempdir=lambda: str(tmp_path / 'sys_tmp')))
    (tmp_path / 'sys_tmp').mkdir()
    monkeypatch.setattr(
        ebooklib, 'time', SimpleNamespace(time=lambda: next(counter)))
    return tmp_path


@pytest.fixture
def calls(monkeypatch):
    recorded = []

    def fake_call(args, *a, **kw):
        recorded.append(list(args))
        return 0

    monkeypatch.setattr(ebooklib.subprocess, 'call', fake_call)
    return recorded


@pytest.fixture
def cover_file(tmp_path):
    p = tmp_path / 'mycover.jpg'
    p.write_bytes(b'\xff\xd8\xff fake jpg')
    return p


def make_book():
    book = Ebook('My Book', author='Someone')
    c1 = book.create_chapter('Chapter1')
    c1.set_content('<html>one</html>')
    c1.create_subchapter('Sub1a').set_content('<html>1a</html>')
    c1.create_subchapter('Sub1b').set_content('<html>1b</html>')
    c2 = book.create_chapter('Chapter2')
    c2.set_content('<html>two</html>')
    return book


def test_format_file_name_strips_illegal_characters():
    assert format_file_name('a/b c+d"e\\f:g|h') == 'abcd-ef' + '-g-h'


def test_ebook_dest_paths():
    book = Ebook('My Book: Part+1')
    assert book.dest_file_path == pathlib.Path('MyBook-Part-1.mobi')
    assert book.full_dest_file_path == book.tmpdir / 'MyBook-Part-1.mobi'
    assert book.tmpdir.is_dir()


def test_set_cover_path_and_content_exclusive():
    book = Ebook('t')
    assert book.set_cover(cover_path='/x/y.jpg') is book
    assert book.cover_path == '/x/y.jpg'
    assert book.cover_content is None
    book.set_cover(cover_content=b'abc')
    assert book.cover_content == b'abc'
    assert book.cover_path is None


def test_set_cover_requires_argument():
    book = Ebook('t')
    with pytest.raises(ValueError):
        book.set_cover()


def test_create_chapter_and_add_chapter():
    book = Ebook('t')
    c = book.create_chapter('c1')
    assert book.chapter_list == [c]
    assert c.is_top_chapter
    assert book.add_chapter(book.create_chapter('c2')) is book
    assert len(book.chapter_list) == 3


def test_add_chapter_rejects_subchapter():
    book = Ebook('t')
    c = book.create_chapter('c1')
    sub = c.create_subchapter('s1')
    with pytest.raises(ValueError):
        book.add_chapter(sub)


def test_subchapter_max_level_enforced():
    book = Ebook('t')
    c = book.create_chapter('c1')
    sub = c.create_subchapter('s1')
    assert not sub.is_top_chapter
    with pytest.raises(Exception):
        sub.create_subchapter('too deep')


def test_chapter_with_missing_file_raises(tmp_path):
    book = Ebook('t')
    with pytest.raises(ValueError):
        book.create_chapter('c1', str(tmp_path / 'nope.html'))


def test_headings_structure_and_play_order():
    book = make_book()
    hs = EbookUtil(book).headings
    assert [h['title'] for h in hs] == ['Chapter1', 'Chapter2']
    assert [h['play_order'] for h in hs] == [2, 5]
    assert hs[0]['file_name'] == 'Chapter1.html'
    subs = hs[0]['sub_headings']
    assert [s['title'] for s in subs] == ['Sub1a', 'Sub1b']
    assert [s['play_order'] for s in subs] == [3, 4]
    assert subs[0]['file_name'] == 'Sub1a.html'
    assert hs[1]['sub_headings'] == []


def test_headings_are_cached():
    book = make_book()
    eu = EbookUtil(book)
    first = eu.headings
    book.create_chapter('Late').set_content('x')
    assert eu.headings is first
    assert len(eu.headings) == 2


def test_render_opf_file_name_uses_formatted_title():
    book = make_book()
    eu = EbookUtil(book)
    name = eu._render_opf()
    assert name == 'MyBook.opf'
    assert (book.tmpdir / name).is_file()
    assert 'My Book' in (book.tmpdir / name).read_text(encoding='utf-8')


def test_render_toc_files_created():
    book = make_book()
    eu = EbookUtil(book)
    assert eu._render_toc_ncx() == 'toc.ncx'
    assert eu._render_toc_html() == 'toc.html'
    ncx = (book.tmpdir / 'toc.ncx').read_text(encoding='utf-8')
    assert 'My Book' in ncx
    assert 'Someone' in ncx
    assert (book.tmpdir / 'toc.html').stat().st_size > 0


def test_render_toc_ncx_default_author():
    book = Ebook('NoAuthor')
    book.create_chapter('c').set_content('x')
    EbookUtil(book)._render_toc_ncx()
    ncx = (book.tmpdir / 'toc.ncx').read_text(encoding='utf-8')
    assert 'jachinlin.github.io' in ncx


def test_save_cover_copies_given_cover(cover_file):
    book = Ebook('t')
    book.set_cover(cover_path=str(cover_file))
    EbookUtil(book)._save_cover()
    assert (book.tmpdir / 'mycover.jpg').read_bytes() == cover_file.read_bytes()


def test_save_cover_with_content_not_implemented():
    book = Ebook('t')
    book.set_cover(cover_content=b'data')
    with pytest.raises(NotImplementedError):
        EbookUtil(book)._save_cover()


def test_create_writes_files_and_calls_kindlegen(calls, cover_file):
    book = make_book()
    book.set_cover(cover_path=str(cover_file))
    EbookUtil(book).create()
    for name in ['toc.ncx', 'toc.html', 'MyBook.opf', 'mycover.jpg',
                 'Chapter1.html', 'Sub1a.html', 'Sub1b.html', 'Chapter2.html']:
        assert (book.tmpdir / name).is_file(), name
    assert (book.tmpdir / 'Sub1a.html').read_text(encoding='utf-8') == '<html>1a</html>'
    assert len(calls) == 1
    assert calls[0][0] == ebooklib.kindlegen
    assert '-dont_append_source' in calls[0]
    assert calls[0][-1] == str(book.tmpdir / 'MyBook.opf')


def test_ebook_save_without_chapters_raises(tmp_path):
    book = Ebook('empty')
    with pytest.raises(ValueError):
        book.save(str(tmp_path / 'out.mobi'))


def test_ebook_save_copies_generated_mobi(tmp_path, monkeypatch, cover_file):
    book = make_book()
    book.set_cover(cover_path=str(cover_file))

    def fake_call(args, *a, **kw):
        book.full_dest_file_path.write_bytes(b'MOBI-DATA')
        return 0

    monkeypatch.setattr(ebooklib.subprocess, 'call', fake_call)
    out = tmp_path / 'result.mobi'
    book.save(str(out))
    assert out.read_bytes() == b'MOBI-DATA'


def test_show_on_linux_uses_xdg_open(monkeypatch, calls, cover_file):
    book = make_book()
    book.set_cover(cover_path=str(cover_file))
    monkeypatch.setattr(ebooklib.platform, 'system', lambda: 'Linux')
    rc = book.show()
    assert rc == 0
    assert calls[-1] == ['xdg-open', str(book.full_dest_file_path)]


def test_ebook_tmpdir_removed_on_delete():
    book = make_book()
    tmpdir = book.tmpdir
    (tmpdir / 'sub').mkdir()
    (tmpdir / 'sub' / 'f.txt').write_text('x')
    assert tmpdir.exists()
    del book
    gc.collect()
    assert not tmpdir.exists()
