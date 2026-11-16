# -*- coding: utf-8 -*-
import itertools
import pathlib
from types import SimpleNamespace

import pytest

from kindle_maker import ebooklib
from kindle_maker.ebooklib import Ebook, EbookUtil, format_file_name


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Isolate temp dirs, unique timestamps and external process calls."""
    counter = itertools.count(1000)
    monkeypatch.setattr(
        ebooklib, 'time', SimpleNamespace(time=lambda: float(next(counter))))
    base = tmp_path / 'tmp'
    base.mkdir()
    monkeypatch.setattr(
        ebooklib, 'tempfile', SimpleNamespace(gettempdir=lambda: str(base)))

    calls = []

    def fake_call(args):
        calls.append(list(args))
        last = pathlib.Path(args[-1])
        if last.suffix == '.opf':
            last.with_suffix('.mobi').write_bytes(b'MOBI')
        return 0

    monkeypatch.setattr(ebooklib, 'subprocess', SimpleNamespace(call=fake_call))
    return SimpleNamespace(calls=calls, base=base, tmp_path=tmp_path,
                           monkeypatch=monkeypatch)


def make_book(env, title='My Book', author=None):
    book = Ebook(title, author)
    c1 = book.create_chapter('Chapter 1')
    c1.set_content('<html>one</html>')
    s1 = c1.create_subchapter('Sub 1')
    s1.set_content('<html>sub one</html>')
    c2 = book.create_chapter('Chapter 2')
    c2.set_content('<html>two</html>')
    return book


def test_format_file_name_strips_illegal_chars():
    assert format_file_name('a b/c+d"e\\f:g|h') == 'abc-def-g-h'


def test_ebook_init_creates_tmpdir(env):
    book = Ebook('T')
    assert book.tmpdir.is_dir()
    assert book.tmpdir.parent == env.base
    assert book.chapter_list == []
    assert book.author is None


def test_ebook_dest_file_path(env):
    book = Ebook('My Book/1')
    assert book.dest_file_path == pathlib.Path('MyBook1.mobi')
    assert book.full_dest_file_path == book.tmpdir / 'MyBook1.mobi'


def test_set_cover_with_path_and_content(env):
    book = Ebook('T')
    assert book.set_cover(cover_path='a.jpg') is book
    assert book.cover_path == 'a.jpg'
    assert book.cover_content is None
    book.set_cover(cover_content=b'bytes')
    assert book.cover_content == b'bytes'
    assert book.cover_path is None


def test_set_cover_requires_argument(env):
    book = Ebook('T')
    with pytest.raises(ValueError):
        book.set_cover()


def test_create_chapter_and_add_chapter_rules(env):
    book = Ebook('T')
    c = book.create_chapter('C')
    assert book.chapter_list == [c]
    sub = c.create_subchapter('S')
    with pytest.raises(ValueError):
        book.add_chapter(sub)
    assert book.chapter_list == [c]


def test_chapter_with_missing_file_raises(env, tmp_path):
    book = Ebook('T')
    with pytest.raises(ValueError):
        book.create_chapter('C', str(tmp_path / 'nope.html'))


def test_subchapter_level_limit(env):
    book = Ebook('T')
    c = book.create_chapter('C')
    s = c.create_subchapter('S')
    assert c.sub_chapters == [s]
    with pytest.raises(Exception):
        s.create_subchapter('too deep')


def test_headings_structure_and_play_order(env):
    book = make_book(env)
    hs = EbookUtil(book).headings
    assert [h['title'] for h in hs] == ['Chapter 1', 'Chapter 2']
    assert hs[0]['play_order'] == 2
    assert hs[0]['file_name'] == 'Chapter1.html'
    assert hs[0]['sub_headings'] == [
        {'title': 'Sub 1', 'play_order': 3, 'file_name': 'Sub1.html'}]
    assert hs[1]['play_order'] == 4
    assert hs[1]['sub_headings'] == []


def test_headings_are_cached(env):
    book = make_book(env)
    eu = book._eu
    first = eu.headings
    book.create_chapter('Late').set_content('x')
    assert eu.headings is first
    assert len(eu.headings) == 2


def test_headings_empty_book(env):
    book = Ebook('T')
    assert EbookUtil(book).headings == []


def test_render_toc_ncx_writes_file(env):
    book = make_book(env, title='NCX Book')
    name = book._eu._render_toc_ncx()
    assert name == 'toc.ncx'
    text = (book.tmpdir / name).read_text(encoding='utf-8')
    assert 'NCX Book' in text


def test_render_toc_html_lists_chapters(env):
    book = make_book(env)
    name = book._eu._render_toc_html()
    assert name == 'toc.html'
    text = (book.tmpdir / name).read_text(encoding='utf-8')
    assert 'Chapter 1' in text
    assert 'Chapter 2' in text
    assert 'Sub 1' in text


def test_render_opf_file_name_and_content(env):
    book = make_book(env, title='Opf Book')
    name = book._eu._render_opf()
    assert name == 'OpfBook.opf'
    text = (book.tmpdir / name).read_text(encoding='utf-8')
    assert 'Opf Book' in text


def test_save_cover_default_uses_template_cover(env):
    book = Ebook('T')
    book._eu._save_cover()
    assert (book.tmpdir / 'cover.jpg').is_file()


def test_save_cover_with_custom_path(env, tmp_path):
    cover = tmp_path / 'mycover.png'
    cover.write_bytes(b'img')
    book = Ebook('T')
    book.set_cover(cover_path=str(cover))
    book._eu._save_cover()
    assert (book.tmpdir / 'mycover.png').read_bytes() == b'img'


def test_save_cover_with_content_not_implemented(env):
    book = Ebook('T')
    book.set_cover(cover_content=b'img')
    with pytest.raises(NotImplementedError):
        book._eu._save_cover()


def test_save_chapters_writes_files(env, tmp_path):
    src = tmp_path / 'src.html'
    src.write_text('<p>from file</p>', encoding='utf-8')
    static = tmp_path / 'style.css'
    static.write_text('body{}', encoding='utf-8')
    book = Ebook('T')
    c = book.create_chapter('C', str(src))
    c.add_static_files(str(static))
    s = c.create_subchapter('S')
    s.set_content('<p>sub</p>')
    book._eu._save_chapters()
    assert (book.tmpdir / 'C.html').read_text(encoding='utf-8') == '<p>from file</p>'
    assert (book.tmpdir / 'S.html').read_text(encoding='utf-8') == '<p>sub</p>'
    assert (book.tmpdir / 'style.css').is_file()


def test_create_invokes_kindlegen(env):
    book = make_book(env, title='Gen Book')
    book._eu.create()
    assert len(env.calls) == 1
    args = env.calls[0]
    assert '-dont_append_source' in args
    assert args[-1] == str(book.tmpdir / 'GenBook.opf')
    for fn in ('toc.ncx', 'toc.html', 'GenBook.opf', 'cover.jpg',
               'Chapter1.html', 'Sub1.html', 'Chapter2.html'):
        assert (book.tmpdir / fn).is_file()


def test_save_requires_chapters(env, tmp_path):
    book = Ebook('Empty')
    with pytest.raises(ValueError):
        book.save(str(tmp_path / 'out.mobi'))


def test_save_copies_result(env, tmp_path):
    book = make_book(env, title='Save Book')
    out = tmp_path / 'out.mobi'
    book.save(str(out))
    assert out.read_bytes() == b'MOBI'


def test_show_linux_uses_xdg_open(env):
    env.monkeypatch.setattr(
        ebooklib, 'platform', SimpleNamespace(system=lambda: 'Linux'))
    book = make_book(env, title='Show Book')
    rc = book.show()
    assert rc == 0
    assert env.calls[-1] == ['xdg-open', str(book.tmpdir / 'ShowBook.mobi')]
