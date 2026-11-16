# -*- coding: utf-8 -*-
import types

import pytest

from kindle_maker import ebooklib
from kindle_maker.ebooklib import Ebook, EbookUtil, format_file_name


@pytest.fixture
def calls(monkeypatch):
    """Replace subprocess.call in ebooklib so kindlegen is never run."""
    recorded = []

    def fake_call(args, *a, **kw):
        recorded.append(list(args))
        return 0

    monkeypatch.setattr(ebooklib.subprocess, "call", fake_call)
    return recorded


@pytest.fixture
def ebook(tmp_path, monkeypatch):
    """An Ebook whose temp dir lives under pytest's tmp_path."""
    monkeypatch.setattr(ebooklib.tempfile, "gettempdir", lambda: str(tmp_path))
    counter = iter(range(1, 10000))
    monkeypatch.setattr(
        ebooklib, "time",
        types.SimpleNamespace(time=lambda: float(next(counter))))
    return Ebook("My Book: Test", author="Someone")


def test_format_file_name_strips_illegal_chars():
    assert format_file_name('a/b c+d"e\\f:g|h') == 'abcd-ef-g-h'


def test_format_file_name_plain_unchanged():
    assert format_file_name('plain-name') == 'plain-name'


def test_ebook_init_creates_tmpdir(ebook):
    assert ebook.tmpdir.is_dir()
    assert ebook.chapter_list == []
    assert ebook.cover_path is None
    assert ebook.cover_content is None
    assert ebook.author == "Someone"


def test_ebook_dest_file_paths(ebook):
    assert str(ebook.dest_file_path) == 'MyBook-Test.mobi'
    assert ebook.full_dest_file_path == ebook.tmpdir / 'MyBook-Test.mobi'


def test_set_cover_path_and_content(ebook):
    assert ebook.set_cover(cover_path='x.jpg') is ebook
    assert ebook.cover_path == 'x.jpg'
    assert ebook.cover_content is None
    ebook.set_cover(cover_content=b'bytes')
    assert ebook.cover_content == b'bytes'
    assert ebook.cover_path is None


def test_set_cover_requires_argument(ebook):
    with pytest.raises(ValueError):
        ebook.set_cover()


def test_create_chapter_adds_to_list(ebook):
    c = ebook.create_chapter('Chapter 1')
    assert ebook.chapter_list == [c]
    assert c.is_top_chapter
    assert str(c.dest_file_path) == 'Chapter1.html'


def test_add_chapter_rejects_subchapter(ebook):
    c = ebook.create_chapter('Top')
    sub = c.create_subchapter('Sub')
    with pytest.raises(ValueError):
        ebook.add_chapter(sub)


def test_create_subchapter_level_limit(ebook):
    c = ebook.create_chapter('Top')
    sub = c.create_subchapter('Sub')
    assert c.sub_chapters == [sub]
    assert not sub.is_top_chapter
    with pytest.raises(Exception):
        sub.create_subchapter('Too deep')


def test_chapter_with_missing_file_raises(ebook, tmp_path):
    with pytest.raises(ValueError):
        ebook.create_chapter('Bad', str(tmp_path / 'nope.html'))


def test_headings_play_order_and_structure(ebook):
    a = ebook.create_chapter('A')
    a.create_subchapter('A 1')
    b = ebook.create_chapter('B')
    hs = EbookUtil(ebook).headings
    assert [h['title'] for h in hs] == ['A', 'B']
    assert hs[0]['play_order'] == 2
    assert hs[0]['file_name'] == 'A.html'
    assert hs[0]['sub_headings'] == [
        {'title': 'A 1', 'play_order': 3, 'file_name': 'A1.html'}]
    assert hs[1]['play_order'] == 4
    assert hs[1]['sub_headings'] == []
    assert b.title == 'B'


def test_headings_are_cached(ebook):
    ebook.create_chapter('A')
    eu = EbookUtil(ebook)
    first = eu.headings
    ebook.create_chapter('B')
    assert eu.headings is first
    assert len(eu.headings) == 1


def test_save_cover_content_not_implemented(ebook):
    ebook.set_cover(cover_content=b'abc')
    with pytest.raises(NotImplementedError):
        EbookUtil(ebook)._save_cover()


def test_save_cover_copies_given_path(ebook, tmp_path):
    cover = tmp_path / 'mycover.jpg'
    cover.write_bytes(b'img')
    ebook.set_cover(cover_path=str(cover))
    EbookUtil(ebook)._save_cover()
    assert (ebook.tmpdir / 'mycover.jpg').read_bytes() == b'img'


def test_save_cover_default_template(ebook):
    EbookUtil(ebook)._save_cover()
    assert (ebook.tmpdir / 'cover.jpg').is_file()


def test_create_renders_files_and_calls_kindlegen(ebook, calls):
    c = ebook.create_chapter('Intro')
    c.set_content('<html>intro</html>')
    c.create_subchapter('Part').set_content('<html>part</html>')
    EbookUtil(ebook).create()
    for name in ('toc.ncx', 'toc.html', 'MyBook-Test.opf', 'cover.jpg',
                 'Intro.html', 'Part.html'):
        assert (ebook.tmpdir / name).is_file(), name
    assert (ebook.tmpdir / 'Intro.html').read_text(
        encoding='utf-8') == '<html>intro</html>'
    assert len(calls) == 1
    assert calls[0][1] == '-dont_append_source'
    assert calls[0][2] == str(ebook.tmpdir / 'MyBook-Test.opf')


def test_ebook_save_without_chapters_raises(ebook, calls, tmp_path):
    with pytest.raises(ValueError):
        ebook.save(str(tmp_path / 'out.mobi'))
    assert calls == []


def test_ebook_save_copies_result(ebook, monkeypatch, tmp_path):
    def fake_call(args, *a, **kw):
        ebook.full_dest_file_path.write_bytes(b'MOBI')
        return 0

    monkeypatch.setattr(ebooklib.subprocess, "call", fake_call)
    ebook.create_chapter('One').set_content('hello')
    out = tmp_path / 'out.mobi'
    ebook.save(str(out))
    assert out.read_bytes() == b'MOBI'


def test_show_on_linux_uses_xdg_open(ebook, calls, monkeypatch):
    monkeypatch.setattr(ebooklib.platform, "system", lambda: 'Linux')
    ebook.create_chapter('One').set_content('hello')
    rc = ebook.show()
    assert rc == 0
    assert calls[-1] == ['xdg-open', str(ebook.full_dest_file_path)]
