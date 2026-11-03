# -*- coding: utf-8 -*-
import itertools
import pathlib
import tempfile

import pytest

from kindle_maker import ebooklib
from kindle_maker.ebooklib import Ebook, EbookUtil, format_file_name


class _FakeTime(object):
    """Deterministic, strictly increasing replacement for the time module."""

    def __init__(self):
        self._counter = itertools.count(1)

    def time(self):
        return float(next(self._counter))


@pytest.fixture(autouse=True)
def isolated_tmp(tmp_path, monkeypatch):
    base = tmp_path / "sysTmp"
    base.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(base))
    monkeypatch.setattr(ebooklib, "time", _FakeTime())
    return base


@pytest.fixture
def fake_kindlegen(monkeypatch):
    calls = []

    def fake_call(args, *a, **kw):
        calls.append(list(args))
        if args and str(args[0]) != "xdg-open" and str(args[0]) != "open":
            opf = pathlib.Path(args[-1])
            opf.with_suffix(".mobi").write_bytes(b"MOBI")
        return 0

    monkeypatch.setattr(ebooklib.subprocess, "call", fake_call)
    return calls


def _make_book(title="My Book", author=None):
    book = Ebook(title, author)
    c1 = book.create_chapter("Chapter 1")
    c1.set_content("<html><body>one</body></html>")
    sub = c1.create_subchapter("Section 1.1")
    sub.set_content("<html><body>one-one</body></html>")
    c2 = book.create_chapter("Chapter 2")
    c2.set_content("<html><body>two</body></html>")
    return book, c1, sub, c2


def test_format_file_name_strips_illegal_characters():
    assert format_file_name("a b") == "ab"
    assert format_file_name("a/b") == "ab"
    assert format_file_name("a+b") == "a-b"
    assert format_file_name('a"b') == "ab"
    assert format_file_name("a\\b") == "ab"
    assert format_file_name("a:b") == "a-b"
    assert format_file_name("a|b") == "a-b"


def test_format_file_name_leaves_clean_names_untouched():
    assert format_file_name("Chapter_1.html") == "Chapter_1.html"
    assert format_file_name("") == ""


def test_ebook_init_creates_tmpdir_and_defaults(isolated_tmp):
    book = Ebook("Title")
    assert book.title == "Title"
    assert book.author is None
    assert book.cover_path is None
    assert book.cover_content is None
    assert book.chapter_list == []
    assert book.tmpdir.is_dir()
    assert book.tmpdir.parent == isolated_tmp


def test_ebook_dest_file_paths():
    book = Ebook("My Book: Vol+1")
    assert book.dest_file_path == pathlib.Path("MyBook-Vol-1.mobi")
    assert book.full_dest_file_path == book.tmpdir / "MyBook-Vol-1.mobi"


def test_set_cover_with_path_clears_content():
    book = Ebook("T")
    book.set_cover(cover_content=b"bytes")
    result = book.set_cover(cover_path="/some/cover.jpg")
    assert result is book
    assert book.cover_path == "/some/cover.jpg"
    assert book.cover_content is None


def test_set_cover_with_content_clears_path():
    book = Ebook("T")
    book.set_cover(cover_path="/some/cover.jpg")
    result = book.set_cover(cover_content=b"bytes")
    assert result is book
    assert book.cover_content == b"bytes"
    assert book.cover_path is None


def test_set_cover_requires_an_argument():
    book = Ebook("T")
    with pytest.raises(ValueError):
        book.set_cover()


def test_create_chapter_registers_in_order():
    book = Ebook("T")
    a = book.create_chapter("A")
    b = book.create_chapter("B")
    assert book.chapter_list == [a, b]
    assert a.is_top_chapter


def test_add_chapter_rejects_subchapters():
    book = Ebook("T")
    top = book.create_chapter("Top")
    sub = top.create_subchapter("Sub")
    with pytest.raises(ValueError):
        book.add_chapter(sub)
    assert book.chapter_list == [top]


def test_add_chapter_returns_self_for_chaining():
    book = Ebook("T")
    top = book.create_chapter("Top")
    other = ebooklib.Chapter("Other", book)
    assert book.add_chapter(other) is book
    assert book.chapter_list == [top, other]


def test_save_with_no_chapters_raises(tmp_path):
    book = Ebook("T")
    with pytest.raises(ValueError):
        book.save(str(tmp_path / "out.mobi"))


def test_headings_play_order_and_structure():
    book, c1, sub, c2 = _make_book()
    hs = EbookUtil(book).headings
    assert [h["title"] for h in hs] == ["Chapter 1", "Chapter 2"]
    assert [h["play_order"] for h in hs] == [2, 4]
    assert hs[0]["file_name"] == "Chapter1.html"
    assert hs[0]["sub_headings"] == [
        {"title": "Section 1.1", "play_order": 3, "file_name": "Section1.1.html"}
    ]
    assert hs[1]["sub_headings"] == []


def test_headings_are_cached_once_computed():
    book, c1, sub, c2 = _make_book()
    util = EbookUtil(book)
    first = util.headings
    book.create_chapter("Late").set_content("x")
    assert util.headings is first
    assert len(util.headings) == 2


def test_headings_empty_book():
    book = Ebook("T")
    assert EbookUtil(book).headings == []


def test_render_toc_ncx_uses_default_author_and_titles():
    book, c1, sub, c2 = _make_book("Great Book")
    util = EbookUtil(book)
    name = util._render_toc_ncx()
    assert name == "toc.ncx"
    text = (book.tmpdir / "toc.ncx").read_text(encoding="utf-8")
    assert "Great Book" in text
    assert "jachinlin.github.io" in text
    assert "Chapter 1" in text and "Section 1.1" in text
    assert 'playOrder="3"' in text


def test_render_toc_ncx_uses_custom_author():
    book, _, _, _ = _make_book("Great Book", author="Jane Doe")
    EbookUtil(book)._render_toc_ncx()
    text = (book.tmpdir / "toc.ncx").read_text(encoding="utf-8")
    assert "Jane Doe" in text
    assert "jachinlin.github.io" not in text


def test_render_toc_html_links_chapters():
    book, _, _, _ = _make_book()
    name = EbookUtil(book)._render_toc_html()
    assert name == "toc.html"
    text = (book.tmpdir / "toc.html").read_text(encoding="utf-8")
    assert 'href="Chapter1.html"' in text
    assert 'href="Section1.1.html"' in text
    assert 'href="Chapter2.html"' in text


def test_render_opf_file_name_and_content():
    book, _, _, _ = _make_book("My Book", author="Zed")
    opf = EbookUtil(book)._render_opf()
    assert opf == "MyBook.opf"
    text = (book.tmpdir / opf).read_text(encoding="utf-8")
    assert "<dc:title>My Book</dc:title>" in text
    assert "<dc:creator>Zed</dc:creator>" in text
    assert 'href="Chapter1.html"' in text


def test_save_cover_default_copies_template():
    book, _, _, _ = _make_book()
    EbookUtil(book)._save_cover()
    assert (book.tmpdir / "cover.jpg").is_file()


def test_save_cover_with_custom_path(tmp_path):
    src = tmp_path / "mycover.png"
    src.write_bytes(b"png-data")
    book, _, _, _ = _make_book()
    book.set_cover(cover_path=str(src))
    EbookUtil(book)._save_cover()
    assert (book.tmpdir / "mycover.png").read_bytes() == b"png-data"


def test_save_cover_with_content_not_implemented():
    book, _, _, _ = _make_book()
    book.set_cover(cover_content=b"raw")
    with pytest.raises(NotImplementedError):
        EbookUtil(book)._save_cover()


def test_create_writes_all_files_and_invokes_kindlegen(fake_kindlegen):
    book, _, _, _ = _make_book("My Book")
    book._eu.create()
    for name in ["toc.ncx", "toc.html", "MyBook.opf", "cover.jpg",
                 "Chapter1.html", "Section1.1.html", "Chapter2.html"]:
        assert (book.tmpdir / name).is_file(), name
    assert len(fake_kindlegen) == 1
    cmd = fake_kindlegen[0]
    assert cmd[0] == ebooklib.kindlegen
    assert "-dont_append_source" in cmd
    assert cmd[-1] == str(book.tmpdir / "MyBook.opf")


def test_create_tolerates_nonzero_kindlegen_exit(monkeypatch):
    monkeypatch.setattr(ebooklib.subprocess, "call", lambda *a, **k: 1)
    book, _, _, _ = _make_book()
    assert book._eu.create() is None


def test_ebook_save_copies_result_to_destination(fake_kindlegen, tmp_path):
    book, _, _, _ = _make_book("My Book")
    dest = tmp_path / "out.mobi"
    book.save(str(dest))
    assert dest.read_bytes() == b"MOBI"
    saved = (book.tmpdir / "Chapter1.html").read_text(encoding="utf-8")
    assert "one" in saved


@pytest.mark.parametrize("system,expected", [("Linux", "xdg-open"),
                                             ("Darwin", "open")])
def test_show_uses_platform_viewer(monkeypatch, fake_kindlegen, system, expected):
    monkeypatch.setattr(ebooklib.platform, "system", lambda: system)
    book, _, _, _ = _make_book("My Book")
    rc = book.show()
    assert rc == 0
    last = fake_kindlegen[-1]
    assert last[0] == expected
    assert last[1] == str(book.full_dest_file_path)
