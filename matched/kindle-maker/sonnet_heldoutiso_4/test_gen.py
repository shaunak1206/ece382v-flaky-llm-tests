# -*- coding: utf-8 -*-
import itertools
import pathlib
import types

import pytest

from kindle_maker import ebooklib
from kindle_maker.ebooklib import Ebook, EbookUtil, format_file_name


@pytest.fixture(autouse=True)
def isolated_tmp(tmp_path, monkeypatch):
    """Make every Ebook use a unique temp dir under pytest's tmp_path."""
    monkeypatch.setattr(ebooklib.tempfile, "gettempdir",
                        lambda: str(tmp_path))
    counter = itertools.count(1)
    fake_time = types.SimpleNamespace(time=lambda: next(counter))
    monkeypatch.setattr(ebooklib, "time", fake_time)
    return tmp_path


@pytest.fixture
def calls(monkeypatch):
    """Record subprocess.call invocations instead of running them."""
    recorded = []

    def fake_call(args, *a, **kw):
        recorded.append(list(args))
        return 0

    monkeypatch.setattr(ebooklib.subprocess, "call", fake_call)
    return recorded


def make_book(title="MyBook", author=None):
    book = Ebook(title, author)
    c1 = book.create_chapter("Chapter1").set_content("<p>one</p>")
    c1.create_subchapter("Sub1").set_content("<p>sub1</p>")
    c1.create_subchapter("Sub2").set_content("<p>sub2</p>")
    book.create_chapter("Chapter2").set_content("<p>two</p>")
    return book


def test_format_file_name_strips_illegal_chars():
    assert format_file_name('a b/c+d"e\\f:g|h') == 'abc-def-g-h'


def test_ebook_init_creates_tmpdir_and_defaults():
    book = Ebook("Title", "Me")
    assert book.title == "Title"
    assert book.author == "Me"
    assert book.tmpdir.is_dir()
    assert book.chapter_list == []
    assert book.cover_path is None and book.cover_content is None


def test_ebook_dest_file_paths():
    book = Ebook("My Book: 1")
    assert book.dest_file_path == pathlib.Path("MyBook-1.mobi")
    assert book.full_dest_file_path == book.tmpdir / "MyBook-1.mobi"


def test_set_cover_path_then_content_resets_other():
    book = Ebook("T")
    assert book.set_cover(cover_path="x.jpg") is book
    assert book.cover_path == "x.jpg" and book.cover_content is None
    book.set_cover(cover_content=b"abc")
    assert book.cover_content == b"abc" and book.cover_path is None


def test_set_cover_requires_argument():
    book = Ebook("T")
    with pytest.raises(ValueError):
        book.set_cover()


def test_add_chapter_rejects_sub_chapter():
    book = Ebook("T")
    top = book.create_chapter("top")
    sub = top.create_subchapter("sub")
    with pytest.raises(ValueError):
        book.add_chapter(sub)
    assert book.chapter_list == [top]


def test_create_chapter_with_missing_file_raises(tmp_path):
    book = Ebook("T")
    with pytest.raises(ValueError):
        book.create_chapter("c", str(tmp_path / "nope.html"))


def test_subchapter_max_level():
    book = Ebook("T")
    sub = book.create_chapter("top").create_subchapter("sub")
    with pytest.raises(Exception):
        sub.create_subchapter("too deep")


def test_headings_structure_and_play_order():
    book = make_book()
    hs = EbookUtil(book).headings
    assert [h["title"] for h in hs] == ["Chapter1", "Chapter2"]
    assert hs[0]["play_order"] == 2
    assert [s["play_order"] for s in hs[0]["sub_headings"]] == [3, 4]
    assert [s["title"] for s in hs[0]["sub_headings"]] == ["Sub1", "Sub2"]
    assert hs[0]["sub_headings"][0]["file_name"] == "Sub1.html"
    assert hs[1]["play_order"] == 5
    assert hs[1]["sub_headings"] == []
    assert hs[0]["file_name"] == "Chapter1.html"


def test_headings_are_cached():
    book = make_book()
    eu = EbookUtil(book)
    first = eu.headings
    book.create_chapter("Late").set_content("x")
    assert eu.headings is first
    assert len(eu.headings) == 2


def test_headings_empty_book():
    assert EbookUtil(Ebook("T")).headings == []


def test_render_toc_ncx_toc_html_and_opf_create_files():
    book = make_book("My Book")
    eu = EbookUtil(book)
    assert eu._render_toc_ncx() == "toc.ncx"
    assert eu._render_toc_html() == "toc.html"
    assert eu._render_opf() == "MyBook.opf"
    for name in ("toc.ncx", "toc.html", "MyBook.opf"):
        f = book.tmpdir / name
        assert f.is_file()
        assert f.read_text(encoding="utf-8").strip() != ""


def test_render_toc_html_mentions_chapter_titles():
    book = make_book()
    eu = EbookUtil(book)
    eu._render_toc_html()
    text = (book.tmpdir / "toc.html").read_text(encoding="utf-8")
    assert "Chapter1" in text and "Chapter2" in text


def test_save_cover_with_custom_path(tmp_path):
    cover = tmp_path / "mycover.jpg"
    cover.write_bytes(b"fakejpg")
    book = Ebook("T")
    book.set_cover(cover_path=str(cover))
    EbookUtil(book)._save_cover()
    assert (book.tmpdir / "mycover.jpg").read_bytes() == b"fakejpg"


def test_save_cover_default_and_content_not_implemented():
    book = Ebook("T")
    EbookUtil(book)._save_cover()
    assert (book.tmpdir / "cover.jpg").is_file()

    book2 = Ebook("T2")
    book2.set_cover(cover_content=b"bytes")
    with pytest.raises(NotImplementedError):
        EbookUtil(book2)._save_cover()


def test_save_chapters_writes_chapter_and_subchapter_files():
    book = make_book()
    EbookUtil(book)._save_chapters()
    assert (book.tmpdir / "Chapter1.html").read_text(encoding="utf-8") == "<p>one</p>"
    assert (book.tmpdir / "Sub1.html").read_text(encoding="utf-8") == "<p>sub1</p>"
    assert (book.tmpdir / "Sub2.html").is_file()
    assert (book.tmpdir / "Chapter2.html").is_file()


def test_create_invokes_kindlegen_with_opf(calls):
    book = make_book("My Book", author="Someone")
    EbookUtil(book).create()
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == ebooklib.kindlegen
    assert "-dont_append_source" in cmd
    assert cmd[-1] == str(book.tmpdir / "MyBook.opf")
    assert (book.tmpdir / "MyBook.opf").is_file()
    assert (book.tmpdir / "toc.ncx").is_file()
    assert (book.tmpdir / "toc.html").is_file()
    assert (book.tmpdir / "Chapter1.html").is_file()


def test_ebook_save_empty_raises_value_error(tmp_path, calls):
    book = Ebook("T")
    with pytest.raises(ValueError):
        book.save(str(tmp_path / "out.mobi"))
    assert calls == []


def test_ebook_save_copies_generated_mobi(tmp_path, monkeypatch):
    book = make_book("My Book")

    def fake_call(args, *a, **kw):
        (book.tmpdir / "MyBook.mobi").write_bytes(b"MOBI")
        return 0

    monkeypatch.setattr(ebooklib.subprocess, "call", fake_call)
    out = tmp_path / "out.mobi"
    book.save(str(out))
    assert out.read_bytes() == b"MOBI"


@pytest.mark.parametrize("system,opener", [("Linux", "xdg-open"),
                                           ("Darwin", "open")])
def test_ebook_show_uses_platform_opener(monkeypatch, calls, system, opener):
    book = make_book("My Book")
    monkeypatch.setattr(ebooklib.platform, "system", lambda: system)
    rc = book.show()
    assert rc == 0
    assert calls[-1] == [opener, str(book.full_dest_file_path)]


def test_ebook_del_removes_tmpdir():
    book = Ebook("T")
    book.create_chapter("c").set_content("x")
    EbookUtil(book)._save_chapters()
    tmp = book.tmpdir
    assert tmp.is_dir()
    book.__del__()
    assert not tmp.exists()
    # avoid a second removal attempt at garbage collection
    book._tempdir = pathlib.Path(tmp.parent / "recreated")
    book._tempdir.mkdir()
