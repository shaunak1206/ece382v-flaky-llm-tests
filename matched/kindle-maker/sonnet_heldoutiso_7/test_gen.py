# -*- coding: utf-8 -*-
import itertools
import pathlib
import tempfile
from types import SimpleNamespace

import pytest

from kindle_maker import ebooklib
from kindle_maker.ebooklib import Ebook, EbookUtil, format_file_name


@pytest.fixture
def make_ebook(tmp_path, monkeypatch):
    """Factory creating Ebooks whose temp dirs live under pytest's tmp_path."""
    root = tmp_path / "sys_tmp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(root))
    counter = itertools.count(1000)
    monkeypatch.setattr(
        ebooklib, "time", SimpleNamespace(time=lambda: next(counter))
    )

    def factory(title="Book", author=None):
        return Ebook(title, author)

    return factory


@pytest.fixture
def fake_subprocess(monkeypatch):
    calls = []

    def call(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return 0

    monkeypatch.setattr(ebooklib, "subprocess", SimpleNamespace(call=call))
    return calls


def _build_book(make_ebook):
    book = make_ebook("My Book", "Someone")
    c1 = book.create_chapter("Chapter 1").set_content("<p>one</p>")
    c1.create_subchapter("Sub 1.1").set_content("<p>1.1</p>")
    c1.create_subchapter("Sub 1.2").set_content("<p>1.2</p>")
    book.create_chapter("Chapter 2").set_content("<p>two</p>")
    return book


def test_format_file_name_strips_illegal_chars():
    assert format_file_name('a/b c+d"e\\f:g|h') == 'abc-def-g-h'


def test_ebook_dest_file_path_is_sanitized(make_ebook):
    book = make_ebook("My Book: v1/2")
    assert book.dest_file_path == pathlib.Path("MyBook-v12.mobi")


def test_ebook_tmpdir_created_and_full_dest_path(make_ebook):
    book = make_ebook("Book")
    assert book.tmpdir.is_dir()
    assert book.full_dest_file_path == book.tmpdir / "Book.mobi"


def test_set_cover_with_path_clears_content(make_ebook):
    book = make_ebook()
    book.set_cover(cover_content=b"abc")
    result = book.set_cover(cover_path="/some/cover.jpg")
    assert result is book
    assert book.cover_path == "/some/cover.jpg"
    assert book.cover_content is None


def test_set_cover_with_content_clears_path(make_ebook):
    book = make_ebook()
    book.set_cover(cover_path="/some/cover.jpg")
    book.set_cover(cover_content=b"abc")
    assert book.cover_content == b"abc"
    assert book.cover_path is None


def test_set_cover_without_arguments_raises(make_ebook):
    book = make_ebook()
    with pytest.raises(ValueError):
        book.set_cover()


def test_add_chapter_rejects_non_top_level(make_ebook):
    book = make_ebook()
    top = book.create_chapter("Top")
    sub = top.create_subchapter("Sub")
    with pytest.raises(ValueError):
        book.add_chapter(sub)
    assert book.chapter_list == [top]


def test_create_chapter_with_missing_file_raises(make_ebook, tmp_path):
    book = make_ebook()
    with pytest.raises(ValueError):
        book.create_chapter("Ch", str(tmp_path / "missing.html"))
    assert book.chapter_list == []


def test_save_with_no_chapters_raises(make_ebook, tmp_path):
    book = make_ebook()
    with pytest.raises(ValueError):
        book.save(str(tmp_path / "out.mobi"))


def test_headings_structure_and_play_order(make_ebook):
    book = _build_book(make_ebook)
    headings = EbookUtil(book).headings
    assert [h["title"] for h in headings] == ["Chapter 1", "Chapter 2"]
    assert [h["play_order"] for h in headings] == [2, 5]
    subs = headings[0]["sub_headings"]
    assert [s["title"] for s in subs] == ["Sub 1.1", "Sub 1.2"]
    assert [s["play_order"] for s in subs] == [3, 4]
    assert headings[0]["file_name"] == "Chapter1.html"
    assert subs[0]["file_name"] == "Sub1.1.html"
    assert headings[1]["sub_headings"] == []


def test_headings_are_cached(make_ebook):
    book = _build_book(make_ebook)
    util = EbookUtil(book)
    assert util.headings is util.headings


def test_render_toc_ncx_writes_file(make_ebook):
    book = _build_book(make_ebook)
    util = EbookUtil(book)
    name = util._render_toc_ncx()
    assert name == "toc.ncx"
    content = (book.tmpdir / name).read_text(encoding="utf-8")
    assert "My Book" in content
    assert "Someone" in content
    assert "Chapter 1" in content


def test_render_toc_ncx_default_author(make_ebook):
    book = make_ebook("Anon")
    book.create_chapter("Ch").set_content("x")
    util = EbookUtil(book)
    name = util._render_toc_ncx()
    content = (book.tmpdir / name).read_text(encoding="utf-8")
    assert "jachinlin.github.io" in content


def test_render_toc_html_and_opf_files(make_ebook):
    book = _build_book(make_ebook)
    util = EbookUtil(book)
    assert util._render_toc_html() == "toc.html"
    opf = util._render_opf()
    assert opf == "MyBook.opf"
    assert (book.tmpdir / "toc.html").is_file()
    assert "Chapter 2" in (book.tmpdir / "toc.html").read_text(encoding="utf-8")
    assert "My Book" in (book.tmpdir / opf).read_text(encoding="utf-8")


def test_save_cover_default_and_custom(make_ebook, tmp_path):
    book = make_ebook()
    EbookUtil(book)._save_cover()
    assert (book.tmpdir / "cover.jpg").is_file()

    book2 = make_ebook("Other")
    custom = tmp_path / "mycover.png"
    custom.write_bytes(b"png")
    book2.set_cover(cover_path=str(custom))
    EbookUtil(book2)._save_cover()
    assert (book2.tmpdir / "mycover.png").read_bytes() == b"png"


def test_save_cover_with_content_not_implemented(make_ebook):
    book = make_ebook()
    book.set_cover(cover_content=b"bytes")
    with pytest.raises(NotImplementedError):
        EbookUtil(book)._save_cover()


def test_save_chapters_writes_all_files(make_ebook):
    book = _build_book(make_ebook)
    EbookUtil(book)._save_chapters()
    for name in ("Chapter1.html", "Sub1.1.html", "Sub1.2.html", "Chapter2.html"):
        assert (book.tmpdir / name).is_file()
    assert (book.tmpdir / "Chapter2.html").read_text(encoding="utf-8") == "<p>two</p>"


def test_create_runs_kindlegen_on_opf(make_ebook, fake_subprocess):
    book = _build_book(make_ebook)
    book._eu.create()
    assert len(fake_subprocess) == 1
    cmd = fake_subprocess[0]
    assert cmd[0] == ebooklib.kindlegen
    assert "-dont_append_source" in cmd
    assert cmd[-1] == str(book.tmpdir / "MyBook.opf")
    assert (book.tmpdir / "MyBook.opf").is_file()
    assert (book.tmpdir / "toc.ncx").is_file()


def test_save_copies_generated_mobi(make_ebook, tmp_path, monkeypatch):
    book = make_ebook("Gen")
    book.create_chapter("Ch").set_content("x")

    def fake_create():
        (book.tmpdir / "Gen.mobi").write_bytes(b"MOBI")

    monkeypatch.setattr(book._eu, "create", fake_create)
    out = tmp_path / "out.mobi"
    book.save(str(out))
    assert out.read_bytes() == b"MOBI"


@pytest.mark.parametrize("system,expected", [("Linux", "xdg-open"),
                                             ("Darwin", "open")])
def test_show_uses_platform_viewer(make_ebook, monkeypatch, system, expected):
    book = make_ebook("Show")
    book.create_chapter("Ch").set_content("x")
    monkeypatch.setattr(book._eu, "create", lambda: None)
    calls = []

    def call(cmd):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(ebooklib, "subprocess", SimpleNamespace(call=call))
    monkeypatch.setattr(ebooklib, "platform",
                        SimpleNamespace(system=lambda: system))
    assert book.show() == 0
    assert calls == [[expected, str(book.full_dest_file_path)]]


def test_del_removes_tmpdir(make_ebook):
    book = make_ebook("Del")
    (book.tmpdir / "sub").mkdir()
    (book.tmpdir / "sub" / "f.txt").write_text("x")
    (book.tmpdir / "a.txt").write_text("y")
    book.__del__()
    assert not book.tmpdir.exists()
    # recreate so the real finalizer at GC time does not fail
    book.tmpdir.mkdir()
