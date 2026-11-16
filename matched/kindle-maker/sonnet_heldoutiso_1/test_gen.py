# -*- coding: utf-8 -*-
import itertools
import pathlib
from types import SimpleNamespace

import pytest

from kindle_maker import ebooklib
from kindle_maker.ebooklib import Ebook, EbookUtil, format_file_name


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Isolate temp dirs and external process calls."""
    counter = itertools.count(1)
    monkeypatch.setattr(
        ebooklib, "time",
        SimpleNamespace(time=lambda: "book-{}".format(next(counter))))
    monkeypatch.setattr(
        ebooklib, "tempfile",
        SimpleNamespace(gettempdir=lambda: str(tmp_path)))

    calls = []

    def fake_call(args, *a, **kw):
        calls.append(list(args))
        first = str(args[0])
        if first not in ("xdg-open", "open"):
            opf = pathlib.Path(args[-1])
            (opf.parent / (opf.stem + ".mobi")).write_bytes(b"MOBI")
        return 0

    monkeypatch.setattr(ebooklib, "subprocess",
                        SimpleNamespace(call=fake_call))
    return SimpleNamespace(calls=calls, tmp_path=tmp_path)


def make_book(title="Book", author=None):
    book = Ebook(title, author)
    a = book.create_chapter("Chap A")
    a.set_content("<html>A</html>")
    a1 = a.create_subchapter("Sub A1")
    a1.set_content("<html>A1</html>")
    a2 = a.create_subchapter("Sub A2")
    a2.set_content("<html>A2</html>")
    b = book.create_chapter("Chap B")
    b.set_content("<html>B</html>")
    return book


def test_format_file_name_strips_illegal_chars():
    assert format_file_name('a b/c+d"e\\f:g|h') == 'abc-def-g-h'


def test_ebook_init_creates_tmpdir(sandbox):
    book = Ebook("Title", "Me")
    assert book.title == "Title"
    assert book.author == "Me"
    assert book.chapter_list == []
    assert book.cover_path is None and book.cover_content is None
    assert book.tmpdir.is_dir()
    assert isinstance(book._eu, EbookUtil)


def test_dest_file_paths(sandbox):
    book = Ebook("My Book: v+1")
    assert book.dest_file_path == pathlib.Path("MyBook-v-1.mobi")
    assert book.full_dest_file_path == book.tmpdir / "MyBook-v-1.mobi"


def test_set_cover_path_and_content(sandbox):
    book = Ebook("T")
    assert book.set_cover(cover_path="x.jpg") is book
    assert book.cover_path == "x.jpg" and book.cover_content is None
    book.set_cover(cover_content=b"abc")
    assert book.cover_content == b"abc" and book.cover_path is None


def test_set_cover_requires_argument(sandbox):
    book = Ebook("T")
    with pytest.raises(ValueError):
        book.set_cover()


def test_create_chapter_and_add_chapter(sandbox):
    book = Ebook("T")
    c = book.create_chapter("One")
    assert book.chapter_list == [c]
    assert c.is_top_chapter
    sub = c.create_subchapter("Sub")
    with pytest.raises(ValueError):
        book.add_chapter(sub)
    assert book.chapter_list == [c]


def test_create_empty_book_raises(sandbox):
    book = Ebook("T")
    with pytest.raises(ValueError):
        book.save(str(sandbox.tmp_path / "out.mobi"))
    assert sandbox.calls == []


def test_headings_structure_and_play_order(sandbox):
    book = make_book()
    hs = book._eu.headings
    assert [h["title"] for h in hs] == ["Chap A", "Chap B"]
    assert [h["play_order"] for h in hs] == [2, 5]
    assert hs[0]["file_name"] == "ChapA.html"
    subs = hs[0]["sub_headings"]
    assert [s["title"] for s in subs] == ["Sub A1", "Sub A2"]
    assert [s["play_order"] for s in subs] == [3, 4]
    assert subs[0]["file_name"] == "SubA1.html"
    assert hs[1]["sub_headings"] == []


def test_headings_are_cached(sandbox):
    book = make_book()
    first = book._eu.headings
    book.create_chapter("Late")
    assert book._eu.headings is first
    assert len(first) == 2


def test_headings_empty_book(sandbox):
    assert Ebook("T")._eu.headings == []


def test_render_toc_ncx_default_author(sandbox):
    book = make_book(title="NcxTitle")
    name = book._eu._render_toc_ncx()
    assert name == "toc.ncx"
    text = (book.tmpdir / name).read_text(encoding="utf-8")
    assert "NcxTitle" in text
    assert "jachinlin.github.io" in text


def test_render_toc_html_and_opf(sandbox):
    book = make_book(title="My Book", author="Alice")
    assert book._eu._render_toc_html() == "toc.html"
    assert (book.tmpdir / "toc.html").is_file()
    opf = book._eu._render_opf()
    assert opf == "MyBook.opf"
    text = (book.tmpdir / opf).read_text(encoding="utf-8")
    assert "My Book" in text
    assert "Alice" in text


def test_save_cover_default_and_custom(sandbox, tmp_path):
    book = Ebook("T")
    book._eu._save_cover()
    assert (book.tmpdir / "cover.jpg").is_file()

    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    custom = custom_dir / "mycover.png"
    custom.write_bytes(b"png")
    book2 = Ebook("T2")
    book2.set_cover(cover_path=str(custom))
    book2._eu._save_cover()
    assert (book2.tmpdir / "mycover.png").read_bytes() == b"png"


def test_save_cover_with_content_not_implemented(sandbox):
    book = Ebook("T")
    book.set_cover(cover_content=b"data")
    with pytest.raises(NotImplementedError):
        book._eu._save_cover()


def test_create_writes_files_and_calls_kindlegen(sandbox):
    book = make_book(title="Gen")
    book._eu.create()
    for name in ("toc.ncx", "toc.html", "Gen.opf", "cover.jpg",
                 "ChapA.html", "SubA1.html", "SubA2.html", "ChapB.html"):
        assert (book.tmpdir / name).is_file(), name
    assert (book.tmpdir / "SubA1.html").read_text(
        encoding="utf-8") == "<html>A1</html>"
    assert len(sandbox.calls) == 1
    args = sandbox.calls[0]
    assert args[0] == ebooklib.kindlegen
    assert "-dont_append_source" in args
    assert args[-1] == str(book.tmpdir / "Gen.opf")


def test_save_copies_mobi_to_destination(sandbox):
    book = make_book(title="Saved")
    dest = sandbox.tmp_path / "out.mobi"
    book.save(str(dest))
    assert dest.read_bytes() == b"MOBI"


def test_show_uses_xdg_open_on_linux(sandbox, monkeypatch):
    monkeypatch.setattr(ebooklib, "platform",
                        SimpleNamespace(system=lambda: "Linux"))
    book = make_book(title="Shown")
    rc = book.show()
    assert rc == 0
    assert sandbox.calls[-1] == ["xdg-open", str(book.full_dest_file_path)]


def test_chapter_nonexistent_file_raises(sandbox, tmp_path):
    book = Ebook("T")
    with pytest.raises(ValueError):
        book.create_chapter("X", str(tmp_path / "missing.html"))


def test_chapter_max_level_and_content_from_file(sandbox, tmp_path):
    src = tmp_path / "src.html"
    src.write_text("hello é", encoding="utf-8")
    book = Ebook("T")
    c = book.create_chapter("X", str(src))
    assert c.content == "hello é"
    sub = c.create_subchapter("Y")
    assert not sub.is_top_chapter
    with pytest.raises(Exception):
        sub.create_subchapter("Z")
