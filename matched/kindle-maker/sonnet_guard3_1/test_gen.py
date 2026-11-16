# -*- coding: utf-8 -*-
import itertools
import pathlib
from types import SimpleNamespace

import pytest

from kindle_maker import ebooklib
from kindle_maker.ebooklib import Ebook, EbookUtil, format_file_name


@pytest.fixture
def make_ebook(monkeypatch, tmp_path):
    """Build Ebooks with deterministic, unique temp dirs under tmp_path."""
    counter = itertools.count(1000)
    monkeypatch.setattr(
        ebooklib, "time",
        SimpleNamespace(time=lambda: float(next(counter))))
    monkeypatch.setattr(
        ebooklib, "tempfile",
        SimpleNamespace(gettempdir=lambda: str(tmp_path)))
    created = []

    def factory(title="My Book", author=None):
        book = Ebook(title, author)
        created.append(book)
        return book

    yield factory

    # make __del__ harmless regardless of cleanup order
    for i, book in enumerate(created):
        empty = tmp_path / "_empty_{}".format(i)
        empty.mkdir(exist_ok=True)
        book._tempdir = empty


def _fake_subprocess(calls, rc=0, on_call=None):
    def call(args):
        calls.append(list(args))
        if on_call:
            on_call(args)
        return rc
    return SimpleNamespace(call=call)


def test_format_file_name_strips_illegal_characters():
    assert format_file_name('a b/c+d"e\\f:g|h') == 'abc-def-g-h'


def test_ebook_init_creates_tmpdir_and_paths(make_ebook, tmp_path):
    book = make_ebook("Hello World")
    assert book.tmpdir.is_dir()
    assert book.tmpdir.parent == tmp_path
    assert book.dest_file_path == pathlib.Path("HelloWorld.mobi")
    assert book.full_dest_file_path == book.tmpdir / "HelloWorld.mobi"
    assert book.chapter_list == []


def test_set_cover_path_and_content_exclusive(make_ebook):
    book = make_ebook()
    assert book.set_cover(cover_content=b"abc") is book
    assert book.cover_content == b"abc"
    assert book.cover_path is None
    book.set_cover(cover_path="/some/path.jpg")
    assert book.cover_path == "/some/path.jpg"
    assert book.cover_content is None


def test_set_cover_requires_argument(make_ebook):
    book = make_ebook()
    with pytest.raises(ValueError):
        book.set_cover()


def test_create_chapter_adds_to_list(make_ebook):
    book = make_ebook()
    c1 = book.create_chapter("One")
    c2 = book.create_chapter("Two")
    assert book.chapter_list == [c1, c2]
    assert c1.is_top_chapter


def test_create_chapter_with_missing_file_raises(make_ebook, tmp_path):
    book = make_ebook()
    with pytest.raises(ValueError):
        book.create_chapter("One", str(tmp_path / "missing.html"))
    assert book.chapter_list == []


def test_add_chapter_rejects_non_top_level(make_ebook):
    book = make_ebook()
    top = book.create_chapter("One")
    sub = top.create_subchapter("Sub")
    assert not sub.is_top_chapter
    with pytest.raises(ValueError):
        book.add_chapter(sub)
    assert book.chapter_list == [top]


def test_subchapter_max_level(make_ebook):
    book = make_ebook()
    top = book.create_chapter("One")
    sub = top.create_subchapter("Sub")
    assert top.sub_chapters == [sub]
    with pytest.raises(Exception):
        sub.create_subchapter("Too deep")


def test_headings_structure_and_play_order(make_ebook):
    book = make_ebook()
    c1 = book.create_chapter("Ch 1")
    c1.create_subchapter("Sub A")
    c1.create_subchapter("Sub B")
    book.create_chapter("Ch 2")
    hs = EbookUtil(book).headings
    assert [h["title"] for h in hs] == ["Ch 1", "Ch 2"]
    assert [h["play_order"] for h in hs] == [2, 5]
    assert hs[0]["file_name"] == "Ch1.html"
    subs = hs[0]["sub_headings"]
    assert [s["title"] for s in subs] == ["Sub A", "Sub B"]
    assert [s["play_order"] for s in subs] == [3, 4]
    assert subs[0]["file_name"] == "SubA.html"
    assert hs[1]["sub_headings"] == []


def test_headings_are_cached(make_ebook):
    book = make_ebook()
    book.create_chapter("One")
    util = EbookUtil(book)
    first = util.headings
    book.create_chapter("Two")
    assert util.headings is first
    assert len(util.headings) == 1


def test_save_with_no_chapters_raises(make_ebook, tmp_path):
    book = make_ebook()
    with pytest.raises(ValueError):
        book.save(str(tmp_path / "out.mobi"))


def test_render_files_written_to_tmpdir(make_ebook):
    book = make_ebook("A+B Title", author="Someone")
    book.create_chapter("One").set_content("<p>hi</p>")
    util = EbookUtil(book)
    assert util._render_toc_ncx() == "toc.ncx"
    assert util._render_toc_html() == "toc.html"
    assert util._render_opf() == "A-BTitle.opf"
    for name in ("toc.ncx", "toc.html", "A-BTitle.opf"):
        assert (book.tmpdir / name).is_file()


def test_save_cover_from_cover_path(make_ebook, tmp_path):
    cover = tmp_path / "mycover.jpg"
    cover.write_bytes(b"JPEGDATA")
    book = make_ebook()
    book.set_cover(cover_path=str(cover))
    EbookUtil(book)._save_cover()
    assert (book.tmpdir / "mycover.jpg").read_bytes() == b"JPEGDATA"


def test_save_cover_default_template(make_ebook):
    book = make_ebook()
    EbookUtil(book)._save_cover()
    assert (book.tmpdir / "cover.jpg").is_file()


def test_save_cover_with_content_not_implemented(make_ebook):
    book = make_ebook()
    book.set_cover(cover_content=b"raw")
    with pytest.raises(NotImplementedError):
        EbookUtil(book)._save_cover()


def test_chapter_save_writes_content_and_static_files(make_ebook, tmp_path):
    static = tmp_path / "style.css"
    static.write_text("body{}", encoding="utf-8")
    src = tmp_path / "src.html"
    src.write_text(u"<p>héllo</p>", encoding="utf-8")
    book = make_ebook()
    c = book.create_chapter("From File", str(src))
    c.add_static_files(str(static))
    assert c.content == u"<p>héllo</p>"
    c.save()
    out = book.tmpdir / "FromFile.html"
    assert out.read_text(encoding="utf-8") == u"<p>héllo</p>"
    assert (book.tmpdir / "style.css").read_text() == "body{}"


def test_create_runs_kindlegen_and_saves_chapters(make_ebook, monkeypatch):
    book = make_ebook("Book X")
    c = book.create_chapter("One").set_content("<p>1</p>")
    c.create_subchapter("Sub").set_content("<p>2</p>")
    calls = []
    monkeypatch.setattr(ebooklib, "subprocess", _fake_subprocess(calls))
    book._create()
    assert len(calls) == 1
    assert calls[0][0] == ebooklib.kindlegen
    assert "-dont_append_source" in calls[0]
    assert calls[0][-1] == str(book.tmpdir / "BookX.opf")
    for name in ("toc.ncx", "toc.html", "BookX.opf", "cover.jpg",
                 "One.html", "Sub.html"):
        assert (book.tmpdir / name).is_file()
    assert (book.tmpdir / "Sub.html").read_text() == "<p>2</p>"


def test_save_copies_generated_mobi(make_ebook, monkeypatch, tmp_path):
    book = make_ebook("Book Y")
    book.create_chapter("One").set_content("<p>1</p>")

    def produce(args):
        book.full_dest_file_path.write_bytes(b"MOBI")

    calls = []
    monkeypatch.setattr(ebooklib, "subprocess",
                        _fake_subprocess(calls, on_call=produce))
    out = tmp_path / "result.mobi"
    book.save(str(out))
    assert out.read_bytes() == b"MOBI"


def test_show_on_linux_uses_xdg_open(make_ebook, monkeypatch):
    book = make_ebook("Book Z")
    book.create_chapter("One").set_content("x")
    calls = []
    monkeypatch.setattr(ebooklib, "subprocess", _fake_subprocess(calls))
    monkeypatch.setattr(ebooklib, "platform",
                        SimpleNamespace(system=lambda: "Linux"))
    assert book.show() == 0
    assert calls[-1] == ["xdg-open", str(book.full_dest_file_path)]


def test_show_on_darwin_uses_open(make_ebook, monkeypatch):
    book = make_ebook("Book Z")
    book.create_chapter("One").set_content("x")
    calls = []
    monkeypatch.setattr(ebooklib, "subprocess", _fake_subprocess(calls))
    monkeypatch.setattr(ebooklib, "platform",
                        SimpleNamespace(system=lambda: "Darwin"))
    assert book.show() == 0
    assert calls[-1] == ["open", str(book.full_dest_file_path)]
