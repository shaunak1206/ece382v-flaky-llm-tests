# -*- coding: utf-8 -*-
import itertools
import pathlib
from types import SimpleNamespace

import pytest

from kindle_maker import ebooklib
from kindle_maker.ebooklib import Ebook, EbookUtil, format_file_name


@pytest.fixture(autouse=True)
def unique_time(monkeypatch):
    """Make sure every Ebook gets a unique temp dir name."""
    counter = itertools.count(1)
    base = "kmtest_{}_".format(id(object()))
    fake_time = SimpleNamespace(time=lambda: base + str(next(counter)))
    monkeypatch.setattr(ebooklib, "time", fake_time)


@pytest.fixture
def calls(monkeypatch):
    recorded = []

    def fake_call(args, *a, **kw):
        recorded.append(list(args))
        return 0

    monkeypatch.setattr(ebooklib, "subprocess", SimpleNamespace(call=fake_call))
    return recorded


@pytest.fixture
def ebook():
    return Ebook("My Book", author="Tester")


def test_format_file_name_strips_illegal_characters():
    assert format_file_name('a b/c+d"e\\f:g|h') == "abc-def-g-h"


def test_ebook_dest_paths(ebook):
    assert ebook.dest_file_path == pathlib.Path("MyBook.mobi")
    assert ebook.full_dest_file_path == ebook.tmpdir / "MyBook.mobi"
    assert ebook.tmpdir.is_dir()


def test_set_cover_with_path_and_content(ebook):
    assert ebook.set_cover(cover_path="a.jpg") is ebook
    assert ebook.cover_path == "a.jpg"
    assert ebook.cover_content is None
    ebook.set_cover(cover_content=b"bytes")
    assert ebook.cover_content == b"bytes"
    assert ebook.cover_path is None


def test_set_cover_requires_argument(ebook):
    with pytest.raises(ValueError):
        ebook.set_cover()


def test_create_chapter_adds_to_list(ebook):
    c = ebook.create_chapter("Chap 1")
    assert ebook.chapter_list == [c]
    assert c.is_top_chapter
    assert c.dest_file_path == pathlib.Path("Chap1.html")


def test_add_chapter_rejects_non_top_level(ebook):
    c = ebook.create_chapter("Top")
    sub = c.create_subchapter("Sub")
    with pytest.raises(ValueError):
        ebook.add_chapter(sub)
    assert ebook.chapter_list == [c]


def test_subchapter_max_level_enforced(ebook):
    c = ebook.create_chapter("Top")
    sub = c.create_subchapter("Sub")
    assert not sub.is_top_chapter
    with pytest.raises(Exception):
        sub.create_subchapter("Too deep")


def test_chapter_invalid_file_path_raises(ebook, tmp_path):
    with pytest.raises(ValueError):
        ebook.create_chapter("Bad", str(tmp_path / "missing.html"))


def test_chapter_content_from_file_and_set_content(ebook, tmp_path):
    f = tmp_path / "c.html"
    f.write_text(u"<p>héllo</p>", encoding="utf-8")
    c = ebook.create_chapter("C", str(f))
    assert c.content == u"<p>héllo</p>"
    c.set_content("<p>new</p>")
    assert c.content == "<p>new</p>"


def test_chapter_save_writes_content_and_static_files(ebook, tmp_path):
    static = tmp_path / "img.png"
    static.write_bytes(b"png")
    c = ebook.create_chapter("Chap")
    c.set_content("<p>x</p>").add_static_files(str(static))
    c.save()
    assert c.full_dest_file_path.read_text(encoding="utf-8") == "<p>x</p>"
    assert (ebook.tmpdir / "img.png").read_bytes() == b"png"


def test_add_static_files_requires_existing_file(ebook, tmp_path):
    c = ebook.create_chapter("Chap")
    with pytest.raises(AssertionError):
        c.add_static_files(str(tmp_path / "nope.png"))


def test_headings_structure_and_play_order(ebook):
    c1 = ebook.create_chapter("One")
    c1.create_subchapter("One A")
    c1.create_subchapter("One B")
    ebook.create_chapter("Two")
    hs = EbookUtil(ebook).headings
    assert [h["title"] for h in hs] == ["One", "Two"]
    assert [h["play_order"] for h in hs] == [2, 5]
    assert hs[0]["file_name"] == "One.html"
    subs = hs[0]["sub_headings"]
    assert [s["title"] for s in subs] == ["One A", "One B"]
    assert [s["play_order"] for s in subs] == [3, 4]
    assert subs[0]["file_name"] == "OneA.html"
    assert hs[1]["sub_headings"] == []


def test_headings_are_cached(ebook):
    ebook.create_chapter("One")
    eu = EbookUtil(ebook)
    first = eu.headings
    ebook.create_chapter("Two")
    assert eu.headings is first
    assert len(eu.headings) == 1


def test_save_cover_default_copies_template_cover(ebook):
    EbookUtil(ebook)._save_cover()
    assert (ebook.tmpdir / "cover.jpg").is_file()


def test_save_cover_with_cover_path(ebook, tmp_path):
    cover = tmp_path / "mycover.jpg"
    cover.write_bytes(b"data")
    ebook.set_cover(cover_path=str(cover))
    EbookUtil(ebook)._save_cover()
    assert (ebook.tmpdir / "mycover.jpg").read_bytes() == b"data"


def test_save_cover_with_content_not_implemented(ebook):
    ebook.set_cover(cover_content=b"abc")
    with pytest.raises(NotImplementedError):
        EbookUtil(ebook)._save_cover()


def test_ebookutil_create_generates_files_and_calls_kindlegen(ebook, calls):
    c = ebook.create_chapter("Chap One").set_content("<p>1</p>")
    c.create_subchapter("Sub").set_content("<p>2</p>")
    EbookUtil(ebook).create()
    for name in ("toc.ncx", "toc.html", "MyBook.opf", "cover.jpg",
                 "ChapOne.html", "Sub.html"):
        assert (ebook.tmpdir / name).is_file(), name
    assert len(calls) == 1
    assert "-dont_append_source" in calls[0]
    assert calls[0][-1] == str(ebook.tmpdir / "MyBook.opf")


def test_ebook_create_with_empty_chapter_list_raises(ebook):
    with pytest.raises(ValueError):
        ebook._create()
    with pytest.raises(ValueError):
        ebook.save("out.mobi")


def test_ebook_save_copies_result(ebook, tmp_path, monkeypatch):
    ebook.create_chapter("C").set_content("x")

    def fake_create():
        ebook.full_dest_file_path.write_bytes(b"MOBI")

    monkeypatch.setattr(ebook, "_create", fake_create)
    out = tmp_path / "out.mobi"
    ebook.save(str(out))
    assert out.read_bytes() == b"MOBI"


def test_ebook_show_on_linux_uses_xdg_open(ebook, monkeypatch):
    ebook.create_chapter("C").set_content("x")
    recorded = []
    monkeypatch.setattr(ebook, "_create", lambda: None)
    monkeypatch.setattr(ebooklib, "platform",
                        SimpleNamespace(system=lambda: "Linux"))
    monkeypatch.setattr(
        ebooklib, "subprocess",
        SimpleNamespace(call=lambda args: recorded.append(args) or 0))
    assert ebook.show() == 0
    assert recorded == [["xdg-open", str(ebook.full_dest_file_path)]]


def test_del_removes_tmpdir(tmp_path):
    e = Ebook("Temp")
    d = e.tmpdir
    (d / "sub").mkdir()
    (d / "sub" / "f.txt").write_text("x")
    e.__del__()
    assert not d.exists()
    d.mkdir()  # so the real finalizer has something to remove
