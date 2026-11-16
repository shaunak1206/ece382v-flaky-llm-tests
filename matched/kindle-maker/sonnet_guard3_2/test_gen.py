# -*- coding: utf-8 -*-
import itertools
import pathlib
from types import SimpleNamespace

import pytest

from kindle_maker import ebooklib
from kindle_maker.ebooklib import Ebook, EbookUtil, format_file_name


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Make Ebook temp dirs unique and located under tmp_path; record subprocess calls."""
    counter = itertools.count(1000)
    root = tmp_path / "tmproot"
    root.mkdir()
    monkeypatch.setattr(
        ebooklib, "time", SimpleNamespace(time=lambda: float(next(counter))))
    monkeypatch.setattr(
        ebooklib, "tempfile", SimpleNamespace(gettempdir=lambda: str(root)))

    calls = []

    def fake_call(cmd):
        calls.append(list(cmd))
        if cmd and str(cmd[-1]).endswith(".opf"):
            pathlib.Path(cmd[-1]).with_suffix(".mobi").write_bytes(b"MOBI")
        return 0

    monkeypatch.setattr(ebooklib, "subprocess", SimpleNamespace(call=fake_call))
    src = tmp_path / "src"
    src.mkdir()
    return SimpleNamespace(calls=calls, src=src, out=tmp_path / "out",
                           monkeypatch=monkeypatch)


def _make_cover(env):
    cover = env.src / "mycover.jpg"
    cover.write_bytes(b"jpgdata")
    return cover


def test_format_file_name_strips_illegal_characters():
    assert format_file_name('a b/c+d"e\\f:g|h') == 'abc-def-g-h'


def test_ebook_dest_file_path_and_tmpdir(env):
    book = Ebook("My Book: part+1")
    assert book.dest_file_path == pathlib.Path("MyBook-part-1.mobi")
    assert book.tmpdir.is_dir()
    assert book.full_dest_file_path == book.tmpdir / "MyBook-part-1.mobi"


def test_two_ebooks_have_distinct_tmpdirs(env):
    a = Ebook("A")
    b = Ebook("B")
    assert a.tmpdir != b.tmpdir


def test_set_cover_path_and_content_switch(env):
    book = Ebook("T")
    assert book.set_cover(cover_path="/x/y.jpg") is book
    assert book.cover_path == "/x/y.jpg"
    assert book.cover_content is None
    book.set_cover(cover_content=b"abc")
    assert book.cover_content == b"abc"
    assert book.cover_path is None


def test_set_cover_requires_argument(env):
    book = Ebook("T")
    with pytest.raises(ValueError):
        book.set_cover()


def test_create_chapter_adds_to_list_and_add_chapter_rejects_sub(env):
    book = Ebook("T")
    c = book.create_chapter("Ch1")
    assert book.chapter_list == [c]
    sub = c.create_subchapter("Sub")
    with pytest.raises(ValueError):
        book.add_chapter(sub)
    assert book.chapter_list == [c]


def test_create_chapter_with_missing_file_raises(env):
    book = Ebook("T")
    with pytest.raises(ValueError):
        book.create_chapter("Ch", str(env.src / "missing.html"))
    assert book.chapter_list == []


def test_subchapter_level_limit(env):
    book = Ebook("T")
    c = book.create_chapter("Ch1")
    sub = c.create_subchapter("Sub")
    with pytest.raises(Exception):
        sub.create_subchapter("Too deep")
    assert c.sub_chapters == [sub]


def test_chapter_content_from_file_and_set_content(env):
    f = env.src / "c.html"
    f.write_text(u"<p>héllo</p>", encoding="utf-8")
    book = Ebook("T")
    c = book.create_chapter("Ch", str(f))
    assert c.content == u"<p>héllo</p>"
    c.set_content("new")
    assert c.content == "new"


def test_headings_play_order_and_structure(env):
    book = Ebook("T")
    c1 = book.create_chapter("One")
    c1.create_subchapter("One A")
    c1.create_subchapter("One B")
    book.create_chapter("Two")
    hs = EbookUtil(book).headings
    assert [h["title"] for h in hs] == ["One", "Two"]
    assert [h["play_order"] for h in hs] == [2, 5]
    assert hs[0]["file_name"] == "One.html"
    subs = hs[0]["sub_headings"]
    assert [s["title"] for s in subs] == ["One A", "One B"]
    assert [s["play_order"] for s in subs] == [3, 4]
    assert subs[0]["file_name"] == "OneA.html"
    assert hs[1]["sub_headings"] == []


def test_headings_are_cached(env):
    book = Ebook("T")
    book.create_chapter("One")
    eu = EbookUtil(book)
    first = eu.headings
    book.create_chapter("Two")
    assert eu.headings is first
    assert len(eu.headings) == 1


def test_render_toc_ncx_default_author_and_context(env):
    book = Ebook("T")
    book.create_chapter("One")
    eu = EbookUtil(book)
    rendered = []
    env.monkeypatch.setattr(
        eu, "_render_file",
        lambda template, ctx, to: rendered.append((template, ctx, to)))
    assert eu._render_toc_ncx() == "toc.ncx"
    template, ctx, to = rendered[0]
    assert template == "toc.xml"
    assert to == "toc.ncx"
    assert ctx["title"] == "T"
    assert ctx["author"] == "jachinlin.github.io"
    assert ctx["headings"] == eu.headings


def test_render_opf_uses_formatted_title_and_author(env):
    book = Ebook("My Book", author="Alice")
    book.create_chapter("One")
    eu = EbookUtil(book)
    rendered = []
    env.monkeypatch.setattr(
        eu, "_render_file",
        lambda template, ctx, to: rendered.append((template, ctx, to)))
    assert eu._render_opf() == "MyBook.opf"
    template, ctx, to = rendered[0]
    assert template == "opf.xml"
    assert to == "MyBook.opf"
    assert ctx["author"] == "Alice"


def test_render_toc_html_writes_file(env):
    book = Ebook("T")
    book.create_chapter("One")
    eu = EbookUtil(book)
    assert eu._render_toc_html() == "toc.html"
    assert (book.tmpdir / "toc.html").is_file()


def test_save_cover_copies_given_cover_and_rejects_content(env):
    book = Ebook("T")
    cover = _make_cover(env)
    book.set_cover(cover_path=str(cover))
    eu = EbookUtil(book)
    eu._save_cover()
    assert (book.tmpdir / "mycover.jpg").read_bytes() == b"jpgdata"

    book2 = Ebook("T2")
    book2.set_cover(cover_content=b"raw")
    with pytest.raises(NotImplementedError):
        EbookUtil(book2)._save_cover()


def test_save_chapters_writes_chapter_and_subchapter_files(env):
    book = Ebook("T")
    c = book.create_chapter("One").set_content("<p>one</p>")
    c.create_subchapter("Sub").set_content("<p>sub</p>")
    EbookUtil(book)._save_chapters()
    assert (book.tmpdir / "One.html").read_text(encoding="utf-8") == "<p>one</p>"
    assert (book.tmpdir / "Sub.html").read_text(encoding="utf-8") == "<p>sub</p>"


def test_save_with_no_chapters_raises(env):
    book = Ebook("T")
    with pytest.raises(ValueError):
        book.save(str(env.out))
    assert env.calls == []


def test_create_invokes_kindlegen_with_opf(env):
    book = Ebook("My Book")
    book.set_cover(cover_path=str(_make_cover(env)))
    book.create_chapter("One").set_content("<p>x</p>")
    EbookUtil(book).create()
    assert len(env.calls) == 1
    cmd = env.calls[0]
    assert cmd[0] == ebooklib.kindlegen
    assert "-dont_append_source" in cmd
    assert cmd[-1] == str(book.tmpdir / "MyBook.opf")
    assert (book.tmpdir / "MyBook.opf").is_file()
    assert (book.tmpdir / "toc.ncx").is_file()
    assert (book.tmpdir / "toc.html").is_file()
    assert (book.tmpdir / "One.html").is_file()


def test_ebook_save_copies_mobi_to_destination(env):
    book = Ebook("My Book")
    book.set_cover(cover_path=str(_make_cover(env)))
    book.create_chapter("One").set_content("<p>x</p>")
    dest = env.src / "result.mobi"
    book.save(str(dest))
    assert dest.read_bytes() == b"MOBI"


@pytest.mark.parametrize("system,opener", [("Linux", "xdg-open"),
                                            ("Darwin", "open")])
def test_show_uses_platform_opener(env, system, opener):
    env.monkeypatch.setattr(ebooklib, "platform",
                            SimpleNamespace(system=lambda: system))
    book = Ebook("My Book")
    book.set_cover(cover_path=str(_make_cover(env)))
    book.create_chapter("One").set_content("<p>x</p>")
    rc = book.show()
    assert rc == 0
    assert env.calls[-1] == [opener, str(book.tmpdir / "MyBook.mobi")]
