# -*- coding: utf-8 -*-

import gc
import pathlib

import pytest

from kindle_maker import ebooklib
from kindle_maker.ebooklib import Chapter, Ebook, EbookUtil, format_file_name
from kindle_maker.kindlegen import kindlegen as KINDLEGEN


@pytest.fixture
def ebook():
    eb = Ebook("My Book", author="Tester")
    yield eb
    # Temp directory cleanup is handled by Ebook.__del__.


@pytest.fixture
def fake_kindlegen(monkeypatch):
    """
    Replace subprocess.call used by ebooklib so no real kindlegen runs.
    The fake writes a .mobi file next to the opf it was asked to process.
    """
    calls = []
    state = {"rc": 0}

    def fake_call(args, *a, **kw):
        calls.append(list(args))
        opf = pathlib.Path(args[-1])
        mobi = opf.parent / (opf.stem + ".mobi")
        mobi.write_bytes(b"MOBI")
        return state["rc"]

    monkeypatch.setattr(ebooklib.subprocess, "call", fake_call)
    fake_call.state = state
    fake_call.calls = calls
    return fake_call


def _html(tmp_path, name, text="<p>body</p>"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_format_file_name_strips_illegal_characters():
    assert format_file_name('a/b c+d"e\\f:g|h') == "abc-def-g-h"


def test_ebook_init_sets_fields_and_creates_tmpdir(ebook):
    assert ebook.title == "My Book"
    assert ebook.author == "Tester"
    assert ebook.cover_path is None
    assert ebook.cover_content is None
    assert ebook.chapter_list == []
    assert ebook.tmpdir.is_dir()


def test_ebook_dest_file_path_is_sanitized_mobi(ebook):
    assert ebook.dest_file_path == pathlib.Path("MyBook.mobi")
    assert ebook.full_dest_file_path == ebook.tmpdir / "MyBook.mobi"


def test_set_cover_with_path_returns_self_and_clears_content(ebook):
    ebook.cover_content = b"old"
    result = ebook.set_cover(cover_path="/some/cover.jpg")
    assert result is ebook
    assert ebook.cover_path == "/some/cover.jpg"
    assert ebook.cover_content is None


def test_set_cover_with_content_returns_self_and_clears_path(ebook):
    ebook.cover_path = "/old.jpg"
    result = ebook.set_cover(cover_content=b"\xff\xd8")
    assert result is ebook
    assert ebook.cover_content == b"\xff\xd8"
    assert ebook.cover_path is None


def test_set_cover_without_arguments_raises(ebook):
    with pytest.raises(ValueError):
        ebook.set_cover()


def test_add_chapter_rejects_non_top_level_chapter(ebook):
    sub = Chapter("Sub", ebook, level=2)
    with pytest.raises(ValueError):
        ebook.add_chapter(sub)
    assert ebook.chapter_list == []


def test_create_chapter_appends_and_returns_chapter(ebook, tmp_path):
    path = _html(tmp_path, "intro.html")
    ch = ebook.create_chapter("Intro", path)
    assert isinstance(ch, Chapter)
    assert ch.title == "Intro"
    assert ch.is_top_chapter
    assert ebook.chapter_list == [ch]


def test_chapter_with_missing_file_raises(ebook, tmp_path):
    with pytest.raises(ValueError):
        Chapter("Missing", ebook, file_path=str(tmp_path / "nope.html"))


def test_chapter_reads_content_from_file(ebook, tmp_path):
    path = _html(tmp_path, "ch.html", "hello file")
    ch = ebook.create_chapter("Ch", path)
    assert ch.content == "hello file"


def test_set_content_overrides_file_path(ebook, tmp_path):
    path = _html(tmp_path, "ch.html", "from file")
    ch = ebook.create_chapter("Ch", path)
    result = ch.set_content("from memory")
    assert result is ch
    assert ch.content == "from memory"


def test_create_subchapter_increases_level(ebook, tmp_path):
    ch = ebook.create_chapter("Top", _html(tmp_path, "top.html"))
    sub = ch.create_subchapter("Sub", _html(tmp_path, "sub.html"))
    assert ch.sub_chapters == [sub]
    assert sub.title == "Sub"
    assert not sub.is_top_chapter


def test_create_subchapter_beyond_max_level_raises(ebook):
    deepest = Chapter("Deep", ebook, level=Chapter.max_level)
    with pytest.raises(Exception):
        deepest.create_subchapter("TooDeep")


def test_chapter_save_writes_content_to_tmpdir(ebook):
    ch = Chapter("Intro", ebook)
    ch.set_content("<h1>Hi</h1>")
    ch.save()
    out = ebook.tmpdir / "Intro.html"
    assert out.read_text(encoding="utf-8") == "<h1>Hi</h1>"


def test_chapter_save_copies_static_files(ebook, tmp_path):
    css = tmp_path / "style.css"
    css.write_text("body{}", encoding="utf-8")
    ch = Chapter("Styled", ebook)
    ch.set_content("x")
    result = ch.add_static_files(str(css))
    assert result is ch
    ch.save()
    assert (ebook.tmpdir / "style.css").read_text(encoding="utf-8") == "body{}"


def test_add_static_files_rejects_missing_file(ebook, tmp_path):
    ch = Chapter("Ch", ebook)
    with pytest.raises(AssertionError):
        ch.add_static_files(str(tmp_path / "missing.css"))


def test_headings_numbering_includes_subchapters(ebook, tmp_path):
    a = ebook.create_chapter("A", _html(tmp_path, "a.html"))
    a.create_subchapter("A1", _html(tmp_path, "a1.html"))
    ebook.create_chapter("B", _html(tmp_path, "b.html"))

    headings = ebook._eu.headings
    assert [h["title"] for h in headings] == ["A", "B"]
    assert headings[0]["play_order"] == 2
    assert headings[0]["file_name"] == "A.html"
    assert headings[0]["sub_headings"][0]["title"] == "A1"
    assert headings[0]["sub_headings"][0]["play_order"] == 3
    assert headings[1]["play_order"] == 4
    assert headings[1]["sub_headings"] == []


def test_render_opf_creates_sanitized_opf_file(ebook, tmp_path):
    ebook.create_chapter("Intro", _html(tmp_path, "intro.html"))
    opf_name = ebook._eu._render_opf()
    assert opf_name == "MyBook.opf"
    assert (ebook.tmpdir / opf_name).is_file()


def test_render_toc_ncx_creates_file(ebook, tmp_path):
    ebook.create_chapter("Intro", _html(tmp_path, "intro.html"))
    ncx = ebook._eu._render_toc_ncx()
    assert ncx == "toc.ncx"
    assert (ebook.tmpdir / ncx).is_file()


def test_save_cover_default_copies_packaged_cover(ebook):
    ebook._eu._save_cover()
    assert (ebook.tmpdir / "cover.jpg").is_file()


def test_save_cover_copies_given_cover_path(ebook, tmp_path):
    img = tmp_path / "mine.png"
    img.write_bytes(b"PNGDATA")
    ebook.set_cover(cover_path=str(img))
    ebook._eu._save_cover()
    assert (ebook.tmpdir / "mine.png").read_bytes() == b"PNGDATA"


def test_save_cover_with_content_is_not_implemented(ebook):
    ebook.set_cover(cover_content=b"raw")
    with pytest.raises(NotImplementedError):
        ebook._eu._save_cover()


def test_create_invokes_kindlegen_on_opf(ebook, tmp_path, fake_kindlegen):
    ebook.create_chapter("Intro", _html(tmp_path, "intro.html"))
    ebook._eu.create()
    assert len(fake_kindlegen.calls) == 1
    args = fake_kindlegen.calls[0]
    assert args[0] == KINDLEGEN
    assert args[1] == "-dont_append_source"
    assert args[2] == str(ebook.tmpdir / "MyBook.opf")


def test_create_does_not_raise_when_kindlegen_fails(ebook, tmp_path, fake_kindlegen):
    fake_kindlegen.state["rc"] = 1
    ebook.create_chapter("Intro", _html(tmp_path, "intro.html"))
    ebook._eu.create()  # should not raise
    assert len(fake_kindlegen.calls) == 1


def test_create_without_chapters_raises(ebook, fake_kindlegen):
    with pytest.raises(ValueError):
        ebook._create()
    assert fake_kindlegen.calls == []


def test_save_writes_mobi_to_destination(ebook, tmp_path, fake_kindlegen):
    ebook.create_chapter("Intro", _html(tmp_path, "intro.html"))
    out = tmp_path / "out.mobi"
    ebook.save(str(out))
    assert out.read_bytes() == b"MOBI"
    assert len(fake_kindlegen.calls) == 1


def test_ebook_del_removes_tmpdir():
    eb = Ebook("Gone Soon")
    tmp = eb.tmpdir
    assert tmp.is_dir()
    del eb
    gc.collect()
    assert not tmp.exists()
