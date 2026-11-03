import gc
import io
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from kindle_maker.ebooklib import (
    Chapter,
    Ebook,
    EbookUtil,
    _parse_headers,
    format_file_name,
    make_ebook,
)


class DummyTemplate:
    def __init__(self):
        self.calls = []

    def render(self, **context):
        self.calls.append(context)
        return "rendered:" + ",".join(sorted(context.keys()))


class DummyEnv:
    def __init__(self, template):
        self.template = template

    def get_template(self, template_name):
        return self.template


@pytest.fixture
def ebook():
    e = Ebook("My Book", author="Author")
    yield e
    # Avoid destructor cleanup issues affecting other tests
    tmpdir = e.tmpdir
    del e
    gc.collect()
    if tmpdir.exists():
        shutil.rmtree(str(tmpdir), ignore_errors=True)


def test_format_file_name_removes_and_replaces_illegal_chars():
    assert format_file_name('a/b c+"d\\e:f|g') == 'abc-d-ef-g'


def test_chapter_dest_file_path_uses_sanitized_title(ebook):
    ch = Chapter('A / B + C', ebook)
    assert str(ch.dest_file_path) == "AB-C.html"


def test_chapter_set_content_clears_file_path_and_returns_self(ebook):
    ch = Chapter("Chapter 1", ebook, file_path=None)
    out = ch.set_content("hello")
    assert out is ch
    assert ch.content == "hello"


def test_chapter_add_static_files_requires_files(tmp_path, ebook):
    ch = Chapter("Chapter 1", ebook)
    existing = tmp_path / "a.txt"
    existing.write_text("x", encoding="utf-8")
    ch.add_static_files(str(existing))
    assert str(existing) in ch._static_files

    with pytest.raises(AssertionError):
        ch.add_static_files(str(tmp_path / "missing.txt"))


def test_create_subchapter_enforces_max_level(ebook):
    parent = Chapter("Parent", ebook)
    child = parent.create_subchapter("Child")
    assert child.title == "Child"
    assert child.is_top_chapter is False

    with pytest.raises(Exception, match="max level num"):
        child.create_subchapter("Too deep")


def test_ebook_add_chapter_accepts_only_top_level_chapters(ebook):
    top = Chapter("Top", ebook)
    ebook.add_chapter(top)
    assert ebook.chapter_list == [top]

    sub = top.create_subchapter("Sub")
    with pytest.raises(ValueError, match="only chapters with level 1"):
        ebook.add_chapter(sub)


def test_ebook_set_cover_accepts_path_or_content(ebook, tmp_path):
    cover = tmp_path / "cover.jpg"
    cover.write_bytes(b"img")

    returned = ebook.set_cover(cover_path=str(cover))
    assert returned is ebook
    assert ebook.cover_path == str(cover)
    assert ebook.cover_content is None

    returned = ebook.set_cover(cover_content=b"bytes")
    assert returned is ebook
    assert ebook.cover_content == b"bytes"
    assert ebook.cover_path is None

    with pytest.raises(ValueError, match="both empty"):
        Ebook("Empty").set_cover()


def test_ebook_create_chapter_adds_to_chapter_list(ebook):
    ch = ebook.create_chapter("Intro", "ignored.html")
    assert ch.title == "Intro"
    assert ebook.chapter_list == [ch]


def test_ebook_create_requires_at_least_one_chapter(ebook):
    with pytest.raises(ValueError, match="chapter list is empty"):
        ebook._create()


def test_ebookutil_headings_are_built_and_cached(ebook):
    ch1 = ebook.create_chapter("One")
    ch1.create_subchapter("One-1")
    ch2 = ebook.create_chapter("Two")

    util = EbookUtil(ebook)
    first = util.headings
    second = util.headings

    assert first is second
    assert len(first) == 2
    assert first[0]["title"] == "One"
    assert first[0]["sub_headings"][0]["title"] == "One-1"
    assert first[1]["title"] == "Two"
    assert first[0]["play_order"] == 2
    assert first[0]["sub_headings"][0]["play_order"] == 3
    assert first[1]["play_order"] == 4


def test_ebookutil_render_file_writes_to_tmpdir(tmp_path, ebook):
    ebook._tempdir = tmp_path
    util = EbookUtil(ebook)
    template = DummyTemplate()
    util._templates_env = DummyEnv(template)

    util._render_file("anything.xml", {"x": 1}, "out.txt")

    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "rendered:x"
    assert template.calls == [{"x": 1}]


def test_ebookutil_render_toc_and_opf_use_expected_defaults(tmp_path):
    ebook = Ebook("Title With Spaces")
    ebook._tempdir = tmp_path
    ebook.create_chapter("Chapter 1")

    util = EbookUtil(ebook)
    template = DummyTemplate()
    util._templates_env = DummyEnv(template)

    ncx = util._render_toc_ncx()
    html = util._render_toc_html()
    opf = util._render_opf()

    assert ncx == "toc.ncx"
    assert html == "toc.html"
    assert opf == "TitleWithSpaces.opf"
    assert len(template.calls) == 3
    assert template.calls[0]["author"] == "jachinlin.github.io"
    assert template.calls[2]["title"] == "Title With Spaces"


def test_ebookutil_save_cover_prefers_cover_path(tmp_path, ebook):
    ebook._tempdir = tmp_path
    src = tmp_path / "cover.jpg"
    src.write_bytes(b"cover-bytes")
    ebook.cover_path = str(src)

    util = EbookUtil(ebook)
    util._save_cover()

    assert (tmp_path / "cover.jpg").read_bytes() == b"cover-bytes"


def test_ebookutil_save_cover_raises_for_cover_content(tmp_path, ebook):
    ebook._tempdir = tmp_path
    ebook.cover_content = b"bytes"
    util = EbookUtil(ebook)

    with pytest.raises(NotImplementedError):
        util._save_cover()


def test_ebookutil_save_chapters_writes_chapter_and_static_files(tmp_path):
    ebook = Ebook("Book")
    ebook._tempdir = tmp_path
    ch = ebook.create_chapter("Chap 1")
    ch.set_content("chapter content")
    static = tmp_path / "static.css"
    static.write_text("body{}", encoding="utf-8")
    ch.add_static_files(str(static))

    util = EbookUtil(ebook)
    util._save_chapters()

    assert (tmp_path / "Chap1.html").read_text(encoding="utf-8") == "chapter content"
    assert (tmp_path / "static.css").exists()


def test_ebookutil_create_invokes_kindlegen_and_writes_files(monkeypatch, tmp_path):
    ebook = Ebook("Book")
    ebook._tempdir = tmp_path
    ch = ebook.create_chapter("Chapter 1")
    ch.set_content("content")

    util = EbookUtil(ebook)
    template = DummyTemplate()
    util._templates_env = DummyEnv(template)

    calls = []

    def fake_call(cmd):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)

    util.create()

    assert (tmp_path / "toc.ncx").exists()
    assert (tmp_path / "toc.html").exists()
    assert (tmp_path / "Book.opf").exists()
    assert (tmp_path / "cover.jpg").exists()
    assert (tmp_path / "Chapter1.html").read_text(encoding="utf-8") == "content"
    assert calls and calls[0][0]  # kindlegen path set
    assert calls[0][1:] == ["-dont_append_source", str(tmp_path / "Book.opf")]


def test_parse_headers_reads_title_and_headings(tmp_path):
    toc = tmp_path / "toc.md"
    toc.write_text(
        "\n\nMy Title\n"
        "# Chapter One\n"
        "## Sub One\n"
        "# Chapter Two\n",
        encoding="utf-8",
    )

    title, headers = _parse_headers(str(toc))

    assert title == "My Title"
    assert headers == [
        {"title": "Chapter One", "next_headers": [{"title": "Sub One"}]},
        {"title": "Chapter Two", "next_headers": []},
    ]


def test_parse_headers_raises_for_missing_or_invalid_files(tmp_path):
    with pytest.raises(ValueError, match="not exists"):
        _parse_headers(str(tmp_path / "missing.md"))

    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="title is empty"):
        _parse_headers(str(empty))

    no_headers = tmp_path / "no_headers.md"
    no_headers.write_text("Title only\n", encoding="utf-8")
    with pytest.raises(ValueError, match="headings is empty"):
        _parse_headers(str(no_headers))


def test_make_ebook_creates_mobi_and_warns(monkeypatch, tmp_path):
    source = tmp_path / "src"
    output = tmp_path / "out"
    source.mkdir()
    output.mkdir()

    (source / "toc.md").write_text("Book Title\n# Chapter 1\n", encoding="utf-8")
    (source / "Chapter 1.html").write_text("<p>hi</p>", encoding="utf-8")
    (source / "cover.jpg").write_bytes(b"cover")
    (source / "extra.txt").write_text("extra", encoding="utf-8")

    # make ebook.save a no-op around creation output copy destination
    calls = []

    def fake_call(cmd):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)

    out_file = make_ebook(str(source), str(output))

    assert out_file == str(output / "Book Title.mobi")
    assert calls  # kindlegen was invoked
    assert (output / "Book Title.mobi").exists()
