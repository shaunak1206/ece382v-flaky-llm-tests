import os
import pathlib
import shutil
import subprocess
import warnings

import pytest

import kindle_maker.ebooklib as ebooklib
from kindle_maker.ebooklib import (
    Ebook,
    EbookUtil,
    Chapter,
    format_file_name,
    _parse_headers,
    make_ebook,
)


class DummyTemplate:
    def __init__(self, name):
        self.name = name

    def render(self, **context):
        # deterministic output for assertions
        keys = ",".join(sorted(context.keys()))
        return f"{self.name}|{keys}|{context.get('title', '')}|{context.get('author', '')}"


class DummyEnv:
    def __init__(self):
        self.requested = []

    def get_template(self, template_name):
        self.requested.append(template_name)
        return DummyTemplate(template_name)


def make_ebook_with_tmpdir(tmp_path, title="My Title", author=None):
    e = Ebook(title, author=author)
    # Redirect temp dir to test-controlled location
    old_tmpdir = e._tempdir
    e._tempdir = tmp_path / "ebook-tmp"
    e._tempdir.mkdir()
    return e, old_tmpdir


def test_format_file_name_replaces_and_removes_illegal_chars():
    assert format_file_name('a/b c+d"e\\f:g|h') == "abcd-e-f-g-h"


def test_chapter_dest_and_full_dest_file_path(tmp_path):
    ebook, _ = make_ebook_with_tmpdir(tmp_path, title="Book Title")
    ch = Chapter("Hello / World", ebook)
    assert ch.dest_file_path == pathlib.Path("HelloWorld.html")
    assert ch.full_dest_file_path == ebook.tmpdir / "HelloWorld.html"


def test_chapter_init_rejects_missing_file(tmp_path):
    ebook, _ = make_ebook_with_tmpdir(tmp_path)
    with pytest.raises(ValueError):
        Chapter("X", ebook, file_path=str(tmp_path / "missing.html"))


def test_chapter_set_content_and_content_property(tmp_path):
    ebook, _ = make_ebook_with_tmpdir(tmp_path)
    ch = Chapter("X", ebook).set_content("<p>hello</p>")
    assert ch.content == "<p>hello</p>"
    assert ch._file_path is None


def test_chapter_content_reads_from_file(tmp_path):
    ebook, _ = make_ebook_with_tmpdir(tmp_path)
    src = tmp_path / "chapter.html"
    src.write_text("from-file", encoding="utf-8")
    ch = Chapter("X", ebook, file_path=str(src))
    assert ch.content == "from-file"


def test_chapter_add_static_files_and_save_copies_files(tmp_path):
    ebook, _ = make_ebook_with_tmpdir(tmp_path)
    content_file = tmp_path / "static.txt"
    content_file.write_text("static-content", encoding="utf-8")

    ch = Chapter("Hello", ebook).set_content("chapter-content")
    ch.add_static_files(str(content_file))
    ch.save()

    assert (ebook.tmpdir / "Hello.html").read_text(encoding="utf-8") == "chapter-content"
    assert (ebook.tmpdir / "static.txt").is_file()
    assert (ebook.tmpdir / "static.txt").read_text(encoding="utf-8") == "static-content"


def test_chapter_create_subchapter_rejects_level_over_max(tmp_path):
    ebook, _ = make_ebook_with_tmpdir(tmp_path)
    ch = Chapter("Top", ebook)
    sub = ch.create_subchapter("Sub")
    assert sub.is_top_chapter is False
    with pytest.raises(Exception, match="max level num"):
        sub.create_subchapter("Too deep")


def test_ebook_set_cover_accepts_path_and_content_and_rejects_empty(tmp_path):
    ebook, _ = make_ebook_with_tmpdir(tmp_path)
    cover = tmp_path / "cover.jpg"
    cover.write_bytes(b"img")

    assert ebook.set_cover(cover_path=str(cover)) is ebook
    assert ebook.cover_path == str(cover)
    assert ebook.cover_content is None

    assert ebook.set_cover(cover_content=b"bytes") is ebook
    assert ebook.cover_content == b"bytes"
    assert ebook.cover_path is None

    with pytest.raises(ValueError):
        ebook.set_cover()


def test_ebook_add_chapter_only_accepts_top_level_chapters(tmp_path):
    ebook, _ = make_ebook_with_tmpdir(tmp_path)
    top = Chapter("Top", ebook)
    ebook.add_chapter(top)
    assert ebook.chapter_list == [top]

    sub = Chapter("Sub", ebook, level=2)
    with pytest.raises(ValueError, match="only chapters with level 1"):
        ebook.add_chapter(sub)


def test_ebook_create_chapter_appends_and_returns_chapter(tmp_path):
    ebook, _ = make_ebook_with_tmpdir(tmp_path)
    ch = ebook.create_chapter("Top")
    assert isinstance(ch, Chapter)
    assert ch in ebook.chapter_list
    assert ch.title == "Top"


def test_ebookutil_headings_builds_nested_structure_and_caches(tmp_path):
    ebook, _ = make_ebook_with_tmpdir(tmp_path)
    ch1 = ebook.create_chapter("Chapter 1")
    ch1.create_subchapter("Sub 1")
    ch2 = ebook.create_chapter("Chapter 2")

    eu = EbookUtil(ebook)
    headings = eu.headings
    assert headings[0]["title"] == "Chapter 1"
    assert headings[0]["play_order"] == 2
    assert headings[0]["sub_headings"][0]["title"] == "Sub 1"
    assert headings[0]["sub_headings"][0]["play_order"] == 3
    assert headings[1]["title"] == "Chapter 2"
    assert eu.headings is headings  # cached object


def test_ebookutil_render_file_writes_template_output(tmp_path, monkeypatch):
    ebook, _ = make_ebook_with_tmpdir(tmp_path)
    eu = EbookUtil(ebook)
    eu._templates_env = DummyEnv()

    eu._render_file("sample.txt", {"title": "T", "author": "A"}, "out.txt")
    assert (ebook.tmpdir / "out.txt").read_text(encoding="utf-8") == "sample.txt|author,title|T|A"


def test_ebookutil_render_opf_uses_default_author_when_missing(tmp_path):
    ebook, _ = make_ebook_with_tmpdir(tmp_path, title="Book", author=None)
    ebook.create_chapter("Chapter")
    eu = EbookUtil(ebook)
    eu._templates_env = DummyEnv()

    opf = eu._render_opf()
    assert opf == "Book.opf"
    assert (ebook.tmpdir / opf).read_text(encoding="utf-8").endswith("|Book|jachinlin.github.io")


def test_ebookutil_save_cover_uses_default_cover_when_none_set(tmp_path, monkeypatch):
    ebook, _ = make_ebook_with_tmpdir(tmp_path)
    eu = EbookUtil(ebook)

    copied = {}

    def fake_copy(src, dst):
        copied["src"] = str(src)
        copied["dst"] = str(dst)

    monkeypatch.setattr(shutil, "copy", fake_copy)
    eu._save_cover()

    assert copied["dst"] == str(ebook.tmpdir)
    assert copied["src"].endswith(os.path.join("templates", "cover.jpg"))


def test_ebookutil_save_cover_copies_custom_cover_path(tmp_path, monkeypatch):
    ebook, _ = make_ebook_with_tmpdir(tmp_path)
    cover = tmp_path / "cover.jpg"
    cover.write_bytes(b"img")
    ebook.cover_path = str(cover)

    copied = {}
    monkeypatch.setattr(shutil, "copy", lambda src, dst: copied.update(src=str(src), dst=str(dst)))

    eu = EbookUtil(ebook)
    eu._save_cover()

    assert copied["src"] == str(cover)
    assert copied["dst"] == str(ebook.tmpdir)


def test_ebookutil_create_invokes_kindlegen_and_steps(tmp_path, monkeypatch):
    ebook, _ = make_ebook_with_tmpdir(tmp_path, title="Book", author="Author")
    ebook.create_chapter("Chapter 1").set_content("content")
    ebook.cover_path = None

    eu = EbookUtil(ebook)

    calls = []
    monkeypatch.setattr(eu, "_render_toc_ncx", lambda: calls.append("ncx") or "toc.ncx")
    monkeypatch.setattr(eu, "_render_toc_html", lambda: calls.append("html") or "toc.html")
    monkeypatch.setattr(eu, "_render_opf", lambda: calls.append("opf") or "Book.opf")
    monkeypatch.setattr(eu, "_save_cover", lambda: calls.append("cover"))
    monkeypatch.setattr(eu, "_save_chapters", lambda: calls.append("chapters"))
    monkeypatch.setattr(ebooklib.subprocess, "call", lambda cmd: calls.append(cmd) or 0)

    eu.create()

    assert calls[:5] == ["ncx", "html", "opf", "cover", "chapters"]
    assert calls[5][0] == ebooklib.kindlegen
    assert calls[5][1:] == ["-dont_append_source", str(ebook.tmpdir / "Book.opf")]


def test_ebook_save_copies_final_mobi_file(tmp_path, monkeypatch):
    ebook, _ = make_ebook_with_tmpdir(tmp_path, title="Book Title")
    ebook.create_chapter("Chapter")

    created = ebook.tmpdir / ebook.dest_file_path
    created.write_text("mobi-data", encoding="utf-8")

    monkeypatch.setattr(ebook, "_create", lambda: None)
    target = tmp_path / "out.mobi"
    ebook.save(str(target))

    assert target.read_text(encoding="utf-8") == "mobi-data"


@pytest.mark.parametrize("system,expected", [
    ("Linux", ["xdg-open"]),
    ("Darwin", ["open"]),
])
def test_ebook_show_uses_platform_specific_command(tmp_path, monkeypatch, system, expected):
    ebook, _ = make_ebook_with_tmpdir(tmp_path, title="Book")
    ebook.create_chapter("Chapter")

    monkeypatch.setattr(ebook, "_create", lambda: None)
    monkeypatch.setattr(ebooklib.platform, "system", lambda: system)

    called = {}
    monkeypatch.setattr(ebooklib.subprocess, "call", lambda cmd: called.update(cmd=cmd) or 7)

    rc = ebook.show()

    assert rc == 7
    assert called["cmd"][:1] == expected
    assert called["cmd"][1] == str(ebook.full_dest_file_path)


def test_ebook_show_uses_os_startfile_on_windows(tmp_path, monkeypatch):
    ebook, _ = make_ebook_with_tmpdir(tmp_path, title="Book")
    ebook.create_chapter("Chapter")

    monkeypatch.setattr(ebook, "_create", lambda: None)
    monkeypatch.setattr(ebooklib.platform, "system", lambda: "Windows")
    called = {}
    monkeypatch.setattr(ebooklib.os, "startfile", lambda p: called.setdefault("path", p) or 9)

    rc = ebook.show()

    assert rc == 9
    assert called["path"] == str(ebook.full_dest_file_path)


def test_parse_headers_parses_title_and_nested_headings(tmp_path):
    toc = tmp_path / "toc.md"
    toc.write_text(
        "\n\nMy Book\n# Chapter 1\n## Sub 1\n# Chapter 2\n",
        encoding="utf-8",
    )

    title, headers = _parse_headers(str(toc))

    assert title == "My Book"
    assert headers[0]["title"] == "Chapter 1"
    assert headers[0]["next_headers"][0]["title"] == "Sub 1"
    assert headers[1]["title"] == "Chapter 2"


def test_parse_headers_rejects_missing_or_empty_files(tmp_path):
    missing = tmp_path / "missing.md"
    with pytest.raises(ValueError, match="toc.md file not exists"):
        _parse_headers(str(missing))

    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="title is empty"):
        _parse_headers(str(empty))


def test_make_ebook_orchestrates_generation_and_returns_output_path(tmp_path, monkeypatch):
    source = tmp_path / "src"
    output = tmp_path / "out"
    source.mkdir()
    output.mkdir()

    (source / "toc.md").write_text(
        "Book Title\n# Chapter A\n## Sub A1\n# Chapter B\n",
        encoding="utf-8",
    )
    (source / "cover.jpg").write_bytes(b"cover")
    (source / "extra.bin").write_bytes(b"bin")
    (source / "Chapter A.html").write_text("A", encoding="utf-8")
    (source / "Sub A1.html").write_text("SA1", encoding="utf-8")
    (source / "Chapter B.html").write_text("B", encoding="utf-8")

    monkeypatch.setattr(ebooklib, "_parse_headers", lambda path: (
        "Book Title",
        [
            {"title": "Chapter A", "next_headers": [{"title": "Sub A1"}]},
            {"title": "Chapter B", "next_headers": []},
        ],
    ))

    saved = {}
    real_save = Ebook.save

    def fake_save(self, file_path):
        saved["file_path"] = file_path
        # create the output so the caller can verify behavior
        pathlib.Path(file_path).write_text("mobi", encoding="utf-8")
        return real_save if False else None

    monkeypatch.setattr(Ebook, "save", fake_save)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        result = make_ebook(str(source), str(output))

    assert result == str(output / "Book Title.mobi")
    assert saved["file_path"] == str(output / "Book Title.mobi")
    assert any("deprecated" in str(w.message) for w in rec)
