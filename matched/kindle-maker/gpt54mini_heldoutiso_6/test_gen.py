import os
import pathlib
import platform
import subprocess

import pytest

import kindle_maker.ebooklib as ebooklib


def _make_text_file(tmp_path, name, content="content"):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_format_file_name_strips_and_replaces_illegal_chars():
    assert ebooklib.format_file_name('a/b c+d"x\\y:z|w') == 'abcd-xyz-w'


def test_ebook_initializes_tempdir_and_dest_paths():
    ebook = ebooklib.Ebook("My Book Title")
    assert ebook.title == "My Book Title"
    assert ebook.dest_file_path == pathlib.Path("MyBookTitle.mobi")
    assert ebook.full_dest_file_path == ebook.tmpdir / "MyBookTitle.mobi"
    assert ebook.tmpdir.exists()


def test_set_cover_with_path_and_content_mutually_exclusive():
    ebook = ebooklib.Ebook("Book")
    ebook.set_cover(cover_path="/tmp/cover.jpg")
    assert ebook.cover_path == "/tmp/cover.jpg"
    assert ebook.cover_content is None

    ebook.set_cover(cover_content=b"abc")
    assert ebook.cover_content == b"abc"
    assert ebook.cover_path is None


def test_set_cover_without_anything_raises():
    ebook = ebooklib.Ebook("Book")
    with pytest.raises(ValueError):
        ebook.set_cover()


def test_create_chapter_adds_top_level_chapter():
    ebook = ebooklib.Ebook("Book")
    chapter = ebook.create_chapter("Chapter 1")
    assert chapter in ebook.chapter_list
    assert chapter.is_top_chapter
    assert chapter.title == "Chapter 1"


def test_add_chapter_rejects_non_top_level_chapter(tmp_path):
    ebook = ebooklib.Ebook("Book")
    chapter_file = _make_text_file(tmp_path, "c.html", "hello")
    top = ebook.create_chapter("Top")
    sub = top.create_subchapter("Sub", str(chapter_file))
    with pytest.raises(ValueError):
        ebook.add_chapter(sub)


def test_create_chapter_with_existing_file_sets_content_property(tmp_path):
    ebook = ebooklib.Ebook("Book")
    chapter_file = _make_text_file(tmp_path, "chapter.html", "<p>hello</p>")
    chapter = ebook.create_chapter("Chapter", str(chapter_file))
    assert chapter.content == "<p>hello</p>"


def test_chapter_dest_file_path_sanitizes_title():
    ebook = ebooklib.Ebook("Book")
    chapter = ebook.create_chapter('A/B C+D:"E\\F|G')
    assert chapter.dest_file_path == pathlib.Path("ABCD-EFG.html")


def test_add_static_files_requires_existing_files(tmp_path):
    ebook = ebooklib.Ebook("Book")
    chapter = ebook.create_chapter("Chapter")
    existing = _make_text_file(tmp_path, "asset.txt", "asset")
    chapter.add_static_files(str(existing))
    assert str(existing) in chapter._static_files

    with pytest.raises(AssertionError):
        chapter.add_static_files(str(tmp_path / "missing.txt"))


def test_create_subchapter_enforces_max_level(tmp_path):
    ebook = ebooklib.Ebook("Book")
    root = ebook.create_chapter("Root")
    sub = root.create_subchapter("Child")
    with pytest.raises(Exception, match="max level"):
        sub.create_subchapter("Too Deep")


def test_heading_property_builds_and_caches_structure(tmp_path):
    ebook = ebooklib.Ebook("Book")
    ch1 = ebook.create_chapter("Chapter 1")
    ch1.create_subchapter("Sub 1")
    ch2 = ebook.create_chapter("Chapter 2")

    util = ebook._eu
    headings1 = util.headings
    headings2 = util.headings

    assert headings1 is headings2
    assert headings1 == [
        {
            "title": "Chapter 1",
            "play_order": 2,
            "file_name": str(ch1.dest_file_path),
            "sub_headings": [
                {
                    "title": "Sub 1",
                    "play_order": 3,
                    "file_name": str(ch1.sub_chapters[0].dest_file_path),
                }
            ],
        },
        {
            "title": "Chapter 2",
            "play_order": 4,
            "file_name": str(ch2.dest_file_path),
            "sub_headings": [],
        },
    ]


def test_render_file_writes_to_tempdir(monkeypatch, tmp_path):
    ebook = ebooklib.Ebook("Book")
    (ebook.tmpdir / "x.txt").unlink(missing_ok=True)

    class DummyTemplate:
        def render(self, **context):
            return "hello " + context["name"]

    monkeypatch.setattr(ebook._eu._templates_env, "get_template", lambda name: DummyTemplate())
    ebook._eu._render_file("any.txt", {"name": "world"}, "x.txt")

    assert (ebook.tmpdir / "x.txt").read_text(encoding="utf-8") == "hello world"


def test_render_toc_ncx_uses_default_author_when_missing(monkeypatch):
    ebook = ebooklib.Ebook("Book")
    ebook.create_chapter("Chapter 1")
    captured = {}

    def fake_render(template_name, context, to_filename):
        captured["template_name"] = template_name
        captured["context"] = context
        captured["to_filename"] = to_filename

    monkeypatch.setattr(ebook._eu, "_render_file", fake_render)
    assert ebook._eu._render_toc_ncx() == "toc.ncx"
    assert captured["template_name"] == "toc.xml"
    assert captured["to_filename"] == "toc.ncx"
    assert captured["context"]["author"] == "jachinlin.github.io"


def test_save_cover_prefers_cover_path(monkeypatch, tmp_path):
    ebook = ebooklib.Ebook("Book")
    cover_src = _make_text_file(tmp_path, "cover.jpg", "img")
    ebook.set_cover(cover_path=str(cover_src))

    copied = {}

    def fake_copy(src, dst):
        copied["src"] = src
        copied["dst"] = dst

    monkeypatch.setattr(ebooklib.shutil, "copy", fake_copy)
    ebook._eu._save_cover()

    assert copied["src"] == str(cover_src)
    assert copied["dst"] == str(ebook.tmpdir)


def test_save_cover_falls_back_to_default_cover(monkeypatch):
    ebook = ebooklib.Ebook("Book")
    copied = {}

    def fake_copy(src, dst):
        copied["src"] = src
        copied["dst"] = dst

    monkeypatch.setattr(ebooklib.shutil, "copy", fake_copy)
    ebook._eu._save_cover()

    assert pathlib.Path(copied["src"]).name == "cover.jpg"
    assert copied["dst"] == str(ebook.tmpdir)


def test_save_cover_with_cover_content_is_not_implemented():
    ebook = ebooklib.Ebook("Book")
    ebook.cover_content = b"bytes"
    with pytest.raises(NotImplementedError):
        ebook._eu._save_cover()


def test_create_invokes_render_save_and_kindlegen(monkeypatch):
    ebook = ebooklib.Ebook("Book")
    chapter = ebook.create_chapter("Chapter 1")
    chapter.set_content("<html>chapter</html>")

    calls = {"ncx": 0, "html": 0, "opf": 0, "cover": 0, "chapters": 0, "subprocess": 0}

    monkeypatch.setattr(ebook._eu, "_render_toc_ncx", lambda: calls.__setitem__("ncx", calls["ncx"] + 1) or "toc.ncx")
    monkeypatch.setattr(ebook._eu, "_render_toc_html", lambda: calls.__setitem__("html", calls["html"] + 1) or "toc.html")
    monkeypatch.setattr(ebook._eu, "_render_opf", lambda: calls.__setitem__("opf", calls["opf"] + 1) or "book.opf")
    monkeypatch.setattr(ebook._eu, "_save_cover", lambda: calls.__setitem__("cover", calls["cover"] + 1))
    monkeypatch.setattr(ebook._eu, "_save_chapters", lambda: calls.__setitem__("chapters", calls["chapters"] + 1))
    monkeypatch.setattr(ebooklib.subprocess, "call", lambda args: calls.__setitem__("subprocess", calls["subprocess"] + 1) or 0)

    ebook._eu.create()

    assert calls == {"ncx": 1, "html": 1, "opf": 1, "cover": 1, "chapters": 1, "subprocess": 1}


def test_ebook_create_raises_when_chapter_list_empty():
    ebook = ebooklib.Ebook("Book")
    with pytest.raises(ValueError, match="chapter list is empty"):
        ebook._create()


def test_show_linux_uses_xdg_open(monkeypatch):
    ebook = ebooklib.Ebook("Book")
    ebook.create_chapter("Chapter 1")

    monkeypatch.setattr(ebook._eu, "create", lambda: None)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    captured = {}
    monkeypatch.setattr(ebooklib.subprocess, "call", lambda args: captured.setdefault("args", args) or 0)

    rc = ebook.show()

    assert rc == 0
    assert captured["args"][0] == "xdg-open"
    assert str(ebook.full_dest_file_path) in captured["args"][1]


def test_show_darwin_uses_open(monkeypatch):
    ebook = ebooklib.Ebook("Book")
    ebook.create_chapter("Chapter 1")

    monkeypatch.setattr(ebook._eu, "create", lambda: None)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    captured = {}
    monkeypatch.setattr(ebooklib.subprocess, "call", lambda args: captured.setdefault("args", args) or 0)

    rc = ebook.show()

    assert rc == 0
    assert captured["args"][0] == "open"


def test_show_other_platform_uses_startfile(monkeypatch):
    ebook = ebooklib.Ebook("Book")
    ebook.create_chapter("Chapter 1")

    monkeypatch.setattr(ebook._eu, "create", lambda: None)
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    captured = {}
    monkeypatch.setattr(ebooklib.os, "startfile", lambda path: captured.setdefault("path", path) or 123)

    rc = ebook.show()

    assert rc == 123
    assert captured["path"] == str(ebook.full_dest_file_path)
