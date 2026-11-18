import os
import pathlib
import subprocess
import warnings

import pytest

import kindle_maker.ebooklib as ebooklib


class FakeTemplate:
    def __init__(self, name):
        self.name = name

    def render(self, **context):
        return f"TEMPLATE={self.name}\nCONTEXT={sorted(context.keys())}\n"


class FakeTemplateEnv:
    def get_template(self, name):
        return FakeTemplate(name)


def make_ebook(title="My Title", author=None):
    ebook = ebooklib.Ebook(title, author=author)
    return ebook


def make_text_file(path, content="content"):
    path = pathlib.Path(path)
    path.write_text(content, encoding="utf-8")
    return path


def test_format_file_name_strips_and_replaces_illegal_characters():
    assert ebooklib.format_file_name('a/b c+d"e\\f:g|h') == 'abcd-ef-gh'


def test_chapter_init_rejects_nonexistent_file_path(tmp_path):
    ebook = make_ebook()
    missing = tmp_path / "missing.html"
    with pytest.raises(ValueError, match="is not a file"):
        ebooklib.Chapter("T", ebook, str(missing))


def test_chapter_dest_and_full_dest_file_path(tmp_path):
    ebook = make_ebook()
    chapter = ebooklib.Chapter("Hello World+Foo", ebook)
    assert chapter.dest_file_path == pathlib.Path("HelloWorld-Foo.html")
    assert chapter.full_dest_file_path == ebook.tmpdir / "HelloWorld-Foo.html"


def test_chapter_set_content_and_content_property():
    ebook = make_ebook()
    chapter = ebooklib.Chapter("Title", ebook).set_content("abc")
    assert chapter.content == "abc"
    assert chapter._file_path is None


def test_chapter_add_static_files_requires_existing_files(tmp_path):
    ebook = make_ebook()
    chapter = ebooklib.Chapter("Title", ebook)
    existing = make_text_file(tmp_path / "a.txt", "x")
    missing = tmp_path / "missing.txt"

    chapter.add_static_files(str(existing))
    assert chapter._static_files == [str(existing)]

    with pytest.raises(AssertionError, match="is not a file"):
        chapter.add_static_files(str(missing))


def test_chapter_create_subchapter_enforces_max_level():
    ebook = make_ebook()
    top = ebooklib.Chapter("Top", ebook)
    sub = top.create_subchapter("Sub")
    assert sub.title == "Sub"
    assert sub.is_top_chapter is False
    with pytest.raises(Exception, match="max level num"):
        sub.create_subchapter("Too deep")


def test_ebook_set_cover_path_and_content_and_error(tmp_path):
    ebook = make_ebook()
    cover = make_text_file(tmp_path / "cover.jpg", "img")

    assert ebook.set_cover(cover_path=str(cover)) is ebook
    assert ebook.cover_path == str(cover)
    assert ebook.cover_content is None

    assert ebook.set_cover(cover_content=b"bytes") is ebook
    assert ebook.cover_content == b"bytes"
    assert ebook.cover_path is None

    with pytest.raises(ValueError, match="both empty"):
        ebook.set_cover()


def test_ebook_add_chapter_only_accepts_top_level_chapter():
    ebook = make_ebook()
    top = ebooklib.Chapter("Top", ebook)
    ebook.add_chapter(top)
    assert ebook.chapter_list == [top]

    sub = ebooklib.Chapter("Sub", ebook, level=2)
    with pytest.raises(ValueError, match="only chapters with level 1"):
        ebook.add_chapter(sub)


def test_ebook_create_chapter_adds_to_chapter_list():
    ebook = make_ebook()
    chapter = ebook.create_chapter("Chapter 1")
    assert chapter in ebook.chapter_list
    assert chapter.title == "Chapter 1"
    assert chapter.is_top_chapter


def test_ebook_create_raises_when_chapter_list_empty():
    ebook = make_ebook()
    with pytest.raises(ValueError, match="chapter list is empty"):
        ebook._create()


def test_ebookutil_headings_builds_nested_structure_and_caches():
    ebook = make_ebook()
    c1 = ebook.create_chapter("One")
    c1.create_subchapter("One-A")
    c2 = ebook.create_chapter("Two")

    util = ebooklib.EbookUtil(ebook)
    headings1 = util.headings
    assert [h["title"] for h in headings1] == ["One", "Two"]
    assert headings1[0]["sub_headings"][0]["title"] == "One-A"
    assert headings1[0]["play_order"] == 2
    assert headings1[0]["sub_headings"][0]["play_order"] == 3
    assert headings1[1]["play_order"] == 4

    ebook.create_chapter("Three")
    headings2 = util.headings
    assert headings2 is headings1
    assert [h["title"] for h in headings2] == ["One", "Two"]


def test_ebookutil_render_file_writes_rendered_template(tmp_path):
    ebook = make_ebook()
    ebook._tempdir = tmp_path
    util = ebooklib.EbookUtil(ebook)
    util._templates_env = FakeTemplateEnv()

    util._render_file("sample.xml", {"a": 1, "b": 2}, "out.txt")

    written = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert "TEMPLATE=sample.xml" in written
    assert "CONTEXT=['a', 'b']" in written


def test_ebookutil_save_cover_uses_explicit_cover_path(tmp_path, monkeypatch):
    ebook = make_ebook()
    ebook._tempdir = tmp_path
    cover = make_text_file(tmp_path / "cover.jpg", "img")
    ebook.cover_path = str(cover)

    copied = []

    def fake_copy(src, dst):
        copied.append((src, dst))

    monkeypatch.setattr(ebooklib.shutil, "copy", fake_copy)
    util = ebooklib.EbookUtil(ebook)
    util._save_cover()

    assert copied == [(str(cover), str(tmp_path))]


def test_ebookutil_save_cover_uses_default_cover_when_no_cover_set(tmp_path, monkeypatch):
    ebook = make_ebook()
    ebook._tempdir = tmp_path

    copied = []

    def fake_copy(src, dst):
        copied.append((src, dst))

    monkeypatch.setattr(ebooklib.shutil, "copy", fake_copy)
    util = ebooklib.EbookUtil(ebook)
    util._save_cover()

    assert len(copied) == 1
    assert copied[0][1] == str(tmp_path)
    assert copied[0][0].endswith(os.path.join("templates", "cover.jpg"))


def test_ebookutil_create_calls_kindlegen_and_saves_chapters(tmp_path, monkeypatch):
    ebook = make_ebook("Title", author="Author")
    ebook._tempdir = tmp_path
    chapter = ebook.create_chapter("Chapter 1").set_content("hello")
    sub = chapter.create_subchapter("Sub 1").set_content("sub")

    util = ebooklib.EbookUtil(ebook)
    util._templates_env = FakeTemplateEnv()

    calls = []
    saved = []

    monkeypatch.setattr(ebooklib.shutil, "copy", lambda src, dst: calls.append(("copy", src, dst)))
    monkeypatch.setattr(ebooklib.subprocess, "call", lambda args: calls.append(("call", args)) or 0)
    monkeypatch.setattr(chapter, "save", lambda: saved.append("chapter"))
    monkeypatch.setattr(sub, "save", lambda: saved.append("sub"))

    util.create()

    assert any(item[0] == "call" for item in calls)
    call_args = next(item[1] for item in calls if item[0] == "call")
    assert call_args[0] == ebooklib.kindlegen
    assert "-dont_append_source" in call_args
    assert str(tmp_path / "Title.opf") in call_args
    assert saved == ["chapter", "sub"]


def test_parse_headers_valid_file(tmp_path):
    toc = tmp_path / "toc.md"
    toc.write_text(
        "\n\nMy Book\n# Chapter One\n## Sub One\n# Chapter Two\n",
        encoding="utf-8",
    )

    title, headers = ebooklib._parse_headers(str(toc))
    assert title == "My Book"
    assert headers == [
        {"title": "Chapter One", "next_headers": [{"title": "Sub One"}]},
        {"title": "Chapter Two", "next_headers": []},
    ]


def test_parse_headers_missing_file_raises(tmp_path):
    with pytest.raises(ValueError, match="toc.md file not exists"):
        ebooklib._parse_headers(str(tmp_path / "missing.md"))


def test_parse_headers_empty_title_raises(tmp_path):
    toc = tmp_path / "toc.md"
    toc.write_text("\n\n   \n", encoding="utf-8")
    with pytest.raises(ValueError, match="title is empty"):
        ebooklib._parse_headers(str(toc))


def test_parse_headers_no_headings_raises(tmp_path):
    toc = tmp_path / "toc.md"
    toc.write_text("My Book\nJust text\n", encoding="utf-8")
    with pytest.raises(ValueError, match="headings is empty"):
        ebooklib._parse_headers(str(toc))


def test_make_ebook_warns_and_returns_output_path(tmp_path, monkeypatch):
    source = tmp_path / "src"
    source.mkdir()
    output = tmp_path / "out"
    output.mkdir()

    (source / "toc.md").write_text("Book Title\n# Chapter 1\n## Sub 1\n", encoding="utf-8")
    make_text_file(source / "cover.jpg", "cover")
    make_text_file(source / "Chapter 1.html", "<h1>chapter</h1>")
    make_text_file(source / "Sub 1.html", "<h1>sub</h1>")
    make_text_file(source / "extra.txt", "x")

    saved = []

    def fake_save(self, file_path):
        saved.append(file_path)

    monkeypatch.setattr(ebooklib.Ebook, "save", fake_save, raising=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = ebooklib.make_ebook(str(source), str(output))

    assert result == str(output / "Book Title.mobi")
    assert any("deprecated" in str(w.message) for w in caught)
    assert saved == [str(output / "Book Title.mobi")]
