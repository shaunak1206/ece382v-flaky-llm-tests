import gc
import os
import pathlib
import subprocess
import tempfile

import pytest

from kindle_maker.ebooklib import (
    Chapter,
    Ebook,
    EbookUtil,
    _parse_headers,
    format_file_name,
)


def _make_ebook(monkeypatch, tmp_path, title="My Book", author=None):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr("kindle_maker.ebooklib.time.time", lambda: 123456.0)
    return Ebook(title, author=author)


def test_format_file_name_removes_and_replaces_illegal_characters():
    assert format_file_name('a/b c+d"e\\f:g|h') == "abc-d-ef-g-h"


def test_ebook_dest_and_full_dest_file_paths_use_sanitized_title(monkeypatch, tmp_path):
    ebook = _make_ebook(monkeypatch, tmp_path, title='A/B + C: D|E')
    assert ebook.dest_file_path == pathlib.Path("AB-C-D-E.mobi")
    assert ebook.full_dest_file_path == ebook.tmpdir / pathlib.Path("AB-C-D-E.mobi")


def test_chapter_dest_and_full_dest_file_paths_use_sanitized_title(monkeypatch, tmp_path):
    ebook = _make_ebook(monkeypatch, tmp_path)
    chapter = Chapter('Ch 1 / + "Intro"', ebook)
    assert chapter.dest_file_path == pathlib.Path("Ch1-Intro.html")
    assert chapter.full_dest_file_path == ebook.tmpdir / pathlib.Path("Ch1-Intro.html")


def test_chapter_is_top_chapter_reflects_level(monkeypatch, tmp_path):
    ebook = _make_ebook(monkeypatch, tmp_path)
    top = Chapter("Top", ebook, level=1)
    sub = Chapter("Sub", ebook, level=2)
    assert top.is_top_chapter is True
    assert sub.is_top_chapter is False


def test_create_chapter_adds_top_level_chapter(monkeypatch, tmp_path):
    ebook = _make_ebook(monkeypatch, tmp_path)
    chapter = ebook.create_chapter("Chapter 1")
    assert ebook.chapter_list == [chapter]
    assert chapter.is_top_chapter is True


def test_add_chapter_rejects_non_top_level_chapter(monkeypatch, tmp_path):
    ebook = _make_ebook(monkeypatch, tmp_path)
    chapter = Chapter("Nested", ebook, level=2)
    with pytest.raises(ValueError, match="only chapters with level 1 are accepted"):
        ebook.add_chapter(chapter)


def test_create_subchapter_appends_child_and_sets_level(monkeypatch, tmp_path):
    ebook = _make_ebook(monkeypatch, tmp_path)
    chapter = ebook.create_chapter("Chapter 1")
    sub = chapter.create_subchapter("Section 1.1")
    assert chapter.sub_chapters == [sub]
    assert sub.is_top_chapter is False
    assert sub._level == 2


def test_create_subchapter_rejects_level_above_max(monkeypatch, tmp_path):
    ebook = _make_ebook(monkeypatch, tmp_path)
    chapter = Chapter("Top", ebook, level=2)
    with pytest.raises(Exception, match="chapter level is great than"):
        chapter.create_subchapter("Too deep")


def test_set_content_clears_file_path_and_returns_self(monkeypatch, tmp_path):
    ebook = _make_ebook(monkeypatch, tmp_path)
    chapter = Chapter("Chapter", ebook, file_path=None)
    out = chapter.set_content("hello")
    assert out is chapter
    assert chapter.content == "hello"
    assert chapter._file_path is None


def test_chapter_content_reads_from_file_when_file_path_is_set(monkeypatch, tmp_path):
    ebook = _make_ebook(monkeypatch, tmp_path)
    source = tmp_path / "chapter.html"
    source.write_text("file content", encoding="utf-8")
    chapter = Chapter("Chapter", ebook, file_path=str(source))
    assert chapter.content == "file content"


def test_add_static_files_validates_existing_files(monkeypatch, tmp_path):
    ebook = _make_ebook(monkeypatch, tmp_path)
    chapter = Chapter("Chapter", ebook)
    static_file = tmp_path / "style.css"
    static_file.write_text("body {}", encoding="utf-8")
    out = chapter.add_static_files(str(static_file))
    assert out is chapter
    assert chapter._static_files == [str(static_file)]


def test_headings_property_builds_nested_structure_and_caches_result(monkeypatch, tmp_path):
    ebook = _make_ebook(monkeypatch, tmp_path)
    ch1 = ebook.create_chapter("Chapter 1")
    ch1.create_subchapter("Section 1.1")
    util = EbookUtil(ebook)

    headings1 = util.headings
    headings2 = util.headings

    assert headings1 is headings2
    assert headings1 == [
        {
            "title": "Chapter 1",
            "play_order": 2,
            "file_name": "Chapter1.html",
            "sub_headings": [
                {
                    "title": "Section 1.1",
                    "play_order": 3,
                    "file_name": "Section1.1.html",
                }
            ],
        }
    ]


def test_render_file_writes_rendered_template_to_tmpdir(monkeypatch, tmp_path):
    ebook = _make_ebook(monkeypatch, tmp_path)
    util = EbookUtil(ebook)

    class DummyTemplate:
        def __init__(self):
            self.calls = []

        def render(self, **context):
            self.calls.append(context)
            return "TITLE=" + context["title"]

    dummy_template = DummyTemplate()
    monkeypatch.setattr(util._templates_env, "get_template", lambda name: dummy_template)

    util._render_file("anything.xml", {"title": "Hello"}, "out.txt")

    assert (ebook.tmpdir / "out.txt").read_text(encoding="utf-8") == "TITLE=Hello"
    assert dummy_template.calls == [{"title": "Hello"}]


def test_render_toc_ncx_uses_default_author_when_missing(monkeypatch, tmp_path):
    ebook = _make_ebook(monkeypatch, tmp_path, title="My Book", author=None)
    ebook.create_chapter("Chapter 1")
    util = EbookUtil(ebook)

    captured = {}

    def fake_render_file(template_name, context, to_filename):
        captured["template_name"] = template_name
        captured["context"] = context
        captured["to_filename"] = to_filename

    monkeypatch.setattr(util, "_render_file", fake_render_file)

    result = util._render_toc_ncx()

    assert result == "toc.ncx"
    assert captured["template_name"] == "toc.xml"
    assert captured["to_filename"] == "toc.ncx"
    assert captured["context"]["title"] == "My Book"
    assert captured["context"]["author"] == "jachinlin.github.io"
    assert captured["context"]["headings"] == util.headings


def test_create_calls_subprocess_and_saves_cover_and_chapters(monkeypatch, tmp_path):
    ebook = _make_ebook(monkeypatch, tmp_path, title="Book", author="Author")
    chapter = ebook.create_chapter("Chapter 1")
    chapter.set_content("chapter body")

    util = ebook._eu

    calls = []

    monkeypatch.setattr(util, "_render_toc_ncx", lambda: calls.append("ncx") or "toc.ncx")
    monkeypatch.setattr(util, "_render_toc_html", lambda: calls.append("html") or "toc.html")
    monkeypatch.setattr(util, "_render_opf", lambda: calls.append("opf") or "book.opf")
    monkeypatch.setattr(util, "_save_cover", lambda: calls.append("cover"))
    monkeypatch.setattr(util, "_save_chapters", lambda: calls.append("chapters"))

    called = {}

    def fake_call(cmd):
        called["cmd"] = cmd
        return 0

    monkeypatch.setattr("kindle_maker.ebooklib.subprocess.call", fake_call)
    monkeypatch.setattr("kindle_maker.ebooklib.kindlegen", "/fake/kindlegen")

    ebook._create()

    assert calls == ["ncx", "html", "opf", "cover", "chapters"]
    assert called["cmd"] == ["/fake/kindlegen", "-dont_append_source", str(ebook.tmpdir / "book.opf")]


def test_save_copies_generated_mobi_to_requested_destination(monkeypatch, tmp_path):
    ebook = _make_ebook(monkeypatch, tmp_path, title="Book")
    generated = ebook.full_dest_file_path
    generated.write_text("mobi-bytes", encoding="utf-8")

    monkeypatch.setattr(ebook, "_create", lambda: None)

    out_file = tmp_path / "out.mobi"
    ebook.save(str(out_file))

    assert out_file.read_text(encoding="utf-8") == "mobi-bytes"


def test_show_uses_linux_xdg_open(monkeypatch, tmp_path):
    ebook = _make_ebook(monkeypatch, tmp_path, title="Book")
    monkeypatch.setattr(ebook, "_create", lambda: None)
    monkeypatch.setattr("kindle_maker.ebooklib.platform.system", lambda: "Linux")

    called = {}

    def fake_call(cmd):
        called["cmd"] = cmd
        return 7

    monkeypatch.setattr("kindle_maker.ebooklib.subprocess.call", fake_call)

    rc = ebook.show()

    assert rc == 7
    assert called["cmd"] == ["xdg-open", str(ebook.full_dest_file_path)]


def test_show_uses_darwin_open(monkeypatch, tmp_path):
    ebook = _make_ebook(monkeypatch, tmp_path, title="Book")
    monkeypatch.setattr(ebook, "_create", lambda: None)
    monkeypatch.setattr("kindle_maker.ebooklib.platform.system", lambda: "Darwin")

    called = {}

    def fake_call(cmd):
        called["cmd"] = cmd
        return 3

    monkeypatch.setattr("kindle_maker.ebooklib.subprocess.call", fake_call)

    rc = ebook.show()

    assert rc == 3
    assert called["cmd"] == ["open", str(ebook.full_dest_file_path)]


def test_parse_headers_parses_title_and_nested_headings(monkeypatch, tmp_path):
    toc = tmp_path / "toc.md"
    toc.write_text(
        "\n\nMy Title\n# Chapter 1\n## Section 1.1\n# Chapter 2\n",
        encoding="utf-8",
    )

    title, headers = _parse_headers(str(toc))

    assert title == "My Title"
    assert headers == [
        {"title": "Chapter 1", "next_headers": [{"title": "Section 1.1"}]},
        {"title": "Chapter 2", "next_headers": []},
    ]


def test_parse_headers_raises_for_empty_or_missing_title(monkeypatch, tmp_path):
    toc = tmp_path / "toc.md"
    toc.write_text("\n   \n", encoding="utf-8")

    with pytest.raises(ValueError, match="title is empty"):
        _parse_headers(str(toc))


def test_make_ebook_warns_for_deprecated_api(monkeypatch, tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "toc.md").write_text("My Title\n# Chapter 1\n", encoding="utf-8")
    (source / "Chapter 1.html").write_text("<p>chapter</p>", encoding="utf-8")
    (source / "cover.jpg").write_bytes(b"cover")
    (source / "asset.txt").write_text("asset", encoding="utf-8")

    monkeypatch.setattr("kindle_maker.ebooklib.Ebook.save", lambda self, fp: None)

    with pytest.warns(UserWarning, match="make_ebook"):
        from kindle_maker.ebooklib import make_ebook

        out = make_ebook(str(source), output_dir=str(tmp_path))

    assert out == os.path.join(str(tmp_path), "My Title.mobi")
