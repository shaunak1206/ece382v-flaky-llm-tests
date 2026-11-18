import io
import os
import pathlib
import shutil
import types

import pytest

import kindle_maker.ebooklib as ebooklib


class DummyTemplate:
    def __init__(self, text):
        self.text = text

    def render(self, **context):
        # Minimal deterministic rendering for tests.
        return self.text.format(**context)


class DummyTemplateEnv:
    def __init__(self, template_map):
        self.template_map = template_map

    def get_template(self, name):
        return DummyTemplate(self.template_map[name])


@pytest.fixture
def ebook(tmp_path):
    # Use a simple ebook instance and ensure cleanup runs during the test.
    e = ebooklib.Ebook("My Book", author="Alice")
    yield e
    try:
        del e
    except Exception:
        pass


def test_format_file_name_strips_and_replaces_illegal_chars():
    s = 'a/b c+d"e\\f:g|h'
    assert ebooklib.format_file_name(s) == 'ab-cdef-gh'


def test_chapter_dest_and_full_dest_file_path(ebook):
    c = ebooklib.Chapter("Hello World+Test", ebook)
    assert c.dest_file_path == pathlib.Path("HelloWorld-Test.html")
    assert c.full_dest_file_path == ebook.tmpdir / "HelloWorld-Test.html"


def test_chapter_set_content_overrides_file_path(ebook):
    p = ebook.tmpdir / "chapter.txt"
    p.write_text("from file", encoding="utf-8")
    c = ebooklib.Chapter("T", ebook, str(p))
    assert c.content == "from file"
    c.set_content("manual content")
    assert c.content == "manual content"


def test_chapter_initial_file_path_must_exist(ebook):
    with pytest.raises(ValueError):
        ebooklib.Chapter("Bad", ebook, file_path=str(ebook.tmpdir / "missing.html"))


def test_add_static_files_accepts_existing_files_and_rejects_missing(ebook, tmp_path):
    c = ebooklib.Chapter("T", ebook)
    f1 = tmp_path / "a.css"
    f2 = tmp_path / "b.js"
    f1.write_text("x", encoding="utf-8")
    f2.write_text("y", encoding="utf-8")
    c.add_static_files(str(f1), str(f2))
    assert c._static_files == [str(f1), str(f2)]
    with pytest.raises(AssertionError):
        c.add_static_files(str(tmp_path / "nope.png"))


def test_create_subchapter_sets_level_and_parent_relationship(ebook):
    c = ebooklib.Chapter("Top", ebook)
    sub = c.create_subchapter("Sub")
    assert sub.title == "Sub"
    assert sub._level == 2
    assert sub.is_top_chapter is False
    assert c.sub_chapters == [sub]


def test_create_subchapter_respects_max_level(ebook):
    top = ebooklib.Chapter("Top", ebook)
    sub = top.create_subchapter("Sub")
    with pytest.raises(Exception, match="max level"):
        sub.create_subchapter("Too deep")


def test_ebook_set_cover_requires_one_of_path_or_content(ebook, tmp_path):
    cover = tmp_path / "cover.jpg"
    cover.write_bytes(b"jpg")
    assert ebook.set_cover(cover_path=str(cover)) is ebook
    assert ebook.cover_path == str(cover)
    assert ebook.cover_content is None

    assert ebook.set_cover(cover_content=b"data") is ebook
    assert ebook.cover_content == b"data"
    assert ebook.cover_path is None

    with pytest.raises(ValueError):
        ebooklib.Ebook("Empty").set_cover()


def test_ebook_add_chapter_only_accepts_top_level(ebook):
    top = ebooklib.Chapter("Top", ebook)
    sub = top.create_subchapter("Sub")
    ebook.add_chapter(top)
    assert ebook.chapter_list == [top]
    with pytest.raises(ValueError, match="level 1"):
        ebook.add_chapter(sub)


def test_ebook_create_chapter_adds_top_level_chapter(ebook):
    c = ebook.create_chapter("Chapter 1")
    assert c in ebook.chapter_list
    assert c.is_top_chapter
    assert c.title == "Chapter 1"


def test_ebookutil_headings_caches_and_includes_subheadings(ebook):
    c1 = ebook.create_chapter("C1")
    c1.create_subchapter("C1.1")
    c2 = ebook.create_chapter("C2")
    util = ebooklib.EbookUtil(ebook)

    hs1 = util.headings
    hs2 = util.headings
    assert hs1 is hs2
    assert hs1 == [
        {
            "title": "C1",
            "play_order": 2,
            "file_name": str(c1.dest_file_path),
            "sub_headings": [
                {
                    "title": "C1.1",
                    "play_order": 3,
                    "file_name": str(c1.sub_chapters[0].dest_file_path),
                }
            ],
        },
        {
            "title": "C2",
            "play_order": 4,
            "file_name": str(c2.dest_file_path),
            "sub_headings": [],
        },
    ]


def test_render_file_writes_template_output_to_tmpdir(ebook, monkeypatch):
    util = ebooklib.EbookUtil(ebook)
    util._templates_env = DummyTemplateEnv({"x.txt": "title={title}, count={count}"})
    util._render_file("x.txt", {"title": "A", "count": 3}, "out.txt")
    assert (ebook.tmpdir / "out.txt").read_text(encoding="utf-8") == "title=A, count=3"


def test_render_toc_and_opf_use_default_author_when_missing(ebook):
    ebook.author = None
    util = ebooklib.EbookUtil(ebook)
    captured = {}

    def fake_render(template_name, context, to_filename):
        captured[to_filename] = (template_name, context)

    util._render_file = fake_render
    assert util._render_toc_ncx() == "toc.ncx"
    assert util._render_opf().endswith(".opf")
    assert captured["toc.ncx"][0] == "toc.xml"
    assert captured["toc.ncx"][1]["author"] == "jachinlin.github.io"
    assert captured[next(k for k in captured if k.endswith(".opf"))][1]["author"] == "jachinlin.github.io"


def test_save_cover_uses_custom_cover_and_fallback_cover(ebook, tmp_path, monkeypatch):
    util = ebooklib.EbookUtil(ebook)
    custom = tmp_path / "custom.jpg"
    custom.write_bytes(b"custom")
    ebook.cover_path = str(custom)
    util._save_cover()
    assert (ebook.tmpdir / "custom.jpg").read_bytes() == b"custom"

    ebook2 = ebooklib.Ebook("Book2")
    util2 = ebooklib.EbookUtil(ebook2)
    copied = {}

    def fake_copy(src, dst):
        copied["src"] = src
        copied["dst"] = dst

    monkeypatch.setattr(ebooklib.shutil, "copy", fake_copy)
    util2._save_cover()
    assert copied["src"].endswith(os.path.join("templates", "cover.jpg"))
    assert copied["dst"] == str(ebook2.tmpdir)


def test_ebook_create_raises_when_no_chapters(ebook):
    with pytest.raises(ValueError, match="chapter list is empty"):
        ebook._create()


def test_ebook_save_copies_generated_mobi(monkeypatch, ebook, tmp_path):
    out = tmp_path / "result.mobi"

    def fake_create():
        (ebook.tmpdir / ebook.dest_file_path).write_text("mobi-bytes", encoding="utf-8")

    ebook._create = fake_create
    copied = {}

    def fake_copy(src, dst):
        copied["src"] = src
        copied["dst"] = dst

    monkeypatch.setattr(ebooklib.shutil, "copy", fake_copy)
    ebook.save(str(out))
    assert copied["src"] == str(ebook.tmpdir / ebook.dest_file_path)
    assert copied["dst"] == str(out)


def test_ebook_show_uses_platform_specific_openers(monkeypatch, ebook):
    ebook._create = lambda: None

    monkeypatch.setattr(ebooklib.platform, "system", lambda: "Linux")
    calls = []
    monkeypatch.setattr(ebooklib.subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    assert ebook.show() == 0
    assert calls[-1][0] == "xdg-open"

    monkeypatch.setattr(ebooklib.platform, "system", lambda: "Darwin")
    assert ebook.show() == 0
    assert calls[-1][0] == "open"

    monkeypatch.setattr(ebooklib.platform, "system", lambda: "Windows")
    monkeypatch.setattr(ebooklib.os, "startfile", lambda path: calls.append(["startfile", path]) or 7)
    assert ebook.show() == 7
    assert calls[-1][0] == "startfile"


def test_parse_headers_parses_title_top_and_subheadings(tmp_path):
    toc = tmp_path / "toc.md"
    toc.write_text(
        "\n\nMy Title\n# Chapter A\n## Sub A1\n# Chapter B\n## Sub B1\n",
        encoding="utf-8",
    )
    title, headers = ebooklib._parse_headers(str(toc))
    assert title == "My Title"
    assert headers == [
        {"title": "Chapter A", "next_headers": [{"title": "Sub A1"}]},
        {"title": "Chapter B", "next_headers": [{"title": "Sub B1"}]},
    ]


def test_parse_headers_errors_for_missing_or_invalid_files(tmp_path):
    with pytest.raises(ValueError, match="not exists"):
        ebooklib._parse_headers(str(tmp_path / "missing.md"))

    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="title is empty"):
        ebooklib._parse_headers(str(empty))

    no_headings = tmp_path / "no_headings.md"
    no_headings.write_text("Title only\n", encoding="utf-8")
    with pytest.raises(ValueError, match="headings is empty"):
        ebooklib._parse_headers(str(no_headings))
