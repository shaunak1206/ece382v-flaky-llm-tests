import io
import os
import pathlib
import shutil
import subprocess

import pytest

from kindle_maker import ebooklib


class FakeTemplate:
    def __init__(self, name):
        self.name = name

    def render(self, **context):
        # deterministic, simple output for assertions
        keys = ",".join(sorted(context.keys()))
        return f"TEMPLATE={self.name};KEYS={keys};CTX={context!r}"


class FakeTemplateEnv:
    def __init__(self):
        self.requested = []

    def get_template(self, template_name):
        self.requested.append(template_name)
        return FakeTemplate(template_name)


def make_ebook(monkeypatch, title="My Book", author=None):
    ebook = ebooklib.Ebook(title, author=author)
    monkeypatch.setattr(ebook._eu, "_templates_env", FakeTemplateEnv())
    return ebook


def test_format_file_name_strips_and_replaces_illegal_characters():
    assert ebooklib.format_file_name('a/b c+"d\\e:f|g') == "abcd-e-f-g"


def test_chapter_dest_file_path_uses_sanitized_title():
    ebook = make_ebook(pytest.MonkeyPatch(), title="x")
    chapter = ebooklib.Chapter('A/B + C:"|\\', ebook)
    assert chapter.dest_file_path == pathlib.Path("AB-C--.html")


def test_chapter_rejects_nonexistent_file_path(tmp_path):
    ebook = make_ebook(pytest.MonkeyPatch(), title="x")
    missing = tmp_path / "missing.html"
    with pytest.raises(ValueError, match="is not a file"):
        ebooklib.Chapter("Title", ebook, str(missing))


def test_add_chapter_accepts_top_level_only():
    ebook = make_ebook(pytest.MonkeyPatch(), title="x")
    top = ebooklib.Chapter("Top", ebook, level=1)
    ebook.add_chapter(top)
    assert ebook.chapter_list == [top]

    sub = ebooklib.Chapter("Sub", ebook, level=2)
    with pytest.raises(ValueError, match="only chapters with level 1 are accepted"):
        ebook.add_chapter(sub)


def test_create_chapter_adds_to_ebook_and_returns_chapter():
    ebook = make_ebook(pytest.MonkeyPatch(), title="x")
    chapter = ebook.create_chapter("Intro")
    assert chapter.title == "Intro"
    assert chapter in ebook.chapter_list
    assert chapter.is_top_chapter is True


def test_create_subchapter_respects_max_level():
    ebook = make_ebook(pytest.MonkeyPatch(), title="x")
    top = ebook.create_chapter("Top")
    sub = top.create_subchapter("Sub")
    assert sub.title == "Sub"
    assert sub.is_top_chapter is False
    with pytest.raises(Exception, match="max level num"):
        sub.create_subchapter("Too Deep")


def test_set_cover_requires_one_of_path_or_content():
    ebook = make_ebook(pytest.MonkeyPatch(), title="x")
    with pytest.raises(ValueError, match="both empty"):
        ebook.set_cover()


def test_set_cover_with_path_clears_cover_content(tmp_path):
    ebook = make_ebook(pytest.MonkeyPatch(), title="x")
    cover = tmp_path / "cover.jpg"
    cover.write_bytes(b"cover")
    ebook.cover_content = b"old"
    ret = ebook.set_cover(cover_path=str(cover))
    assert ret is ebook
    assert ebook.cover_path == str(cover)
    assert ebook.cover_content is None


def test_parse_headers_valid_file(tmp_path):
    toc = tmp_path / "toc.md"
    toc.write_text(
        "\n\nMy Title\n"
        "# Chapter 1\n"
        "## Sub 1\n"
        "# Chapter 2\n",
        encoding="utf-8",
    )
    title, headers = ebooklib._parse_headers(str(toc))
    assert title == "My Title"
    assert headers == [
        {"title": "Chapter 1", "next_headers": [{"title": "Sub 1"}]},
        {"title": "Chapter 2", "next_headers": []},
    ]


def test_parse_headers_missing_file_raises(tmp_path):
    toc = tmp_path / "missing.md"
    with pytest.raises(ValueError, match="toc.md file not exists"):
        ebooklib._parse_headers(str(toc))


def test_parse_headers_empty_title_raises(tmp_path):
    toc = tmp_path / "toc.md"
    toc.write_text("\n\n   \n", encoding="utf-8")
    with pytest.raises(ValueError, match="title is empty"):
        ebooklib._parse_headers(str(toc))


def test_parse_headers_without_headings_raises(tmp_path):
    toc = tmp_path / "toc.md"
    toc.write_text("My Title\n\nJust text\n", encoding="utf-8")
    with pytest.raises(ValueError, match="headings is empty"):
        ebooklib._parse_headers(str(toc))


def test_ebookutil_headings_builds_expected_structure_and_caches(monkeypatch):
    ebook = make_ebook(monkeypatch, title="Title")
    top = ebook.create_chapter("Chapter 1")
    top.create_subchapter("Sub 1")
    top.create_subchapter("Sub 2")

    util = ebook._eu
    headings1 = util.headings
    headings2 = util.headings

    assert headings1 is headings2  # cached
    assert headings1 == [
        {
            "title": "Chapter 1",
            "play_order": 2,
            "file_name": "Chapter1.html",
            "sub_headings": [
                {"title": "Sub 1", "play_order": 3, "file_name": "Sub1.html"},
                {"title": "Sub 2", "play_order": 4, "file_name": "Sub2.html"},
            ],
        }
    ]


def test_render_toc_and_opf_create_expected_files(monkeypatch):
    ebook = make_ebook(monkeypatch, title='A/B + "C":D|E')
    ebook.create_chapter("Chapter 1")
    util = ebook._eu

    ncx = util._render_toc_ncx()
    toc = util._render_toc_html()
    opf = util._render_opf()

    assert ncx == "toc.ncx"
    assert toc == "toc.html"
    assert opf == "AB-C-DEF.opf"

    ncx_path = ebook.tmpdir / ncx
    toc_path = ebook.tmpdir / toc
    opf_path = ebook.tmpdir / opf
    assert ncx_path.is_file()
    assert toc_path.is_file()
    assert opf_path.is_file()

    assert "toc.xml" in ncx_path.read_text(encoding="utf-8")
    assert "toc.html" in toc_path.read_text(encoding="utf-8")
    assert "opf.xml" in opf_path.read_text(encoding="utf-8")


def test_save_cover_uses_default_cover_when_no_cover_set(monkeypatch):
    ebook = make_ebook(monkeypatch, title="Title")
    copied = []

    def fake_copy(src, dst):
        copied.append((str(src), str(dst)))
        # emulate copy creating destination file
        dst_path = pathlib.Path(dst) / pathlib.Path(src).name
        dst_path.write_bytes(b"copied")

    monkeypatch.setattr(ebooklib.shutil, "copy", fake_copy)
    ebook._eu._save_cover()

    assert len(copied) == 1
    assert copied[0][0].endswith(os.path.join("templates", "cover.jpg"))
    assert (ebook.tmpdir / "cover.jpg").is_file()


def test_save_chapters_writes_chapter_and_static_files(monkeypatch, tmp_path):
    ebook = make_ebook(monkeypatch, title="Title")
    chapter = ebook.create_chapter("Chapter 1")
    chapter.set_content("hello world")

    static = tmp_path / "static.txt"
    static.write_text("static", encoding="utf-8")
    chapter.add_static_files(str(static))

    copied = []

    def fake_copy(src, dst):
        copied.append((str(src), str(dst)))
        src_path = pathlib.Path(src)
        dst_path = pathlib.Path(dst) / src_path.name
        dst_path.write_bytes(src_path.read_bytes())

    monkeypatch.setattr(ebooklib.shutil, "copy", fake_copy)
    ebook._eu._save_chapters()

    chapter_file = ebook.tmpdir / "Chapter1.html"
    assert chapter_file.read_text(encoding="utf-8") == "hello world"
    assert (ebook.tmpdir / "static.txt").read_text(encoding="utf-8") == "static"
    assert copied == [(str(static), str(ebook.tmpdir))]


def test_ebook_save_calls_create_and_copies_mobi(monkeypatch, tmp_path):
    ebook = make_ebook(monkeypatch, title="My Book")
    ebook.create_chapter("Chapter 1")

    # Create a fake generated mobi inside tmpdir during _create.
    def fake_create():
        (ebook.tmpdir / ebook.dest_file_path).write_text("MOBI", encoding="utf-8")

    monkeypatch.setattr(ebook, "_create", fake_create)

    copied = []

    def fake_copy(src, dst):
        copied.append((str(src), str(dst)))
        pathlib.Path(dst).write_text(pathlib.Path(src).read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(ebooklib.shutil, "copy", fake_copy)

    out = tmp_path / "out.mobi"
    ebook.save(str(out))

    assert out.read_text(encoding="utf-8") == "MOBI"
    assert copied == [(str(ebook.tmpdir / ebook.dest_file_path), str(out))]


def test_show_calls_linux_viewer(monkeypatch):
    ebook = make_ebook(monkeypatch, title="My Book")
    ebook.create_chapter("Chapter 1")

    def fake_create():
        (ebook.tmpdir / ebook.dest_file_path).write_text("MOBI", encoding="utf-8")

    monkeypatch.setattr(ebook, "_create", fake_create)
    monkeypatch.setattr(ebooklib.platform, "system", lambda: "Linux")

    calls = []
    monkeypatch.setattr(ebooklib.subprocess, "call", lambda args: calls.append(args) or 0)

    rc = ebook.show()
    assert rc == 0
    assert calls == [["xdg-open", str(ebook.full_dest_file_path)]]


def test_show_calls_darwin_viewer(monkeypatch):
    ebook = make_ebook(monkeypatch, title="My Book")
    ebook.create_chapter("Chapter 1")

    def fake_create():
        (ebook.tmpdir / ebook.dest_file_path).write_text("MOBI", encoding="utf-8")

    monkeypatch.setattr(ebook, "_create", fake_create)
    monkeypatch.setattr(ebooklib.platform, "system", lambda: "Darwin")

    calls = []
    monkeypatch.setattr(ebooklib.subprocess, "call", lambda args: calls.append(args) or 0)

    rc = ebook.show()
    assert rc == 0
    assert calls == [["open", str(ebook.full_dest_file_path)]]


def test_show_calls_startfile_on_other_platform(monkeypatch):
    ebook = make_ebook(monkeypatch, title="My Book")
    ebook.create_chapter("Chapter 1")

    def fake_create():
        (ebook.tmpdir / ebook.dest_file_path).write_text("MOBI", encoding="utf-8")

    monkeypatch.setattr(ebook, "_create", fake_create)
    monkeypatch.setattr(ebooklib.platform, "system", lambda: "Windows")

    calls = []

    def fake_startfile(path):
        calls.append(path)
        return 123

    monkeypatch.setattr(ebooklib.os, "startfile", fake_startfile, raising=False)

    rc = ebook.show()
    assert rc == 123
    assert calls == [str(ebook.full_dest_file_path)]
