import pathlib
import types

import pytest

from kindle_maker.ebooklib import (
    Ebook,
    EbookUtil,
    Chapter,
    format_file_name,
)


@pytest.fixture
def ebook(tmp_path, monkeypatch):
    # Make temp dir deterministic and isolated for tests.
    monkeypatch.setattr(
        "kindle_maker.ebooklib.tempfile.gettempdir", lambda: str(tmp_path)
    )
    monkeypatch.setattr("kindle_maker.ebooklib.time.time", lambda: 123456.0)
    e = Ebook("My Book", author="Alice")
    yield e
    # Avoid relying on __del__ timing; clean up explicitly.
    if e.tmpdir.exists():
        for p in sorted(e.tmpdir.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
        if e.tmpdir.exists():
            e.tmpdir.rmdir()


def test_format_file_name_strips_and_replaces_special_chars():
    assert format_file_name('a/b c+d"e\\f:g|h') == "abcd-e-f-g-h"


def test_ebook_dest_and_full_dest_file_paths(ebook):
    assert ebook.dest_file_path.name == "MyBook.mobi"
    assert ebook.full_dest_file_path == ebook.tmpdir / "MyBook.mobi"


def test_create_chapter_adds_top_level_chapter(ebook):
    ch = ebook.create_chapter("Chapter 1")
    assert isinstance(ch, Chapter)
    assert ebook.chapter_list == [ch]
    assert ch.is_top_chapter is True
    assert ch.dest_file_path.name == "Chapter1.html"


def test_add_chapter_rejects_non_top_level_chapter(ebook):
    ch = Chapter("Child", ebook, level=2)
    with pytest.raises(ValueError, match="only chapters with level 1 are accepted"):
        ebook.add_chapter(ch)


def test_set_cover_accepts_cover_path_and_clears_cover_content(ebook, tmp_path):
    cover = tmp_path / "cover.jpg"
    cover.write_bytes(b"img")
    ebook.cover_content = b"old"
    result = ebook.set_cover(cover_path=str(cover))
    assert result is ebook
    assert ebook.cover_path == str(cover)
    assert ebook.cover_content is None


def test_set_cover_accepts_cover_content_and_clears_cover_path(ebook):
    ebook.cover_path = "something.jpg"
    result = ebook.set_cover(cover_content=b"bytes")
    assert result is ebook
    assert ebook.cover_content == b"bytes"
    assert ebook.cover_path is None


def test_set_cover_rejects_empty_arguments(ebook):
    with pytest.raises(ValueError, match="img_path and img_content are both empty"):
        ebook.set_cover()


def test_chapter_init_rejects_missing_file_path(ebook, tmp_path):
    missing = tmp_path / "missing.html"
    with pytest.raises(ValueError, match="is not a file"):
        Chapter("X", ebook, file_path=str(missing))


def test_chapter_content_reads_from_file_path(ebook, tmp_path):
    src = tmp_path / "chapter.html"
    src.write_text("hello", encoding="utf-8")
    ch = Chapter("X", ebook, file_path=str(src))
    assert ch.content == "hello"


def test_chapter_set_content_overrides_file_path_and_returns_self(ebook):
    ch = Chapter("X", ebook)
    out = ch.set_content("abc")
    assert out is ch
    assert ch.content == "abc"


def test_add_static_files_validates_files_and_save_copies_them(ebook, tmp_path):
    ch = ebook.create_chapter("Chapter 1").set_content("chapter body")
    static_file = tmp_path / "style.css"
    static_file.write_text("body{}", encoding="utf-8")
    ch.add_static_files(str(static_file))

    ch.save()

    assert (ebook.tmpdir / "Chapter1.html").read_text(encoding="utf-8") == "chapter body"
    assert (ebook.tmpdir / "style.css").is_file()
    assert (ebook.tmpdir / "style.css").read_text(encoding="utf-8") == "body{}"


def test_create_subchapter_respects_max_level(ebook):
    parent = ebook.create_chapter("Chapter 1")
    sub = parent.create_subchapter("Subchapter 1")
    assert sub.is_top_chapter is False
    assert sub in parent.sub_chapters
    with pytest.raises(Exception, match="chapter level is great than"):
        sub.create_subchapter("Too deep")


def test_ebookutil_headings_are_generated_and_cached(ebook):
    c1 = ebook.create_chapter("Chapter 1")
    c1.create_subchapter("Section 1.1")
    c2 = ebook.create_chapter("Chapter 2")

    util = EbookUtil(ebook)
    headings_1 = util.headings
    headings_2 = util.headings

    assert headings_1 is headings_2
    assert headings_1 == [
        {
            "title": "Chapter 1",
            "play_order": 2,
            "file_name": str(c1.dest_file_path),
            "sub_headings": [
                {
                    "title": "Section 1.1",
                    "play_order": 3,
                    "file_name": str(c1.sub_chapters[0].dest_file_path),
                }
            ],
        },
        {
            "title": "Chapter 2",
            "play_order": 4,
            "file_name": str(c2.dest_file_path),
            "sub_headings": [],
        },
    ]


def test_ebookutil_render_file_writes_template_output(ebook, monkeypatch):
    util = EbookUtil(ebook)

    class DummyTemplate:
        def render(self, **context):
            return f"{context['title']}|{context['author']}"

    util._templates_env = types.SimpleNamespace(
        get_template=lambda name: DummyTemplate()
    )
    out_name = "out.txt"
    util._render_file("dummy.tpl", {"title": "T", "author": "A"}, out_name)

    assert (ebook.tmpdir / out_name).read_text(encoding="utf-8") == "T|A"


def test_render_toc_ncx_uses_default_author_when_missing(ebook, monkeypatch):
    ebook.author = None
    util = EbookUtil(ebook)
    util._headings = [{"title": "Chap", "play_order": 1, "file_name": "x", "sub_headings": []}]

    captured = {}

    class DummyTemplate:
        def render(self, **context):
            captured.update(context)
            return "ncx"

    util._templates_env = types.SimpleNamespace(
        get_template=lambda name: DummyTemplate()
    )

    assert util._render_toc_ncx() == "toc.ncx"
    assert captured["author"] == "jachinlin.github.io"
    assert (ebook.tmpdir / "toc.ncx").read_text(encoding="utf-8") == "ncx"


def test_ebook_create_raises_when_no_chapters(ebook):
    with pytest.raises(ValueError, match="chapter list is empty"):
        ebook._create()


def test_ebook_create_invokes_util_and_kindlegen(monkeypatch, ebook, tmp_path):
    # Prepare a fake opf file that kindlegen will receive.
    util = EbookUtil(ebook)

    calls = []

    def fake_render_toc_ncx():
        calls.append("ncx")
        (ebook.tmpdir / "toc.ncx").write_text("ncx", encoding="utf-8")
        return "toc.ncx"

    def fake_render_toc_html():
        calls.append("html")
        (ebook.tmpdir / "toc.html").write_text("html", encoding="utf-8")
        return "toc.html"

    def fake_render_opf():
        calls.append("opf")
        opf = "MyBook.opf"
        (ebook.tmpdir / opf).write_text("opf", encoding="utf-8")
        return opf

    def fake_save_cover():
        calls.append("cover")
        (ebook.tmpdir / "cover.jpg").write_bytes(b"cover")

    def fake_save_chapters():
        calls.append("chapters")
        (ebook.tmpdir / "Chapter1.html").write_text("chapter", encoding="utf-8")

    util._render_toc_ncx = fake_render_toc_ncx
    util._render_toc_html = fake_render_toc_html
    util._render_opf = fake_render_opf
    util._save_cover = fake_save_cover
    util._save_chapters = fake_save_chapters
    ebook._eu = util

    monkeypatch.setattr("kindle_maker.ebooklib.subprocess.call", lambda cmd: 0)
    monkeypatch.setattr("kindle_maker.ebooklib.kindlegen", "/fake/kindlegen")

    ebook.create_chapter("Chapter 1").set_content("body")
    ebook._create()

    assert calls == ["ncx", "html", "opf", "cover", "chapters"]


def test_ebook_save_copies_generated_mobi_to_destination(monkeypatch, ebook, tmp_path):
    ebook.create_chapter("Chapter 1").set_content("body")

    fake_mobi = ebook.tmpdir / ebook.dest_file_path
    fake_mobi.write_bytes(b"mobi-bytes")

    monkeypatch.setattr(ebook, "_create", lambda: None)

    dst = tmp_path / "out.mobi"
    ebook.save(str(dst))

    assert dst.read_bytes() == b"mobi-bytes"
