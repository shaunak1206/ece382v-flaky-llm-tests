import os
import pathlib
import tempfile
import shutil

import pytest

import kindle_maker.ebooklib as ebooklib


@pytest.fixture
def temp_dir():
    d = pathlib.Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(str(d), ignore_errors=True)


@pytest.fixture
def ebook(tmp_path):
    e = ebooklib.Ebook("My Book", author="Alice")
    yield e
    shutil.rmtree(str(e.tmpdir), ignore_errors=True)


def test_format_file_name_strips_and_replaces_invalid_chars():
    assert ebooklib.format_file_name('a/b c+"d\\e:f|g') == 'abcd-ef-g'


def test_chapter_dest_and_full_dest_file_path_use_sanitized_title(ebook):
    c = ebooklib.Chapter('Chap 1/2 + "x"', ebook)
    assert c.dest_file_path == pathlib.Path('Chap12-x.html')
    assert c.full_dest_file_path == ebook.tmpdir / pathlib.Path('Chap12-x.html')


def test_chapter_set_content_clears_file_path(ebook):
    c = ebooklib.Chapter("Intro", ebook, file_path=None)
    c.set_content("hello")
    assert c.content == "hello"
    assert c._file_path is None


def test_chapter_reads_content_from_file(temp_dir, ebook):
    p = temp_dir / "chapter.html"
    p.write_text("file content", encoding="utf-8")
    c = ebooklib.Chapter("File Chapter", ebook, file_path=str(p))
    assert c.content == "file content"


def test_chapter_add_static_files_accepts_existing_files(temp_dir, ebook):
    p1 = temp_dir / "a.css"
    p2 = temp_dir / "b.js"
    p1.write_text("x", encoding="utf-8")
    p2.write_text("y", encoding="utf-8")
    c = ebooklib.Chapter("Static", ebook)
    c.add_static_files(str(p1), str(p2))
    assert c._static_files == [str(p1), str(p2)]


def test_chapter_add_static_files_raises_for_missing_file(ebook):
    c = ebooklib.Chapter("Static", ebook)
    with pytest.raises(AssertionError):
        c.add_static_files("/does/not/exist.css")


def test_create_subchapter_enforces_max_level(ebook):
    top = ebooklib.Chapter("Top", ebook, level=2)
    with pytest.raises(Exception, match="max level num"):
        top.create_subchapter("Too deep")


def test_create_subchapter_sets_level_and_links_to_parent(ebook):
    top = ebooklib.Chapter("Top", ebook)
    sub = top.create_subchapter("Sub")
    assert sub._level == 2
    assert sub.title == "Sub"
    assert sub in top.sub_chapters
    assert not sub.is_top_chapter


def test_ebook_set_cover_accepts_cover_path_and_cover_content(ebook):
    ebook.set_cover(cover_path="/tmp/cover.jpg")
    assert ebook.cover_path == "/tmp/cover.jpg"
    assert ebook.cover_content is None

    ebook.set_cover(cover_content=b"bytes")
    assert ebook.cover_content == b"bytes"
    assert ebook.cover_path is None


def test_ebook_set_cover_rejects_empty_values(ebook):
    with pytest.raises(ValueError, match="img_path and img_content are both empty"):
        ebook.set_cover()


def test_ebook_add_chapter_rejects_non_top_level_chapter(ebook):
    sub = ebooklib.Chapter("Sub", ebook, level=2)
    with pytest.raises(ValueError, match="only chapters with level 1 are accepted"):
        ebook.add_chapter(sub)


def test_ebook_create_chapter_adds_top_level_chapter(ebook):
    ch = ebook.create_chapter("Intro")
    assert ch in ebook.chapter_list
    assert ch.is_top_chapter
    assert ch.title == "Intro"


def test_ebook_create_raises_when_chapter_list_empty(ebook):
    with pytest.raises(ValueError, match="chapter list is empty"):
        ebook._create()


def test_ebookutil_headings_builds_nested_structure_and_caches(ebook):
    c1 = ebook.create_chapter("Chapter 1")
    s1 = c1.create_subchapter("Section 1.1")
    c2 = ebook.create_chapter("Chapter 2")

    util = ebooklib.EbookUtil(ebook)
    headings1 = util.headings
    headings2 = util.headings

    assert headings1 is headings2
    assert headings1 == [
        {
            "title": "Chapter 1",
            "play_order": 2,
            "file_name": str(c1.dest_file_path),
            "sub_headings": [
                {
                    "title": "Section 1.1",
                    "play_order": 3,
                    "file_name": str(s1.dest_file_path),
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


def test_ebookutil_render_opf_uses_default_author_when_missing(ebook, monkeypatch):
    ebook.author = None
    util = ebooklib.EbookUtil(ebook)

    captured = {}

    def fake_render_file(template_name, context, to_filename):
        captured["template_name"] = template_name
        captured["context"] = context
        captured["to_filename"] = to_filename

    monkeypatch.setattr(util, "_render_file", fake_render_file)

    result = util._render_opf()
    assert result == "MyBook.opf"
    assert captured["template_name"] == "opf.xml"
    assert captured["to_filename"] == "MyBook.opf"
    assert captured["context"]["author"] == "jachinlin.github.io"
    assert captured["context"]["title"] == "My Book"


def test_ebookutil_save_cover_uses_default_cover_when_none_set(ebook, monkeypatch):
    util = ebooklib.EbookUtil(ebook)

    copied = []

    def fake_copy(src, dst):
        copied.append((str(src), str(dst)))

    monkeypatch.setattr(ebooklib.shutil, "copy", fake_copy)

    util._save_cover()
    assert len(copied) == 1
    assert copied[0][1] == str(ebook.tmpdir)
    assert copied[0][0].endswith(os.path.join("templates", "cover.jpg"))


def test_ebookutil_save_cover_copies_custom_cover(ebook, temp_dir, monkeypatch):
    cover = temp_dir / "cover.jpg"
    cover.write_bytes(b"img")
    ebook.set_cover(cover_path=str(cover))

    util = ebooklib.EbookUtil(ebook)
    copied = []

    def fake_copy(src, dst):
        copied.append((str(src), str(dst)))

    monkeypatch.setattr(ebooklib.shutil, "copy", fake_copy)

    util._save_cover()
    assert copied == [(str(cover), str(ebook.tmpdir))]


def test_ebook_save_invokes_create_and_copies_mobi(monkeypatch, ebook, temp_dir):
    created = []

    def fake_create():
        created.append(True)
        out = ebook.tmpdir / ebook.dest_file_path
        out.write_text("mobi", encoding="utf-8")

    monkeypatch.setattr(ebook, "_create", fake_create)

    copied = []

    def fake_copy(src, dst):
        copied.append((str(src), str(dst)))

    monkeypatch.setattr(ebooklib.shutil, "copy", fake_copy)

    out_file = temp_dir / "out.mobi"
    ebook.save(str(out_file))

    assert created == [True]
    assert copied == [(str(ebook.tmpdir / ebook.dest_file_path), str(out_file))]


def test_ebook_show_uses_linux_viewer(monkeypatch, ebook):
    monkeypatch.setattr(ebook, "_create", lambda: None)
    monkeypatch.setattr(ebooklib.platform, "system", lambda: "Linux")

    calls = []

    def fake_call(args):
        calls.append(args)
        return 0

    monkeypatch.setattr(ebooklib.subprocess, "call", fake_call)

    rc = ebook.show()
    assert rc == 0
    assert calls == [["xdg-open", str(ebook.full_dest_file_path)]]


def test_parse_headers_parses_title_and_nested_headers(temp_dir):
    toc = temp_dir / "toc.md"
    toc.write_text(
        "\n".join(
            [
                "",
                "My Title",
                "# Chapter 1",
                "## Section 1.1",
                "# Chapter 2",
            ]
        ),
        encoding="utf-8",
    )

    title, headers = ebooklib._parse_headers(str(toc))
    assert title == "My Title"
    assert headers == [
        {"title": "Chapter 1", "next_headers": [{"title": "Section 1.1"}]},
        {"title": "Chapter 2", "next_headers": []},
    ]


def test_parse_headers_raises_for_missing_file():
    with pytest.raises(ValueError, match="toc.md file not exists"):
        ebooklib._parse_headers("/no/such/toc.md")


def test_parse_headers_raises_for_empty_title(temp_dir):
    toc = temp_dir / "toc.md"
    toc.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="title is empty"):
        ebooklib._parse_headers(str(toc))


def test_make_ebook_builds_and_saves_output(monkeypatch, temp_dir):
    source = temp_dir / "src"
    output = temp_dir / "out"
    source.mkdir()
    output.mkdir()

    (source / "toc.md").write_text(
        "\n".join(["Title", "# Chapter A", "## Section A.1"]),
        encoding="utf-8",
    )
    (source / "cover.jpg").write_bytes(b"cover")
    (source / "Chapter A.html").write_text("chapter", encoding="utf-8")
    (source / "Section A.1.html").write_text("section", encoding="utf-8")
    (source / "asset.css").write_text("css", encoding="utf-8")

    # Avoid invoking templates and external kindlegen
    monkeypatch.setattr(ebooklib.EbookUtil, "_render_toc_ncx", lambda self: "toc.ncx")
    monkeypatch.setattr(ebooklib.EbookUtil, "_render_toc_html", lambda self: "toc.html")
    monkeypatch.setattr(ebooklib.EbookUtil, "_render_opf", lambda self: "Title.opf")
    monkeypatch.setattr(ebooklib.EbookUtil, "_save_cover", lambda self: None)
    monkeypatch.setattr(ebooklib.EbookUtil, "_save_chapters", lambda self: None)

    copied = []

    def fake_copy(src, dst):
        copied.append((str(src), str(dst)))

    monkeypatch.setattr(ebooklib.shutil, "copy", fake_copy)
    monkeypatch.setattr(ebooklib.os.path, "isfile", lambda p: True)
    monkeypatch.setattr(ebooklib.subprocess, "call", lambda args: 0)

    result = ebooklib.make_ebook(str(source), str(output))
    assert result == str(output / "Title.mobi")
    assert any(src.endswith("asset.css") for src, _ in copied)
    assert any(dst == str(output / "Title.mobi") for _, dst in copied)
