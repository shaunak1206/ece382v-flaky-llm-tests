import os
import pathlib
import shutil
import subprocess
import tempfile

import pytest

import kindle_maker.ebooklib as ebooklib


class DummyTemplate:
    def __init__(self, rendered_text):
        self.rendered_text = rendered_text

    def render(self, **context):
        # simple deterministic rendering that exposes context keys/values
        parts = [self.rendered_text]
        for k in sorted(context):
            parts.append(f"{k}={context[k]}")
        return "\n".join(parts)


def make_ebook(title="My Book", author=None):
    return ebooklib.Ebook(title, author=author)


def test_format_file_name_strips_and_replaces_illegal_chars():
    assert ebooklib.format_file_name('a/b c+d"e\\f:g|h') == 'abcd-ef-gh'


def test_ebook_dest_and_full_dest_file_path_use_formatted_title():
    eb = make_ebook('My /Title + Test:1')
    assert eb.dest_file_path == pathlib.Path('MyTitle-Test-1.mobi')
    assert eb.full_dest_file_path == eb.tmpdir / eb.dest_file_path


def test_set_cover_with_path_updates_cover_path_and_clears_content(tmp_path):
    eb = make_ebook()
    cover = tmp_path / "cover.jpg"
    cover.write_bytes(b"img")
    eb.cover_content = b"old"
    returned = eb.set_cover(cover_path=str(cover))
    assert returned is eb
    assert eb.cover_path == str(cover)
    assert eb.cover_content is None


def test_set_cover_with_content_updates_cover_content_and_clears_path():
    eb = make_ebook()
    returned = eb.set_cover(cover_content=b"bytes")
    assert returned is eb
    assert eb.cover_content == b"bytes"
    assert eb.cover_path is None


def test_set_cover_without_arguments_raises_value_error():
    eb = make_ebook()
    with pytest.raises(ValueError):
        eb.set_cover()


def test_create_chapter_adds_top_level_chapter():
    eb = make_ebook()
    chapter = eb.create_chapter("Intro")
    assert chapter in eb.chapter_list
    assert chapter.is_top_chapter is True
    assert chapter.title == "Intro"


def test_add_chapter_rejects_non_top_level_chapter():
    eb = make_ebook()
    top = eb.create_chapter("Top")
    sub = top.create_subchapter("Child")
    with pytest.raises(ValueError, match="only chapters with level 1 are accepted"):
        eb.add_chapter(sub)


def test_chapter_dest_file_paths_are_formatted():
    eb = make_ebook()
    chapter = ebooklib.Chapter('A/B + C', eb)
    assert chapter.dest_file_path == pathlib.Path("ABC-.html")
    assert chapter.full_dest_file_path == eb.tmpdir / pathlib.Path("ABC-.html")


def test_chapter_content_reads_from_file_when_file_path_is_given(tmp_path):
    eb = make_ebook()
    fp = tmp_path / "chapter.html"
    fp.write_text("hello world", encoding="utf-8")
    chapter = ebooklib.Chapter("Title", eb, file_path=str(fp))
    assert chapter.content == "hello world"


def test_chapter_set_content_overrides_file_path_and_returns_self(tmp_path):
    eb = make_ebook()
    fp = tmp_path / "chapter.html"
    fp.write_text("from file", encoding="utf-8")
    chapter = ebooklib.Chapter("Title", eb, file_path=str(fp))
    returned = chapter.set_content("from memory")
    assert returned is chapter
    assert chapter.content == "from memory"
    assert chapter._file_path is None


def test_chapter_create_subchapter_enforces_max_level():
    eb = make_ebook()
    top = eb.create_chapter("Top")
    sub = top.create_subchapter("Sub")
    with pytest.raises(Exception, match="chapter level is great than"):
        sub.create_subchapter("Too deep")


def test_chapter_save_writes_content_and_copies_static_files(tmp_path):
    eb = make_ebook()
    chapter = ebooklib.Chapter("Chapter 1", eb).set_content("<p>content</p>")

    static_file = tmp_path / "asset.txt"
    static_file.write_text("asset", encoding="utf-8")
    chapter.add_static_files(str(static_file))

    chapter.save()

    saved = chapter.full_dest_file_path
    assert saved.is_file()
    assert saved.read_text(encoding="utf-8") == "<p>content</p>"
    copied_asset = saved.parent / static_file.name
    assert copied_asset.is_file()
    assert copied_asset.read_text(encoding="utf-8") == "asset"


def test_headings_property_builds_nested_heading_structure():
    eb = make_ebook()
    ch1 = eb.create_chapter("Chapter 1")
    ch1.create_subchapter("Sub 1")
    ch2 = eb.create_chapter("Chapter 2")

    util = ebooklib.EbookUtil(eb)
    headings = util.headings

    assert headings == [
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


def test_render_toc_ncx_uses_default_author_when_missing(monkeypatch):
    eb = make_ebook(title="Book Title")
    eb.create_chapter("Chapter 1")
    util = ebooklib.EbookUtil(eb)

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
    assert captured["context"]["title"] == "Book Title"
    assert captured["context"]["author"] == "jachinlin.github.io"


def test_render_opf_uses_formatted_title_and_custom_author(monkeypatch):
    eb = make_ebook(title='A /B + C', author="Author Name")
    eb.create_chapter("Chapter 1")
    util = ebooklib.EbookUtil(eb)

    captured = {}

    def fake_render_file(template_name, context, to_filename):
        captured["template_name"] = template_name
        captured["context"] = context
        captured["to_filename"] = to_filename

    monkeypatch.setattr(util, "_render_file", fake_render_file)

    result = util._render_opf()

    assert result == "ABC-.opf"
    assert captured["template_name"] == "opf.xml"
    assert captured["to_filename"] == "ABC-.opf"
    assert captured["context"]["title"] == 'A /B + C'
    assert captured["context"]["author"] == "Author Name"


def test_save_cover_uses_default_cover_when_no_cover_is_set(monkeypatch):
    eb = make_ebook()
    util = ebooklib.EbookUtil(eb)

    copied = {}

    def fake_copy(src, dst):
        copied["src"] = src
        copied["dst"] = dst

    monkeypatch.setattr(ebooklib.shutil, "copy", fake_copy)

    util._save_cover()

    assert copied["src"].endswith(os.path.join("templates", "cover.jpg"))
    assert copied["dst"] == str(eb.tmpdir)


def test_save_cover_raises_not_implemented_for_cover_content_only():
    eb = make_ebook()
    eb.cover_content = b"bytes"
    util = ebooklib.EbookUtil(eb)
    with pytest.raises(NotImplementedError):
        util._save_cover()


def test_parse_headers_valid_and_invalid_inputs(tmp_path):
    toc = tmp_path / "toc.md"
    toc.write_text(
        "\n\nMy Title\n"
        "# Chapter 1\n"
        "## Section 1\n"
        "# Chapter 2\n",
        encoding="utf-8",
    )
    title, headers = ebooklib._parse_headers(str(toc))
    assert title == "My Title"
    assert headers == [
        {"title": "Chapter 1", "next_headers": [{"title": "Section 1"}]},
        {"title": "Chapter 2", "next_headers": []},
    ]

    missing = tmp_path / "missing.md"
    with pytest.raises(ValueError, match="toc.md file not exists"):
        ebooklib._parse_headers(str(missing))

    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid toc md file: title is empty"):
        ebooklib._parse_headers(str(empty))


def test_make_ebook_deprecated_api_creates_output_file_and_copies_assets(tmp_path, monkeypatch):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()

    (source / "toc.md").write_text(
        "Book Title\n"
        "# Chapter 1\n"
        "## Sub 1\n",
        encoding="utf-8",
    )
    (source / "cover.jpg").write_bytes(b"cover-bytes")
    (source / "extra.css").write_text("body{}", encoding="utf-8")
    (source / "Chapter 1.html").write_text("<h1>Chapter 1</h1>", encoding="utf-8")
    (source / "Sub 1.html").write_text("<h2>Sub 1</h2>", encoding="utf-8")

    # Avoid invoking external kindlegen; emulate a successful mobi build.
    def fake_create(self):
        dest = self.tmpdir / self.dest_file_path
        dest.write_text("mobi-bytes", encoding="utf-8")

    monkeypatch.setattr(ebooklib.Ebook, "_create", fake_create)

    out_file = ebooklib.make_ebook(str(source), str(output))

    expected = output / "Book Title.mobi"
    assert out_file == str(expected)
    assert expected.is_file()
    assert expected.read_text(encoding="utf-8") == "mobi-bytes"
