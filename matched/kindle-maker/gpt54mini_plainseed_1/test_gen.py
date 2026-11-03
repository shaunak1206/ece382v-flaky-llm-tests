import io
import os
import pathlib
import shutil
import tempfile

import pytest

import kindle_maker.ebooklib as ebooklib


class DummyTemplate:
    def __init__(self, name):
        self.name = name

    def render(self, **context):
        # simple deterministic output that includes template name and context keys
        return f"{self.name}:{sorted(context.keys())}"


class DummyEnv:
    def get_template(self, template_name):
        return DummyTemplate(template_name)


@pytest.fixture
def ebook(tmp_path, monkeypatch):
    # Avoid relying on the package templates directory for unit tests.
    monkeypatch.setattr(ebooklib, "Environment", lambda loader=None: DummyEnv())
    monkeypatch.setattr(ebooklib, "FileSystemLoader", lambda *args, **kwargs: object())
    e = ebooklib.Ebook("My Book", author="Alice")
    # Put tempdir under pytest temp for deterministic cleanup and easier assertions.
    new_tmp = tmp_path / "ebooktmp"
    new_tmp.mkdir()
    e._tempdir = new_tmp
    e._eu = ebooklib.EbookUtil(e)
    return e


def test_format_file_name_replaces_illegal_characters():
    assert ebooklib.format_file_name(r'a/b c+"d\e:f|g') == "abc-d-e-f-g"


def test_chapter_dest_and_full_dest_file_path(ebook):
    c = ebooklib.Chapter("Hello World", ebook)
    assert c.dest_file_path == pathlib.Path("HelloWorld.html")
    assert c.full_dest_file_path == ebook.tmpdir / "HelloWorld.html"


def test_chapter_set_content_and_is_top_chapter(ebook):
    c = ebooklib.Chapter("Intro", ebook)
    assert c.is_top_chapter is True
    c.set_content("chapter body")
    assert c.content == "chapter body"


def test_chapter_content_reads_from_file(tmp_path, ebook):
    f = tmp_path / "src.html"
    f.write_text("file content", encoding="utf-8")
    c = ebooklib.Chapter("FromFile", ebook, file_path=str(f))
    assert c.content == "file content"


def test_chapter_init_raises_for_nonexistent_file(ebook):
    with pytest.raises(ValueError, match="is not a file"):
        ebooklib.Chapter("Bad", ebook, file_path="does-not-exist.html")


def test_add_static_files_requires_existing_files(tmp_path, ebook):
    c = ebooklib.Chapter("Static", ebook)
    existing = tmp_path / "asset.css"
    existing.write_text("body{}", encoding="utf-8")
    c.add_static_files(str(existing))
    assert c._static_files == [str(existing)]
    with pytest.raises(AssertionError, match="is not a file"):
        c.add_static_files(str(tmp_path / "missing.css"))


def test_create_subchapter_increments_level_and_appends(ebook):
    c = ebooklib.Chapter("Parent", ebook)
    sc = c.create_subchapter("Child")
    assert sc.title == "Child"
    assert sc._level == 2
    assert c.sub_chapters == [sc]
    assert sc.is_top_chapter is False


def test_create_subchapter_raises_beyond_max_level(ebook):
    c = ebooklib.Chapter("Parent", ebook)
    sc = c.create_subchapter("Child")
    with pytest.raises(Exception, match="max level num"):
        sc.create_subchapter("Grandchild")


def test_save_writes_content_and_copies_static_files(tmp_path, ebook, monkeypatch):
    src = tmp_path / "extra.txt"
    src.write_text("x", encoding="utf-8")

    c = ebooklib.Chapter("Save Me", ebook)
    c.set_content("saved content")
    c.add_static_files(str(src))

    copied = []

    def fake_copy(src_path, dst_path):
        copied.append((src_path, dst_path))
        # simulate copy into destination directory
        shutil.copyfile(src_path, os.path.join(dst_path, os.path.basename(src_path)))

    monkeypatch.setattr(ebooklib.shutil, "copy", fake_copy)
    c.save()

    assert (ebook.tmpdir / "SaveMe.html").read_text(encoding="utf-8") == "saved content"
    assert copied and copied[0][0] == str(src)


def test_ebook_add_chapter_rejects_non_top_chapter(ebook):
    c = ebooklib.Chapter("Parent", ebook)
    sc = c.create_subchapter("Child")
    with pytest.raises(ValueError, match="only chapters with level 1"):
        ebook.add_chapter(sc)


def test_ebook_create_chapter_adds_top_level_chapter(ebook):
    c = ebook.create_chapter("Top Chapter")
    assert c in ebook.chapter_list
    assert c.title == "Top Chapter"
    assert c.is_top_chapter is True


def test_ebook_create_raises_when_no_chapters(ebook):
    with pytest.raises(ValueError, match="chapter list is empty"):
        ebook._create()


def test_ebookutil_headings_caches_and_includes_subchapters(ebook):
    c1 = ebook.create_chapter("Chapter 1")
    c1.create_subchapter("Sub 1")
    c2 = ebook.create_chapter("Chapter 2")

    hs1 = ebook._eu.headings
    hs2 = ebook._eu.headings

    assert hs1 is hs2
    assert hs1[0]["title"] == "Chapter 1"
    assert hs1[0]["play_order"] == 2
    assert hs1[0]["sub_headings"][0]["title"] == "Sub 1"
    assert hs1[0]["sub_headings"][0]["play_order"] == 3
    assert hs1[1]["title"] == "Chapter 2"
    assert hs1[1]["play_order"] == 4
    assert hs1[1]["sub_headings"] == []


def test_ebookutil_render_toc_and_opf_files(tmp_path, ebook):
    ebook.create_chapter("Chapter 1")
    ebook.author = None  # exercise default author value

    ncx = ebook._eu._render_toc_ncx()
    html = ebook._eu._render_toc_html()
    opf = ebook._eu._render_opf()

    assert ncx == "toc.ncx"
    assert html == "toc.html"
    assert opf == "MyBook.opf"

    assert (ebook.tmpdir / "toc.ncx").exists()
    assert (ebook.tmpdir / "toc.html").exists()
    assert (ebook.tmpdir / "MyBook.opf").exists()

    ncx_text = (ebook.tmpdir / "toc.ncx").read_text(encoding="utf-8")
    assert "toc.xml" in ncx_text
    assert "headings" in ncx_text
    assert "author" in ncx_text
    assert "jachinlin.github.io" in ncx_text


def test_ebookutil_save_cover_prefers_cover_path(tmp_path, ebook, monkeypatch):
    cover = tmp_path / "cover.jpg"
    cover.write_bytes(b"cover-bytes")
    ebook.cover_path = str(cover)

    copied = {}

    def fake_copy(src_path, dst_path):
        copied["src"] = src_path
        copied["dst"] = dst_path

    monkeypatch.setattr(ebooklib.shutil, "copy", fake_copy)
    ebook._eu._save_cover()

    assert copied["src"] == str(cover)
    assert copied["dst"] == str(ebook.tmpdir)


def test_ebookutil_save_cover_uses_default_cover_when_none(tmp_path, ebook, monkeypatch):
    copied = {}

    def fake_copy(src_path, dst_path):
        copied["src"] = src_path
        copied["dst"] = dst_path

    monkeypatch.setattr(ebooklib.shutil, "copy", fake_copy)
    ebook._eu._save_cover()

    assert copied["src"].endswith(os.path.join("templates", "cover.jpg"))
    assert copied["dst"] == str(ebook.tmpdir)


def test_ebookutil_save_cover_raises_for_cover_content(ebook):
    ebook.cover_content = b"abc"
    with pytest.raises(NotImplementedError):
        ebook._eu._save_cover()


def test_ebook_create_invokes_kindlegen_and_save_chapters(tmp_path, ebook, monkeypatch):
    c = ebook.create_chapter("Chapter 1")
    c.set_content("body")

    calls = []

    monkeypatch.setattr(ebooklib.subprocess, "call", lambda cmd: calls.append(cmd) or 0)

    def fake_render_file(template_name, context, to_filename):
        (ebook.tmpdir / to_filename).write_text(template_name, encoding="utf-8")

    monkeypatch.setattr(ebook._eu, "_render_file", fake_render_file)

    saved = []

    def fake_save():
        saved.append(True)

    monkeypatch.setattr(ebooklib.Chapter, "save", lambda self: fake_save())

    ebook._create()

    assert saved
    assert calls and calls[0][0] == ebooklib.kindlegen
    assert calls[0][1] == "-dont_append_source"
    assert calls[0][2].endswith(".opf")


def test_ebook_save_copies_mobi_to_destination(tmp_path, ebook, monkeypatch):
    c = ebook.create_chapter("Chapter 1")
    c.set_content("body")

    # Create the expected mobi file in tempdir.
    (ebook.tmpdir / ebook.dest_file_path).write_text("mobi-bytes", encoding="utf-8")

    monkeypatch.setattr(ebook, "_create", lambda: None)

    copied = {}

    def fake_copy(src, dst):
        copied["src"] = src
        copied["dst"] = dst

    monkeypatch.setattr(ebooklib.shutil, "copy", fake_copy)

    out = tmp_path / "out.mobi"
    ebook.save(str(out))

    assert copied["src"] == str(ebook.tmpdir / ebook.dest_file_path)
    assert copied["dst"] == str(out)


def test_ebook_show_uses_platform_specific_openers(tmp_path, ebook, monkeypatch):
    c = ebook.create_chapter("Chapter 1")
    c.set_content("body")
    (ebook.tmpdir / ebook.dest_file_path).write_text("mobi-bytes", encoding="utf-8")
    monkeypatch.setattr(ebook, "_create", lambda: None)

    monkeypatch.setattr(ebooklib.platform, "system", lambda: "Linux")
    linux_calls = []
    monkeypatch.setattr(ebooklib.subprocess, "call", lambda cmd: linux_calls.append(cmd) or 0)
    assert ebook.show() == 0
    assert linux_calls[0][0] == "xdg-open"

    monkeypatch.setattr(ebooklib.platform, "system", lambda: "Darwin")
    darwin_calls = []
    monkeypatch.setattr(ebooklib.subprocess, "call", lambda cmd: darwin_calls.append(cmd) or 0)
    assert ebook.show() == 0
    assert darwin_calls[0][0] == "open"

    monkeypatch.setattr(ebooklib.platform, "system", lambda: "Windows")
    monkeypatch.setattr(ebooklib.os, "startfile", lambda path: 123, raising=False)
    assert ebook.show() == 123


def test_parse_headers_reads_title_and_hierarchy(tmp_path):
    toc = tmp_path / "toc.md"
    toc.write_text(
        "\n\nMy Title\n# Chapter A\n## Sub A1\n# Chapter B\n",
        encoding="utf-8",
    )
    title, headers = ebooklib._parse_headers(str(toc))
    assert title == "My Title"
    assert headers == [
        {"title": "Chapter A", "next_headers": [{"title": "Sub A1"}]},
        {"title": "Chapter B", "next_headers": []},
    ]


def test_parse_headers_raises_on_missing_file(tmp_path):
    with pytest.raises(ValueError, match="toc.md file not exists"):
        ebooklib._parse_headers(str(tmp_path / "missing.md"))


def test_make_ebook_warns_and_returns_mobi_path(tmp_path, monkeypatch):
    source_dir = tmp_path / "src"
    output_dir = tmp_path / "out"
    source_dir.mkdir()
    output_dir.mkdir()

    (source_dir / "toc.md").write_text("Book\n# Chapter 1\n## Sub 1\n", encoding="utf-8")
    (source_dir / "cover.jpg").write_bytes(b"cover")
    (source_dir / "Chapter 1.html").write_text("chapter", encoding="utf-8")
    (source_dir / "Sub 1.html").write_text("subchapter", encoding="utf-8")
    (source_dir / "asset.css").write_text("css", encoding="utf-8")

    monkeypatch.setattr(ebooklib.subprocess, "call", lambda cmd: 0)

    # Ensure save() can copy the generated mobi file.
    real_save = ebooklib.Ebook.save

    def fake_save(self, file_path):
        (self.tmpdir / self.dest_file_path).write_text("mobi", encoding="utf-8")
        return real_save(self, file_path)

    monkeypatch.setattr(ebooklib.Ebook, "save", fake_save)

    with pytest.warns(UserWarning):
        out = ebooklib.make_ebook(str(source_dir), str(output_dir))

    assert out == str(output_dir / "Book.mobi")
    assert (output_dir / "Book.mobi").read_text(encoding="utf-8") == "mobi"
