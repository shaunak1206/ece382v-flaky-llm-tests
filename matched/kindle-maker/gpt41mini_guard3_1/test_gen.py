import io
import os
import pathlib
import random
import shutil
import string
import tempfile
import time

import pytest

from kindle_maker import ebooklib
from kindle_maker.ebooklib import Ebook, EbookUtil, Chapter


def _create_temp_file_with_content(tmp_path, content="example content"):
    file_path = tmp_path / "testfile.txt"
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


def _create_temp_html_file(tmp_path, name="chapter"):
    content = f"<html><body><h1>{name}</h1></body></html>"
    file_path = tmp_path / f"{name}.html"
    file_path.write_text(content, encoding="utf-8")
    return str(file_path), content


def _seed_all():
    # seed std lib random since code uses none explicitly here but we must do as per instructions
    random.seed(1234)


def test_format_file_name_basic():
    _seed_all()
    s = "a/b c+d\"e\\f:g|h"
    expected = "abcdefgh"
    replaced = ebooklib.format_file_name(s)
    assert "/" not in replaced
    assert " " not in replaced
    assert '"' not in replaced
    assert "\\" not in replaced
    assert ":" not in replaced
    assert "|" not in replaced
    # + replaced with -
    assert "-" in replaced
    # plus sign replaced by a dash
    assert replaced.count("-") == 1


def test_chapter_init_and_properties(tmp_path):
    _seed_all()
    # valid file_path
    fpath, content = _create_temp_html_file(tmp_path, "ch1")
    ch = Chapter("Title", ebook=Ebook("Book"), file_path=fpath)
    assert ch.title == "Title"
    assert ch.is_top_chapter is True
    assert ch.content == content
    assert ch.dest_file_path.name.endswith(".html")
    assert ch._level == 1
    assert ch.sub_chapters == []


def test_chapter_init_invalid_file_path(tmp_path):
    _seed_all()
    invalid_file = tmp_path / "notexist.html"
    with pytest.raises(ValueError):
        Chapter("T", ebook=Ebook("Book"), file_path=str(invalid_file))


def test_chapter_set_content_and_content_property(tmp_path):
    _seed_all()
    ch = Chapter("T", ebook=Ebook("Book"))
    text = "some new content"
    ch.set_content(text)
    assert ch.content == text
    # file_path cleared after set_content
    assert ch._file_path is None


def test_chapter_add_static_files_valid(tmp_path):
    _seed_all()
    ch = Chapter("T", ebook=Ebook("Book"))
    f1 = _create_temp_file_with_content(tmp_path, "data1")
    f2 = _create_temp_file_with_content(tmp_path, "data2")
    result = ch.add_static_files(f1, f2)
    assert result is ch
    assert len(ch._static_files) == 2
    assert all(str(tmp_path) in f for f in ch._static_files)


def test_chapter_add_static_files_invalid_file(tmp_path):
    _seed_all()
    ch = Chapter("T", ebook=Ebook("Book"))
    invalid_path = tmp_path / "nope.txt"
    with pytest.raises(AssertionError):
        ch.add_static_files(str(invalid_path))


def test_chapter_create_subchapter_and_levels(tmp_path):
    _seed_all()
    eb = Ebook("Book")
    ch1 = Chapter("Top Level", ebook=eb)
    subch = ch1.create_subchapter("Sub chapter")
    assert isinstance(subch, Chapter)
    assert subch._level == 2
    # adding third-level chapter should raise exception (max_level=2)
    with pytest.raises(Exception):
        subch.create_subchapter("Too deep")


def test_chapter_save_creates_file_and_copies_static_files(tmp_path):
    _seed_all()
    eb = Ebook("Book")
    eb._tempdir = tmp_path
    # Create chapter with content and static file
    ch = Chapter("MyChapter", ebook=eb).set_content("Hello World")
    # Create static file
    sf_path = _create_temp_file_with_content(tmp_path, "static data")
    ch.add_static_files(sf_path)
    ch.save()
    dest_fp = ch.full_dest_file_path
    # File saved and contains content
    assert dest_fp.is_file()
    saved_content = dest_fp.read_text(encoding="utf-8")
    assert saved_content == "Hello World"
    # Static file copied into tempdir (same folder as dest file)
    copied_static = dest_fp.parent / pathlib.Path(sf_path).name
    assert copied_static.is_file()
    copied_content = copied_static.read_text(encoding="utf-8")
    assert copied_content == "static data"


def test_ebook_set_cover_behaviour(tmp_path):
    _seed_all()
    eb = Ebook("Title")
    file_cover = _create_temp_file_with_content(tmp_path, "cover data")
    # path cover
    res = eb.set_cover(cover_path=file_cover)
    assert res is eb
    assert eb.cover_path == file_cover
    assert eb.cover_content is None
    # content cover
    res2 = eb.set_cover(cover_content=b"abc")
    assert res2 is eb
    assert eb.cover_path is None
    assert eb.cover_content == b"abc"
    # no args raises
    with pytest.raises(ValueError):
        eb.set_cover()


def test_ebook_add_chapter_accepts_only_top_level(tmp_path):
    _seed_all()
    eb = Ebook("Title")
    ch1 = Chapter("Top1", eb)
    eb.add_chapter(ch1)
    assert len(eb.chapter_list) == 1
    ch2 = Chapter("Lvl2", eb, level=2)
    with pytest.raises(ValueError):
        eb.add_chapter(ch2)


def test_ebook_create_chapter_auto_adds_to_chapter_list(tmp_path):
    _seed_all()
    eb = Ebook("Title")
    ch = eb.create_chapter("Chap1")
    assert ch.title == "Chap1"
    assert ch in eb.chapter_list


def test_ebook_dest_file_path_and_full_path(tmp_path):
    _seed_all()
    eb = Ebook("My ebook+title")
    fname = eb.dest_file_path.name
    # Should remove spaces, slashes and plus signs replaced with dash
    assert " " not in fname
    assert "/" not in fname
    assert "+" not in fname
    assert fname.endswith(".mobi")
    # full_dest_file_path includes tmpdir
    assert str(eb.full_dest_file_path).startswith(str(eb.tmpdir))


def test_ebook_util_headings_property(tmp_path):
    _seed_all()
    eb = Ebook("MyBook")
    # Create chapters with subchapters
    ch1 = eb.create_chapter("Chapter1").set_content("c1 content")
    ch2 = eb.create_chapter("Chapter2").set_content("c2 content")
    sub1 = ch1.create_subchapter("Sub1").set_content("sub1 content")
    sub2 = ch1.create_subchapter("Sub2").set_content("sub2 content")
    eu = EbookUtil(eb)
    headings = eu.headings
    # headings is list with length==len(chapter_list)
    assert isinstance(headings, list)
    assert len(headings) == 2
    # Each heading dict has keys including 'title', 'play_order', 'file_name', 'sub_headings'
    for h in headings:
        assert isinstance(h, dict)
        assert {"title", "play_order", "file_name", "sub_headings"} <= h.keys()
        assert isinstance(h['sub_headings'], list)
        for sh in h['sub_headings']:
            assert {"title", "play_order", "file_name"} <= sh.keys()
    # play_order increments properly: headings and sub_headings increasing in order field
    po_values = []
    for h in headings:
        po_values.append(h['play_order'])
        for sh in h['sub_headings']:
            po_values.append(sh['play_order'])
    # strictly increasing play_order starting from 2 (since order starts 1 and increased before appending)
    assert po_values == sorted(po_values)
    assert all(po >= 2 for po in po_values)


@pytest.mark.parametrize("template_name,expected_filename", [
    ("toc.xml", "toc.ncx"),
    ("toc.html", "toc.html"),
    ("opf.xml", None),  # filename depends on ebook title formatting
])
def test_ebook_util_render_file_and_render_methods(tmp_path, monkeypatch, template_name, expected_filename):
    _seed_all()
    # Setup a dummy Ebook with one chapter
    eb = Ebook("Book Title")
    ch = eb.create_chapter("Chapter One").set_content("Content")
    eu = EbookUtil(eb)

    tmp_templates_path = pathlib.Path(__file__).parent / "kindle_maker" / "templates"
    # monkeypatch templates_env to a minimal environment with dummy template strings
    dummy_template_text = "Rendered: {{ title }}"

    class DummyTemplate:
        def render(self, **context):
            return dummy_template_text

    class DummyEnvironment:
        def get_template(self, name):
            assert name == template_name
            return DummyTemplate()

    monkeypatch.setattr(eu, "_templates_env", DummyEnvironment())

    # create needed tmpdir if missing
    eb._tempdir.mkdir(exist_ok=True)

    # Call _render_file explicitly and check file is created with dummy content
    to_filename = "testfile.txt"
    eu._render_file(to_filename=to_filename, template_name=template_name, context={"title": "X"})
    fpath = eb.tmpdir / to_filename
    assert fpath.is_file()
    content = fpath.read_text(encoding="utf-8")
    assert "Rendered" in content

    # Test _render_toc_ncx and _render_toc_html return expected files
    if expected_filename:
        if expected_filename == "toc.ncx":
            fname = eu._render_toc_ncx()
        elif expected_filename == "toc.html":
            fname = eu._render_toc_html()
        else:
            fname = None
        if fname:
            fpath2 = eb.tmpdir / fname
            # file should exist after rendering
            # But since template env is dummy, render will still create files with dummy data
            assert fpath2.is_file()
    # Test _render_opf returns filename matching pattern
    if template_name == "opf.xml":
        fname = eu._render_opf()
        # Format file name from title
        assert fname.startswith("BookTitle") or fname.startswith("Book-Title")
        assert fname.endswith(".opf")
        fpath3 = eb.tmpdir / fname
        assert fpath3.is_file()


def test_ebook_util_save_cover_behavior(tmp_path, monkeypatch):
    _seed_all()
    eb = Ebook("Book")
    eb._tempdir = tmp_path

    eu = EbookUtil(eb)

    # No cover set: copies default template cover file
    # monkeypatch copy to check it's called with right args without actually copying files
    called = {}
    def fake_copy(src, dst):
        called['src'] = src
        called['dst'] = dst

    monkeypatch.setattr(shutil, "copy", fake_copy)

    eu._save_cover()
    # Default cover file is inside templates folder in project dir
    assert "cover.jpg" in str(called['src'])
    assert str(called['dst']) == str(eb.tmpdir)

    # When cover_path set
    eb.cover_path = str(tmp_path / "mycover.jpg")
    (tmp_path / "mycover.jpg").write_text("imgdata", encoding="utf-8")
    called.clear()
    eu._save_cover()
    assert called['src'] == eb.cover_path
    assert str(called['dst']) == str(eb.tmpdir)

    # When cover_content set, NotImplementedError raised
    eb.cover_path = None
    eb.cover_content = b"somecontent"
    with pytest.raises(NotImplementedError):
        eu._save_cover()


def test_ebook_util_save_chapters_calls_save(monkeypatch):
    _seed_all()
    eb = Ebook("Book")
    ch = eb.create_chapter("C").set_content("some text")
    sub = ch.create_subchapter("Sub").set_content("sub text")
    eu = EbookUtil(eb)
    # patch both chapter and subchapter save methods to record calls
    calls = []
    monkeypatch.setattr(ch, "save", lambda: calls.append("ch"))
    monkeypatch.setattr(sub, "save", lambda: calls.append("sub"))
    eu._save_chapters()
    assert "ch" in calls
    assert "sub" in calls


def test_ebook_create_method_invokes_util_create(monkeypatch):
    _seed_all()
    eb = Ebook("Book")
    eb.create_chapter("C").set_content("c")
    called = {}

    def fake_create():
        called['yes'] = True

    monkeypatch.setattr(eb._eu, "create", fake_create)
    eb._create()
    assert called.get('yes')


def test_ebook_create_raises_without_chapters():
    _seed_all()
    eb = Ebook("No Chapters")
    eb.chapter_list = []
    with pytest.raises(ValueError):
        eb._create()


def test_ebook_save_creates_and_copies(monkeypatch, tmp_path):
    _seed_all()
    eb = Ebook("Book")
    eb._tempdir = tmp_path
    ch = eb.create_chapter("C").set_content("c")
    # monkeypatch _create to create dummy mobi file
    mobi_file = tmp_path / eb.dest_file_path.name
    mobi_file.write_text("mobi content", encoding="utf-8")
    monkeypatch.setattr(eb, "_create", lambda: None)
    monkeypatch.setattr(shutil, "copy", lambda src, dst: None)
    # patch tmpdir to tmp_path
    eb._tempdir = tmp_path
    # Real shutil.copy called, we patch to dummy so avoid actual copying
    eb.save(str(tmp_path / "final.mobi"))


@pytest.mark.parametrize("platform_name,expected_call_args", [
    ("Linux", ["xdg-open"]),
    ("Darwin", ["open"]),
])
def test_ebook_show_platform(monkeypatch, platform_name):
    _seed_all()
    eb = Ebook("Book")
    eb._tempdir = pathlib.Path(tempfile.gettempdir())
    eb.create_chapter("C").set_content("c")
    monkeypatch.setattr(eb, "_create", lambda: None)
    monkeypatch.setattr(eb, "full_dest_file_path", pathlib.Path("/fake/path/book.mobi"))
    called_args = []
    def fake_call(args):
        called_args.append(args)
        return 0

    monkeypatch.setattr(eb, "full_dest_file_path", pathlib.Path("/fake/path/book.mobi"))
    monkeypatch.setattr(ebooklib.platform, "system", lambda: platform_name)
    monkeypatch.setattr(ebooklib.subprocess, "call", fake_call)
    ret = eb.show()
    assert ret == 0
    args = called_args[0]
    assert args[0] == expected_call_args[0]
    assert args[1] == str(eb.full_dest_file_path)


def test_ebook_show_windows(monkeypatch):
    _seed_all()
    eb = Ebook("Book")
    eb._tempdir = pathlib.Path(tempfile.gettempdir())
    eb.create_chapter("C").set_content("c")
    monkeypatch.setattr(eb, "_create", lambda: None)
    monkeypatch.setattr(ebooklib.platform, "system", lambda: "Windows")
    monkeypatch.setattr(eb, "full_dest_file_path", pathlib.Path("/fake/path/book.mobi"))
    called = {}
    def fake_startfile(path):
        called['path'] = path

    monkeypatch.setattr(ebooklib.os, "startfile", fake_startfile)
    ret = eb.show()
    assert ret is None
    assert called['path'] == str(eb.full_dest_file_path)


def test_ebook_del_removes_tmpdir(monkeypatch, tmp_path):
    _seed_all()
    eb = Ebook("Book")
    # Setup tempdir with files and directories
    tdir = tmp_path / "tempdir"
    tdir.mkdir()
    (tdir / "file1.txt").write_text("1")
    subdir = tdir / "subdir"
    subdir.mkdir()
    (subdir / "file2.txt").write_text("2")

    eb._tempdir = tdir
    # Replace tmpdir to test rmtree deletes recursively
    # Call __del__ explicitly
    eb.__del__()
    assert not tdir.exists()


def test__parse_headers_valid(tmp_path):
    _seed_all()
    toc_content = """
My Book Title

# Chapter 1
## Subchapter 1.1
# Chapter 2
## Subchapter 2.1
"""
    tocfile = tmp_path / "toc.md"
    tocfile.write_text(toc_content, encoding="utf-8")
    title, headers = ebooklib._parse_headers(str(tocfile))
    assert title == "My Book Title"
    assert isinstance(headers, list)
    assert any(h['title'].startswith("Chapter") for h in headers)
    # subheadings exist in some headers
    found_sub = any("next_headers" in h and len(h["next_headers"]) > 0 for h in headers)
    assert found_sub


def test__parse_headers_invalid_file(tmp_path):
    _seed_all()
    tocfile = tmp_path / "nonexist.md"
    with pytest.raises(ValueError):
        ebooklib._parse_headers(str(tocfile))


def test__parse_headers_empty_title(tmp_path):
    _seed_all()
    tocfile = tmp_path / "toc.md"
    tocfile.write_text("\n\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ebooklib._parse_headers(str(tocfile))


def test__parse_headers_no_headings(tmp_path):
    _seed_all()
    tocfile = tmp_path / "toc.md"
    tocfile.write_text("Title is here\nJust some text\nNo headings", encoding="utf-8")
    with pytest.raises(ValueError):
        ebooklib._parse_headers(str(tocfile))


def test_make_ebook_smoke(tmp_path, monkeypatch):
    _seed_all()
    # Setup source_dir with toc.md, cover.jpg, chapters, and other files
    source_dir = tmp_path / "src"
    source_dir.mkdir()

    toc_md_content = """
My Title

# Chapter1
## Sub1
"""
    (source_dir / "toc.md").write_text(toc_md_content, encoding="utf-8")
    (source_dir / "cover.jpg").write_text("coverdata", encoding="utf-8")
    # Create html files for chapters
    (source_dir / "Chapter1.html").write_text("<html>ch1</html>", encoding="utf-8")
    (source_dir / "Sub1.html").write_text("<html>sub1</html>", encoding="utf-8")
    # Create a non-html file to get copied
    other_file = source_dir / "file.dat"
    other_file.write_text("data", encoding="utf-8")

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    # Patch shutil.copy to dummy to avoid actual copying
    monkeypatch.setattr(shutil, "copy", lambda src, dst: None)
    # Patch Ebook.save to dummy to avoid running the whole create process
    called = {}
    def fake_save(self, fp):
        called['called'] = True
        called['fp'] = fp

    monkeypatch.setattr(Ebook, "save", fake_save)

    dest_path = ebooklib.make_ebook(str(source_dir), str(output_dir))
    assert dest_path.endswith(".mobi")
    assert called.get('called')
    assert called.get('fp').startswith(str(output_dir))


# Test that Ebook.__del__ does not raise if tmpdir already removed
def test_ebook_del_tmpdir_already_removed(tmp_path):
    _seed_all()
    eb = Ebook("Book")
    tdir = tmp_path / "tempdir"
    tdir.mkdir()
    eb._tempdir = tdir
    shutil.rmtree(tdir)
    # Calling del should not raise even if directory missing
    eb.__del__()


# Test EbookUtil headings caches result
def test_ebook_util_headings_caching(tmp_path):
    _seed_all()
    eb = Ebook("Book")
    ch = eb.create_chapter("C").set_content("c")
    eu = EbookUtil(eb)
    h1 = eu.headings
    # modify chapter list after headings computed
    eb.create_chapter("NewCh").set_content("new")
    h2 = eu.headings
    # headings property returns cached value, unchanged after chapters modified
    assert h1 == h2


# Test Chapter dest_file_path sanitizes title filename
def test_chapter_dest_file_path_formatting():
    _seed_all()
    eb = Ebook("Book")
    ch = Chapter('a/b c+d"e\\f:g|h', ebook=eb)
    fname = ch.dest_file_path.name
    # should not contain /, spaces, quotes, slashes, colons, pipes
    for c in ('/', ' ', '"', '\\', ':', '|'):
        assert c not in fname
    # plus replaced by dash => check that dash present
    assert '-' in fname
    # endswith .html
    assert fname.endswith(".html")
