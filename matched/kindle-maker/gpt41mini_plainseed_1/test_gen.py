import io
import os
import platform
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

import pytest

import kindle_maker.ebooklib as ebooklib
from kindle_maker.ebooklib import Ebook, EbookUtil, Chapter, format_file_name


@pytest.fixture
def temp_file():
    fd, path = tempfile.mkstemp(text=True)
    os.close(fd)
    yield path
    if os.path.isfile(path):
        os.remove(path)


@pytest.fixture
def temp_dir():
    d = pathlib.Path(tempfile.mkdtemp())
    yield d
    if d.exists():
        shutil.rmtree(d)


@pytest.fixture
def sample_ebook():
    ebook = Ebook("Test Title", author="Test Author")
    yield ebook
    # cleanup explicitly (in case _del doesn't trigger)
    if ebook.tmpdir.exists():
        shutil.rmtree(ebook.tmpdir)


def test_format_file_name_basic():
    s = "hello/world + test:\"pipe|slash\\"
    expected = "helloworld--test-pipe-slash"
    assert format_file_name(s) == expected


def test_chapter_init_and_properties(temp_file, sample_ebook):
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write("chapter content")
    ch = Chapter("Chapter 1", sample_ebook, temp_file)
    assert ch.title == "Chapter 1"
    assert ch._level == 1
    assert ch.content == "chapter content"
    assert ch.is_top_chapter is True
    # dest_file_path uses sanitized title
    assert str(ch.dest_file_path).endswith(".html")
    assert "Chapter1" in str(ch.dest_file_path)
    # full_dest_file_path points to correct temp dir + file
    assert (sample_ebook.tmpdir / ch.dest_file_path) == ch.full_dest_file_path


def test_chapter_init_raises_with_nonfile(tmp_path, sample_ebook):
    non_file = tmp_path / "nofile.txt"
    with pytest.raises(ValueError):
        Chapter("bad chapter", sample_ebook, str(non_file))


def test_chapter_set_content_resets_file_path():
    ch = Chapter("Title", None)
    ch.set_content("new content")
    assert ch._content == "new content"
    assert ch._file_path is None
    assert ch.content == "new content"


def test_chapter_add_static_files(tmp_path):
    # create a tmp file to add as static file
    static_file = tmp_path / "static.png"
    static_file.write_text("binarydata", encoding="utf-8")
    ch = Chapter("Title", None)
    ch.add_static_files(str(static_file))
    assert str(static_file) in ch._static_files

    # adding invalid static file path raises assertion
    with pytest.raises(AssertionError):
        ch.add_static_files("nonexistent.file")


def test_chapter_create_subchapter_and_level_limit(sample_ebook):
    ch = Chapter("Main", sample_ebook)
    sub = ch.create_subchapter("Sub chapter")
    assert isinstance(sub, Chapter)
    assert sub._level == 2
    # Creating subchapter beyond max_level raises Exception
    with pytest.raises(Exception):
        sub.create_subchapter("Too deep")


def test_chapter_save_writes_content_and_copies_static(tmp_path):
    ch = Chapter("SaveTest", None)
    file_path = tmp_path / "chapter.html"
    file_path.write_text("This is content", encoding="utf-8")
    ch.set_content("content to save")
    ch._ebook = type("MockEbook", (), {"tmpdir": tmp_path})()
    # Add static file
    static_file = tmp_path / "staticfile.txt"
    static_file.write_text("static stuff", encoding="utf-8")
    ch.add_static_files(str(static_file))
    ch.save()
    saved_file = tmp_path / ch.dest_file_path
    assert saved_file.exists()
    assert saved_file.read_text(encoding="utf-8") == "content to save"
    # Static file copied to same dir
    copied_static = tmp_path / static_file.name
    assert copied_static.exists()


def test_ebook_set_cover_with_path_and_content(tmp_path):
    cover_path = tmp_path / "cover.jpg"
    cover_path.write_text("fakeimage", encoding="utf-8")
    ebook = Ebook("Title")
    # set with cover_path
    ebook.set_cover(cover_path=str(cover_path))
    assert ebook.cover_content is None
    assert ebook.cover_path == str(cover_path)
    # set with cover_content
    ebook.set_cover(cover_content=b"bytes")
    assert ebook.cover_path is None
    assert ebook.cover_content == b"bytes"
    # setting with nothing raises ValueError
    with pytest.raises(ValueError):
        ebook.set_cover()


def test_ebook_add_chapter_accepts_only_top_level(sample_ebook):
    top_ch = Chapter("Top", sample_ebook)
    sample_ebook.add_chapter(top_ch)
    assert top_ch in sample_ebook.chapter_list
    # Non top level raises error
    sub = Chapter("Sub", sample_ebook, level=2)
    with pytest.raises(ValueError):
        sample_ebook.add_chapter(sub)


def test_ebook_create_chapter_adds_to_list(sample_ebook):
    ch = sample_ebook.create_chapter("New Chapter")
    assert ch.title == "New Chapter"
    assert ch.is_top_chapter is True
    assert ch in sample_ebook.chapter_list


def test_ebook__create_raises_without_chapters(sample_ebook):
    sample_ebook.chapter_list.clear()
    with pytest.raises(ValueError):
        sample_ebook._create()


def test_ebook_dest_file_path_and_full(sample_ebook):
    fname = sample_ebook.dest_file_path
    assert str(fname).endswith(".mobi")
    assert isinstance(sample_ebook.full_dest_file_path, pathlib.Path)
    assert sample_ebook.full_dest_file_path.parent == sample_ebook.tmpdir


def test_ebook_del_cleans_tempdir(tmp_path):
    ebook = Ebook("DelTest")
    d = ebook.tmpdir
    assert d.exists()
    # Trigger delete and check folder removed
    del ebook
    # The tempdir removal happens asynchronously, but likely immediate
    assert not d.exists()


def test_ebook_show_calls_correct_subprocess(monkeypatch, sample_ebook):
    called = {}

    def fake_call(cmd):
        called["cmd"] = cmd
        return 42

    def fake_startfile(path):
        called["startfile"] = path
        return 99

    monkeypatch.setattr(subprocess, "call", fake_call)
    monkeypatch.setattr(os, "startfile", fake_startfile)
    # Patch platform.system temporarily
    original_platform = platform.system

    platform.system = lambda: "Linux"
    r = sample_ebook.show()
    assert called["cmd"][0] == "xdg-open"
    assert r == 42

    called.clear()
    platform.system = lambda: "Darwin"
    r = sample_ebook.show()
    assert called["cmd"][0] == "open"
    assert r == 42

    called.clear()
    platform.system = lambda: "Windows"
    r = sample_ebook.show()
    assert "startfile" in called
    assert r == 99

    platform.system = original_platform


def test_ebookutil_headings_property_creates_structure(sample_ebook):
    # Add chapters with sub chapters
    ch1 = sample_ebook.create_chapter("Top1")
    sub1 = ch1.create_subchapter("Sub1")
    ch2 = sample_ebook.create_chapter("Top2")

    util = EbookUtil(sample_ebook)
    headings = util.headings
    # Should contain both top-level chapters
    assert any(h['title'] == "Top1" for h in headings)
    assert any(h['title'] == "Top2" for h in headings)
    # The first should have sub_headings
    first = headings[0]
    assert "sub_headings" in first
    assert any(sh['title'] == "Sub1" for sh in first['sub_headings'])


def test_ebookutil_render_file_creates_file(tmp_path, sample_ebook):
    util = EbookUtil(sample_ebook)
    sample_ebook._tempdir = tmp_path
    # We create a dummy template in templates/ to test render_file
    templates_dir = pathlib.Path(__file__).parent.parent / "kindle_maker" / "templates"
    template_file = templates_dir / "dummy_template.txt"
    # create dummy template file for test
    template_file.write_text("Hello {{ thing }}", encoding="utf-8")
    try:
        util._render_file("dummy_template.txt", {"thing": "world"}, "output.txt")
        output_path = tmp_path / "output.txt"
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "Hello world" in content
    finally:
        template_file.unlink()


def test_ebookutil_render_toc_ncx_and_html_and_opf(tmp_path, sample_ebook):
    # Setup chapters so headings not empty
    ch1 = sample_ebook.create_chapter("Chapter 1")
    util = EbookUtil(sample_ebook)
    sample_ebook._tempdir = tmp_path
    ncx = util._render_toc_ncx()
    html = util._render_toc_html()
    opf = util._render_opf()
    # Files should exist
    assert (tmp_path / ncx).exists()
    assert (tmp_path / html).exists()
    assert (tmp_path / opf).exists()
    # The opf filename should be sanitized title + .opf
    assert opf.endswith(".opf")
    assert sample_ebook.title.replace(" ", "") in opf


def test_ebookutil_save_cover_copies_cover_file_and_default(tmp_path, sample_ebook):
    util = EbookUtil(sample_ebook)
    sample_ebook._tempdir = tmp_path
    # Case: cover_path given: file copied
    cover_file = tmp_path / "mycover.jpg"
    cover_file.write_text("cover contents", encoding="utf-8")
    sample_ebook.cover_path = str(cover_file)
    sample_ebook.cover_content = None
    util._save_cover()
    assert any(f.name == "mycover.jpg" for f in tmp_path.iterdir())

    # Case: cover_content raises NotImplementedError
    sample_ebook.cover_path = None
    sample_ebook.cover_content = b"bytes"
    with pytest.raises(NotImplementedError):
        util._save_cover()

    # Case: no cover, default copied
    sample_ebook.cover_path = None
    sample_ebook.cover_content = None
    util._save_cover()
    found = False
    for f in tmp_path.iterdir():
        if f.name == "cover.jpg":
            found = True
    assert found


def test_ebookutil_save_chapters_calls_save(tmp_path, sample_ebook, monkeypatch):
    ch = sample_ebook.create_chapter("Chap")
    monkeypatch.setattr(ch, "save", lambda : setattr(ch, "_saved", True))
    sub = ch.create_subchapter("Sub")
    monkeypatch.setattr(sub, "save", lambda : setattr(sub, "_saved", True))

    util = EbookUtil(sample_ebook)
    sample_ebook._tempdir = tmp_path
    util._save_chapters()
    assert getattr(ch, "_saved", False) is True
    assert getattr(sub, "_saved", False) is True
