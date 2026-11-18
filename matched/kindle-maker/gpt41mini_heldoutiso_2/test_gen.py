import os
import pathlib
import platform
import subprocess
import sys
import tempfile
import time
import shutil

import pytest

from kindle_maker.ebooklib import Ebook, EbookUtil, Chapter, format_file_name


@pytest.fixture
def temp_dir():
    path = pathlib.Path(tempfile.mkdtemp())
    yield path
    if path.exists():
        shutil.rmtree(path)


@pytest.fixture
def ebook(temp_dir):
    # Create Ebook with a unique tempdir in the temp_dir fixture
    e = Ebook("Test Ebook", author="Author Name")
    # override tmpdir to use temp_dir so cleanup works smoothly
    e._tempdir = temp_dir
    return e


@pytest.fixture
def chapter_file(tmp_path):
    file = tmp_path / "chapter.html"
    file.write_text("<h1>Chapter Content</h1>", encoding="utf-8")
    return str(file)


@pytest.fixture
def static_file(tmp_path):
    file = tmp_path / "static.txt"
    file.write_text("static content", encoding="utf-8")
    return str(file)


def test_format_file_name_basic():
    s = 'a/b c+d"\\:|'
    out = format_file_name(s)
    assert '/' not in out
    assert ' ' not in out
    assert '+' not in out
    assert '"' not in out
    assert '\\' not in out
    assert ':' not in out
    assert '|' not in out
    assert '-' in out


def test_chapter_init_valid_file(chapter_file, ebook):
    ch = Chapter("Title", ebook, file_path=chapter_file)
    assert ch.title == "Title"
    assert ch._file_path == chapter_file
    assert ch._level == 1
    assert ch.is_top_chapter is True


def test_chapter_init_invalid_file(tmp_path, ebook):
    invalid_path = tmp_path / "nonexistent.html"
    with pytest.raises(ValueError):
        Chapter("Title", ebook, file_path=str(invalid_path))


def test_chapter_dest_file_name_and_full_path(ebook):
    ch = Chapter("My Title*<>?", ebook)
    n = ch.dest_file_path.name
    assert n.endswith(".html")
    # dest_file_path contains only allowed chars (format_file_name)
    for c in r'/ +"\:|':
        assert c not in n
    full_path = ch.full_dest_file_path
    assert full_path.name == n
    assert full_path.parent == ebook.tmpdir


def test_chapter_set_content_and_content_property():
    ch = Chapter("A", ebook=None)
    html_text = "<p>Hello World</p>"
    ch.set_content(html_text)
    assert ch.content == html_text
    # file_path should be None after set_content
    assert ch._file_path is None


def test_chapter_add_static_files(static_file):
    ch = Chapter("Title", ebook=None)
    ch.add_static_files(static_file)
    assert static_file in ch._static_files


def test_chapter_add_static_files_invalid(tmp_path):
    ch = Chapter("Title", ebook=None)
    invalid_path = tmp_path / "doesnotexist.txt"
    with pytest.raises(AssertionError):
        ch.add_static_files(str(invalid_path))


def test_chapter_create_subchapter_and_level(tmp_path, ebook):
    ch = Chapter("Main", ebook)
    sub = ch.create_subchapter("Sub", chapter_file_path=None)
    assert sub._level == ch._level + 1
    with pytest.raises(Exception):
        # Exceed max_level = 2
        sub.create_subchapter("SubSub", None)


def test_chapter_save_writes_file_and_copies_static(tmp_path):
    # Create chapter with content and static file
    ch = Chapter("SaveTest", ebook=None)
    content = "<h1>Content</h1>"
    ch.set_content(content)
    ch._ebook = type('DummyEbook', (), {})()
    tmpdir = tmp_path
    setattr(ch._ebook, 'tmpdir', tmpdir)
    # Override full_dest_file_path property dynamically
    ch._ebook.tmpdir = tmpdir
    save_path = tmpdir / ch.dest_file_path.name
    ch.add_static_files = lambda *args, **kwargs: None  # noop add_static_files
    # Create a dummy static file and add it
    static_file = tmp_path / "staticfile.txt"
    static_file.write_text("hello", encoding="utf-8")
    ch._static_files = [str(static_file)]
    ch.save()
    assert save_path.is_file()
    assert save_path.read_text(encoding="utf-8") == content
    # static file copied into the tmpdir directory
    copied_static = tmpdir / static_file.name
    assert copied_static.is_file()
    assert copied_static.read_text(encoding="utf-8") == "hello"


def test_ebook_set_cover_path(tmp_path):
    tmp_cover = tmp_path / "cover.jpg"
    tmp_cover.write_bytes(b"jpgbytes")
    ebook = Ebook("Title")
    ebook._tempdir = tmp_path
    returned = ebook.set_cover(cover_path=str(tmp_cover))
    assert returned is ebook
    assert ebook.cover_path == str(tmp_cover)
    assert ebook.cover_content is None


def test_ebook_set_cover_content(tmp_path):
    ebook = Ebook("Title")
    ebook._tempdir = tmp_path
    content = b"abcd"
    returned = ebook.set_cover(cover_content=content)
    assert returned is ebook
    assert ebook.cover_content == content
    assert ebook.cover_path is None


def test_ebook_set_cover_value_error():
    ebook = Ebook("Title")
    with pytest.raises(ValueError):
        ebook.set_cover()


def test_ebook_add_chapter_accepts_only_top_level(ebook):
    top_ch = Chapter("Top", ebook)
    ebook.add_chapter(top_ch)
    assert top_ch in ebook.chapter_list
    sub_ch = Chapter("Sub", ebook, level=2)
    with pytest.raises(ValueError):
        ebook.add_chapter(sub_ch)


def test_ebook_create_chapter_and_add_to_list(ebook):
    ch = ebook.create_chapter("Chapter 1")
    assert ch.title == "Chapter 1"
    assert ch.is_top_chapter is True
    assert ch in ebook.chapter_list


def test_ebook_dest_file_name_and_full_path(ebook):
    ebook._tempdir = pathlib.Path(tempfile.mkdtemp())
    name = ebook.dest_file_path.name
    assert name.endswith(".mobi")
    for c in r'/ +"\:|':
        assert c not in name
    full_path = ebook.full_dest_file_path
    assert full_path.name == name
    assert full_path.parent == ebook.tmpdir


def test_ebook_create_without_chapters_raises(ebook):
    # Clear chapters
    ebook.chapter_list.clear()
    with pytest.raises(ValueError):
        ebook._create()


def test_ebook_util_headings_property_order_and_content(temp_dir):
    ebook = Ebook("Title")
    ebook._tempdir = temp_dir
    ch1 = ebook.create_chapter("Ch1")
    ch2 = ebook.create_chapter("Ch2")
    sub = ch1.create_subchapter("Sub1")
    eu = EbookUtil(ebook)
    headings = eu.headings
    # headings include sub_headings as list
    assert isinstance(headings, list)
    assert all('title' in h for h in headings)
    # play_order increases by 1 for each chapter and subchapter
    po = [h['play_order'] for h in headings]
    assert all(po[i] < po[i+1] for i in range(len(po)-1))
    # sub_headings is list with play_order
    assert isinstance(headings[0]['sub_headings'], list)
    assert len(headings[0]['sub_headings']) == 1
    assert headings[0]['sub_headings'][0]['title'] == 'Sub1'


def test_ebook_util_render_files_are_created(tmp_path, ebook):
    ebook._tempdir = tmp_path
    ch = ebook.create_chapter("A")
    eu = EbookUtil(ebook)

    # Prepare template files in templates folder
    templates_dir = pathlib.Path(__file__).parent.parent / 'kindle_maker' / 'templates'
    assert templates_dir.is_dir()

    toc_ncx = eu._render_toc_ncx()
    assert (tmp_path / toc_ncx).is_file()

    toc_html = eu._render_toc_html()
    assert (tmp_path / toc_html).is_file()

    opf_file = eu._render_opf()
    assert (tmp_path / opf_file).is_file()


def test_ebook_util_save_cover_copies_cover_file(tmp_path, ebook):
    ebook._tempdir = tmp_path
    cover_file = tmp_path / "cover.jpg"
    cover_file.write_bytes(b"xxx")
    ebook.cover_path = str(cover_file)
    eu = EbookUtil(ebook)
    eu._save_cover()
    assert (tmp_path / cover_file.name).exists()


def test_ebook_util_save_cover_default_copies_template(tmp_path, monkeypatch, ebook):
    ebook._tempdir = tmp_path
    ebook.cover_path = None
    ebook.cover_content = None
    eu = EbookUtil(ebook)
    # Patch _save_cover to catch shutil.copy call without error
    monkeypatch.setattr(shutil, "copy", lambda src, dst: None)
    eu._save_cover()  # no error should occur


def test_ebook_util_save_cover_content_not_implemented(tmp_path, ebook):
    ebook._tempdir = tmp_path
    ebook.cover_content = b"data"
    ebook.cover_path = None
    eu = EbookUtil(ebook)
    with pytest.raises(NotImplementedError):
        eu._save_cover()


def test_ebook_util_create_calls_subprocess(monkeypatch, tmp_path, ebook):
    ebook._tempdir = tmp_path
    ch = ebook.create_chapter("Chapter 1")
    eu = EbookUtil(ebook)

    # Patch methods to avoid actual file operations
    monkeypatch.setattr(eu, "_render_toc_ncx", lambda: "toc_file")
    monkeypatch.setattr(eu, "_render_toc_html", lambda: "toc_html")
    monkeypatch.setattr(eu, "_render_opf", lambda: "opf_file.opf")
    monkeypatch.setattr(eu, "_save_cover", lambda: None)
    monkeypatch.setattr(eu, "_save_chapters", lambda: None)

    called_args = {}
    def fake_call(args):
        called_args['args'] = args
        return 0  # success
    monkeypatch.setattr(subprocess, "call", fake_call)

    eu.create()
    assert called_args['args'][0] == 'kindlegen'
    assert str(eu._ebook.tmpdir / "opf_file.opf") in called_args['args']


def test_ebook_show_calls_subprocess(monkeypatch, tmp_path, ebook):
    ebook._tempdir = tmp_path
    ch = ebook.create_chapter("Chapter 1")

    called = {}
    def fake_call(args):
        called['args'] = args
        return 4
    monkeypatch.setattr(subprocess, "call", fake_call)

    # Patch platform.system to Linux
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    ret = ebook.show()
    assert ret == 4
    assert called['args'][0] == "xdg-open"

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    ret = ebook.show()
    assert ret == 4
    assert called['args'][0] == "open"

    monkeypatch.setattr(platform, "system", lambda: "Windows")
    # Patch os.startfile to a dummy function that records call
    startfile_called = {}
    def fake_startfile(path):
        startfile_called['path'] = path
    monkeypatch.setattr(os, "startfile", fake_startfile)
    ret = ebook.show()
    assert startfile_called['path'].endswith(".mobi")


def test_ebook_del_removes_tmpdir(tmp_path):
    # Create a new Ebook with its own unique tmpdir inside tmp_path
    ebook = Ebook("Test Del")
    # Override tempdir inside the tmp_path
    ebook._tempdir = tmp_path / "ebook_tmp"
    ebook._tempdir.mkdir()
    # Create some files and dirs inside ebook.tmpdir
    file1 = ebook._tempdir / "file1.txt"
    file1.write_text("abc")
    subdir = ebook._tempdir / "subdir"
    subdir.mkdir()
    file2 = subdir / "file2.txt"
    file2.write_text("xyz")
    # cache tmpdir path to test if it exists later
    tmpdir_path = ebook._tempdir
    # call __del__ explicitly (may raise errors if already deleted)
    ebook.__del__()
    # tmpdir should be removed
    assert not tmpdir_path.exists()
