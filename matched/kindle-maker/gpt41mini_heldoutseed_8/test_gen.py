import io
import os
import pathlib
import platform
import shutil
import subprocess
import tempfile
import time

import pytest

from kindle_maker.ebooklib import Ebook, EbookUtil, Chapter, format_file_name


@pytest.fixture
def empty_ebook():
    return Ebook("Test Book")


@pytest.fixture
def ebook_with_chapters(tmp_path):
    ebook = Ebook("Sample Book", author="Author Name")
    # create temp html files for chapters
    c1_path = tmp_path / "Chapter1.html"
    c1_path.write_text("<h1>Chapter 1</h1>", encoding="utf-8")
    c2_path = tmp_path / "Chapter2.html"
    c2_path.write_text("<h1>Chapter 2</h1>", encoding="utf-8")
    c1 = ebook.create_chapter("Chapter1", str(c1_path))
    c2 = ebook.create_chapter("Chapter2", str(c2_path))
    # Create subchapter from c1
    sc1_path = tmp_path / "Subchapter1.html"
    sc1_path.write_text("<h2>Subchapter 1</h2>", encoding="utf-8")
    c1.create_subchapter("Subchapter1", str(sc1_path))
    return ebook


def test_format_file_name_basic():
    s = 'file/na me+"\\:|test'
    formatted = format_file_name(s)
    assert '/' not in formatted
    assert ' ' not in formatted
    assert '+' not in formatted
    assert '"' not in formatted
    assert '\\' not in formatted
    assert ':' not in formatted
    assert '|' not in formatted
    # plus replaced by dash
    assert '-' in formatted


def test_chapter_init_and_properties(tmp_path):
    fp = tmp_path / "file.html"
    fp.write_text("content", encoding="utf-8")
    ebook = Ebook("Title")
    ch = Chapter("Title1", ebook, str(fp))
    assert ch.title == "Title1"
    assert ch._file_path == str(fp)
    assert ch.is_top_chapter
    # dest_file_path ends with .html and sanitized
    assert str(ch.dest_file_path).endswith(".html")
    # full_dest_file_path includes tmpdir path
    full_path = ch.full_dest_file_path
    assert full_path.parent == ebook.tmpdir
    # content reads file content
    assert ch.content == "content"


def test_chapter_init_invalid_file():
    ebook = Ebook("Test")
    with pytest.raises(ValueError):
        Chapter("BadChapter", ebook, "nonexistent.html")


def test_chapter_set_content_and_content_property(tmp_path):
    ebook = Ebook("T")
    ch = Chapter("TestChapter", ebook)
    # initially no content
    assert ch.content is None
    result = ch.set_content("my content")
    assert result is ch
    assert ch.content == "my content"
    # after setting content, file path is None
    assert ch._file_path is None


def test_chapter_add_static_files_and_save(tmp_path):
    ebook = Ebook("T")
    ch = Chapter("StaticTest", ebook)
    # Create dummy static files
    static_file1 = tmp_path / "file1.txt"
    static_file2 = tmp_path / "file2.txt"
    static_file1.write_text("data1", encoding="utf-8")
    static_file2.write_text("data2", encoding="utf-8")
    ch.set_content("Static content")
    ch.add_static_files(str(static_file1), str(static_file2))
    # Save chapter should write content and copy static files
    ch.save()
    saved_path = ch.full_dest_file_path
    assert saved_path.exists()
    assert saved_path.read_text(encoding="utf-8") == "Static content"
    # static files copied to chapter file dir
    for sf in [static_file1, static_file2]:
        dest = saved_path.parent / sf.name
        assert dest.exists()
        assert dest.read_text(encoding="utf-8") == sf.read_text(encoding="utf-8")


def test_chapter_create_subchapter_increments_level(tmp_path):
    ebook = Ebook("Test")
    ch = Chapter("TopChapter", ebook, level=1)
    sub = ch.create_subchapter("SubChapter1")
    assert sub._level == 2
    # creating sub-subchapter should fail (max_level=2)
    with pytest.raises(Exception):
        sub.create_subchapter("TooDeep")


def test_ebook_add_chapter_level_check():
    ebook = Ebook("T")
    ch = Chapter("Level2", ebook, level=2)
    with pytest.raises(ValueError):
        ebook.add_chapter(ch)
    # only top level accepted
    top_ch = Chapter("Level1", ebook, level=1)
    result = ebook.add_chapter(top_ch)
    assert result is ebook
    assert top_ch in ebook.chapter_list


def test_ebook_create_chapter_adds_to_list(tmp_path):
    ebook = Ebook("T")
    fp = tmp_path / "f.html"
    fp.write_text("content", encoding="utf-8")
    ch = ebook.create_chapter("MyChapter", str(fp))
    assert isinstance(ch, Chapter)
    assert ch in ebook.chapter_list
    assert ch.title == "MyChapter"
    assert ch._file_path == str(fp)


def test_ebook_set_cover_path_and_content(tmp_path):
    ebook = Ebook("Book")
    # cover_path set
    fake_cover = tmp_path / "cover.jpg"
    fake_cover.write_bytes(b"binarycontent")
    ebook.set_cover(cover_path=str(fake_cover))
    assert ebook.cover_path == str(fake_cover)
    assert ebook.cover_content is None
    # cover_content set
    ebook.set_cover(cover_content=b"imgbytes")
    assert ebook.cover_content == b"imgbytes"
    assert ebook.cover_path is None
    # error if both empty
    with pytest.raises(ValueError):
        ebook.set_cover()


def test_ebook_properties_and_tmpdir_unique():
    ebook = Ebook("Book Title")
    dest_file = ebook.dest_file_path
    assert dest_file.suffix == ".mobi"
    # sanitized file name
    assert " " not in str(dest_file)
    # full_dest_file_path under tmpdir
    full_path = ebook.full_dest_file_path
    assert full_path.parent == ebook.tmpdir
    # tmpdir exists
    assert ebook.tmpdir.exists()


def test_ebookutil_headings_structure(ebook_with_chapters):
    eu = ebook_with_chapters._eu
    headings = eu.headings
    # headings is list of dict
    assert isinstance(headings, list)
    assert all(isinstance(h, dict) for h in headings)
    first = headings[0]
    assert "title" in first and "play_order" in first and "file_name" in first
    assert "sub_headings" in first
    # Subheadings present
    assert len(first['sub_headings']) > 0
    for subh in first['sub_headings']:
        assert "title" in subh
        assert "file_name" in subh


def test_ebookutil_render_file_creates_file(tmp_path, ebook_with_chapters):
    eu = ebook_with_chapters._eu
    context = {"headings": [], "title": "title", "author": "author"}
    # create dummy template file
    template_dir = pathlib.Path(__file__).parent / "kindle_maker" / "templates"
    # Just ensure template folder exists, skip actual jinja rendering test (complex)
    # Instead test that the render method writes file with any output
    fname = "test_output.html"
    # Patch environment to use a dummy template that returns fixed string
    class DummyTemplate:
        def render(self, **kwargs):
            return "rendered-content"

    class DummyEnv:
        def get_template(self, name):
            assert name == "dummy_template"
            return DummyTemplate()

    eu._templates_env = DummyEnv()
    out_path = ebook_with_chapters.tmpdir / fname
    assert not out_path.exists()
    eu._render_file("dummy_template", context, fname)
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == "rendered-content"


def test_ebookutil_render_toc_methods(ebook_with_chapters):
    eu = ebook_with_chapters._eu
    # toc.ncx file generated
    ncx_file = eu._render_toc_ncx()
    assert isinstance(ncx_file, str)
    assert (ebook_with_chapters.tmpdir / ncx_file).exists()
    # toc.html file generated
    toc_html = eu._render_toc_html()
    assert isinstance(toc_html, str)
    assert (ebook_with_chapters.tmpdir / toc_html).exists()
    # opf.xml generated with title sanitized
    opf_file = eu._render_opf()
    assert opf_file.startswith(format_file_name(ebook_with_chapters.title))
    assert opf_file.endswith(".opf")
    assert (ebook_with_chapters.tmpdir / opf_file).exists()


def test_ebookutil_save_cover_default(tmp_path, monkeypatch):
    ebook = Ebook("Book Default Cover")
    eu = ebook._eu
    # No cover_path or cover_content set -> copy default from templates
    # Setup dummy default cover.jpg in templates folder for copy
    templates_dir = pathlib.Path(__file__).parent / "kindle_maker" / "templates"
    default_cover = templates_dir / "cover.jpg"
    # Create dummy cover if not exist
    if not default_cover.exists():
        default_cover.parent.mkdir(parents=True, exist_ok=True)
        default_cover.write_bytes(b"dummycover")
    eu._save_cover()
    # Check copied cover to tmpdir
    copied_cover = ebook.tmpdir / "cover.jpg"
    assert copied_cover.exists()
    assert copied_cover.read_bytes() == default_cover.read_bytes()


def test_ebookutil_save_cover_with_cover_path(tmp_path):
    ebook = Ebook("Book with cover_path")
    fake_cover = tmp_path / "mycover.jpg"
    fake_cover.write_bytes(b"coverdata")
    ebook.cover_path = str(fake_cover)
    ebook.cover_content = None
    eu = ebook._eu
    eu._save_cover()
    copied_cover = ebook.tmpdir / fake_cover.name
    assert copied_cover.exists()
    assert copied_cover.read_bytes() == b"coverdata"


def test_ebookutil_save_cover_with_cover_content_raises_notimplemented():
    ebook = Ebook("Book with cover content")
    ebook.cover_path = None
    ebook.cover_content = b"cover bytes"
    eu = ebook._eu
    with pytest.raises(NotImplementedError):
        eu._save_cover()


def test_ebookutil_save_chapters_calls_save(tmp_path):
    ebook = Ebook("Book")
    # Add chapter with set content
    ch = ebook.create_chapter("Chap1")
    ch.set_content("chap1 content")
    sub = ch.create_subchapter("Sub1")
    sub.set_content("sub1 content")
    eu = ebook._eu
    eu._save_chapters()
    # saved html files exist
    assert (ch.full_dest_file_path).exists()
    assert (sub.full_dest_file_path).exists()


def test_ebook_create_calls_all_methods(monkeypatch, ebook_with_chapters):
    called = {}

    def dummy_render_ncx():
        called['ncx'] = True
        return "toc.ncx"

    def dummy_render_html():
        called['html'] = True
        return "toc.html"

    def dummy_render_opf():
        called['opf'] = True
        return "book.opf"

    def dummy_save_cover():
        called['cover'] = True

    def dummy_save_chapters():
        called['chapters'] = True

    def dummy_subprocess_call(cmd):
        called['subprocess'] = cmd
        return 0

    eu = ebook_with_chapters._eu
    monkeypatch.setattr(eu, "_render_toc_ncx", dummy_render_ncx)
    monkeypatch.setattr(eu, "_render_toc_html", dummy_render_html)
    monkeypatch.setattr(eu, "_render_opf", dummy_render_opf)
    monkeypatch.setattr(eu, "_save_cover", dummy_save_cover)
    monkeypatch.setattr(eu, "_save_chapters", dummy_save_chapters)
    monkeypatch.setattr(subprocess, "call", dummy_subprocess_call)

    ebook_with_chapters._eu.create()

    assert called.get('ncx')
    assert called.get('html')
    assert called.get('opf')
    assert called.get('cover')
    assert called.get('chapters')
    assert 'kindlegen' in called.get('subprocess', [])
    assert called.get('subprocess', [-1])[0] == 'kindlegen'


def test_ebook_save_calls_create_and_copies(tmp_path, monkeypatch):
    ebook = Ebook("TestSave")
    ch = ebook.create_chapter("Ch1")
    ch.set_content("content")
    called = {}

    def dummy_create():
        called['create'] = True

    monkeypatch.setattr(ebook._eu, "create", dummy_create)
    # patch shutil.copy to record call
    def dummy_copy(src, dst):
        called['copy'] = (src, dst)

    monkeypatch.setattr(shutil, "copy", dummy_copy)
    # provide a path to save
    target_file = tmp_path / "output.mobi"
    ebook.save(str(target_file))
    # create called and copy called
    assert called.get('create')
    assert called.get('copy')
    # copy destination is target_file
    _, dst = called['copy']
    assert pathlib.Path(dst) == target_file


def test_ebook_show_calls_platform_methods(monkeypatch, ebook_with_chapters):
    ebook = ebook_with_chapters
    called = {}

    def dummy_create():
        called['create'] = True

    monkeypatch.setattr(ebook._eu, "create", dummy_create)

    def dummy_subprocess_call(args):
        called['subprocess_args'] = args
        return 0

    monkeypatch.setattr(subprocess, "call", dummy_subprocess_call)

    def dummy_startfile(path):
        called['startfile'] = path

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    rc = ebook.show()
    assert rc == 0
    assert 'xdg-open' in called.get('subprocess_args', [])

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    rc = ebook.show()
    assert rc == 0
    assert 'open' in called.get('subprocess_args', [])

    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(os, "startfile", dummy_startfile)
    rc = ebook.show()
    assert rc is None
    assert pathlib.Path(ebook.full_dest_file_path) in pathlib.Path(called['startfile'])


def test_ebook_del_removes_tmpdir(monkeypatch):
    ebook = Ebook("DelTest")
    tmpdir = ebook.tmpdir
    assert tmpdir.exists()

    called_paths = []

    # patch unlink and rmdir to track calls
    class DummyPath(pathlib.Path):
        def unlink(self):
            called_paths.append(str(self))

        def rmdir(self):
            called_paths.append(str(self))

    # monkeypatch tmpdir iterdir to simulate files and dirs
    file1 = tmpdir / "f1"
    file1.write_text("f1")
    file2 = tmpdir / "f2"
    file2.write_text("f2")
    subdir = tmpdir / "d1"
    subdir.mkdir()
    subfile = subdir / "sf"
    subfile.write_text("sf")

    # save original iterdir
    orig_iterdir = tmpdir.iterdir

    def dummy_iterdir():
        # returns files and subdir
        yield file1
        yield file2
        yield subdir

    # patch tmpdir.iterdir to dummy_iterdir
    monkeypatch.setattr(tmpdir, "iterdir", dummy_iterdir)

    # patch Path methods on actual Path objects
    # instead of hacking Path class (not trivial),
    # we forcibly call __del__ and check that tmpdir is removed
    path_list = []

    # call __del__ safely (it should remove files and dir)
    del ebook
    # Give time for __del__ to process (usually immediate)
    # tmpdir should be removed
    assert not tmpdir.exists()
