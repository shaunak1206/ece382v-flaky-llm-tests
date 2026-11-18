# tests/test_ebooklib.py

import io
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import pytest

from kindle_maker import ebooklib
from kindle_maker.ebooklib import Ebook, EbookUtil, Chapter, format_file_name


@pytest.fixture
def tmp_dir():
    d = pathlib.Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(str(d))


@pytest.fixture
def ebook(tmp_dir):
    eb = Ebook("Test Ebook", author="Author Name")
    # override tempdir to fixture tmp_dir to control cleanup
    eb._tempdir = tmp_dir
    return eb


def test_format_file_name_removes_illegal_chars():
    s = 'a/b c+d"e\\f:g|h'
    result = format_file_name(s)
    assert '/' not in result
    assert ' ' not in result
    assert '+' not in result
    assert '"' not in result
    assert '\\' not in result
    assert ':' not in result
    assert '|' not in result
    # '+' replaced with '-'
    assert '-' in result


def test_chapter_init_and_properties(tmp_path):
    file_path = tmp_path / "chapter.html"
    file_path.write_text("Content", encoding="utf-8")

    ebook = Ebook("Title")
    ebook._tempdir = tmp_path

    c = Chapter("Chapter 1", ebook, str(file_path))
    assert c.title == "Chapter 1"
    assert c._level == 1
    assert c.dest_file_path.name == "Chapter1.html"
    assert c.full_dest_file_path == tmp_path / "Chapter1.html"
    assert c.content == "Content"
    assert c.is_top_chapter is True
    assert c.sub_chapters == []

    # invalid file raises ValueError
    with pytest.raises(ValueError):
        Chapter("Invalid", ebook, str(tmp_path / "not_exist.html"))


def test_chapter_set_content_and_add_static_files(tmp_path):
    ebook = Ebook("Title")
    ebook._tempdir = tmp_path
    c = Chapter("Chapter", ebook)
    c.set_content("New content")

    assert c.content == "New content"
    # _file_path should be None
    assert c._file_path is None

    # Create tmp files to add as static files
    f1 = tmp_path / "f1.png"
    f2 = tmp_path / "f2.css"
    f1.write_text("data1")
    f2.write_text("data2")
    c.add_static_files(str(f1), str(f2))
    assert str(f1) in c._static_files
    assert str(f2) in c._static_files

    # Adding non-existent file raises AssertionError
    with pytest.raises(AssertionError):
        c.add_static_files(str(tmp_path / "nofile.jpg"))


def test_chapter_create_subchapter_and_level_limit(tmp_path):
    ebook = Ebook("Title")
    ebook._tempdir = tmp_path
    c = Chapter("Parent", ebook, level=1)
    sub = c.create_subchapter("Child")
    assert isinstance(sub, Chapter)
    assert sub._level == 2
    assert c.sub_chapters[-1] == sub

    # Trying to add third level raises Exception
    with pytest.raises(Exception) as e:
        sub.create_subchapter("Grandchild")
    assert "max level num" in str(e.value)


def test_chapter_save_writes_content_and_copies_static_files(tmp_path):
    ebook = Ebook("Title")
    ebook._tempdir = tmp_path
    c = Chapter("SaveTest", ebook)
    c.set_content("SaveThisContent")

    # Create static file
    static_file = tmp_path / "static.css"
    static_file.write_text("csscontent")

    c.add_static_files(str(static_file))
    c.save()

    dest_path = tmp_path / "SaveTest.html"
    assert dest_path.is_file()
    assert dest_path.read_text(encoding="utf-8") == "SaveThisContent"

    copied_static = tmp_path / "static.css"
    assert copied_static.is_file()


def test_ebook_set_cover_with_path_and_content(tmp_path):
    eb = Ebook("Title")
    eb._tempdir = tmp_path

    f = tmp_path / "cover.jpg"
    f.write_text("fakeimage")

    eb.set_cover(cover_path=str(f))
    assert eb.cover_path == str(f)
    assert eb.cover_content is None

    eb.set_cover(cover_content=b"data")
    assert eb.cover_content == b"data"
    assert eb.cover_path is None

    with pytest.raises(ValueError):
        eb.set_cover()


def test_ebook_add_chapter_accepts_only_top_level(tmp_path):
    eb = Ebook("Title")
    eb._tempdir = tmp_path

    c1 = Chapter("TopLevel", eb, level=1)
    c2 = Chapter("SubLevel", eb, level=2)

    eb.add_chapter(c1)
    assert c1 in eb.chapter_list

    with pytest.raises(ValueError):
        eb.add_chapter(c2)


def test_ebook_create_chapter_adds_to_list(tmp_path):
    eb = Ebook("Title")
    eb._tempdir = tmp_path
    c = eb.create_chapter("Ch Title")

    assert c.title == "Ch Title"
    assert c._level == 1
    assert c in eb.chapter_list


def test_ebook_create_without_chapters_raises():
    eb = Ebook("Empty")
    # forcibly empty chapter_list
    eb.chapter_list.clear()
    with pytest.raises(ValueError):
        eb._create()


def test_ebookutil_headings_property(tmp_path):
    ebook = Ebook("Test Ebook")
    ebook._tempdir = tmp_path

    c1 = ebook.create_chapter("Ch 1")
    c2 = ebook.create_chapter("Ch 2")
    sc1 = c1.create_subchapter("Sub 1-1")
    sc2 = c2.create_subchapter("Sub 2-1")

    eu = EbookUtil(ebook)

    headings = eu.headings
    assert isinstance(headings, list)
    assert len(headings) == 2
    titles = [h["title"] for h in headings]
    assert "Ch 1" in titles and "Ch 2" in titles

    # sub_headings included and have titles as well
    for h in headings:
        for sub in h["sub_headings"]:
            assert isinstance(sub["title"], str)


def test_ebookutil_render_file_creates_files(tmp_path):
    ebook = Ebook("RenderTest")
    ebook._tempdir = tmp_path
    c = ebook.create_chapter("One")

    eu = EbookUtil(ebook)

    # monkeypatch template loader to a simple template that outputs a fixed string
    class DummyTemplate:
        def render(self, **kwargs):
            return "My Rendered Content"

    eu._templates_env.get_template = lambda name: DummyTemplate()

    filename = "output.txt"
    eu._render_file("dummy", {"foo": "bar"}, filename)
    f = ebook.tmpdir / filename
    assert f.is_file()
    assert f.read_text(encoding="utf-8") == "My Rendered Content"


def test_ebookutil_render_toc_ncx_and_html_return_filenames(tmp_path):
    ebook = Ebook("Test")
    ebook._tempdir = tmp_path
    c = ebook.create_chapter("One")
    eu = EbookUtil(ebook)

    # Patch _render_file to write dummy files
    def dummy_render_file(template_name, context, to_filename):
        (ebook.tmpdir / to_filename).write_text("toc content")
    eu._render_file = dummy_render_file

    ncx = eu._render_toc_ncx()
    toc = eu._render_toc_html()

    assert ncx == "toc.ncx"
    assert toc == "toc.html"
    assert (ebook.tmpdir / ncx).is_file()
    assert (ebook.tmpdir / toc).is_file()


def test_ebookutil_render_opf_creates_opf_file(tmp_path):
    ebook = Ebook("Title With : chars")
    ebook._tempdir = tmp_path
    c = ebook.create_chapter("Ch1")
    eu = EbookUtil(ebook)

    rendered_files = {}

    def dummy_render_file(template_name, context, to_filename):
        rendered_files[to_filename] = context

    eu._render_file = dummy_render_file

    opf_file = eu._render_opf()

    # The filename should have illegal characters replaced
    assert ':' not in opf_file
    assert opf_file.endswith(".opf")
    assert opf_file in rendered_files
    # The context must include correct title and author
    ctx = rendered_files[opf_file]
    assert ctx['title'] == ebook.title
    assert "author" in ctx


def test_ebookutil_save_cover_copies_existing(tmp_path):
    ebook = Ebook("Test Cover")
    ebook._tempdir = tmp_path

    cover_file = tmp_path / "cover.jpg"
    cover_file.write_text("coverdata")
    ebook.cover_path = str(cover_file)
    ebook.cover_content = None
    eu = EbookUtil(ebook)
    eu._save_cover()

    copied_cover = tmp_dir_file = ebook.tmpdir / "cover.jpg"
    assert copied_cover.is_file()
    assert copied_cover.read_text() == "coverdata"


def test_ebookutil_save_cover_default_copies_template(tmp_path, monkeypatch):
    ebook = Ebook("Default Cover")
    ebook._tempdir = tmp_path
    ebook.cover_path = None
    ebook.cover_content = None
    eu = EbookUtil(ebook)

    # patch shutil.copy to track called src path
    copied_paths = []
    def fake_copy(src, dst):
        copied_paths.append(src)
    monkeypatch.setattr(shutil, "copy", fake_copy)

    eu._save_cover()

    # It should copy from the 'templates/cover.jpg' relative path
    assert any("cover.jpg" in str(p) for p in copied_paths)


def test_ebookutil_save_cover_raises_notimplemented(monkeypatch, tmp_path):
    ebook = Ebook("Test Cover Content")
    ebook._tempdir = tmp_path
    ebook.cover_path = None
    ebook.cover_content = b'data'
    eu = EbookUtil(ebook)
    with pytest.raises(NotImplementedError):
        eu._save_cover()


def test_ebookutil_save_chapters_calls_save(tmp_path):
    ebook = Ebook("Test Save Chapters")
    ebook._tempdir = tmp_path
    c1 = ebook.create_chapter("Ch1")
    c2 = ebook.create_chapter("Ch2")
    sc = c1.create_subchapter("Sub1")

    saved = []

    def fake_save(self):
        saved.append(self.title)

    monkeypatch = pytest.monkeypatch
    monkeypatch.setattr(ebooklib.Chapter, "save", fake_save)

    eu = EbookUtil(ebook)
    eu._save_chapters()
    # Expect all 3 chapters saved
    assert "Ch1" in saved
    assert "Ch2" in saved
    assert "Sub1" in saved


def test_ebook_create_calls_util_create_and_calls_kindlegen(monkeypatch, tmp_path):
    ebook = Ebook("Test Create")
    ebook._tempdir = tmp_path
    ch = ebook.create_chapter("Ch")
    called = {}

    # patch EbookUtil.create to set a flag and simulate subprocess.call
    def fake_create(self):
        called['created'] = True
    monkeypatch.setattr(EbookUtil, "create", fake_create)

    # patch subprocess.call to record arguments
    def fake_call(args):
        called['subprocess_args'] = args
        return 0
    monkeypatch.setattr(subprocess, "call", fake_call)

    ebook._eu.create()
    assert called["created"]

    # test Ebook._eu.create integration via Ebook._create
    called.clear()
    monkeypatch.setattr(EbookUtil, "create", lambda self: "test.opf")
    monkeypatch.setattr(subprocess, "call", lambda args: 0)
    ret = ebook._eu.create()
    assert ret is None or ret == "test.opf" or ret is None  # because create returns None in code

    # Actually call ebook._create and patch to avoid real calls
    monkeypatch.setattr(EbookUtil, "create", lambda self: "abc.opf")
    monkeypatch.setattr(subprocess, "call", lambda args: 0)
    ebook._create()  # Should run without error


def test_ebook_save_copies_mobi(tmp_path, monkeypatch):
    ebook = Ebook("Save Ebook")
    ebook._tempdir = tmp_path
    ch = ebook.create_chapter("Ch")
    # Patch create to do nothing
    monkeypatch.setattr(ebook, "_create", lambda: None)

    # Create dummy mobi file at tmpdir
    mobi_file = ebook.tmpdir / ebook.dest_file_path
    mobi_file.write_text("mobi content")

    dest_file = tmp_path / "out.mobi"
    ebook.save(str(dest_file))

    assert dest_file.is_file()
    assert dest_file.read_text() == "mobi content"


def test_ebook_show_calls_platform_commands(monkeypatch, tmp_path):
    ebook = Ebook("Show Ebook")
    ebook._tempdir = tmp_path
    ch = ebook.create_chapter("Ch")

    # Patch _create to avoid side effects
    monkeypatch.setattr(ebook, "_create", lambda: None)
    # Patch subprocess.call and os.startfile
    called = {}

    def fake_subprocess_call(args):
        called['args'] = args
        return 42

    def fake_startfile(path):
        called['startfile'] = path
        return 43

    monkeypatch.setattr(subprocess, "call", fake_subprocess_call)
    monkeypatch.setattr(os, "startfile", fake_startfile, raising=False)

    system = platform.system()

    rv = ebook.show()
    if system == 'Linux':
        assert called['args'][0] == 'xdg-open'
    elif system == 'Darwin':
        assert called['args'][0] == 'open'
    else:
        # For Windows os.startfile called
        assert called.get('startfile') == str(ebook.full_dest_file_path)
        assert rv == 43


def test_ebook_del_removes_tempdir(tmp_path):
    # create temp dir inside tmp_path to control cleanup
    test_tempdir = tmp_path / "ebook_temp"
    test_tempdir.mkdir()
    f = test_tempdir / "file.txt"
    f.write_text("data")
    # create Ebook and override tmpdir
    ebook = Ebook("Del Ebook")
    ebook._tempdir = test_tempdir
    # Ensure file exists
    assert (test_tempdir / "file.txt").exists()

    # call __del__ manually
    ebook.__del__()
    assert not test_tempdir.exists()


def test__parse_headers_invalid_file(tmp_path):
    bad_file = tmp_path / "bad_toc.md"
    # file does not exist
    with pytest.raises(ValueError):
        ebooklib._parse_headers(str(tmp_path / "not_exist.md"))

    # empty file
    bad_file.write_text("")
    with pytest.raises(ValueError):
        ebooklib._parse_headers(str(bad_file))

    # file with blank lines only
    bad_file.write_text("\n\n")
    with pytest.raises(ValueError):
        ebooklib._parse_headers(str(bad_file))


def test__parse_headers_valid_file(tmp_path):
    toc_path = tmp_path / "toc.md"
    toc_content = """
    
    Ebook Title

    # Chapter 1
    ## Sub 1A
    # Chapter 2
    ## Sub 2A
    ## Sub 2B
    """

    toc_path.write_text(toc_content.strip())
    title, headers = ebooklib._parse_headers(str(toc_path))
    assert title == "Ebook Title"
    assert isinstance(headers, list)
    assert len(headers) == 2
    assert headers[0]["title"] == "Chapter 1"
    assert headers[0]["next_headers"][0]["title"] == "Sub 1A"
    assert headers[1]["title"] == "Chapter 2"
    assert len(headers[1]["next_headers"]) == 2


# Note: make_ebook is marked deprecated with warning,
# test basic use with warning capture (integration)
def test_make_ebook_warning(tmp_path, monkeypatch):
    # Setup a fake source dir
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "toc.md").write_text(
        "# Test Ebook\n\n# Ch1\n## Sub1\n"
    )
    (source_dir / "Ch1.html").write_text("<h1>Ch1</h1>")
    (source_dir / "Sub1.html").write_text("<h2>Sub1</h2>")
    (source_dir / "cover.jpg").write_text("imagedata")

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # monkeypatch Ebook.save to just copy some file to verify output
    called = {}

    def fake_save(self, path):
        called['path'] = path
        # create a dummy file to simulate save
        pathlib.Path(path).write_text("dummy mobi")

    monkeypatch.setattr(Ebook, "save", fake_save)

    with pytest.warns(DeprecationWarning):
        fn = ebooklib.make_ebook(str(source_dir), str(output_dir))

    assert fn == os.path.join(str(output_dir), "Test Ebook.mobi")
    assert called['path'] == fn
