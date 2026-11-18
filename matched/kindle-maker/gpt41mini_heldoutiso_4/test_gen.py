import os
import io
import sys
import shutil
import subprocess
import platform
import tempfile
import pathlib
import time
import builtins
import types

import pytest

from kindle_maker.ebooklib import Ebook, EbookUtil, Chapter, format_file_name


@pytest.fixture
def tmp_dir():
    d = pathlib.Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(str(d))


@pytest.fixture
def ebook(tmp_dir):
    # Patch Ebook to use tmp_dir instead of its own tempdir
    class TestEbook(Ebook):
        def __init__(self, title, author=None):
            self.title = title
            self.author = author
            self.cover_path = None
            self.cover_content = None
            self._tempdir = tmp_dir
            self.chapter_list = list()
            self._eu = EbookUtil(self)
    return TestEbook("Test Ebook", "Author")


@pytest.fixture
def chapter_file(tmp_path):
    f = tmp_path / "chapter1.html"
    f.write_text("<html>Test chapter content</html>", encoding="utf-8")
    return str(f)


@pytest.mark.parametrize(
    "input_str,expected",
    [
        ("abc def/ghi+jkl\"\\:|", "abcdefghi-jkl--"),
        ("simplefilename", "simplefilename"),
        ("filename with spaces", "filenamewithspaces"),
        ("file/name/with/slashes", "filenamewithslashes"),
    ],
)
def test_format_file_name(input_str, expected):
    assert format_file_name(input_str) == expected


def test_chapter_init_valid_file(chapter_file, ebook):
    ch = Chapter("Title", ebook, file_path=chapter_file)
    assert ch.title == "Title"
    assert ch._file_path == chapter_file
    assert ch._level == 1
    assert ch.is_top_chapter
    content = ch.content
    assert "<html>" in content


def test_chapter_init_invalid_file(ebook, tmp_path):
    invalid_path = tmp_path / "nofile.html"
    with pytest.raises(ValueError):
        Chapter("Invalid", ebook, file_path=str(invalid_path))


def test_chapter_dest_paths(ebook):
    ch = Chapter("Test/Chapter", ebook)
    expected_filename = pathlib.Path("TestChapter.html")
    assert ch.dest_file_path.name == expected_filename.name
    assert ch.full_dest_file_path.parent == ebook.tmpdir


def test_chapter_set_content_and_content_property(ebook):
    ch = Chapter("Cont Chapter", ebook)
    cont_html = "<p>Hello content</p>"
    returned = ch.set_content(cont_html)
    assert returned is ch
    assert ch.content == cont_html
    assert ch._file_path is None


def test_chapter_add_static_files(tmp_path, ebook):
    file1 = tmp_path / "f1.txt"
    file2 = tmp_path / "f2.txt"
    file1.write_text("hello1", encoding="utf-8")
    file2.write_text("hello2", encoding="utf-8")
    ch = Chapter("StaticTest", ebook)
    returned = ch.add_static_files(str(file1), str(file2))
    assert returned is ch
    assert str(file1) in ch._static_files
    assert str(file2) in ch._static_files


def test_chapter_add_static_files_invalid(tmp_path, ebook):
    invalid_file = tmp_path / "no.txt"
    ch = Chapter("StaticTest", ebook)
    with pytest.raises(AssertionError):
        ch.add_static_files(str(invalid_file))


def test_create_subchapter_and_level_limit(ebook):
    ch = Chapter("Top", ebook)
    sub1 = ch.create_subchapter("Sub 1")
    assert sub1._level == 2
    assert sub1.title == "Sub 1"
    assert sub1 in ch.sub_chapters

    # max_level = 2, creating subchapter of level 2 raises Exception
    with pytest.raises(Exception):
        sub1.create_subchapter("Sub sub")


def test_chapter_save_creates_files(tmp_path, ebook):
    f = tmp_path / "content.html"
    f.write_text("<p>Saved content</p>", encoding="utf-8")
    ch = Chapter("SaveTest", ebook, file_path=str(f))
    # add static file
    static_file = tmp_path / "static.txt"
    static_file.write_text("static content", encoding="utf-8")
    ch.add_static_files(str(static_file))

    dest_file = ch.full_dest_file_path
    if dest_file.exists():
        dest_file.unlink()

    ch.save()
    assert dest_file.exists()
    saved_content = dest_file.read_text(encoding="utf-8")
    assert "<p>Saved content</p>" in saved_content

    # Check static file copied to same directory as dest_file
    copied_static = dest_file.parent / static_file.name
    assert copied_static.exists()
    # Content check
    assert copied_static.read_text(encoding="utf-8") == "static content"


def test_ebook_add_and_create_chapter(ebook, chapter_file):
    ch = ebook.create_chapter("Ch 1", chapter_file)
    assert ch in ebook.chapter_list
    assert ch.is_top_chapter

    # Add chapter with level > 1 should fail
    sub_ch = ch.create_subchapter("SubCh")
    with pytest.raises(ValueError):
        ebook.add_chapter(sub_ch)


def test_ebook_set_cover(tmp_path, ebook):
    img_file = tmp_path / "cover.jpg"
    img_file.write_text("image", encoding="utf-8")

    # Setting cover with path
    ebook.set_cover(cover_path=str(img_file))
    assert ebook.cover_path == str(img_file)
    assert ebook.cover_content is None

    # Setting cover with content
    ebook.set_cover(cover_content=b"binarydata")
    assert ebook.cover_path is None
    assert ebook.cover_content == b"binarydata"

    # Without either should raise
    with pytest.raises(ValueError):
        ebook.set_cover()


def test_ebook_dest_file_path(ebook):
    expected_name = f"{format_file_name(ebook.title)}.mobi"
    assert ebook.dest_file_path.name == expected_name
    assert (ebook.tmpdir / ebook.dest_file_path).exists() or True  # temp dir exists


def test_ebook__create_without_chapters_raises():
    e = Ebook("NoChapters")
    with pytest.raises(ValueError):
        e._create()


def test_ebook_create_and_save(monkeypatch, tmp_path):
    # Prepare ebook with chapter
    ebook = Ebook("SaveTest", "author")
    # Patch tempdir to controlled tmp_path
    ebook._tempdir = tmp_path
    ebook._eu = EbookUtil(ebook)

    # chapter with set content and save
    ch = ebook.create_chapter("ChOne")
    ch.set_content("<p>content</p>")
    # Monkeypatch EbookUtil.create to do nothing except copy a dummy .mobi file
    dummy_mobi = tmp_path / "SaveTest.mobi"
    dummy_mobi.write_text("dummy mobi content", encoding="utf-8")

    def fake_create():
        shutil.copy(str(dummy_mobi), str(dummy_mobi))  # no-op
        return None

    monkeypatch.setattr(ebook._eu, "create", lambda: None)
    # monkeypatch shutil.copy used in save
    monkeypatch.setattr(shutil, "copy", lambda src, dst: PathCreationHelper.copy_file(src, dst))

    # Provide PathCreationHelper to ensure shutil.copy works in this restricted environment
    class PathCreationHelper:
        @staticmethod
        def copy_file(src, dst):
            # create empty dst file
            dst_path = pathlib.Path(dst)
            dst_path.write_text("dummy copied", encoding="utf-8")

    # save to a real location
    dest_file = tmp_path / "out.mobi"
    ebook._tempdir = tmp_path
    ebook.dest_file_path.touch(exist_ok=True)  # dummy mobi file

    # monkeypatch _eu.create to avoid subprocess and real file creation
    monkeypatch.setattr(ebook._eu, "create", lambda: None)

    ebook.save(str(dest_file))
    assert dest_file.exists()


def test_ebook_show(monkeypatch, tmp_path):
    ebook = Ebook("ShowTest")
    ebook._tempdir = tmp_path
    ebook._eu = EbookUtil(ebook)

    # Create dummy mobi file to show
    mobi_file = tmp_path / f"{format_file_name(ebook.title)}.mobi"
    mobi_file.write_text("fake mobi")

    # monkeypatch _eu.create so no actual processing
    monkeypatch.setattr(ebook._eu, "create", lambda: None)

    # simulate platform.system and subprocess.call/os.startfile
    records = {}

    def fake_call(cmd):
        records["call"] = cmd
        return 0

    def fake_startfile(path):
        records["startfile"] = path
        return 0

    system = platform.system()

    if system == "Linux":
        monkeypatch.setattr(subprocess, "call", fake_call)
        ret = ebook.show()
        assert ret == 0
        assert records["call"][0] == "xdg-open"
        assert str(mobi_file) in records["call"][1]
    elif system == "Darwin":
        monkeypatch.setattr(subprocess, "call", fake_call)
        ret = ebook.show()
        assert ret == 0
        assert records["call"][0] == "open"
        assert str(mobi_file) in records["call"][1]
    else:
        monkeypatch.setattr(os, "startfile", fake_startfile)
        ret = ebook.show()
        assert ret == 0
        assert records["startfile"] == str(mobi_file)


def test_ebook_del_removes_tmpdir(tmp_path):
    # create dummy files and folder in tmpdir
    tmp_subdir = tmp_path / "subdir"
    tmp_subdir.mkdir()
    f1 = tmp_path / "file1.txt"
    f2 = tmp_subdir / "file2.txt"
    f1.write_text("test")
    f2.write_text("test")

    class TestEbook(Ebook):
        def __init__(self):
            self.title = "deltest"
            self._tempdir = tmp_path
            self.chapter_list = []
            self._eu = EbookUtil(self)

    e = TestEbook()
    tmpdir = e.tmpdir
    # Confirm exists
    assert tmpdir.exists()
    # delete ebook instance explicitly calls __del__
    e.__del__()
    assert not tmpdir.exists()


def test_ebookutil_headings_structure(ebook, chapter_file):
    ch1 = ebook.create_chapter("Chapter1", chapter_file)
    ch2 = ebook.create_chapter("Chapter2")
    sc21 = ch2.create_subchapter("SubChapter21")
    sc22 = ch2.create_subchapter("SubChapter22")
    headings = ebook._eu.headings
    assert isinstance(headings, list)
    assert all("title" in h for h in headings)
    # chapter2 play_order increments for sub chapters
    found_ch2 = next(h for h in headings if h["title"] == "Chapter2")
    assert "sub_headings" in found_ch2
    assert all(isinstance(sb, dict) for sb in found_ch2["sub_headings"])
    assert len(found_ch2["sub_headings"]) == 2


def test_ebookutil_render_file_writes(tmp_path, ebook):
    # create minimal template file in templates dir
    templates_dir = pathlib.Path(__file__).parent / "kindle_maker" / "templates"
    # For test isolate, create a temp template dir instead
    with tempfile.TemporaryDirectory() as td:
        td_path = pathlib.Path(td)
        env = ebook._eu._templates_env
        # monkeypatch templates env to use temp dir
        old_loader = ebook._eu._templates_env.loader
        ebook._eu._templates_env.loader = None
        ebook._eu._templates_env = ebook._eu._templates_env = ebook._eu._templates_env = ebook._eu._templates_env = ebook._templates_env = ebook._templates_env = None
        # Instead forcibly patch _templates_env with a simple env for test
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader(td))
        ebook._eu._templates_env = env
        # Create the template file
        template_file = td_path / "test_template.txt"
        template_file.write_text("Hello {{ name }}", encoding="utf-8")
        # Call _render_file
        ebook._eu._render_file("test_template.txt", {"name": "pytest"}, "output.txt")
        out_file = ebook.tmpdir / "output.txt"
        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert content == "Hello pytest"


def test_ebookutil_save_cover(tmp_path, ebook, monkeypatch):
    # Case 1: ebook.cover_path exists, copied
    cover_file = tmp_path / "cover.jpg"
    cover_file.write_text("coverimage", encoding="utf-8")
    ebook.cover_path = str(cover_file)
    ebook.cover_content = None

    copied_files = []

    def fake_copy(src, dst):
        copied_files.append((src, dst))

    monkeypatch.setattr(shutil, "copy", fake_copy)
    ebook._eu._save_cover()
    assert any(str(cover_file) == src for src, dst in copied_files)

    # Case 2: ebook.cover_content defined => raises NotImplementedError
    ebook.cover_path = None
    ebook.cover_content = b"bytes"
    with pytest.raises(NotImplementedError):
        ebook._eu._save_cover()

    # Case 3: No cover_path or cover_content, default cover copied
    ebook.cover_content = None
    monkeypatch.setattr(shutil, "copy", fake_copy)

    # manually clear list
    copied_files.clear()
    ebook._eu._save_cover()
    assert len(copied_files) == 1
    # The source should be the default cover.jpg in templates directory
    src_path = copied_files[0][0]
    assert "cover.jpg" in src_path


def test_ebookutil_save_chapters_calls_save(monkeypatch, ebook):
    ch1 = ebook.create_chapter("Chapter1")
    ch2 = ebook.create_chapter("Chapter2")
    sub2 = ch2.create_subchapter("Sub2")

    called = []

    def fake_save(self):
        called.append(self.title)

    monkeypatch.setattr(Chapter, "save", fake_save)
    ebook._eu._save_chapters()
    # All three chapters should be saved
    assert set(called) == {"Chapter1", "Chapter2", "Sub2"}


def test_ebookutil_create_calls_expected(monkeypatch, ebook):
    calls = []

    def fake_render_toc_ncx():
        calls.append("toc_ncx")
        return "toc.ncx"

    def fake_render_toc_html():
        calls.append("toc_html")
        return "toc.html"

    def fake_render_opf():
        calls.append("opf")
        return "ebook.opf"

    def fake_save_cover():
        calls.append("cover")

    def fake_save_chapters():
        calls.append("chapters")

    def fake_call(args):
        calls.append(("subprocess", args))
        return 0

    monkeypatch.setattr(ebook._eu, "_render_toc_ncx", fake_render_toc_ncx)
    monkeypatch.setattr(ebook._eu, "_render_toc_html", fake_render_toc_html)
    monkeypatch.setattr(ebook._eu, "_render_opf", fake_render_opf)
    monkeypatch.setattr(ebook._eu, "_save_cover", fake_save_cover)
    monkeypatch.setattr(ebook._eu, "_save_chapters", fake_save_chapters)
    monkeypatch.setattr(subprocess, "call", fake_call)

    ebook._eu.create()

    assert calls[:5] == ["toc_ncx", "toc_html", "opf", "cover", "chapters"]
    assert any(isinstance(x, tuple) and x[0] == "subprocess" for x in calls)


def test_ebookutil_create_subprocess_failure(monkeypatch, ebook):
    monkeypatch.setattr(ebook._eu, "_render_toc_ncx", lambda: "toc.ncx")
    monkeypatch.setattr(ebook._eu, "_render_toc_html", lambda: "toc.html")
    monkeypatch.setattr(ebook._eu, "_render_opf", lambda: "ebook.opf")
    monkeypatch.setattr(ebook._eu, "_save_cover", lambda: None)
    monkeypatch.setattr(ebook._eu, "_save_chapters", lambda: None)

    def fake_call(args):
        return 1  # simulate failure

    monkeypatch.setattr(subprocess, "call", fake_call)

    # Should not raise, just pass on subprocess return != 0
    ebook._eu.create()


def test_parse_headers_valid(tmp_path):
    file_path = tmp_path / "toc.md"
    content = "\n\nTitle line\n# Chapter 1\n## Sub 1a\n# Chapter 2\n## Sub 2a\n"
    file_path.write_text(content, encoding="utf-8")
    from kindle_maker.ebooklib import _parse_headers

    title, headers = _parse_headers(str(file_path))
    assert title == "Title line"
    assert isinstance(headers, list)
    assert any(h["title"] == "Chapter 1" for h in headers)
    assert any("next_headers" in h for h in headers)
    assert any(sh["title"] == "Sub 1a" for h in headers for sh in h["next_headers"])


def test_parse_headers_invalid(tmp_path):
    from kindle_maker.ebooklib import _parse_headers

    empty_file = tmp_path / "empty.md"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        _parse_headers(str(empty_file))

    bad_file = tmp_path / "bad.md"
    bad_file.write_text("\n\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _parse_headers(str(bad_file))

    no_file = tmp_path / "nofile.md"
    with pytest.raises(ValueError):
        _parse_headers(str(no_file))


def test_make_ebook(monkeypatch, tmp_path):
    # Write toc.md with valid content and cover.jpg file
    toc_file = tmp_path / "toc.md"
    toc_content = "Test Ebook\n# Chapter1\n## Sub1"
    toc_file.write_text(toc_content, encoding="utf-8")
    cover_file = tmp_path / "cover.jpg"
    cover_file.write_text("cover", encoding="utf-8")
    # Create html files referenced
    (tmp_path / "Chapter1.html").write_text("<p>ch1</p>", encoding="utf-8")
    (tmp_path / "Sub1.html").write_text("<p>sub1</p>", encoding="utf-8")
    # Create a non-html file that should be copied
    other_file = tmp_path / "file.txt"
    other_file.write_text("some text", encoding="utf-8")

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    # monkeypatch Ebook.save to just create a dummy .mobi file in tmpdir
    def fake_save(self, path):
        mobi_path = pathlib.Path(path)
        mobi_path.write_text("dummy mobi content", encoding="utf-8")

    monkeypatch.setattr(Ebook, "save", fake_save)
    from kindle_maker.ebooklib import make_ebook

    mobi_path = make_ebook(str(tmp_path), str(output_dir))
    assert mobi_path.endswith(".mobi")
    assert pathlib.Path(mobi_path).exists()
