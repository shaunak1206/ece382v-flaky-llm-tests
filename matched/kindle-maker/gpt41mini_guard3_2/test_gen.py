# tests/test_ebooklib.py

import io
import os
import pathlib
import platform
import shutil
import subprocess
import tempfile
import time
import builtins
import pytest
import random

from kindle_maker import ebooklib
from kindle_maker.ebooklib import Ebook, EbookUtil, Chapter, format_file_name


def test_format_file_name_basic():
    # test all replacements: '/', ' ', '+', '"', '\\', ':', '|'
    s = 'ab/c d+e"f\\g:h|i'
    expected = 'abcd-e-fgh-i'
    result = format_file_name(s)
    assert result == expected


def test_chapter_init_invalid_file(tmp_path):
    invalid_file = tmp_path / "nofile.html"
    # file doesn't exist
    with pytest.raises(ValueError):
        Chapter("title", ebook=None, file_path=str(invalid_file))


def test_chapter_dest_file_path_and_full_tmp_path(tmp_path):
    ebook = Ebook("My Title")
    chapter = Chapter("My Chapter", ebook)
    expected_name = pathlib.Path("MyChapter.html")
    assert chapter.dest_file_path == expected_name
    # full path should be inside ebook.tmpdir
    full_path = ebook.tmpdir / expected_name
    assert chapter.full_dest_file_path == full_path


def test_chapter_content_with_file(tmp_path):
    fn = tmp_path / "content.html"
    content_text = "Hello world"
    fn.write_text(content_text, encoding='utf-8')
    ebook = Ebook("Title")
    chapter = Chapter("Test", ebook, file_path=str(fn))
    # content property reads file content
    assert chapter.content == content_text


def test_chapter_content_with_set_content():
    ebook = Ebook("Title")
    chapter = Chapter("Test", ebook)
    set_text = "<html>content</html>"
    chapter.set_content(set_text)
    # After set_content, _file_path should be None
    assert chapter._file_path is None
    # content property returns the set content
    assert chapter.content == set_text


def test_chapter_is_top_chapter_property():
    ebook = Ebook("Title")
    c1 = Chapter("T1", ebook, level=1)
    c2 = Chapter("T2", ebook, level=2)
    assert c1.is_top_chapter is True
    assert c2.is_top_chapter is False


def test_chapter_add_static_files(tmp_path):
    # create some files
    f1 = tmp_path / "f1.txt"
    f2 = tmp_path / "f2.txt"
    f1.write_text("one")
    f2.write_text("two")
    ebook = Ebook("Title")
    chapter = Chapter("Test", ebook)
    # Should accept only existing files
    chapter.add_static_files(str(f1), str(f2))
    assert str(f1) in chapter._static_files
    assert str(f2) in chapter._static_files
    # Non-existent file raises
    nonexistent = tmp_path / "nofile"
    with pytest.raises(AssertionError):
        chapter.add_static_files(str(nonexistent))


def test_chapter_create_subchapter_and_limits():
    ebook = Ebook("Title")
    chapter = Chapter("T1", ebook, level=1)
    sub = chapter.create_subchapter("sub1")
    assert isinstance(sub, Chapter)
    assert sub._level == 2
    # max_level is 2, so another subchapter from sub will fail
    with pytest.raises(Exception):
        sub.create_subchapter("subsub1")


def test_chapter_save(tmp_path):
    ebook = Ebook("Title")
    chapter = Chapter("Test", ebook)
    # Set content directly
    chapter.set_content("<html>abc</html>")
    # Save should create the html file with content in the tmpdir
    chapter.save()
    dest = chapter.full_dest_file_path
    assert dest.is_file()
    text = dest.read_text(encoding='utf-8')
    assert text == "<html>abc</html>"
    # Copying static files is tested by presence in tmpdir
    f_static = tmp_path / "stat.txt"
    f_static.write_text("static content")
    chapter.add_static_files(str(f_static))
    chapter.save()
    # The static file should be copied to the same directory as html file
    copied = dest.parent / f_static.name
    assert copied.is_file()
    assert copied.read_text(encoding='utf-8') == "static content"


def test_ebook_set_cover_path_and_content(tmp_path):
    ebook = Ebook("MyBook")
    f_cover = tmp_path / "cover.jpg"
    f_cover.write_bytes(b"coverdata")
    ebook.set_cover(cover_path=str(f_cover))
    assert ebook.cover_path == str(f_cover)
    assert ebook.cover_content is None

    ebook2 = Ebook("Book2")
    # Using cover_content sets cover_path to None
    content = b"contentbytes"
    ebook2.set_cover(cover_content=content)
    assert ebook2.cover_content == content
    assert ebook2.cover_path is None

    # Neither path nor content raises ValueError
    ebook3 = Ebook("Book3")
    with pytest.raises(ValueError):
        ebook3.set_cover()


def test_ebook_add_chapter_only_top_level():
    ebook = Ebook("Title")
    chapter_top = Chapter("Chapter1", ebook, level=1)
    ebook.add_chapter(chapter_top)  # Should be okay
    chapter_sub = Chapter("Subchapter", ebook, level=2)
    with pytest.raises(ValueError):
        ebook.add_chapter(chapter_sub)


def test_ebook_create_chapter_and_add():
    ebook = Ebook("Title")
    ch = ebook.create_chapter("Chap1")
    assert ch in ebook.chapter_list
    assert ch.title == "Chap1"
    assert ch.is_top_chapter


def test_ebook_util_headings_property(tmp_path):
    # Create ebook with chapters and subchapters
    ebook = Ebook("BookTitle", author="Me")
    ch1 = ebook.create_chapter("Chapter1")
    ch1.set_content("Content 1")
    ch2 = ebook.create_chapter("Chapter2")
    ch2.set_content("Content 2")
    ch1_sub = ch1.create_subchapter("Sub1")
    ch1_sub.set_content("sub content")

    eu = EbookUtil(ebook)
    headings = eu.headings
    # Check headings structure
    assert isinstance(headings, list)
    # First heading title
    assert headings[0]["title"] == "Chapter1"
    # It has one subheading
    assert len(headings[0]["sub_headings"]) == 1
    assert headings[0]["sub_headings"][0]["title"] == "Sub1"
    # Second heading has no subheadings
    assert headings[1]["title"] == "Chapter2"
    assert headings[1]["sub_headings"] == []


def test_ebook_util_render_files(tmp_path, monkeypatch):
    # Setup dummy templates folder with simple templates
    templates_dir = pathlib.Path(__file__).parent / "templates_test"
    templates_dir.mkdir(exist_ok=True)
    # Create minimal templates needed
    (templates_dir / 'toc.xml').write_text(
        "TOC_TEMPLATE {{ title }}", encoding="utf-8")
    (templates_dir / 'toc.html').write_text(
        "TOC_HTML_TEMPLATE {{ headings|length }}", encoding="utf-8")
    (templates_dir / 'opf.xml').write_text(
        "OPF_TEMPLATE {{ author }}", encoding="utf-8")
    (templates_dir / 'cover.jpg').write_bytes(b"coverdefault")

    # Patch the templates env loader to our test dir
    ebook = Ebook("Title", "Author")
    # monkeypatch the templates path in EbookUtil
    def build_env(_self):
        from jinja2 import Environment, FileSystemLoader
        return Environment(loader=FileSystemLoader(str(templates_dir)))
    monkeypatch.setattr(ebooklib.EbookUtil, '__init__', lambda self, ebook: setattr(self, '_templates_env', build_env(self)) or setattr(self, '_ebook', ebook) or setattr(self, '_headings', None))
    # Recreate EbookUtil instance with patched env
    eu = ebooklib.EbookUtil(ebook)

    # Add chapters to have headings
    c1 = ebook.create_chapter("C1")
    c1.set_content("content1")
    c2 = ebook.create_chapter("C2")
    c2.set_content("content2")

    # Create the tmpdir if removed (in case __del__ cleared from previous tests)
    ebook.tmpdir.mkdir(exist_ok=True)

    # _render_file writes the rendered template to a file, test _render_toc_ncx
    res_ncx = eu._render_toc_ncx()
    assert res_ncx == 'toc.ncx'
    out_file = ebook.tmpdir / res_ncx
    assert out_file.is_file()
    text = out_file.read_text(encoding='utf-8')
    assert "TOC_TEMPLATE" in text
    assert "Title" in text

    # _render_toc_html
    res_html = eu._render_toc_html()
    assert res_html == 'toc.html'
    file_html = ebook.tmpdir / res_html
    assert file_html.is_file()
    txt_html = file_html.read_text(encoding='utf-8')
    assert "TOC_HTML_TEMPLATE" in txt_html
    # length of headings should be included in text
    assert str(len(eu.headings)) in txt_html

    # _render_opf
    res_opf = eu._render_opf()
    opf_path = ebook.tmpdir / res_opf
    assert opf_path.is_file()
    opf_content = opf_path.read_text(encoding='utf-8')
    assert "OPF_TEMPLATE" in opf_content
    assert "Author" in opf_content

    # _save_cover when cover_path is set copies that file
    cover_file = tmp_path = pathlib.Path(tempfile.mkdtemp()) / "mycover.jpg"
    cover_file.write_bytes(b"coverdata")
    ebook.cover_path = str(cover_file)
    # Clear any existing files in tmpdir
    for p in ebook.tmpdir.iterdir():
        if p.is_file():
            p.unlink()
    eu._save_cover()
    dest_cover = ebook.tmpdir / cover_file.name
    assert dest_cover.is_file()
    assert dest_cover.read_bytes() == b"coverdata"

    # _save_cover when cover_content is set raises NotImplementedError
    ebook.cover_path = None
    ebook.cover_content = b"bytescontent"
    with pytest.raises(NotImplementedError):
        eu._save_cover()

    # _save_cover when no cover set copies default template cover.jpg
    ebook.cover_content = None
    # Remove tmpdir's cover file if any
    for f in ebook.tmpdir.iterdir():
        if f.is_file():
            f.unlink()
    eu._save_cover()
    found_covers = [p for p in ebook.tmpdir.iterdir() if p.name == 'cover.jpg']
    assert found_covers


def test_ebook_util_save_chapters_calls_save(tmp_path):
    ebook = Ebook("Title")
    c1 = ebook.create_chapter("C1")
    c1.set_content("one")
    c2 = c1.create_subchapter("subc")  # make sure subchapter is saved too
    c2.set_content("two")

    saved = []

    def fake_save(self):
        saved.append(self.title)
    # Patch Chapter.save for both c1 and c2
    monkeypatcher = pytest.MonkeyPatch()
    monkeypatcher.setattr(ebooklib.Chapter, "save", fake_save)
    eu = EbookUtil(ebook)
    eu._save_chapters()
    monkeypatcher.undo()

    # Save should be called on both chapter and subchapter
    assert "C1" in saved
    assert "subc" in saved


def test_ebook_create_calls_kindlegen(monkeypatch, tmp_path):
    ebook = Ebook("MyBook")
    ch = ebook.create_chapter("ChapterOne")
    ch.set_content("test content")

    # Create minimal template files for render methods to not error
    templates_dir = pathlib.Path(__file__).parent / "templates"
    # We do not want to create real files there; monkeypatch _render_file to do nothing
    monkeypatcher = pytest.MonkeyPatch()
    monkeypatcher.setattr(EbookUtil, "_render_file", lambda self, tn, ctx, fn: None)
    monkeypatcher.setattr(EbookUtil, "_save_cover", lambda self: None)
    monkeypatcher.setattr(EbookUtil, "_save_chapters", lambda self: None)

    # Patch subprocess.call to record call args and return 0 (success)
    calls = []
    def fake_call(args):
        calls.append(args)
        return 0
    monkeypatch.setattr(subprocess, "call", fake_call)

    rc = ebook._eu.create()
    # It returns None (actual create returns None on success)
    assert rc is None

    # Check that subprocess.call called kindlegen with expected args containing .opf filename
    assert calls
    assert ebook.title.replace(" ", "") in calls[0][2]

    monkeypatcher.undo()


def test_ebook_create_raises_if_no_chapters():
    ebook = Ebook("EmptyBook")
    # chapter_list is empty, so _create raises ValueError
    with pytest.raises(ValueError):
        ebook._create()


def test_ebook_save_creates_and_copies(monkeypatch, tmp_path):
    ebook = Ebook("BookTitle")
    ch = ebook.create_chapter("Ch1")
    ch.set_content("<html>text</html>")

    # Patch _create to create a dummy mobi file in tmpdir
    def dummy_create():
        mobi_path = ebook.tmpdir / ebook.dest_file_path
        mobi_path.write_text("mobi content")
    monkeypatch.setattr(ebook, "_create", dummy_create)

    outfile = tmp_path / "output.mobi"
    ebook.save(str(outfile))
    assert outfile.is_file()
    assert outfile.read_text() == "mobi content"


@pytest.mark.skipif(platform.system() == "Linux", reason="Platform dependent call")
def test_ebook_show_calls_platform_functions(monkeypatch):
    ebook = Ebook("ShowBook")
    ch = ebook.create_chapter("C")
    ch.set_content("content")

    # Patch _create to no-op
    monkeypatch.setattr(ebook, "_create", lambda: None)

    called = {}

    def fake_call(args):
        called['args'] = args
        return 7

    monkeypatch.setattr(subprocess, "call", fake_call)
    monkeypatch.setattr(ebooklib.os, "startfile", lambda path: 9)

    # Patch platform.system to each supported platform and test return codes
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert ebook.show() == 7
    assert called['args'][0] == 'xdg-open'

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    assert ebook.show() == 7
    assert called['args'][0] == 'open'

    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(ebooklib.os, "startfile", lambda path: 42)
    assert ebook.show() == 42


def test_ebook_del_removes_tmpdir(tmp_path):
    # forcibly create an ebook, tmpdir with files and subdirs, then delete triggers cleanup
    ebook = Ebook("DelBook")

    # Create a file and a subdir + file inside tmpdir
    f1 = ebook.tmpdir / "file1.txt"
    f1.write_text("dummy")
    d1 = ebook.tmpdir / "subdir"
    d1.mkdir()
    f2 = d1 / "file2.txt"
    f2.write_text("dummy2")

    tmp_path = ebook.tmpdir
    # Confirm files/dirs exist
    assert f1.is_file()
    assert d1.is_dir()
    assert f2.is_file()

    # Call __del__ explicitly (normally called on gc)
    ebook.__del__()

    # tmpdir should be removed
    assert not tmp_path.exists()


def test__parse_headers_valid_and_invalid(tmp_path):
    # Create a valid toc.md file
    toc = tmp_path / 'toc.md'
    toc.write_text(
        """

        Book Title

        # Heading 1
        ## Subheading 1.1
        # Heading 2
        ## Subheading 2.1
        """,
        encoding='utf-8'
    )
    title, headers = ebooklib._parse_headers(str(toc))
    assert title == "Book Title"
    assert len(headers) == 2
    assert headers[0]['title'] == 'Heading 1'
    assert headers[0]['next_headers'][0]['title'] == 'Subheading 1.1'

    # Test file not exists
    bad_path = tmp_path / "notexist.md"
    with pytest.raises(ValueError):
        ebooklib._parse_headers(str(bad_path))

    # Test empty file
    empty_file = tmp_path / "empty.md"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        ebooklib._parse_headers(str(empty_file))

    # Test only empty lines
    only_empty = tmp_path / "onlyempty.md"
    only_empty.write_text("\n\n  \n", encoding="utf-8")
    with pytest.raises(ValueError):
        ebooklib._parse_headers(str(only_empty))

    # Test headings empty raises
    toc2 = tmp_path / 'toc2.md'
    toc2.write_text("Title\n", encoding='utf-8')
    with pytest.raises(ValueError):
        ebooklib._parse_headers(str(toc2))

