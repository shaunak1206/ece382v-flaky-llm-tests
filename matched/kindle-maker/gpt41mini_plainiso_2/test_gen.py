# tests/test_ebooklib.py
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

from kindle_maker import ebooklib


@pytest.fixture
def tmp_text_file():
    tmp_path = pathlib.Path(tempfile.gettempdir()) / f"testfile_{time.time()}.txt"
    tmp_path.write_text("Hello World\nContent of file.", encoding="utf-8")
    yield tmp_path
    if tmp_path.exists():
        tmp_path.unlink()


@pytest.fixture
def tmp_html_file():
    tmp_path = pathlib.Path(tempfile.gettempdir()) / f"chapter_{time.time()}.html"
    tmp_path.write_text("<html><body>Test Chapter</body></html>", encoding="utf-8")
    yield tmp_path
    if tmp_path.exists():
        tmp_path.unlink()


@pytest.fixture
def ebook_instance():
    book = ebooklib.Ebook("My Test Book", author="Tester")
    yield book
    # Explicitly delete to trigger cleanup
    del book


def test_format_file_name_replaces_illegal_chars():
    s = 'a/b c+d"e\\f:g|h'
    expected = 'abc-def-g-h'
    assert ebooklib.format_file_name(s) == expected


def test_chapter_init_with_invalid_file_path_raises(tmp_text_file):
    with pytest.raises(ValueError):
        ebooklib.Chapter("Title", ebook=None, file_path=str(tmp_text_file.parent / "nonexistent.file"))


def test_chapter_dest_file_path_and_full_dest_file_path(ebook_instance):
    c = ebooklib.Chapter("Chapter 1: Intro", ebook_instance)
    formatted = ebooklib.format_file_name(c.title)
    assert c.dest_file_path == pathlib.Path(f"{formatted}.html")
    assert c.full_dest_file_path == ebook_instance.tmpdir / c.dest_file_path


def test_chapter_content_reads_file(tmp_html_file, ebook_instance):
    c = ebooklib.Chapter("Title", ebook_instance, file_path=str(tmp_html_file))
    assert "Test Chapter" in c.content


def test_chapter_content_returns_set_content():
    c = ebooklib.Chapter("Title", ebook=None)
    c.set_content("my custom content")
    assert c.content == "my custom content"


def test_chapter_is_top_chapter_property():
    top_ch = ebooklib.Chapter("Top", ebook=None, level=1)
    sub_ch = ebooklib.Chapter("Sub", ebook=None, level=2)
    assert top_ch.is_top_chapter is True
    assert sub_ch.is_top_chapter is False


def test_chapter_set_content_resets_file_path():
    c = ebooklib.Chapter("Test", ebook=None, file_path="foo.html")
    c.set_content("new content")
    assert c._file_path is None
    assert c._content == "new content"


def test_chapter_add_static_files_valid(tmp_text_file):
    c = ebooklib.Chapter("Test", ebook=None)
    c.add_static_files(str(tmp_text_file))
    assert str(tmp_text_file) in c._static_files


def test_chapter_add_static_files_invalid_path_raises():
    c = ebooklib.Chapter("Test", ebook=None)
    with pytest.raises(AssertionError):
        c.add_static_files("nonexistent.file")


def test_chapter_create_subchapter_and_level_restriction(ebook_instance):
    c = ebooklib.Chapter("Parent", ebook_instance, level=1)
    sc = c.create_subchapter("Child")
    assert sc._level == 2
    assert sc in c.sub_chapters
    with pytest.raises(Exception):
        sc.create_subchapter("Grandchild")  # max_level=2 disallows level 3


def test_chapter_save_writes_file(tmp_text_file, ebook_instance):
    c = ebooklib.Chapter("SaveTest", ebook_instance, file_path=str(tmp_text_file))
    c.add_static_files(str(tmp_text_file))
    c.save()
    saved_path = c.full_dest_file_path
    assert saved_path.exists()
    content = saved_path.read_text(encoding="utf-8")
    assert "Hello World" in content
    # Static file copied
    copied_file = saved_path.parent / tmp_text_file.name
    assert copied_file.exists()
    # Cleanup saved files
    saved_path.unlink()
    copied_file.unlink()


def test_ebook_set_cover_path(tmp_text_file, ebook_instance):
    ebook_instance.set_cover(cover_path=str(tmp_text_file))
    assert ebook_instance.cover_path == str(tmp_text_file)
    assert ebook_instance.cover_content is None


def test_ebook_set_cover_content(ebook_instance):
    content = b"cover data"
    ebook_instance.set_cover(cover_content=content)
    assert ebook_instance.cover_content == content
    assert ebook_instance.cover_path is None


def test_ebook_set_cover_raises_on_empty(ebook_instance):
    with pytest.raises(ValueError):
        ebook_instance.set_cover()


def test_ebook_add_chapter_rejects_non_top_chapters(ebook_instance):
    c = ebooklib.Chapter("Subchapter", ebook_instance, level=2)
    with pytest.raises(ValueError):
        ebook_instance.add_chapter(c)


def test_ebook_add_and_create_chapter(ebook_instance):
    c = ebooklib.Chapter("Intro", ebook_instance)
    ebook_instance.add_chapter(c)
    assert c in ebook_instance.chapter_list

    c2 = ebook_instance.create_chapter("Chapter Two")
    assert c2 in ebook_instance.chapter_list
    assert c2.title == "Chapter Two"


def test_ebook_dest_file_paths(ebook_instance):
    formatted = ebooklib.format_file_name(ebook_instance.title)
    assert ebook_instance.dest_file_path == pathlib.Path(f"{formatted}.mobi")
    assert ebook_instance.full_dest_file_path == ebook_instance.tmpdir / ebook_instance.dest_file_path


def test_ebook__create_raises_without_chapters(ebook_instance):
    # remove any chapters from fixture
    ebook_instance.chapter_list.clear()
    with pytest.raises(ValueError):
        ebook_instance._create()


def test_ebook_show_calls_correct_method(monkeypatch, ebook_instance):
    # Add a dummy chapter to avoid _create raise
    c = ebook_instance.create_chapter("Test", None)

    monkeypatch.setattr(ebook_instance._eu, "create", lambda: 0)

    # Patch subprocess.call and os.startfile to test calls
    called = {}

    def fake_call(args):
        called['args'] = args
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)

    if platform.system() == 'Linux':
        ret = ebook_instance.show()
        assert called['args'][0] == 'xdg-open'
        assert ret == 0
    elif platform.system() == 'Darwin':
        ret = ebook_instance.show()
        assert called['args'][0] == 'open'
        assert ret == 0
    else:
        # For other OSes, patch os.startfile
        monkeypatch.setattr(os, "startfile", lambda path: 42)
        ret = ebook_instance.show()
        assert ret == 42


def test_ebook_del_cleans_tempdir(ebook_instance):
    tmpdir = ebook_instance.tmpdir
    # Write a file to tempdir
    file_path = tmpdir / "tempfile.txt"
    file_path.write_text("temp data")
    assert file_path.exists()
    # Call __del__ explicitly
    ebook_instance.__del__()
    assert not tmpdir.exists()


def test_ebookutil_headings_property(ebook_instance):
    c1 = ebook_instance.create_chapter("Ch1")
    c1_sub = c1.create_subchapter("Sub1")
    c2 = ebook_instance.create_chapter("Ch2")
    eu = ebook_instance._eu
    hs = eu.headings
    assert isinstance(hs, list)
    assert any(h['title'] == "Ch1" for h in hs)
    ch1_heading = next(h for h in hs if h['title'] == "Ch1")
    assert any(sc['title'] == "Sub1" for sc in ch1_heading['sub_headings'])


def test_ebookutil_render_file_creates_file(tmp_text_file, ebook_instance, tmp_path):
    # monkeypatch the template to use a dummy template in temp path
    dummy_template_name = "dummy_template.txt"
    dummy_template_path = pathlib.Path(__file__).parent / "templates" / dummy_template_name
    # Create templates directory and file if not exist
    templates_dir = pathlib.Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    dummy_template_path.write_text("Title: {{title}}", encoding="utf-8")

    eu = ebook_instance._eu
    eu._render_file(dummy_template_name, {"title": "MyTitle"}, dummy_template_name)
    written_file = ebook_instance.tmpdir / dummy_template_name
    assert written_file.exists()
    content = written_file.read_text(encoding="utf-8")
    assert "MyTitle" in content

    # Cleanup dummy template
    dummy_template_path.unlink()


def test_ebookutil_save_cover_copies_cover_file(tmp_text_file, ebook_instance):
    ebook_instance.set_cover(cover_path=str(tmp_text_file))
    eu = ebook_instance._eu
    eu._save_cover()
    copied_cover = ebook_instance.tmpdir / pathlib.Path(tmp_text_file).name
    assert copied_cover.exists()


def test_ebookutil_save_cover_raises_not_implemented(ebook_instance):
    ebook_instance.cover_path = None
    ebook_instance.cover_content = b"somebytes"
    eu = ebook_instance._eu
    with pytest.raises(NotImplementedError):
        eu._save_cover()


def test_ebookutil_save_chapters_creates_files(ebook_instance, tmp_html_file):
    c = ebook_instance.create_chapter("Chap1", str(tmp_html_file))
    sc = c.create_subchapter("Sub1")
    sc.set_content("sub chapter content")
    eu = ebook_instance._eu
    eu._save_chapters()
    assert c.full_dest_file_path.exists()
    assert sc.full_dest_file_path.exists()

    # Cleanup created files
    c.full_dest_file_path.unlink()
    sc.full_dest_file_path.unlink()


def test_ebookutil_create_calls_kindlegen(monkeypatch, ebook_instance):
    c = ebook_instance.create_chapter("One")
    eu = ebook_instance._eu
    eu._render_toc_ncx = lambda: "toc.ncx"
    eu._render_toc_html = lambda: "toc.html"
    eu._render_opf = lambda: "ebook.opf"
    eu._save_cover = lambda: None
    eu._save_chapters = lambda: None

    called_with = {}

    def fake_call(args):
        called_with['args'] = args
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)
    ret = eu.create()
    assert "kindlegen" in called_with['args'][0]
    assert str((ebook_instance.tmpdir / "ebook.opf")) in called_with['args'][1]


def test__parse_headers_valid_and_invalid(tmp_path):
    # create a valid toc.md file
    toc_path = tmp_path / "toc.md"
    content = "\n\nMy Book\n# Chapter1\n## Sub1\n# Chapter2\n"
    toc_path.write_text(content, encoding="utf-8")

    title, headers = ebooklib._parse_headers(str(toc_path))
    assert title == "My Book"
    assert any(h['title'] == "Chapter1" for h in headers)
    assert any('next_headers' in h for h in headers)

    # invalid file path
    with pytest.raises(ValueError):
        ebooklib._parse_headers(str(tmp_path / "nonexistent.md"))

    # invalid contents - empty file
    empty_path = tmp_path / "empty.md"
    empty_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        ebooklib._parse_headers(str(empty_path))

    # invalid toc with no headings
    no_heading_path = tmp_path / "noheading.md"
    no_heading_path.write_text("Title only", encoding="utf-8")
    # forcibly pass file with no # headers
    with pytest.raises(ValueError):
        ebooklib._parse_headers(str(no_heading_path))


def test_make_ebook_warns_and_runs(tmp_path, monkeypatch):
    # Prepare source_dir with toc.md and html files
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    toc_md = source_dir / "toc.md"
    toc_md.write_text("# MyBook\n# Chapter1\n", encoding="utf-8")
    html_chapter = source_dir / "Chapter1.html"
    html_chapter.write_text("<html>content</html>", encoding="utf-8")

    # A dummy cover file
    cover = source_dir / "cover.jpg"
    cover.write_bytes(b"coverdata")

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Patch Ebook.save to just create the .mobi file without full processing
    def fake_save(self, file_path):
        pathlib.Path(file_path).write_text("mobi content", encoding="utf-8")

    monkeypatch.setattr(ebooklib.Ebook, 'save', fake_save)

    # Patch warnings.warn to track call
    warnings_called = {}

    def fake_warn(msg, *args, **kwargs):
        warnings_called['msg'] = str(msg)

    monkeypatch.setattr(ebooklib.warnings, 'warn', fake_warn)

    mobi_file = ebooklib.make_ebook(str(source_dir), str(output_dir))
    dest_file = output_dir / "MyBook.mobi"
    assert mobi_file == str(dest_file)
    assert dest_file.exists()
    assert "mobi content" in dest_file.read_text(encoding="utf-8")
    assert "deprecated" in warnings_called.get('msg', '').lower()
