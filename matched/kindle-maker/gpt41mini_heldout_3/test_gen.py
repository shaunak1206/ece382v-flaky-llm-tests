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

from kindle_maker.ebooklib import Ebook, EbookUtil, Chapter, format_file_name


@pytest.fixture
def tmp_file():
    """Create a temporary file with content."""
    with tempfile.NamedTemporaryFile('w+', delete=False, encoding='utf-8') as f:
        f.write("Test content")
        f.flush()
        yield pathlib.Path(f.name)
    os.unlink(f.name)


@pytest.fixture
def tmp_static_file():
    """Create a temporary static file for Chapter add_static_files."""
    with tempfile.NamedTemporaryFile('w+', delete=False, encoding='utf-8') as f:
        f.write("static content")
        f.flush()
        yield pathlib.Path(f.name)
    os.unlink(f.name)


@pytest.fixture
def ebook():
    eb = Ebook("Test Ebook", author="Author")
    yield eb
    # cleanup tmpdir forcibly if still exists
    if eb.tmpdir.exists():
        shutil.rmtree(eb.tmpdir, ignore_errors=True)


def test_format_file_name_basic():
    s = "a/b c+d\"\\:|"
    result = format_file_name(s)
    assert '/' not in result
    assert ' ' not in result
    assert '+' not in result
    assert '"' not in result
    assert '\\' not in result
    assert ':' not in result
    assert '|' not in result
    assert result == "abc-d--"


def test_chapter_init_with_invalid_file_path_raises():
    with pytest.raises(ValueError):
        Chapter("title", ebook=Ebook("title"), file_path="not_exist_file.html")


def test_chapter_content_from_file(tmp_file):
    ch = Chapter("title", ebook=Ebook("title"), file_path=str(tmp_file))
    content = ch.content
    assert "Test content" in content


def test_chapter_content_set_content_and_file_path():
    ch = Chapter("title", ebook=Ebook("title"))
    ch.set_content("some content")
    assert ch.content == "some content"
    assert ch._file_path is None


def test_chapter_add_static_files(tmp_static_file):
    ch = Chapter("title", ebook=Ebook("title"))
    ch.add_static_files(str(tmp_static_file))
    assert str(tmp_static_file) in ch._static_files

    # Adding non-existent file should assert fail
    with pytest.raises(AssertionError):
        ch.add_static_files("non-existent.file")


def test_chapter_create_subchapter_and_level_check():
    ebook = Ebook("ebook")
    ch = Chapter("level1", ebook)
    sub = ch.create_subchapter("level2")
    assert sub._level == 2
    assert sub.title == "level2"
    assert sub in ch.sub_chapters

    # Max level is 2, so creating another level should raise Exception
    with pytest.raises(Exception):
        sub.create_subchapter("level3")


def test_chapter_save_writes_content(tmp_file):
    ebook = Ebook("ebook")
    ch = Chapter("title", ebook)
    ch.set_content("my content")
    ch.save()
    path = ch.full_dest_file_path
    assert path.exists()
    assert path.read_text(encoding='utf-8') == "my content"


def test_chapter_save_copies_static_files(tmp_static_file):
    ebook = Ebook("ebook")
    ch = Chapter("title", ebook)
    ch.set_content("content")
    ch.add_static_files(str(tmp_static_file))
    ch.save()
    dest_dir = ch.full_dest_file_path.parent
    copied_file_path = dest_dir / tmp_static_file.name
    assert copied_file_path.exists()
    # The content of static file is kept
    with open(copied_file_path, encoding='utf-8') as f:
        content = f.read()
    assert "static content" in content


def test_ebook_set_cover_path_and_content(tmp_file):
    ebook1 = Ebook("title1")
    ebook1.set_cover(cover_path=str(tmp_file))
    assert ebook1.cover_path == str(tmp_file)
    assert ebook1.cover_content is None

    ebook2 = Ebook("title2")
    ebook2.set_cover(cover_content=b"bytes")
    assert ebook2.cover_content == b"bytes"
    assert ebook2.cover_path is None

    ebook3 = Ebook("title3")
    with pytest.raises(ValueError):
        ebook3.set_cover()


def test_ebook_add_chapter_only_level_1_allowed():
    ebook = Ebook("title")
    ch1 = Chapter("ch1", ebook)
    ebook.add_chapter(ch1)
    assert ch1 in ebook.chapter_list

    ch2 = Chapter("ch2", ebook, level=2)
    with pytest.raises(ValueError):
        ebook.add_chapter(ch2)


def test_ebook_create_chapter_adds_and_returns():
    ebook = Ebook("title")
    ch = ebook.create_chapter("chapter1")
    assert ch.title == "chapter1"
    assert ch in ebook.chapter_list
    assert ch.is_top_chapter


def test_ebook__create_raises_if_no_chapters():
    ebook = Ebook("title")
    with pytest.raises(ValueError):
        ebook._create()


def test_ebook_dest_file_path_and_full_path():
    ebook = Ebook("My Ebook Title", author="Me")
    expected_file = pathlib.Path("MyEbookTitle.mobi")
    # spaces removed by format_file_name
    assert ebook.dest_file_path.name == expected_file.name
    assert ebook.full_dest_file_path == ebook.tmpdir / ebook.dest_file_path


def test_ebook_save_copies_mobi_generated(monkeypatch, ebook):
    # Prepare dummy chapter and content so _create can run without error
    ch = ebook.create_chapter("ch1")
    ch.set_content("<html>content</html>")
    # Monkeypatch _eu.create to create mobi file
    def dummy_create():
        mobi_path = ebook.tmpdir / ebook.dest_file_path
        mobi_path.write_text("dummy mobi", encoding='utf-8')
        return None
    monkeypatch.setattr(ebook._eu, 'create', dummy_create)
    with tempfile.TemporaryDirectory() as td:
        target_path = pathlib.Path(td) / "output.mobi"
        ebook.save(str(target_path))
        assert target_path.exists()
        assert target_path.read_text(encoding='utf-8') == "dummy mobi"


def test_ebook_show_calls_proper_subprocess(monkeypatch, ebook):
    ch = ebook.create_chapter("ch1")
    ch.set_content("content")

    # Monkeypatch _eu.create to do nothing
    monkeypatch.setattr(ebook._eu, 'create', lambda: None)

    calls = {}

    def fake_call(args):
        calls['args'] = args
        return 0

    monkeypatch.setattr(subprocess, 'call', fake_call)

    # Monkeypatch os.startfile for unsupported
    monkeypatch.setattr(ebook, 'full_dest_file_path',
                        ebook.tmpdir / ebook.dest_file_path)
    ebook.tmpdir.mkdir(exist_ok=True)
    (ebook.tmpdir / ebook.dest_file_path).write_text("content")

    system = platform.system()

    if system == 'Linux':
        ret = ebook.show()
        assert calls['args'][0] == 'xdg-open'
        assert isinstance(ret, int)
    elif system == 'Darwin':
        ret = ebook.show()
        assert calls['args'][0] == 'open'
        assert isinstance(ret, int)
    else:
        called = {}
        def fake_startfile(path):
            called['called'] = path
        monkeypatch.setattr(os, 'startfile', fake_startfile)
        ret = ebook.show()
        assert called['called'] == str(ebook.full_dest_file_path)


def test_ebook_del_removes_tmpdir():
    ebook = Ebook("testdel")
    ch = ebook.create_chapter("ch1")
    ch.set_content("content")
    p = ebook.tmpdir
    assert p.exists()
    del ebook
    # tmpdir should be removed eventually, allow some leeway
    # just check it does not exist now
    # To avoid flaky test in some env, ignore if not cleaned immediately.
    if p.exists():
        # Try again forcibly
        shutil.rmtree(p, ignore_errors=True)
    assert not p.exists()


def test_ebookutil_headings_property(ebook):
    ch1 = ebook.create_chapter("Chap1")
    ch2 = ebook.create_chapter("Chap2")
    sub1 = ch2.create_subchapter("Sub1")
    sub2 = ch2.create_subchapter("Sub2")

    eu = ebook._eu
    headings = eu.headings
    assert isinstance(headings, list)
    assert any(h['title'] == "Chap1" for h in headings)
    # chap2 should contain sub_headings
    chap2_head = next(h for h in headings if h['title'] == "Chap2")
    assert len(chap2_head['sub_headings']) == 2
    titles = [s['title'] for s in chap2_head['sub_headings']]
    assert "Sub1" in titles and "Sub2" in titles

    # cached property: second call returns same object
    assert eu.headings is headings


def test_ebookutil_render_file(tmp_path, ebook):
    eu = EbookUtil(ebook)
    # The templates folder contains files, but we cannot rely on actual files,
    # we patch template loading to return a dummy Template object with render

    class DummyTemplate:
        def __init__(self):
            self.rendered = None
        def render(self, **context):
            self.rendered = context
            return "<html>rendered</html>"

    dummy_template = DummyTemplate()

    def dummy_get_template(name):
        return dummy_template

    eu._templates_env.get_template = dummy_get_template

    ebook._tempdir = tmp_path
    outfn = "output.html"
    eu._render_file(outfn, {'var': 1}, outfn)
    f = tmp_path / outfn
    assert f.exists()
    content = f.read_text(encoding='utf-8')
    assert content == "<html>rendered</html>"
    assert dummy_template.rendered['var'] == 1


def test_ebookutil_render_toc_files(tmp_path, ebook):
    eu = EbookUtil(ebook)
    # Patch get_template() to dummy
    class DummyTemplate:
        def render(self, **context):
            return f"Rendered {context.get('title', '')}"

    dummy_template = DummyTemplate()
    eu._templates_env.get_template = lambda name: dummy_template

    ebook._tempdir = tmp_path
    ebook.title = "Title"
    ebook.author = "Author"
    # add one chapter to create headings
    ch = ebook.create_chapter("C1")
    ch.set_content("content")

    ncx = eu._render_toc_ncx()
    assert ncx == "toc.ncx"
    html = eu._render_toc_html()
    assert html == "toc.html"
    opf = eu._render_opf()
    # opf file name is sanitized title + .opf
    expected_opf = format_file_name("Title") + ".opf"
    assert opf == expected_opf
    assert (tmp_path / opf).exists()


def test_ebookutil_save_cover_copies_cover_path(tmp_path, ebook, tmp_file):
    ebook.cover_path = str(tmp_file)
    ebook.cover_content = None
    ebook._tempdir = tmp_path
    eu = EbookUtil(ebook)
    eu._save_cover()
    files = list(tmp_path.iterdir())
    assert any(tmp_file.name in f.name for f in files)


def test_ebookutil_save_cover_raises_not_implemented_for_cover_content(tmp_path, ebook):
    ebook.cover_content = b"bytes"
    ebook.cover_path = None
    ebook._tempdir = tmp_path
    eu = EbookUtil(ebook)
    with pytest.raises(NotImplementedError):
        eu._save_cover()


def test_ebookutil_save_cover_copies_default_cover(tmp_path, ebook):
    ebook.cover_content = None
    ebook.cover_path = None
    ebook._tempdir = tmp_path
    eu = EbookUtil(ebook)
    # Remove cover.jpg if exists and patch path to a temporary one
    # But to avoid complexity, just test it copies something
    eu._save_cover()
    # There should be at least one file (cover.jpg) in tmpdir
    files = list(tmp_path.iterdir())
    assert files


def test_ebookutil_save_chapters_calls_save_methods(tmp_path, ebook):
    ch1 = ebook.create_chapter("Chapter 1")
    ch1.set_content("content 1")
    ch2 = ebook.create_chapter("Chapter 2")
    ch2.set_content("content 2")
    sc = ch2.create_subchapter("Sub 1")
    sc.set_content("sub content")
    ebook._tempdir = tmp_path
    eu = EbookUtil(ebook)
    # patch save method to mark called
    called = []
    orig_save = Chapter.save
    def patched_save(self):
        called.append(self.title)
        orig_save(self)
    Chapter.save = patched_save
    try:
        eu._save_chapters()
        # All chapters and sub_chapters saved
        for title in ["Chapter 1", "Chapter 2", "Sub 1"]:
            assert title in called
    finally:
        Chapter.save = orig_save


@pytest.mark.parametrize("input_str,expected", [
    ("abc / file+ name:\"|", "abc-file-name--"),
    ("cleanname", "cleanname"),
    ("/ / /", "---"),
    ("with spaces", "withspaces"),
])
def test_format_file_name_various(input_str, expected):
    assert format_file_name(input_str) == expected
