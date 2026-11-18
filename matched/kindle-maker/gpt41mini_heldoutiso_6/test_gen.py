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
def ebook():
    eb = Ebook("Test Ebook", author="Author Name")
    yield eb
    # cleanup tempdir explicitly (in case __del__ doesn't get called timely)
    if eb.tmpdir.exists():
        shutil.rmtree(eb.tmpdir)


@pytest.fixture
def temp_file():
    # create a temporary file with some content
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as f:
        f.write("test content")
        f.flush()
        fname = f.name
    yield fname
    if os.path.exists(fname):
        os.unlink(fname)


def test_format_file_name_removes_illegal_chars():
    s = 'a/b c+d"e\\f:g|h'
    cleaned = format_file_name(s)
    assert '/' not in cleaned
    assert ' ' not in cleaned
    assert '+' not in cleaned
    assert '"' not in cleaned
    assert '\\' not in cleaned
    assert ':' not in cleaned
    assert '|' not in cleaned
    assert '-' in cleaned  # replaced + and : and | by -


def test_chapter_init_with_valid_file(temp_file, ebook):
    c = Chapter("Title", ebook, file_path=temp_file)
    assert c.title == "Title"
    assert c._file_path == temp_file
    assert c._level == 1
    assert c.is_top_chapter


def test_chapter_init_with_invalid_file_raises(ebook):
    with pytest.raises(ValueError):
        Chapter("Title", ebook, file_path="nonexistent_file.html")


def test_chapter_set_content_clears_file_path_and_sets_content(ebook):
    c = Chapter("Title", ebook)
    c.set_content("hello world")
    assert c._content == "hello world"
    assert c._file_path is None
    assert c.content == "hello world"


def test_chapter_content_reads_file(temp_file, ebook):
    c = Chapter("Title", ebook, file_path=temp_file)
    content = c.content
    with open(temp_file, encoding='utf-8') as f:
        expected = f.read()
    assert content == expected


def test_chapter_create_subchapter_and_levels(ebook):
    top = Chapter("Top", ebook)
    sub = top.create_subchapter("Sub")
    assert sub._level == top._level + 1 == 2
    assert sub in top.sub_chapters
    with pytest.raises(Exception):
        # cannot create subchapter beyond max level 2
        sub.create_subchapter("Too Deep")


def test_chapter_add_static_files_and_save(tmp_path, ebook):
    c = Chapter("Test Chapter", ebook)
    # create a temp static file
    static_file = tmp_path / "static.txt"
    static_file.write_text("static content", encoding='utf-8')
    c.add_static_files(str(static_file))
    c.set_content("chapter content")

    # Override ebook.tmpdir to tmp_path for isolation
    ebook._tempdir = tmp_path

    c.save()
    saved_file = tmp_path / c.dest_file_path
    assert saved_file.exists()
    with saved_file.open(encoding='utf-8') as f:
        assert f.read() == "chapter content"
    # static file copied to tmpdir
    copied_static = tmp_path / static_file.name
    assert copied_static.exists()


def test_ebook_add_chapter_only_top_level_allowed(ebook):
    c = Chapter("Top", ebook)
    ebook.add_chapter(c)  # should be fine
    sub = c.create_subchapter("Sub")
    with pytest.raises(ValueError):
        ebook.add_chapter(sub)  # non top-level not allowed


def test_ebook_create_chapter_adds_and_returns_chapter(ebook):
    ch = ebook.create_chapter("Chapter 1")
    assert ch.title == "Chapter 1"
    assert ch in ebook.chapter_list


def test_ebook_set_cover_path_and_content(ebook, tmp_path):
    # Create file for cover
    cover_file = tmp_path / "cover.jpg"
    cover_file.write_bytes(b"jpegdata")
    eb = ebook.set_cover(cover_path=str(cover_file))
    assert eb.cover_path == str(cover_file)
    assert eb.cover_content is None

    eb2 = Ebook("title").set_cover(cover_content=b"bytes")
    assert eb2.cover_content == b"bytes"
    assert eb2.cover_path is None

    with pytest.raises(ValueError):
        Ebook("no cover").set_cover()


def test_ebook_dest_and_full_dest_file_path(ebook):
    fname = format_file_name(ebook.title) + ".mobi"
    assert ebook.dest_file_path == pathlib.Path(fname)
    assert ebook.full_dest_file_path == ebook.tmpdir / fname


def test_ebook__create_raises_without_chapters(ebook):
    with pytest.raises(ValueError):
        ebook._create()


def test_ebook__create_calls_eu_create(monkeypatch, ebook):
    created = []

    def fake_create():
        created.append(True)

    ebook.create_chapter("C1")
    monkeypatch.setattr(ebook._eu, "create", fake_create)
    ebook._create()
    assert created == [True]


def test_ebook_create_and_save(monkeypatch, tmp_path, ebook):
    called = {}

    # monkey patch EbookUtil.create to generate dummy mobi file
    def fake_create():
        # create dummy mobi file in tmpdir
        mobi = ebook.tmpdir / ebook.dest_file_path
        mobi.write_text("mobi content", encoding='utf-8')
        called["create"] = True

    ebook.create_chapter("Test Chapter")
    monkeypatch.setattr(ebook._eu, "create", fake_create)

    out_file = tmp_path / "output.mobi"
    ebook.save(str(out_file))
    assert called.get("create") is True
    assert out_file.exists()
    with out_file.open(encoding='utf-8') as f:
        content = f.read()
    assert content == "mobi content"


def test_ebook_show_returns_platform_call(monkeypatch, ebook):
    ebook.create_chapter("C1")
    monkeypatch.setattr(ebook._eu, "create", lambda: None)

    system = platform.system()
    calls = {}

    def fake_subprocess_call(cmd):
        calls["cmd"] = cmd
        return 42

    def fake_os_startfile(path):
        calls["startfile"] = path
        return 43

    monkeypatch.setattr(subprocess, "call", fake_subprocess_call)
    monkeypatch.setattr(os, "startfile", fake_os_startfile)

    ret = ebook.show()
    if system == 'Linux':
        assert calls["cmd"][0] == 'xdg-open'
        assert ret == 42
    elif system == 'Darwin':
        assert calls["cmd"][0] == 'open'
        assert ret == 42
    else:
        # Windows path
        assert "startfile" in calls
        assert ret == 43


def test_ebookutil_headings_property_builds_structure(ebook):
    c1 = ebook.create_chapter("Chapter1")
    c1.set_content("content1")
    sc1 = c1.create_subchapter("Sub1")
    sc1.set_content("subcontent")
    eu = EbookUtil(ebook)
    hs = eu.headings
    assert isinstance(hs, list)
    assert any(h['title'] == "Chapter1" for h in hs)
    first = hs[0]
    assert "sub_headings" in first
    assert any(sb['title'] == "Sub1" for sb in first['sub_headings'])


def test_format_file_name_writes_expected(monkeypatch):
    s = 'a/b c+d"e\\f:g|h'
    # ensure replaced + : and | as dash
    result = format_file_name(s)
    assert all(c not in result for c in '/ "\\:')
    assert '-' in result


def test_ebookutil_render_file_writes_files(monkeypatch, tmp_path, ebook):
    # patch template to render fixed content
    class DummyTemplate:
        def render(self, **context):
            return "rendered content"

    def fake_get_template(name):
        return DummyTemplate()

    monkeypatch.setattr(ebook._eu._templates_env, "get_template", fake_get_template)
    eu = EbookUtil(ebook)
    monkeypatch.setattr(eu, "_ebook", ebook)
    # override tmpdir to tmp_path
    ebook._tempdir = tmp_path

    eu._render_file("dummy.tpl", {"foo": "bar"}, "output.html")
    output_file = tmp_path / "output.html"
    assert output_file.exists()
    with output_file.open(encoding='utf-8') as f:
        content = f.read()
    assert content == "rendered content"


def test_ebookutil_create_calls_expected_methods(monkeypatch, tmp_path, ebook):
    # Setup dummy methods to track calls
    calls = {}

    def fake_render_toc_ncx(self):
        calls['toc_ncx'] = True
        return "toc.ncx"

    def fake_render_toc_html(self):
        calls['toc_html'] = True
        return "toc.html"

    def fake_render_opf(self):
        calls['opf'] = True
        return "ebook.opf"

    def fake_save_cover(self):
        calls['cover'] = True

    def fake_save_chapters(self):
        calls['chapters'] = True

    def fake_subprocess_call(cmd):
        calls['subprocess'] = cmd
        return 0

    eu = ebook._eu
    monkeypatch.setattr(eu, "_render_toc_ncx", fake_render_toc_ncx)
    monkeypatch.setattr(eu, "_render_toc_html", fake_render_toc_html)
    monkeypatch.setattr(eu, "_render_opf", fake_render_opf)
    monkeypatch.setattr(eu, "_save_cover", fake_save_cover)
    monkeypatch.setattr(eu, "_save_chapters", fake_save_chapters)
    monkeypatch.setattr(subprocess, "call", fake_subprocess_call)

    # create at least one chapter to pass checks
    ebook.create_chapter("Dummy")
    ret = eu.create()
    assert ret == "ebook.opf"
    assert all(k in calls for k in ('toc_ncx', 'toc_html', 'opf', 'cover', 'chapters', 'subprocess'))
    # subprocess call has correct kindlegen and opf file path
    args = calls['subprocess']
    assert kindlegen_in_cmd(args)
    assert str(ebook.tmpdir / "ebook.opf") in args[-1]


def kindlegen_in_cmd(cmd_args):
    """Helper to check if kindlegen is in subprocess call"""
    from kindle_maker.ebooklib import kindlegen
    return cmd_args and kindlegen in cmd_args


def test_chapter_dest_file_path_and_full_dest_file_path(ebook):
    c = ebook.create_chapter("Test Chapter")
    fn = format_file_name(c.title) + ".html"
    assert c.dest_file_path == pathlib.Path(fn)
    assert c.full_dest_file_path == ebook.tmpdir / pathlib.Path(fn)


def test_ebook_del_removes_tempdir():
    eb = Ebook("Title")
    # temp dir exists
    assert eb.tmpdir.exists()
    td = eb.tmpdir
    # explicitly call __del__, forcing cleanup
    eb.__del__()
    assert not td.exists()
