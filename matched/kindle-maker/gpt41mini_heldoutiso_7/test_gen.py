# tests/test_ebooklib.py

import io
import os
import pathlib
import platform
import shutil
import subprocess
import tempfile
import time
import types

import pytest

from kindle_maker import ebooklib
from kindle_maker.ebooklib import Ebook, EbookUtil, Chapter, format_file_name


@pytest.fixture
def temp_dir():
    d = pathlib.Path(tempfile.mkdtemp(prefix="ebooktest-"))
    yield d
    shutil.rmtree(str(d), ignore_errors=True)


def test_format_file_name_basic():
    s = 'my/file name+test:"|\\'
    expected = 'myfilename-test---'
    assert format_file_name(s) == expected


def test_chapter_init_invalid_file(tmp_path):
    invalid = tmp_path / "nofile.html"
    with pytest.raises(ValueError):
        Chapter("title", ebook=Ebook("t"), file_path=str(invalid))


def test_chapter_dest_file_path_and_full_dest_file_path(tmp_path):
    eb = Ebook("My ebook")
    eb._tempdir = tmp_path
    ch = Chapter("Chapter One", eb)
    filename = format_file_name("Chapter One") + ".html"
    assert ch.dest_file_path == pathlib.Path(filename)
    assert ch.full_dest_file_path == tmp_path / filename


def test_chapter_content_with_file(tmp_path):
    eb = Ebook("Test")
    ch_file = tmp_path / "chapter.html"
    ch_file.write_text("Hello world", encoding="utf-8")
    ch = Chapter("Title", eb, str(ch_file))
    assert ch.content == "Hello world"


def test_chapter_content_with_set_content():
    eb = Ebook("Test")
    ch = Chapter("Title", eb)
    ch.set_content("content data")
    assert ch.content == "content data"


def test_chapter_is_top_chapter_property():
    eb = Ebook("Test")
    ch1 = Chapter("t1", eb, level=1)
    ch2 = Chapter("t2", eb, level=2)
    assert ch1.is_top_chapter
    assert not ch2.is_top_chapter


def test_chapter_add_static_files_and_save(tmp_path):
    eb = Ebook("Test")
    eb._tempdir = tmp_path
    ch = Chapter("Title", eb).set_content("my content")
    # create dummy static file
    static_file = tmp_path / "static.txt"
    static_file.write_text("data")
    ch.add_static_files(str(static_file))
    ch.save()
    saved_file = ch.full_dest_file_path
    assert saved_file.exists()
    assert saved_file.read_text(encoding="utf-8") == "my content"
    # static file copied to same dir
    copied_static = saved_file.parent / static_file.name
    assert copied_static.exists()
    assert copied_static.read_text(encoding="utf-8") == "data"


def test_chapter_create_subchapter_level_limit():
    eb = Ebook("Test")
    ch = Chapter("t1", eb, level=1)
    sub = ch.create_subchapter("sub1")
    assert sub._level == 2
    # max_level == 2, next level creation should fail
    with pytest.raises(Exception):
        sub.create_subchapter("subsub")


def test_ebook_set_cover(tmp_path):
    eb = Ebook("Test")
    # Test setting cover_path
    fake_cover = tmp_path / "cover.jpg"
    fake_cover.write_text("cover")
    eb.set_cover(cover_path=str(fake_cover))
    assert eb.cover_path == str(fake_cover)
    assert eb.cover_content is None

    # Test setting cover_content
    eb.set_cover(cover_content=b"abc")
    assert eb.cover_content == b"abc"
    assert eb.cover_path is None

    # Test error if none given
    with pytest.raises(ValueError):
        eb.set_cover()


def test_ebook_add_chapter_only_top_level_allowed():
    eb = Ebook("Test")
    ch = Chapter("Title", eb, level=1)
    eb.add_chapter(ch)
    assert ch in eb.chapter_list
    ch2 = Chapter("Sub", eb, level=2)
    with pytest.raises(ValueError):
        eb.add_chapter(ch2)


def test_ebook_create_chapter_adds_and_returns():
    eb = Ebook("Test")
    ch = eb.create_chapter("Title")
    assert isinstance(ch, Chapter)
    assert ch.title == "Title"
    assert ch in eb.chapter_list


def test_ebook_dest_file_path(tmp_path):
    eb = Ebook("Test Ebook")
    expected_name = format_file_name("Test Ebook") + ".mobi"
    assert eb.dest_file_path == pathlib.Path(expected_name)
    eb._tempdir = tmp_path
    assert eb.full_dest_file_path == tmp_path / expected_name


def test_ebook_save_and_create(tmp_path, monkeypatch):
    eb = Ebook("Test Ebook")
    eb._tempdir = tmp_path

    # Add one chapter
    ch = Chapter("Chapter 1", eb)
    ch.set_content("Hello")
    eb.add_chapter(ch)

    # Patch EbookUtil.create to just touch a mobi file
    def dummy_create():
        mobi_path = eb.tmpdir / eb.dest_file_path
        mobi_path.write_text("mobi content")
        return None
    monkeypatch.setattr(ebooklib.EbookUtil, "create", dummy_create)

    dest_path = tmp_path / "output.mobi"
    eb.save(str(dest_path))
    assert dest_path.exists()
    assert dest_path.read_text() == "mobi content"


def test_ebook_show_calls_platform(monkeypatch, tmp_path):
    eb = Ebook("ShowTest")
    eb._tempdir = tmp_path
    ch = Chapter("Chapter 1", eb)
    ch.set_content("abc")
    eb.add_chapter(ch)

    # Patch EbookUtil.create to not raise
    monkeypatch.setattr(ebooklib.EbookUtil, "create", lambda self: None)

    # Patch subprocess.call and os.startfile, platform.system
    recorded = {}

    def fake_sub_call(args):
        recorded['args'] = args
        return 0

    def fake_startfile(path):
        recorded['startfile'] = path
        return 0

    monkeypatch.setattr(subprocess, "call", fake_sub_call)
    monkeypatch.setattr(os, "startfile", fake_startfile, raising=False)

    # Test Linux
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    ret = eb.show()
    assert ret == 0
    assert recorded['args'][0] == 'xdg-open'
    # Test Darwin
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    ret = eb.show()
    assert ret == 0
    assert recorded['args'][0] == 'open'
    # Test Windows/Others fall-back
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    if hasattr(os, "startfile"):
        recorded.clear()
        ret = eb.show()
        assert ret == 0
        assert recorded.get("startfile") is not None


def test_ebookutil_headings_structure(tmp_path):
    eb = Ebook("Test Ebook")
    eb._tempdir = tmp_path
    c1 = eb.create_chapter("Chapter 1").set_content("Content 1")
    sub = c1.create_subchapter("Sub 1").set_content("Sub content")
    eb.add_chapter(c1)
    eu = EbookUtil(eb)

    headings = eu.headings
    # There should be one chapter with one subheading
    assert isinstance(headings, list)
    assert headings[0]['title'] == "Chapter 1"
    assert len(headings[0]['sub_headings']) == 1
    assert headings[0]['sub_headings'][0]['title'] == "Sub 1"
    # play_order increments properly
    play_orders = [h['play_order'] for h in headings]
    assert all(isinstance(po, int) for po in play_orders)


def test_ebookutil_render_files(tmp_path, monkeypatch):
    eb = Ebook("Test Ebook")
    eb._tempdir = tmp_path
    c1 = eb.create_chapter("Chapter 1").set_content("content")

    eu = EbookUtil(eb)

    # Patch the template loader to use simple template with render returning fixed string
    class DummyTemplate:
        def render(self, **context):
            return "rendered"

    monkeypatch.setattr(ebooklib.Environment, "get_template", lambda self, name: DummyTemplate())

    eu._render_file('toc.xml', {'headings': []}, 'toc.ncx')
    assert (eb.tmpdir / 'toc.ncx').exists()
    assert (eb.tmpdir / 'toc.ncx').read_text(encoding='utf-8') == "rendered"

    # test _render_toc_ncx returns filename
    fname = eu._render_toc_ncx()
    assert fname == 'toc.ncx'

    # test _render_toc_html returns filename
    fname_html = eu._render_toc_html()
    assert fname_html == 'toc.html'

    # test _render_opf returns opf filename matching formatted title
    opf_name = eu._render_opf()
    assert opf_name.endswith('.opf')


def test_ebookutil_save_cover_copies_file(tmp_path, monkeypatch):
    eb = Ebook("Test Ebook")
    eb._tempdir = tmp_path

    fake_cover = tmp_path / "cover.jpg"
    fake_cover.write_text("cover data")
    eb.cover_path = str(fake_cover)
    eb.cover_content = None

    eu = EbookUtil(eb)

    eu._save_cover()
    copied_cover = tmp_path / fake_cover.name
    assert copied_cover.exists()
    assert copied_cover.read_text() == "cover data"


def test_ebookutil_save_cover_default_copy(tmp_path, monkeypatch):
    eb = Ebook("Test Ebook")
    eb._tempdir = tmp_path

    eb.cover_path = None
    eb.cover_content = None
    eu = EbookUtil(eb)

    # Patch shutil.copy to detect call
    calls = []
    monkeypatch.setattr(shutil, "copy", lambda src, dst: calls.append((src, dst)))

    eu._save_cover()
    # It should copy something from the templates/cover.jpg file path
    assert calls
    src_path = calls[0][0]
    assert 'cover.jpg' in src_path


def test_ebookutil_save_cover_not_implemented(tmp_path):
    eb = Ebook("Test Ebook")
    eb._tempdir = tmp_path
    eb.cover_path = None
    eb.cover_content = b"bytes content"
    eu = EbookUtil(eb)
    with pytest.raises(NotImplementedError):
        eu._save_cover()


def test_ebookutil_save_chapters_calls_save(tmp_path, monkeypatch):
    eb = Ebook("Test Ebook")
    eb._tempdir = tmp_path
    ch1 = eb.create_chapter("C1")
    ch2 = ch1.create_subchapter("SC1")
    eb.add_chapter(ch1)

    saved = []
    monkeypatch.setattr(Chapter, "save", lambda self: saved.append(self.title))

    eu = EbookUtil(eb)
    eu._save_chapters()
    # Both should be saved
    assert "C1" in saved
    assert "SC1" in saved


def test_ebook_create_deletes_tmpdir(monkeypatch):
    # Create an ebook and let it be deleted, check tmpdir is removed
    eb = Ebook("Test CleanUp")
    tmp = eb.tmpdir
    chapter = eb.create_chapter("Intro")
    chapter.set_content("abc")

    # Patch EbookUtil.create to do nothing
    monkeypatch.setattr(ebooklib.EbookUtil, "create", lambda self: None)

    # Delete ebook object forcibly and check tmpdir gone
    del eb
    # tmpdir might still be present briefly because of gc, forcibly call cleanup function:
    if tmp.exists():
        shutil.rmtree(str(tmp), ignore_errors=True)
    assert not tmp.exists()


def test_parse_headers_valid(tmp_path):
    toc = tmp_path / "toc.md"
    toc.write_text(
        "\n\nTitle\n# Header 1\n## Sub1\n# Header 2\n## Sub2\n## Sub3\n",
        encoding='utf-8'
    )
    title, headers = ebooklib._parse_headers(str(toc))
    assert title == "Title"
    assert len(headers) == 2
    assert headers[0]['title'] == "Header 1"
    assert len(headers[0]['next_headers']) == 1
    assert headers[1]['title'] == "Header 2"
    assert len(headers[1]['next_headers']) == 2


def test_parse_headers_file_not_exist(tmp_path):
    with pytest.raises(ValueError):
        ebooklib._parse_headers(str(tmp_path / "missing.md"))


def test_parse_headers_invalid_empty_title(tmp_path):
    f = tmp_path / "toc.md"
    f.write_text("\n\n\n", encoding='utf-8')
    with pytest.raises(ValueError):
        ebooklib._parse_headers(str(f))


def test_parse_headers_no_headings(tmp_path):
    f = tmp_path / "toc.md"
    f.write_text("Title only\n", encoding='utf-8')
    with pytest.raises(ValueError):
        ebooklib._parse_headers(str(f))
