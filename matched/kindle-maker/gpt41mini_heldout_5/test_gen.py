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
def ebook_instance():
    eb = Ebook("Test Title", author="Test Author")
    yield eb
    # cleanup if still exists
    if eb.tmpdir.exists():
        shutil.rmtree(eb.tmpdir)


@pytest.fixture
def temp_html_file():
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.html', delete=False) as f:
        f.write("<html><body>Test Content</body></html>")
        f_path = f.name
    yield f_path
    os.unlink(f_path)


def test_format_file_name_removes_illegal_chars():
    s = 'my/illegal: file+name"with|chars\\'
    expected = 'myillegal-file-namewith-chars'
    assert format_file_name(s) == expected


def test_ebook_add_and_create_chapters(ebook_instance, temp_html_file):
    ct = ebook_instance.create_chapter("Chapter 1", temp_html_file)
    assert ct.title == "Chapter 1"
    assert ct._file_path == temp_html_file
    assert ct.is_top_chapter
    # test that non-top chapter rejected when adding directly
    sub_ct = ct.create_subchapter("Subchapter A")
    with pytest.raises(ValueError):
        ebook_instance.add_chapter(sub_ct)
    # top level chapters accepted
    ct2 = Chapter("Chapter 2", ebook_instance, None, level=1)
    ebook_instance.add_chapter(ct2)
    assert ct2 in ebook_instance.chapter_list


def test_chapter_init_invalid_file_path_raises():
    with pytest.raises(ValueError):
        Chapter("Bad Chapter", None, file_path="nonexistentfile.html")


def test_chapter_dest_file_path_formatting():
    ch = Chapter('Test / Title:', None)
    fname = ch.dest_file_path.name
    assert fname.endswith('.html')
    assert '/' not in fname and ':' not in fname


def test_chapter_set_content_and_access_content():
    ch = Chapter("Chapter Content", None)
    content = "<p>Some HTML content</p>"
    ch.set_content(content)
    assert ch._content == content
    assert ch.content == content
    # setting content clears _file_path
    assert ch._file_path is None


def test_chapter_add_static_files(tmp_path):
    f1 = tmp_path / "file1.txt"
    f1.write_text("data1")
    ch = Chapter("StaticFiles", None)
    ch.add_static_files(str(f1))
    assert str(f1) in ch._static_files
    # adding non-existent file raises
    with pytest.raises(AssertionError):
        ch.add_static_files("nonexistent.file")


def test_chapter_create_subchapter_level_limit(ebook_instance):
    top_ch = ebook_instance.create_chapter("Top")
    sub_ch = top_ch.create_subchapter("Sub")
    assert sub_ch._level == 2
    with pytest.raises(Exception):
        sub_ch.create_subchapter("Too Deep")


def test_chapter_save_writes_content_and_copies_static_files(tmp_path):
    content = "<h1>Heading</h1>"
    ch = Chapter("SaveTest", None)
    ch.set_content(content)
    ch._ebook = type('DummyEbook', (), {'tmpdir': tmp_path})()
    ch.save()
    saved_file_path = tmp_path / ch.dest_file_path
    assert saved_file_path.exists()
    assert saved_file_path.read_text(encoding='utf-8') == content
    # test static file copy triggered
    f_static = tmp_path / "static.txt"
    f_static.write_text("data")
    ch.add_static_files(str(f_static))
    ch.save()
    copied_file = tmp_path / f_static.name
    assert copied_file.exists()


def test_ebook_set_cover_accepts_path_and_content(tmp_path):
    eb = Ebook("CoverTest")
    # cover_path sets cover_path and resets content
    f_cover = tmp_path / "cover.jpg"
    f_cover.write_bytes(b"data")
    eb.set_cover(cover_path=str(f_cover))
    assert eb.cover_path == str(f_cover)
    assert eb.cover_content is None
    # cover_content sets cover_content and resets cover_path
    eb.set_cover(cover_content=b"content bytes")
    assert eb.cover_content == b"content bytes"
    assert eb.cover_path is None
    # both empty raises ValueError
    with pytest.raises(ValueError):
        eb.set_cover()


def test_ebook_add_chapter_rejects_non_top_chapter(ebook_instance):
    ch = ebook_instance.create_chapter("Top")
    sub_ch = ch.create_subchapter("Sub")
    with pytest.raises(ValueError):
        ebook_instance.add_chapter(sub_ch)


def test_ebook_dest_file_path_and_full_path(ebook_instance):
    fname = ebook_instance.dest_file_path.name
    assert fname.endswith('.mobi')
    full_path = ebook_instance.full_dest_file_path
    assert full_path.parent == ebook_instance.tmpdir


def test_ebook_create_without_chapters_raises():
    eb = Ebook("EmptyBook")
    with pytest.raises(ValueError):
        eb._create()


def test_ebook_util_headings_property_caches_and_produces_structure(tmp_path):
    eb = Ebook("BookTest")
    ch1 = eb.create_chapter("Chapter One")
    ch2 = eb.create_chapter("Chapter Two")
    subch = ch2.create_subchapter("Sub Two One")
    eu = EbookUtil(eb)
    headings = eu.headings
    assert isinstance(headings, list)
    assert len(headings) == 2
    assert headings[1]['sub_headings'][0]['title'] == "Sub Two One"
    # Accessing again returns cached list
    cached = eu.headings
    assert cached is headings


def test_ebook_util_render_file_creates_file(tmp_path):
    eb = Ebook("RenderTest")
    eb._tempdir = tmp_path
    eu = EbookUtil(eb)
    # create a dummy template file in templates
    templates_dir = pathlib.Path(__file__).parent / 'kindle_maker' / 'templates'
    # Use an existing template name to avoid messing templates folder (to not actually require template)
    # Instead monkeypatch Environment get_template
    def dummy_get_template(name):
        class DummyTemplate:
            def render(self, **kwargs):
                return "rendered content"
        return DummyTemplate()
    eu._templates_env.get_template = dummy_get_template
    eu._render_file('dummy', {}, 'file.txt')
    assert (tmp_path / 'file.txt').read_text(encoding='utf-8') == "rendered content"


def test_ebook_util_render_toc_ncx_and_html(tmp_path):
    eb = Ebook("TOCTitle", author="AuthorName")
    eb._tempdir = tmp_path
    ch = eb.create_chapter("C1")
    eu = EbookUtil(eb)

    # patch _render_file to track calls instead of writing real files
    calls = {}

    def fake_render_file(template_name, context, to_filename):
        calls[to_filename] = (template_name, context)

    eu._render_file = fake_render_file

    ncx = eu._render_toc_ncx()
    toc = eu._render_toc_html()

    assert ncx == 'toc.ncx'
    assert toc == 'toc.html'
    assert 'toc.ncx' in calls
    assert 'toc.html' in calls
    # check context keys
    ctx = calls['toc.ncx'][1]
    assert 'headings' in ctx and 'title' in ctx and 'author' in ctx


def test_ebook_util_render_opf(tmp_path):
    eb = Ebook("OPFTest")
    eb._tempdir = tmp_path
    eb.author = None  # test fallback author
    ch = eb.create_chapter("Ch1")
    eu = EbookUtil(eb)

    recorded = {}

    def fake_render_file(template_name, context, to_filename):
        recorded['template'] = template_name
        recorded['context'] = context
        recorded['filename'] = to_filename

    eu._render_file = fake_render_file
    opf_filename = eu._render_opf()
    assert opf_filename.endswith('.opf')
    assert opf_filename == recorded['filename']
    assert recorded['template'] == 'opf.xml'
    assert recorded['context']['author'] == 'jachinlin.github.io'


def test_ebook_util_save_cover_copies_file(tmp_path):
    eb = Ebook("CoverCopy")
    eb._tempdir = tmp_path
    # place a dummy cover file
    cover_file = tmp_path / "cover.png"
    cover_file.write_bytes(b"coverdata")
    eb.cover_path = str(cover_file)
    eu = EbookUtil(eb)
    eu._save_cover()
    # The file should be copied to tmpdir
    copied = tmp_path / cover_file.name
    assert copied.exists()


def test_ebook_util_save_cover_default_copies_template(tmp_path, monkeypatch):
    eb = Ebook("CoverDefault")
    eb._tempdir = tmp_path
    eb.cover_path = None
    eb.cover_content = None
    eu = EbookUtil(eb)

    # monkeypatch shutil.copy to record args
    copied = []

    def fake_copy(src, dst):
        copied.append((src, dst))

    monkeypatch.setattr(shutil, "copy", fake_copy)

    eu._save_cover()
    assert copied
    # The source should be some templates/cover.jpg path
    assert 'cover.jpg' in copied[0][0]
    assert copied[0][1] == str(tmp_path)


def test_ebook_util_save_cover_raises_not_implemented(tmp_path):
    eb = Ebook("CoverNotImplemented")
    eb._tempdir = tmp_path
    eb.cover_path = None
    eb.cover_content = b"content bytes"
    eu = EbookUtil(eb)
    with pytest.raises(NotImplementedError):
        eu._save_cover()


def test_ebook_util_save_chapters_calls_save(tmp_path):
    eb = Ebook("SaveChapters")
    eb._tempdir = tmp_path
    ch1 = eb.create_chapter("Top1")
    ch2 = eb.create_chapter("Top2")

    # Patch save method to track calls
    calls = []

    def fake_save(self):
        calls.append(self.title)

    Chapter.save = fake_save

    eu = EbookUtil(eb)
    # Adding a subchapter to test save called on sub
    sub = ch1.create_subchapter("Sub1")
    eu._save_chapters()
    assert "Top1" in calls
    assert "Top2" in calls
    # sub chapter also saved
    assert "Sub1" in calls


def test_ebook_create_runs_and_calls_kindlegen(monkeypatch, tmp_path):
    eb = Ebook("RunTest")
    eb._tempdir = tmp_path
    c = eb.create_chapter("Ch1")
    eu = eb._eu

    # Patch methods to avoid actual file creation, but keep expected returns
    monkeypatch.setattr(eb._eu, "_render_toc_ncx", lambda: "tocfile.opf")
    monkeypatch.setattr(eb._eu, "_render_toc_html", lambda: "toc.html")
    monkeypatch.setattr(eb._eu, "_render_opf", lambda: "book.opf")
    monkeypatch.setattr(eb._eu, "_save_cover", lambda: None)
    monkeypatch.setattr(eb._eu, "_save_chapters", lambda: None)

    # Patch subprocess.call to simulate kindlegen runs:
    calls = []

    def fake_subprocess_call(args):
        calls.append(args)
        return 0

    monkeypatch.setattr(subprocess, "call", fake_subprocess_call)
    eb._eu.create()
    assert calls
    assert 'kindlegen' in calls[0][0][0]


def test_ebook_show_platform_specific_calls(monkeypatch, tmp_path):
    eb = Ebook("ShowTest")
    eb._tempdir = tmp_path
    c = eb.create_chapter("Ch1")

    called_args = []

    def fake_call(args):
        called_args.append(args)
        return 42

    def fake_startfile(path):
        called_args.append(path)
        return 99

    monkeypatch.setattr(eb._eu, "_create", lambda: None)

    # Linux
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(subprocess, "call", fake_call)
    ret = eb.show()
    assert ret == 42
    assert called_args[-1][0] == 'xdg-open'

    called_args.clear()
    # MacOS
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    ret = eb.show()
    assert ret == 42
    assert called_args[-1][0] == 'open'

    called_args.clear()
    # Windows fallback
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(os, "startfile", fake_startfile)
    ret = eb.show()
    assert ret == 99
    assert str(eb.full_dest_file_path) == called_args[-1]


def test_ebook_del_removes_tmpdir(tmp_path):
    eb = Ebook("DelTest")
    eb._tempdir = tmp_path
    # create a file inside
    f = tmp_path / "file.txt"
    f.write_text("data")

    # after __del__, tmpdir should be removed
    tmp_dir = eb.tmpdir
    # manually invoke __del__ because pytest won't call automatically
    Ebook.__del__(eb)
    assert not tmp_dir.exists()
