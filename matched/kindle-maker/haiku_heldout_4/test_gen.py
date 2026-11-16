import gc
import pathlib
import tempfile

import pytest

import kindle_maker.ebooklib as ebooklib_mod
from kindle_maker.ebooklib import Chapter, Ebook, EbookUtil, format_file_name


@pytest.fixture(autouse=True)
def isolated_tmp(tmp_path, monkeypatch):
    """Keep Ebook temp directories inside pytest's tmp_path."""
    base = tmp_path / 'tmp'
    base.mkdir()
    monkeypatch.setattr(tempfile, 'tempdir', str(base))
    return base


@pytest.fixture
def kindlegen_calls(monkeypatch):
    """Replace the kindlegen subprocess call with a fake that records args
    and writes the expected .mobi file next to the .opf file."""
    calls = []

    def fake_call(args, *a, **kw):
        calls.append(list(args))
        if '-dont_append_source' in args:
            opf = pathlib.Path(args[-1])
            (opf.parent / (opf.stem + '.mobi')).write_bytes(b'MOBI')
        return 0

    monkeypatch.setattr(ebooklib_mod.subprocess, 'call', fake_call)
    return calls


def test_format_file_name_strips_illegal_chars():
    assert format_file_name('a/b c+d"e\\f:g|h') == 'abcd-efg-h'


def test_ebook_defaults():
    ebook = Ebook('Title')
    assert ebook.title == 'Title'
    assert ebook.author is None
    assert ebook.cover_path is None
    assert ebook.cover_content is None
    assert ebook.chapter_list == []
    assert ebook.tmpdir.is_dir()
    assert isinstance(ebook._eu, EbookUtil)


def test_ebook_dest_paths():
    ebook = Ebook('My Book')
    assert ebook.dest_file_path == pathlib.Path('MyBook.mobi')
    assert ebook.full_dest_file_path == ebook.tmpdir / 'MyBook.mobi'


def test_set_cover_requires_value():
    ebook = Ebook('B')
    with pytest.raises(ValueError):
        ebook.set_cover()


def test_set_cover_path_and_content_are_exclusive():
    ebook = Ebook('B')
    assert ebook.set_cover(cover_content=b'abc') is ebook
    assert ebook.cover_content == b'abc'
    assert ebook.cover_path is None

    ebook.set_cover(cover_path='front.jpg')
    assert ebook.cover_path == 'front.jpg'
    assert ebook.cover_content is None


def test_create_chapter_appends_and_returns_chapter():
    ebook = Ebook('B')
    ch = ebook.create_chapter('One')
    assert isinstance(ch, Chapter)
    assert ch.title == 'One'
    assert ebook.chapter_list == [ch]


def test_add_chapter_rejects_subchapter():
    ebook = Ebook('B')
    top = ebook.create_chapter('Top')
    sub = top.create_subchapter('Sub')
    with pytest.raises(ValueError):
        ebook.add_chapter(sub)


def test_chapter_missing_file_raises(tmp_path):
    ebook = Ebook('B')
    with pytest.raises(ValueError):
        Chapter('C', ebook, str(tmp_path / 'missing.html'))


def test_chapter_content_reads_file(tmp_path):
    f = tmp_path / 'c.html'
    f.write_text('<p>hello</p>', encoding='utf-8')
    ebook = Ebook('B')
    ch = ebook.create_chapter('One', str(f))
    assert ch.content == '<p>hello</p>'


def test_set_content_overrides_file(tmp_path):
    f = tmp_path / 'c.html'
    f.write_text('<p>old</p>', encoding='utf-8')
    ebook = Ebook('B')
    ch = ebook.create_chapter('One', str(f))
    assert ch.set_content('<p>new</p>') is ch
    assert ch.content == '<p>new</p>'
    assert ch._file_path is None


def test_chapter_save_writes_content_and_static_files(tmp_path):
    img = tmp_path / 'pic.png'
    img.write_bytes(b'png-bytes')
    ebook = Ebook('Book')
    ch = ebook.create_chapter('Intro')
    ch.set_content('<p>x</p>').add_static_files(str(img))
    ch.save()
    assert ch.full_dest_file_path.read_text(encoding='utf-8') == '<p>x</p>'
    assert (ebook.tmpdir / 'pic.png').read_bytes() == b'png-bytes'


def test_create_subchapter_limits_depth():
    ebook = Ebook('B')
    top = ebook.create_chapter('Top')
    sub = top.create_subchapter('Sub')
    assert sub.is_top_chapter is False
    with pytest.raises(Exception):
        sub.create_subchapter('SubSub')


def test_save_without_chapters_raises(tmp_path):
    ebook = Ebook('B')
    with pytest.raises(ValueError):
        ebook.save(str(tmp_path / 'out.mobi'))


def test_save_copies_mobi_to_target(tmp_path, kindlegen_calls):
    ebook = Ebook('Target')
    ebook.create_chapter('C').set_content('x')
    out = tmp_path / 'out.mobi'
    ebook.save(str(out))
    assert out.read_bytes() == b'MOBI'


def test_save_invokes_kindlegen_with_opf(kindlegen_calls):
    ebook = Ebook('Target')
    ebook.create_chapter('C').set_content('x')
    ebook._eu.create()
    assert len(kindlegen_calls) == 1
    args = kindlegen_calls[0]
    assert args[0] == ebooklib_mod.kindlegen
    assert args[1] == '-dont_append_source'
    assert args[2] == str(ebook.tmpdir / 'Target.opf')


def test_create_renders_toc_and_opf(kindlegen_calls):
    ebook = Ebook('My Book', author='Ann')
    ebook.create_chapter('Chapter One').set_content('<p>1</p>')
    ebook._eu.create()
    tmp = ebook.tmpdir
    assert (tmp / 'toc.ncx').is_file()
    assert (tmp / 'toc.html').is_file()
    assert (tmp / 'MyBook.opf').is_file()
    assert (tmp / 'ChapterOne.html').read_text(encoding='utf-8') == '<p>1</p>'
    assert 'Chapter One' in (tmp / 'toc.ncx').read_text(encoding='utf-8')
    assert 'My Book' in (tmp / 'MyBook.opf').read_text(encoding='utf-8')


def test_headings_structure():
    ebook = Ebook('Book')
    one = ebook.create_chapter('One')
    one.create_subchapter('One A')
    ebook.create_chapter('Two')
    headings = ebook._eu.headings

    assert len(headings) == 2
    assert headings[0]['title'] == 'One'
    assert headings[0]['play_order'] == 2
    assert headings[0]['file_name'] == 'One.html'
    assert len(headings[0]['sub_headings']) == 1
    assert headings[0]['sub_headings'][0]['title'] == 'One A'
    assert headings[0]['sub_headings'][0]['play_order'] == 3
    assert headings[1]['title'] == 'Two'
    assert headings[1]['play_order'] == 4
    assert headings[1]['sub_headings'] == []


def test_headings_empty_when_no_chapters():
    ebook = Ebook('Empty')
    assert ebook._eu.headings == []


def test_create_copies_cover_file(tmp_path, kindlegen_calls):
    cover = tmp_path / 'front.jpg'
    cover.write_bytes(b'JPG')
    ebook = Ebook('B')
    ebook.create_chapter('C').set_content('x')
    ebook.set_cover(cover_path=str(cover))
    ebook._eu.create()
    assert (ebook.tmpdir / 'front.jpg').read_bytes() == b'JPG'


def test_create_default_cover_from_templates(kindlegen_calls):
    ebook = Ebook('B')
    ebook.create_chapter('C').set_content('x')
    ebook._eu.create()
    assert (ebook.tmpdir / 'cover.jpg').is_file()


def test_cover_content_not_implemented():
    ebook = Ebook('B')
    ebook.create_chapter('C').set_content('x')
    ebook.set_cover(cover_content=b'raw-bytes')
    with pytest.raises(NotImplementedError):
        ebook._eu.create()


def test_show_linux_uses_xdg_open(monkeypatch, kindlegen_calls):
    monkeypatch.setattr(ebooklib_mod.platform, 'system', lambda: 'Linux')
    ebook = Ebook('Show')
    ebook.create_chapter('C').set_content('x')
    result = ebook.show()
    assert result == 0
    assert kindlegen_calls[-1][0] == 'xdg-open'
    assert kindlegen_calls[-1][1] == str(ebook.full_dest_file_path)


def test_del_removes_tmpdir():
    ebook = Ebook('Gone')
    tmp = ebook.tmpdir
    assert tmp.is_dir()
    del ebook
    # Ebook <-> EbookUtil reference cycle requires the cycle collector.
    gc.collect()
    assert not tmp.exists()
