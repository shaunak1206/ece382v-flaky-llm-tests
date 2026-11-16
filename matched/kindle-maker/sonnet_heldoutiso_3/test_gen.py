# -*- coding: utf-8 -*-
import itertools
import pathlib
from types import SimpleNamespace

import pytest

from kindle_maker import ebooklib
from kindle_maker.ebooklib import Ebook, EbookUtil, Chapter


@pytest.fixture
def calls():
    return []


@pytest.fixture
def ebook(tmp_path, monkeypatch, calls):
    counter = itertools.count(1)
    monkeypatch.setattr(
        ebooklib, 'tempfile', SimpleNamespace(gettempdir=lambda: str(tmp_path)))
    monkeypatch.setattr(
        ebooklib, 'time', SimpleNamespace(time=lambda: float(next(counter))))

    def fake_call(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return 0

    monkeypatch.setattr(ebooklib, 'subprocess', SimpleNamespace(call=fake_call))
    return Ebook('My Book', author='Tester')


def _make_chapters(ebook):
    c1 = ebook.create_chapter('Chap 1')
    c1.set_content('<html>one</html>')
    s11 = c1.create_subchapter('Sub 1')
    s11.set_content('<html>sub one</html>')
    s12 = c1.create_subchapter('Sub 2')
    s12.set_content('<html>sub two</html>')
    c2 = ebook.create_chapter('Chap 2')
    c2.set_content('<html>two</html>')
    return c1, c2


def test_ebook_creates_tmpdir(ebook):
    assert ebook.tmpdir.is_dir()
    assert ebook.chapter_list == []
    assert ebook.cover_path is None
    assert ebook.cover_content is None


def test_dest_file_path_strips_illegal_chars(tmp_path, monkeypatch, ebook):
    ebook.title = 'A b/c+d:e|f"g'
    assert ebook.dest_file_path == pathlib.Path('Abc-d-e-fg.mobi')
    assert ebook.full_dest_file_path == ebook.tmpdir / 'Abc-d-e-fg.mobi'


def test_set_cover_path_clears_content(ebook):
    ebook.cover_content = b'xx'
    result = ebook.set_cover(cover_path='/some/cover.jpg')
    assert result is ebook
    assert ebook.cover_path == '/some/cover.jpg'
    assert ebook.cover_content is None


def test_set_cover_content_clears_path(ebook):
    ebook.set_cover(cover_path='/some/cover.jpg')
    ebook.set_cover(cover_content=b'bytes')
    assert ebook.cover_content == b'bytes'
    assert ebook.cover_path is None


def test_set_cover_without_args_raises(ebook):
    with pytest.raises(ValueError):
        ebook.set_cover()


def test_add_chapter_rejects_sub_chapter(ebook):
    c = ebook.create_chapter('Top')
    sub = c.create_subchapter('Sub')
    with pytest.raises(ValueError):
        ebook.add_chapter(sub)


def test_create_chapter_appends_to_list(ebook):
    c = ebook.create_chapter('Top')
    assert ebook.chapter_list == [c]
    assert c.is_top_chapter


def test_save_with_no_chapters_raises(ebook, tmp_path):
    with pytest.raises(ValueError):
        ebook.save(str(tmp_path / 'out.mobi'))


def test_headings_structure_and_play_order(ebook):
    _make_chapters(ebook)
    hs = EbookUtil(ebook).headings
    assert [h['title'] for h in hs] == ['Chap 1', 'Chap 2']
    assert [h['play_order'] for h in hs] == [2, 5]
    assert hs[0]['file_name'] == 'Chap1.html'
    subs = hs[0]['sub_headings']
    assert [s['title'] for s in subs] == ['Sub 1', 'Sub 2']
    assert [s['play_order'] for s in subs] == [3, 4]
    assert subs[0]['file_name'] == 'Sub1.html'
    assert hs[1]['sub_headings'] == []


def test_headings_are_cached(ebook):
    ebook.create_chapter('Chap 1')
    eu = EbookUtil(ebook)
    first = eu.headings
    ebook.create_chapter('Chap 2')
    assert eu.headings is first
    assert len(eu.headings) == 1


def test_render_toc_ncx_uses_default_author(ebook, monkeypatch):
    ebook.author = None
    ebook.create_chapter('Chap 1')
    eu = EbookUtil(ebook)
    rendered = []
    monkeypatch.setattr(
        eu, '_render_file',
        lambda template, ctx, fname: rendered.append((template, ctx, fname)))
    name = eu._render_toc_ncx()
    assert name == 'toc.ncx'
    template, ctx, fname = rendered[0]
    assert template == 'toc.xml'
    assert fname == 'toc.ncx'
    assert ctx['author'] == 'jachinlin.github.io'
    assert ctx['title'] == 'My Book'
    assert ctx['headings'] == eu.headings


def test_render_opf_file_name_and_author(ebook, monkeypatch):
    ebook.create_chapter('Chap 1')
    eu = EbookUtil(ebook)
    rendered = []
    monkeypatch.setattr(
        eu, '_render_file',
        lambda template, ctx, fname: rendered.append((template, ctx, fname)))
    name = eu._render_opf()
    assert name == 'MyBook.opf'
    assert rendered[0][0] == 'opf.xml'
    assert rendered[0][1]['author'] == 'Tester'


def test_save_cover_copies_given_cover(ebook, tmp_path):
    cover = tmp_path / 'mycover.jpg'
    cover.write_bytes(b'JPEGDATA')
    ebook.set_cover(cover_path=str(cover))
    EbookUtil(ebook)._save_cover()
    assert (ebook.tmpdir / 'mycover.jpg').read_bytes() == b'JPEGDATA'


def test_save_cover_with_content_not_implemented(ebook):
    ebook.set_cover(cover_content=b'abc')
    with pytest.raises(NotImplementedError):
        EbookUtil(ebook)._save_cover()


def test_save_cover_default_cover(ebook):
    EbookUtil(ebook)._save_cover()
    assert (ebook.tmpdir / 'cover.jpg').is_file()


def test_save_chapters_writes_all_files(ebook):
    _make_chapters(ebook)
    EbookUtil(ebook)._save_chapters()
    assert (ebook.tmpdir / 'Chap1.html').read_text(encoding='utf-8') == '<html>one</html>'
    assert (ebook.tmpdir / 'Sub1.html').read_text(encoding='utf-8') == '<html>sub one</html>'
    assert (ebook.tmpdir / 'Sub2.html').is_file()
    assert (ebook.tmpdir / 'Chap2.html').is_file()


def test_create_generates_files_and_calls_kindlegen(ebook, calls):
    _make_chapters(ebook)
    EbookUtil(ebook).create()
    for name in ('toc.ncx', 'toc.html', 'MyBook.opf', 'cover.jpg',
                 'Chap1.html', 'Sub1.html', 'Chap2.html'):
        assert (ebook.tmpdir / name).is_file(), name
    assert len(calls) == 1
    assert calls[0][1] == '-dont_append_source'
    assert calls[0][2] == str(ebook.tmpdir / 'MyBook.opf')


def test_save_copies_generated_mobi(ebook, tmp_path, monkeypatch):
    _make_chapters(ebook)

    def fake_call(cmd, *args, **kwargs):
        (ebook.tmpdir / 'MyBook.mobi').write_bytes(b'MOBI')
        return 0

    monkeypatch.setattr(ebooklib, 'subprocess', SimpleNamespace(call=fake_call))
    out = tmp_path / 'out.mobi'
    ebook.save(str(out))
    assert out.read_bytes() == b'MOBI'


def test_show_on_linux_uses_xdg_open(ebook, calls, monkeypatch):
    _make_chapters(ebook)
    monkeypatch.setattr(ebooklib, 'platform', SimpleNamespace(system=lambda: 'Linux'))
    rc = ebook.show()
    assert rc == 0
    assert calls[-1] == ['xdg-open', str(ebook.full_dest_file_path)]


def test_show_on_darwin_uses_open(ebook, calls, monkeypatch):
    _make_chapters(ebook)
    monkeypatch.setattr(ebooklib, 'platform', SimpleNamespace(system=lambda: 'Darwin'))
    ebook.show()
    assert calls[-1] == ['open', str(ebook.full_dest_file_path)]
