import os
import pathlib
import tempfile
from types import SimpleNamespace

import pytest

from kindle_maker import ebooklib as kl
from kindle_maker.ebooklib import (
    Chapter,
    Ebook,
    _parse_headers,
    format_file_name,
    make_ebook,
)


def _install_fake_kindlegen(monkeypatch, mobi_name=None, rc=0):
    """Replace the kindlegen subprocess call with a recorder.

    The fake records the argv it was given and the names of the files in the
    build directory at that moment. If ``mobi_name`` is given it also writes
    that file, standing in for the real kindlegen output.
    """
    calls = []
    seen = {}

    def fake_call(args, *a, **kw):
        calls.append(list(args))
        tmpdir = pathlib.Path(args[-1]).parent
        seen['names'] = sorted(p.name for p in tmpdir.iterdir())
        if mobi_name is not None:
            (tmpdir / mobi_name).write_bytes(b'MOBI')
        return rc

    monkeypatch.setattr(kl, 'subprocess', SimpleNamespace(call=fake_call))
    return calls, seen


def test_format_file_name_strips_illegal_characters():
    assert format_file_name('a/b c/d') == 'abcd'
    assert format_file_name('a+b:c|d') == 'a-b-c-d'
    assert format_file_name('say "hi"\\x') == 'sayhix'


def test_format_file_name_keeps_other_characters():
    assert format_file_name('Chapter_1.(v2)-draft') == 'Chapter_1.(v2)-draft'


def test_ebook_dest_file_path_is_sanitized_mobi_name():
    ebook = Ebook('My/Book: Vol+1')
    assert ebook.dest_file_path == pathlib.Path('MyBook-Vol-1.mobi')


def test_ebook_creates_tmpdir_and_full_dest_path():
    ebook = Ebook('Tmp Test')
    assert ebook.tmpdir.is_dir()
    assert ebook.tmpdir.parent == pathlib.Path(tempfile.gettempdir())
    assert ebook.full_dest_file_path == ebook.tmpdir / 'TmpTest.mobi'


def test_set_cover_keeps_one_source_and_rejects_empty():
    ebook = Ebook('Cover')
    assert ebook.set_cover(cover_content=b'bytes') is ebook
    assert ebook.cover_content == b'bytes'
    assert ebook.cover_path is None

    assert ebook.set_cover(cover_path='/some/cover.jpg') is ebook
    assert ebook.cover_path == '/some/cover.jpg'
    assert ebook.cover_content is None

    with pytest.raises(ValueError):
        Ebook('NoCover').set_cover()


def test_add_chapter_rejects_subchapter_level():
    ebook = Ebook('Levels')
    sub = Chapter('Nested', ebook, level=2)
    with pytest.raises(ValueError):
        ebook.add_chapter(sub)
    assert ebook.chapter_list == []


def test_create_chapter_appends_and_returns_chapter():
    ebook = Ebook('Chapters')
    first = ebook.create_chapter('Intro')
    second = ebook.create_chapter('Body')
    assert isinstance(first, Chapter)
    assert first.is_top_chapter
    assert ebook.chapter_list == [first, second]


def test_create_chapter_rejects_missing_file(tmp_path):
    ebook = Ebook('Missing')
    with pytest.raises(ValueError):
        ebook.create_chapter('Ghost', str(tmp_path / 'nope.html'))
    assert ebook.chapter_list == []


def test_create_subchapter_stops_at_max_level():
    ebook = Ebook('Depth')
    top = ebook.create_chapter('Top')
    sub = top.create_subchapter('Sub')
    assert sub.is_top_chapter is False
    assert top.sub_chapters == [sub]
    with pytest.raises(Exception, match='max level'):
        sub.create_subchapter('Too deep')


def test_chapter_content_reads_utf8_file(tmp_path):
    src = tmp_path / 'ch.html'
    src.write_text('<p>café ✓</p>', encoding='utf-8')
    ebook = Ebook('Utf8')
    ch = ebook.create_chapter('Café', str(src))
    assert ch.content == '<p>café ✓</p>'


def test_chapter_set_content_replaces_file_source(tmp_path):
    src = tmp_path / 'old.html'
    src.write_text('old', encoding='utf-8')
    ch = Ebook('SetContent').create_chapter('X', str(src))
    assert ch.set_content('new') is ch
    assert ch.content == 'new'
    src.write_text('changed on disk', encoding='utf-8')
    assert ch.content == 'new'


def test_headings_report_play_order_and_nesting():
    ebook = Ebook('Headings')
    alpha = ebook.create_chapter('Alpha')
    alpha.create_subchapter('Alpha One')
    alpha.create_subchapter('Alpha Two')
    ebook.create_chapter('Beta')

    headings = ebook._eu.headings
    assert [h['title'] for h in headings] == ['Alpha', 'Beta']
    assert [h['play_order'] for h in headings] == [2, 5]
    assert headings[0]['file_name'] == 'Alpha.html'
    subs = headings[0]['sub_headings']
    assert [s['title'] for s in subs] == ['Alpha One', 'Alpha Two']
    assert [s['play_order'] for s in subs] == [3, 4]
    assert subs[1]['file_name'] == 'AlphaTwo.html'
    assert headings[1]['sub_headings'] == []


def test_chapter_save_writes_html_and_copies_static_files(tmp_path):
    asset = tmp_path / 'style.css'
    asset.write_text('p {}', encoding='utf-8')
    ebook = Ebook('Save Chapter')
    ch = ebook.create_chapter('Part One')
    ch.set_content('<p>hello</p>').add_static_files(str(asset))
    ch.save()
    out = ebook.tmpdir / 'PartOne.html'
    assert out.read_text(encoding='utf-8') == '<p>hello</p>'
    assert (ebook.tmpdir / 'style.css').read_text(encoding='utf-8') == 'p {}'


def test_add_static_files_rejects_missing_file(tmp_path):
    ch = Ebook('Static').create_chapter('S')
    with pytest.raises(AssertionError):
        ch.add_static_files(str(tmp_path / 'missing.png'))


def test_save_refuses_book_without_chapters(tmp_path, monkeypatch):
    calls, _ = _install_fake_kindlegen(monkeypatch)
    ebook = Ebook('Empty Book')
    dest = tmp_path / 'out.mobi'
    with pytest.raises(ValueError):
        ebook.save(str(dest))
    assert calls == []
    assert not dest.exists()


def test_save_copies_generated_mobi_to_destination(tmp_path, monkeypatch):
    ebook = Ebook('Deliver')
    ebook.create_chapter('Intro').set_content('<p>x</p>')
    _install_fake_kindlegen(monkeypatch, mobi_name='Deliver.mobi')
    dest = tmp_path / 'deliver-out.mobi'
    ebook.save(str(dest))
    assert dest.read_bytes() == b'MOBI'


def test_create_renders_toc_opf_and_runs_kindlegen(monkeypatch):
    calls, seen = _install_fake_kindlegen(monkeypatch)
    ebook = Ebook('Render Me', author='Jane Doe')
    ch = ebook.create_chapter('Chapter One').set_content('<p>one</p>')
    ch.create_subchapter('Section 1.1').set_content('<p>sec</p>')
    ebook._eu.create()

    names = set(seen['names'])
    assert {'toc.ncx', 'toc.html', 'RenderMe.opf',
            'ChapterOne.html', 'Section1.1.html'} <= names

    opf = (ebook.tmpdir / 'RenderMe.opf').read_text(encoding='utf-8')
    assert '<dc:title>Render Me</dc:title>' in opf
    assert '<dc:creator>Jane Doe</dc:creator>' in opf

    ncx = (ebook.tmpdir / 'toc.ncx').read_text(encoding='utf-8')
    assert 'playOrder="2"' in ncx
    assert 'playOrder="3"' in ncx

    assert calls == [[kl.kindlegen, '-dont_append_source',
                      str(ebook.tmpdir / 'RenderMe.opf')]]


def test_create_tolerates_kindlegen_nonzero_exit(monkeypatch):
    calls, _ = _install_fake_kindlegen(monkeypatch, rc=1)
    ebook = Ebook('Failing Build')
    ebook.create_chapter('Only').set_content('body')
    assert ebook._eu.create() is None
    assert len(calls) == 1


@pytest.mark.parametrize('system, opener', [
    ('Darwin', 'open'),
    ('Linux', 'xdg-open'),
])
def test_show_opens_mobi_with_platform_viewer(monkeypatch, system, opener):
    ebook = Ebook('Viewer')
    monkeypatch.setattr(ebook, '_create', lambda: None)
    monkeypatch.setattr(kl, 'platform', SimpleNamespace(system=lambda: system))
    launched = []
    monkeypatch.setattr(kl, 'subprocess', SimpleNamespace(
        call=lambda args, *a, **kw: launched.append(list(args)) or 0))

    assert ebook.show() == 0
    assert launched == [[opener, str(ebook.full_dest_file_path)]]


def test_parse_headers_reads_title_and_nested_headings(tmp_path):
    toc = tmp_path / 'toc.md'
    toc.write_text('\n\nMy Book\n# Chapter One\n## Section 1.1\n# Chapter Two\n',
                   encoding='utf-8')
    title, headers = _parse_headers(str(toc))
    assert title == 'My Book'
    assert headers == [
        {'title': 'Chapter One', 'next_headers': [{'title': 'Section 1.1'}]},
        {'title': 'Chapter Two', 'next_headers': []},
    ]


def test_parse_headers_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError):
        _parse_headers(str(tmp_path / 'toc.md'))


def test_parse_headers_requires_at_least_one_chapter(tmp_path):
    toc = tmp_path / 'toc.md'
    toc.write_text('Only A Title\nnothing else here\n', encoding='utf-8')
    with pytest.raises(ValueError, match='headings is empty'):
        _parse_headers(str(toc))


def test_make_ebook_builds_mobi_from_source_folder(tmp_path, monkeypatch):
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'toc.md').write_text('Book\n# Ch1\n## Sec\n', encoding='utf-8')
    (src / 'Ch1.html').write_text('<p>1</p>', encoding='utf-8')
    (src / 'Sec.html').write_text('<p>s</p>', encoding='utf-8')
    (src / 'notes.txt').write_text('extra', encoding='utf-8')
    (src / 'cover.jpg').write_bytes(b'\xff\xd8fake')
    out = tmp_path / 'out'
    out.mkdir()

    _, seen = _install_fake_kindlegen(monkeypatch, mobi_name='Book.mobi')
    with pytest.warns(UserWarning, match='deprecated'):
        result = make_ebook(str(src), str(out))

    assert result == os.path.join(str(out), 'Book.mobi')
    assert pathlib.Path(result).read_bytes() == b'MOBI'
    assert {'Ch1.html', 'Sec.html', 'notes.txt', 'cover.jpg',
            'toc.md', 'toc.ncx', 'toc.html', 'Book.opf'} <= set(seen['names'])
