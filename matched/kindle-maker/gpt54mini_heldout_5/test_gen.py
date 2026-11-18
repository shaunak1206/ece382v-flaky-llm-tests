import os
import pathlib
import tempfile

import pytest

from kindle_maker.ebooklib import (
    Ebook,
    Chapter,
    EbookUtil,
    format_file_name,
    _parse_headers,
    make_ebook,
)


def test_format_file_name_strips_and_replaces_illegal_characters():
    assert format_file_name('a/b c+d"e\\f:g|h') == 'abc-def-gh'


def test_ebook_initialization_creates_tempdir_and_dest_file_paths():
    ebook = Ebook('My Book')
    try:
        assert ebook.title == 'My Book'
        assert ebook.author is None
        assert ebook.tmpdir.exists()
        assert ebook.dest_file_path.name == 'MyBook.mobi'
        assert ebook.full_dest_file_path == ebook.tmpdir / ebook.dest_file_path
    finally:
        del ebook


def test_set_cover_with_cover_path_clears_cover_content(tmp_path):
    cover = tmp_path / 'cover.jpg'
    cover.write_bytes(b'jpg')
    ebook = Ebook('Book')
    try:
        ebook.cover_content = b'old'
        ret = ebook.set_cover(cover_path=str(cover))
        assert ret is ebook
        assert ebook.cover_path == str(cover)
        assert ebook.cover_content is None
    finally:
        del ebook


def test_set_cover_with_cover_content_clears_cover_path():
    ebook = Ebook('Book')
    try:
        ret = ebook.set_cover(cover_content=b'abc')
        assert ret is ebook
        assert ebook.cover_content == b'abc'
        assert ebook.cover_path is None
    finally:
        del ebook


def test_set_cover_without_anything_raises_value_error():
    ebook = Ebook('Book')
    try:
        with pytest.raises(ValueError):
            ebook.set_cover()
    finally:
        del ebook


def test_add_chapter_accepts_top_level_chapter_only():
    ebook = Ebook('Book')
    try:
        chapter = Chapter('Chapter 1', ebook)
        ret = ebook.add_chapter(chapter)
        assert ret is ebook
        assert ebook.chapter_list == [chapter]
    finally:
        del ebook


def test_add_chapter_rejects_non_top_level_chapter():
    ebook = Ebook('Book')
    try:
        chapter = Chapter('Chapter 1', ebook, level=2)
        with pytest.raises(ValueError, match='only chapters with level 1 are accepted'):
            ebook.add_chapter(chapter)
    finally:
        del ebook


def test_create_chapter_adds_and_returns_chapter():
    ebook = Ebook('Book')
    try:
        chapter = ebook.create_chapter('Intro')
        assert isinstance(chapter, Chapter)
        assert chapter.title == 'Intro'
        assert chapter.is_top_chapter is True
        assert ebook.chapter_list == [chapter]
    finally:
        del ebook


def test_chapter_dest_file_path_is_sanitized():
    ebook = Ebook('Book')
    try:
        chapter = Chapter('A/B C+D', ebook)
        assert chapter.dest_file_path == pathlib.Path('ABC-D.html')
        assert chapter.full_dest_file_path == ebook.tmpdir / pathlib.Path('ABC-D.html')
    finally:
        del ebook


def test_chapter_set_content_overrides_file_path():
    ebook = Ebook('Book')
    try:
        chapter = Chapter('Chapter', ebook)
        chapter.set_content('<p>hello</p>')
        assert chapter.content == '<p>hello</p>'
    finally:
        del ebook


def test_chapter_create_subchapter_and_max_level_enforced():
    ebook = Ebook('Book')
    try:
        chapter = Chapter('Top', ebook)
        sub = chapter.create_subchapter('Sub')
        assert sub.title == 'Sub'
        assert sub.is_top_chapter is False
        assert chapter.sub_chapters == [sub]
        with pytest.raises(Exception, match='chapter level is great than'):
            sub.create_subchapter('Too deep')
    finally:
        del ebook


def test_chapter_add_static_files_requires_existing_files(tmp_path):
    ebook = Ebook('Book')
    try:
        chapter = Chapter('Top', ebook)
        static = tmp_path / 'asset.css'
        static.write_text('body{}', encoding='utf-8')
        ret = chapter.add_static_files(str(static))
        assert ret is chapter
        assert chapter._static_files == [str(static)]

        with pytest.raises(AssertionError):
            chapter.add_static_files(str(tmp_path / 'missing.css'))
    finally:
        del ebook


def test_ebookutil_headings_caches_and_builds_expected_structure():
    ebook = Ebook('Book')
    try:
        c1 = ebook.create_chapter('Chapter 1')
        c1.create_subchapter('Section 1.1')
        c2 = ebook.create_chapter('Chapter 2')

        util = EbookUtil(ebook)
        headings1 = util.headings
        headings2 = util.headings

        assert headings1 is headings2
        assert headings1 == [
            {
                'title': 'Chapter 1',
                'play_order': 2,
                'file_name': str(c1.dest_file_path),
                'sub_headings': [
                    {
                        'title': 'Section 1.1',
                        'play_order': 3,
                        'file_name': str(c1.sub_chapters[0].dest_file_path),
                    }
                ],
            },
            {
                'title': 'Chapter 2',
                'play_order': 4,
                'file_name': str(c2.dest_file_path),
                'sub_headings': [],
            },
        ]
    finally:
        del ebook


def test_render_opf_uses_author_fallback_when_missing(tmp_path, monkeypatch):
    ebook = Ebook('My Title')
    try:
        ebook._tempdir = tmp_path
        util = EbookUtil(ebook)

        captured = {}

        def fake_render(template_name, context, to_filename):
            captured['template_name'] = template_name
            captured['context'] = context
            captured['to_filename'] = to_filename
            (tmp_path / to_filename).write_text('opf', encoding='utf-8')

        monkeypatch.setattr(util, '_render_file', fake_render)

        fn = util._render_opf()
        assert fn == 'MyTitle.opf'
        assert captured['template_name'] == 'opf.xml'
        assert captured['context']['author'] == 'jachinlin.github.io'
        assert captured['context']['title'] == 'My Title'
        assert captured['to_filename'] == 'MyTitle.opf'
    finally:
        del ebook


def test_ebookutil_save_cover_falls_back_to_default_cover(tmp_path, monkeypatch):
    ebook = Ebook('Book')
    try:
        ebook._tempdir = tmp_path
        util = EbookUtil(ebook)

        copied = []

        def fake_copy(src, dst):
            copied.append((str(src), str(dst)))

        monkeypatch.setattr('kindle_maker.ebooklib.shutil.copy', fake_copy)

        util._save_cover()
        assert len(copied) == 1
        assert copied[0][1] == str(tmp_path)
        assert copied[0][0].endswith(os.path.join('templates', 'cover.jpg'))
    finally:
        del ebook


def test_ebook_create_calls_render_save_and_kindlegen(monkeypatch, tmp_path):
    ebook = Ebook('Book')
    try:
        ebook._tempdir = tmp_path
        chapter = ebook.create_chapter('Intro')
        chapter.set_content('content')

        calls = []

        monkeypatch.setattr(EbookUtil, '_render_toc_ncx', lambda self: calls.append('ncx') or 'toc.ncx')
        monkeypatch.setattr(EbookUtil, '_render_toc_html', lambda self: calls.append('html') or 'toc.html')
        monkeypatch.setattr(EbookUtil, '_render_opf', lambda self: calls.append('opf') or 'book.opf')
        monkeypatch.setattr(EbookUtil, '_save_cover', lambda self: calls.append('cover'))
        monkeypatch.setattr(EbookUtil, '_save_chapters', lambda self: calls.append('chapters'))
        monkeypatch.setattr('kindle_maker.ebooklib.subprocess.call', lambda args: calls.append(tuple(args)) or 0)

        ebook._create()

        assert calls[0:5] == ['ncx', 'html', 'opf', 'cover', 'chapters']
        assert calls[5][0] == 'kindlegen'
        assert calls[5][1] == '-dont_append_source'
        assert str(tmp_path / 'book.opf') in calls[5]
    finally:
        del ebook


def test_ebook_create_raises_when_no_chapters():
    ebook = Ebook('Book')
    try:
        with pytest.raises(ValueError, match='chapter list is empty'):
            ebook._create()
    finally:
        del ebook


def test_show_uses_platform_specific_viewer(monkeypatch, tmp_path):
    ebook = Ebook('Book')
    try:
        ebook._tempdir = tmp_path
        (tmp_path / ebook.dest_file_path).write_text('x', encoding='utf-8')

        monkeypatch.setattr(Ebook, '_create', lambda self: None)
        monkeypatch.setattr('kindle_maker.ebooklib.platform.system', lambda: 'Linux')
        monkeypatch.setattr('kindle_maker.ebooklib.subprocess.call', lambda args: 123)

        assert ebook.show() == 123
    finally:
        del ebook


def test_make_ebook_parses_toc_and_returns_output_path(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    output = tmp_path / 'output'
    source.mkdir()
    output.mkdir()

    (source / 'toc.md').write_text(
        'My Book\n'
        '# Chapter 1\n'
        '## Section 1.1\n'
        '# Chapter 2\n',
        encoding='utf-8',
    )
    (source / 'cover.jpg').write_bytes(b'cover')
    (source / 'extra.bin').write_bytes(b'extra')
    (source / 'Chapter 1.html').write_text('c1', encoding='utf-8')
    (source / 'Section 1.1.html').write_text('s11', encoding='utf-8')
    (source / 'Chapter 2.html').write_text('c2', encoding='utf-8')

    copied = []
    saved = []

    monkeypatch.setattr('kindle_maker.ebooklib.shutil.copy', lambda src, dst: copied.append((str(src), str(dst))))
    monkeypatch.setattr(Ebook, 'save', lambda self, file_path: saved.append(file_path))
    monkeypatch.setattr('kindle_maker.ebooklib.subprocess.call', lambda args: 0)

    result = make_ebook(str(source), str(output))

    assert result == os.path.join(str(output), 'My Book.mobi')
    assert saved == [os.path.join(str(output), 'My Book.mobi')]
    assert any(str(source / 'cover.jpg') == src for src, _ in copied)
    assert any(str(source / 'extra.bin') == src for src, _ in copied)
