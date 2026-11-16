import pathlib
import subprocess
import tempfile
import time

import pytest

from kindle_maker.ebooklib import Chapter, Ebook, EbookUtil, format_file_name
from kindle_maker.kindlegen import kindlegen


def _new_ebook(title='Test Book', author=None):
    # Ebook names its temp directory after time.time(); a tiny sleep
    # avoids directory collisions between Ebook objects created in a row.
    time.sleep(0.002)
    return Ebook(title, author)


@pytest.fixture
def fake_kindlegen(monkeypatch):
    """Replace kindlegen with a fake that writes the expected .mobi file."""
    calls = []

    def fake_call(args, *a, **kw):
        calls.append(list(args))
        opf = pathlib.Path(args[2])
        (opf.parent / (opf.stem + '.mobi')).write_bytes(b'MOBI')
        return 0

    monkeypatch.setattr(subprocess, 'call', fake_call)
    return calls


def test_format_file_name_strips_illegal_characters():
    assert format_file_name('a/b c+d"e\\f:g|h') == 'abc-de' + 'f-g-h'


def test_ebook_init_stores_metadata_and_creates_tmpdir():
    book = _new_ebook('Test Book', 'Jane')
    assert book.title == 'Test Book'
    assert book.author == 'Jane'
    assert book.cover_path is None
    assert book.cover_content is None
    assert book.chapter_list == []
    assert book.tmpdir.is_dir()
    assert book.tmpdir.parent == pathlib.Path(tempfile.gettempdir())
    assert isinstance(book._eu, EbookUtil)


def test_ebook_dest_file_path_uses_sanitized_title():
    book = _new_ebook('My/Book: A+B')
    assert book.dest_file_path == pathlib.Path('MyBook-A-B.mobi')


def test_set_cover_with_path_clears_content():
    book = _new_ebook()
    book.cover_content = b'old'
    result = book.set_cover(cover_path='/some/cover.jpg')
    assert result is book
    assert book.cover_path == '/some/cover.jpg'
    assert book.cover_content is None


def test_set_cover_with_content_clears_path():
    book = _new_ebook()
    book.cover_path = '/old/cover.jpg'
    result = book.set_cover(cover_content=b'\xff\xd8')
    assert result is book
    assert book.cover_content == b'\xff\xd8'
    assert book.cover_path is None


def test_set_cover_without_arguments_raises():
    book = _new_ebook()
    with pytest.raises(ValueError):
        book.set_cover()


def test_add_chapter_rejects_subchapter_level():
    book = _new_ebook()
    sub = Chapter('Nested', book, level=2)
    with pytest.raises(ValueError):
        book.add_chapter(sub)
    assert book.chapter_list == []


def test_create_chapter_appends_top_level_chapter():
    book = _new_ebook()
    ch = book.create_chapter('Intro')
    assert isinstance(ch, Chapter)
    assert ch.title == 'Intro'
    assert ch.is_top_chapter
    assert book.chapter_list == [ch]


def test_chapter_missing_file_raises(tmp_path):
    book = _new_ebook()
    with pytest.raises(ValueError):
        Chapter('Missing', book, str(tmp_path / 'nope.html'))


def test_chapter_content_read_from_file(tmp_path):
    book = _new_ebook()
    src = tmp_path / 'ch.html'
    src.write_text('<p>hello</p>', encoding='utf-8')
    ch = Chapter('Read', book, str(src))
    assert ch.content == '<p>hello</p>'


def test_set_content_returns_self_and_overrides_file(tmp_path):
    book = _new_ebook()
    src = tmp_path / 'ch.html'
    src.write_text('from file', encoding='utf-8')
    ch = Chapter('Override', book, str(src))
    result = ch.set_content('from memory')
    assert result is ch
    assert ch.content == 'from memory'


def test_create_subchapter_increments_level():
    book = _new_ebook()
    parent = book.create_chapter('Parent')
    sub = parent.create_subchapter('Child')
    assert isinstance(sub, Chapter)
    assert not sub.is_top_chapter
    assert sub._level == 2
    assert parent.sub_chapters == [sub]


def test_create_subchapter_beyond_max_level_raises():
    book = _new_ebook()
    parent = book.create_chapter('Parent')
    sub = parent.create_subchapter('Child')
    with pytest.raises(Exception, match='chapter level'):
        sub.create_subchapter('Grandchild')


def test_add_static_files_rejects_missing_file(tmp_path):
    book = _new_ebook()
    ch = book.create_chapter('Static')
    with pytest.raises(AssertionError):
        ch.add_static_files(str(tmp_path / 'missing.png'))


def test_chapter_save_writes_content_and_copies_static(tmp_path):
    book = _new_ebook()
    ch = book.create_chapter('Saved Chapter')
    ch.set_content('<p>body</p>')
    img = tmp_path / 'img.png'
    img.write_bytes(b'PNG')
    ch.add_static_files(str(img))
    ch.save()
    assert ch.full_dest_file_path.read_text(encoding='utf-8') == '<p>body</p>'
    assert (book.tmpdir / 'img.png').read_bytes() == b'PNG'


def test_headings_structure_and_play_order():
    book = _new_ebook()
    book.create_chapter('Intro')
    usage = book.create_chapter('Usage')
    usage.create_subchapter('Install')
    usage.create_subchapter('Run')
    book.create_chapter('End')

    headings = book._eu.headings
    assert [h['title'] for h in headings] == ['Intro', 'Usage', 'End']
    assert [h['play_order'] for h in headings] == [2, 3, 6]
    assert headings[0]['file_name'] == 'Intro.html'
    assert headings[0]['sub_headings'] == []
    subs = headings[1]['sub_headings']
    assert [s['title'] for s in subs] == ['Install', 'Run']
    assert [s['play_order'] for s in subs] == [4, 5]
    assert subs[0]['file_name'] == 'Install.html'


def test_headings_are_cached():
    book = _new_ebook()
    book.create_chapter('First')
    first = book._eu.headings
    book.create_chapter('Late')
    assert book._eu.headings is first
    assert len(first) == 1


def test_save_cover_default_image():
    book = _new_ebook()
    book._eu._save_cover()
    assert (book.tmpdir / 'cover.jpg').is_file()


def test_save_cover_content_not_implemented():
    book = _new_ebook()
    book.set_cover(cover_content=b'raw bytes')
    with pytest.raises(NotImplementedError):
        book._eu._save_cover()


def test_save_cover_copies_custom_path(tmp_path):
    book = _new_ebook()
    cover = tmp_path / 'my_cover.png'
    cover.write_bytes(b'custom')
    book.set_cover(cover_path=str(cover))
    book._eu._save_cover()
    assert (book.tmpdir / 'my_cover.png').read_bytes() == b'custom'


def test_create_produces_expected_files_and_calls_kindlegen(
        tmp_path, fake_kindlegen):
    book = _new_ebook('Test Book')
    src = tmp_path / 'ch.html'
    src.write_text('<p>one</p>', encoding='utf-8')
    sub_src = tmp_path / 'sub.html'
    sub_src.write_text('<p>sub</p>', encoding='utf-8')
    ch = book.create_chapter('Chapter One', str(src))
    ch.create_subchapter('Sub Part', str(sub_src))

    book._eu.create()

    tmp = book.tmpdir
    assert (tmp / 'toc.ncx').is_file()
    assert (tmp / 'toc.html').is_file()
    assert (tmp / 'TestBook.opf').is_file()
    assert (tmp / 'cover.jpg').is_file()
    assert (tmp / 'ChapterOne.html').read_text(encoding='utf-8') == '<p>one</p>'
    assert (tmp / 'SubPart.html').read_text(encoding='utf-8') == '<p>sub</p>'

    assert len(fake_kindlegen) == 1
    args = fake_kindlegen[0]
    assert args[0] == kindlegen
    assert args[1] == '-dont_append_source'
    assert args[2] == str(tmp / 'TestBook.opf')


def test_create_ignores_kindlegen_failure(monkeypatch):
    book = _new_ebook()
    book.create_chapter('Only')
    monkeypatch.setattr(subprocess, 'call', lambda args, *a, **kw: 1)
    # A non-zero kindlegen return code is currently ignored.
    book._eu.create()
    assert (book.tmpdir / 'toc.ncx').is_file()


def test_save_copies_generated_mobi(tmp_path, fake_kindlegen):
    book = _new_ebook('Test Book')
    book.create_chapter('Chapter One').set_content('<p>x</p>')
    target = tmp_path / 'out.mobi'

    book.save(str(target))

    assert target.read_bytes() == b'MOBI'
    assert len(fake_kindlegen) == 1


def test_save_without_chapters_raises(tmp_path, fake_kindlegen):
    book = _new_ebook()
    with pytest.raises(ValueError, match='chapter list is empty'):
        book.save(str(tmp_path / 'out.mobi'))
    assert fake_kindlegen == []


def test_del_removes_tmpdir():
    book = _new_ebook('Del Test')
    tmpdir = book.tmpdir
    (tmpdir / 'leftover.txt').write_text('x', encoding='utf-8')
    book.__del__()
    assert not tmpdir.exists()
    # Recreate the directory so the automatic __del__ at garbage
    # collection time has something valid to remove without erroring.
    tmpdir.mkdir()
