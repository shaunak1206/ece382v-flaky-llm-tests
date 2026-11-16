import itertools
import pathlib
import tempfile

import pytest

from kindle_maker import ebooklib
from kindle_maker.ebooklib import Chapter, Ebook, EbookUtil, format_file_name


class _FakeTimeModule:
    """Replacement for the ``time`` module used by ebooklib.

    Ebook names its temp directory after ``time.time()``, so every call must
    return a distinct value to avoid collisions between Ebook instances.
    """

    _counter = itertools.count()

    @staticmethod
    def time():
        return 1000.0 + next(_FakeTimeModule._counter)


@pytest.fixture(autouse=True)
def isolated_tempdir(tmp_path, monkeypatch):
    """Send all Ebook temp directories into the per-test tmp_path."""
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    monkeypatch.setattr(ebooklib, 'time', _FakeTimeModule)
    return tmp_path


@pytest.fixture
def kindlegen_calls(monkeypatch):
    """Replace subprocess.call so no real kindlegen binary is needed.

    The fake records every argument list. When kindlegen is invoked it writes
    a fake .mobi file next to the .opf file, mirroring the real tool.
    """
    calls = []

    def fake_call(args, *a, **kw):
        calls.append(list(args))
        if args and args[0] == ebooklib.kindlegen:
            opf = pathlib.Path(args[-1])
            mobi = opf.parent / (opf.stem + '.mobi')
            mobi.write_bytes(b'FAKE-MOBI')
        return 0

    monkeypatch.setattr(ebooklib.subprocess, 'call', fake_call)
    return calls


def test_format_file_name_strips_illegal_characters():
    assert format_file_name('a/b c+d"e\\f:g|h') == 'abc-def-g-h'


def test_ebook_dest_file_path_uses_sanitized_title():
    ebook = Ebook('My Book')
    assert ebook.dest_file_path == pathlib.Path('MyBook.mobi')


def test_ebook_tmpdir_is_created_under_tempdir(isolated_tempdir):
    ebook = Ebook('Book')
    assert ebook.tmpdir.is_dir()
    assert ebook.tmpdir.parent == isolated_tempdir


def test_ebook_full_dest_file_path_is_inside_tmpdir():
    ebook = Ebook('Book')
    assert ebook.full_dest_file_path == ebook.tmpdir / 'Book.mobi'


def test_ebook_init_defaults():
    ebook = Ebook('Book')
    assert ebook.title == 'Book'
    assert ebook.author is None
    assert ebook.cover_path is None
    assert ebook.cover_content is None
    assert ebook.chapter_list == []


def test_set_cover_with_path_clears_content():
    ebook = Ebook('Book')
    ebook.cover_content = b'old'
    result = ebook.set_cover(cover_path='cover.jpg')
    assert result is ebook
    assert ebook.cover_path == 'cover.jpg'
    assert ebook.cover_content is None


def test_set_cover_with_content_clears_path():
    ebook = Ebook('Book')
    ebook.cover_path = 'old.jpg'
    result = ebook.set_cover(cover_content=b'bytes')
    assert result is ebook
    assert ebook.cover_content == b'bytes'
    assert ebook.cover_path is None


def test_set_cover_without_arguments_raises():
    ebook = Ebook('Book')
    with pytest.raises(ValueError):
        ebook.set_cover()


def test_create_chapter_registers_top_level_chapter():
    ebook = Ebook('Book')
    chapter = ebook.create_chapter('Intro')
    assert isinstance(chapter, Chapter)
    assert chapter.title == 'Intro'
    assert chapter.is_top_chapter
    assert ebook.chapter_list == [chapter]


def test_add_chapter_rejects_subchapter():
    ebook = Ebook('Book')
    top = ebook.create_chapter('Top')
    sub = top.create_subchapter('Sub')
    with pytest.raises(ValueError):
        ebook.add_chapter(sub)
    assert ebook.chapter_list == [top]


def test_create_chapter_with_missing_file_raises(tmp_path):
    ebook = Ebook('Book')
    with pytest.raises(ValueError):
        ebook.create_chapter('Missing', str(tmp_path / 'nope.html'))


def test_subchapter_beyond_max_level_raises():
    ebook = Ebook('Book')
    top = ebook.create_chapter('Top')
    sub = top.create_subchapter('Sub')
    with pytest.raises(Exception, match='max level'):
        sub.create_subchapter('Deep')


def test_add_static_files_rejects_missing_file(tmp_path):
    ebook = Ebook('Book')
    chapter = ebook.create_chapter('Ch')
    with pytest.raises(AssertionError):
        chapter.add_static_files(str(tmp_path / 'missing.png'))


def test_chapter_content_read_from_file(tmp_path):
    src = tmp_path / 'src.html'
    src.write_text('hello chapter', encoding='utf-8')
    ebook = Ebook('Book')
    chapter = ebook.create_chapter('Ch', str(src))
    assert chapter.content == 'hello chapter'


def test_save_without_chapters_raises(tmp_path, kindlegen_calls):
    ebook = Ebook('Book')
    with pytest.raises(ValueError, match='chapter list is empty'):
        ebook.save(str(tmp_path / 'out.mobi'))
    assert kindlegen_calls == []


def test_save_writes_mobi_to_destination(tmp_path, kindlegen_calls):
    ebook = Ebook('Book')
    ebook.create_chapter('Ch1').set_content('<p>hi</p>')
    out = tmp_path / 'out.mobi'
    ebook.save(str(out))
    assert out.read_bytes() == b'FAKE-MOBI'
    assert kindlegen_calls[0][0] == ebooklib.kindlegen
    assert '-dont_append_source' in kindlegen_calls[0]


def test_create_writes_toc_opf_chapter_and_default_cover(kindlegen_calls):
    ebook = Ebook('Book', author='Someone')
    ebook.create_chapter('Ch1').set_content('<p>chapter body</p>')
    ebook._create()
    tmpdir = ebook.tmpdir
    assert (tmpdir / 'toc.ncx').is_file()
    assert (tmpdir / 'toc.html').is_file()
    assert (tmpdir / 'Book.opf').is_file()
    assert (tmpdir / 'cover.jpg').is_file()
    assert (tmpdir / 'Ch1.html').read_text(encoding='utf-8') == '<p>chapter body</p>'


def test_create_copies_custom_cover(tmp_path, kindlegen_calls):
    cover = tmp_path / 'mycover.png'
    cover.write_bytes(b'\x89PNG-data')
    ebook = Ebook('Book')
    ebook.set_cover(cover_path=str(cover))
    ebook.create_chapter('Ch1').set_content('x')
    ebook._create()
    assert (ebook.tmpdir / 'mycover.png').read_bytes() == b'\x89PNG-data'


def test_create_with_cover_content_raises_not_implemented(kindlegen_calls):
    ebook = Ebook('Book')
    ebook.set_cover(cover_content=b'raw-bytes')
    ebook.create_chapter('Ch1').set_content('x')
    with pytest.raises(NotImplementedError):
        ebook._create()


def test_create_copies_static_files(tmp_path, kindlegen_calls):
    img = tmp_path / 'pic.jpg'
    img.write_bytes(b'IMG')
    ebook = Ebook('Book')
    chapter = ebook.create_chapter('Ch1').set_content('x')
    chapter.add_static_files(str(img))
    ebook._create()
    assert (ebook.tmpdir / 'pic.jpg').read_bytes() == b'IMG'


def test_create_ignores_kindlegen_failure(monkeypatch):
    monkeypatch.setattr(ebooklib.subprocess, 'call', lambda args, *a, **kw: 1)
    ebook = Ebook('Book')
    ebook.create_chapter('Ch1').set_content('x')
    ebook._create()  # non-zero return code must not raise


def test_show_on_linux_uses_xdg_open(monkeypatch, kindlegen_calls):
    monkeypatch.setattr(ebooklib.platform, 'system', lambda: 'Linux')
    ebook = Ebook('Book')
    ebook.create_chapter('Ch1').set_content('x')
    rc = ebook.show()
    assert rc == 0
    assert kindlegen_calls[-1] == ['xdg-open', str(ebook.full_dest_file_path)]


def test_show_on_macos_uses_open(monkeypatch, kindlegen_calls):
    monkeypatch.setattr(ebooklib.platform, 'system', lambda: 'Darwin')
    ebook = Ebook('Book')
    ebook.create_chapter('Ch1').set_content('x')
    rc = ebook.show()
    assert rc == 0
    assert kindlegen_calls[-1] == ['open', str(ebook.full_dest_file_path)]


def test_del_removes_tmpdir():
    ebook = Ebook('Gone')
    tmpdir = ebook.tmpdir
    (tmpdir / 'f.txt').write_text('x', encoding='utf-8')
    (tmpdir / 'sub').mkdir()
    (tmpdir / 'sub' / 'g.txt').write_text('y', encoding='utf-8')
    ebook.__del__()
    assert not tmpdir.exists()
    # Recreate the directory so the garbage-collected __del__ can clean up cleanly.
    tmpdir.mkdir()


def test_headings_reflect_chapter_structure():
    ebook = Ebook('Book')
    a = ebook.create_chapter('A')
    a.create_subchapter('A1')
    ebook.create_chapter('B')
    headings = ebook._eu.headings
    assert headings == [
        {
            'title': 'A',
            'play_order': 2,
            'file_name': 'A.html',
            'sub_headings': [
                {'title': 'A1', 'play_order': 3, 'file_name': 'A1.html'},
            ],
        },
        {
            'title': 'B',
            'play_order': 4,
            'file_name': 'B.html',
            'sub_headings': [],
        },
    ]


def test_headings_empty_without_chapters():
    ebook = Ebook('Book')
    assert EbookUtil(ebook).headings == []


def test_headings_are_cached_after_first_access():
    ebook = Ebook('Book')
    ebook.create_chapter('A')
    util = EbookUtil(ebook)
    first = util.headings
    ebook.create_chapter('B')
    assert util.headings is first
    assert len(util.headings) == 1


def test_render_opf_uses_sanitized_name_and_default_author():
    ebook = Ebook('My Book')
    ebook.create_chapter('Ch1').set_content('x')
    util = EbookUtil(ebook)
    opf_name = util._render_opf()
    assert opf_name == 'MyBook.opf'
    opf_text = (ebook.tmpdir / opf_name).read_text(encoding='utf-8')
    assert 'jachinlin.github.io' in opf_text
