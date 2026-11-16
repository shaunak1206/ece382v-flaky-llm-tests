import gc
import itertools
import pathlib
import types

import pytest

from kindle_maker import ebooklib
from kindle_maker.ebooklib import Chapter, Ebook, EbookUtil, format_file_name


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """Redirect temp dirs into tmp_path and make Ebook temp dir names unique."""
    base = tmp_path / 'tmp_base'
    base.mkdir()
    monkeypatch.setattr(ebooklib.tempfile, 'gettempdir', lambda: str(base))
    counter = itertools.count(1)
    fake_time = types.SimpleNamespace(time=lambda: float(next(counter)))
    monkeypatch.setattr(ebooklib, 'time', fake_time)


@pytest.fixture
def fake_kindlegen(monkeypatch):
    """Replace the kindlegen subprocess call with a fake that writes a .mobi file."""
    calls = []

    def fake_call(args, *a, **kw):
        calls.append(list(args))
        opf = pathlib.Path(args[2])
        (opf.parent / (opf.stem + '.mobi')).write_bytes(b'MOBI')
        return 0

    monkeypatch.setattr(ebooklib.subprocess, 'call', fake_call)
    return calls


def _ebook_with_chapter(title='Book'):
    ebook = Ebook(title)
    chapter = ebook.create_chapter('Chapter One')
    chapter.set_content('<p>hello</p>')
    return ebook


def test_format_file_name_replaces_illegal_characters():
    assert format_file_name('a/b c+d"e\\f:g|h') == 'abcd-efg-h'


def test_ebook_init_creates_tmpdir_and_defaults(tmp_path):
    ebook = Ebook('My Book', author='Someone')
    assert ebook.title == 'My Book'
    assert ebook.author == 'Someone'
    assert ebook.cover_path is None
    assert ebook.cover_content is None
    assert ebook.chapter_list == []
    assert ebook.tmpdir.is_dir()
    assert str(ebook.tmpdir).startswith(str(tmp_path))


def test_ebook_dest_file_path_is_sanitized():
    ebook = Ebook('My Book')
    assert ebook.dest_file_path == pathlib.Path('MyBook.mobi')
    assert ebook.full_dest_file_path == ebook.tmpdir / 'MyBook.mobi'


def test_set_cover_with_path_clears_content_and_chains():
    ebook = Ebook('Book')
    ebook.cover_content = b'old'
    returned = ebook.set_cover(cover_path='/some/cover.jpg')
    assert returned is ebook
    assert ebook.cover_path == '/some/cover.jpg'
    assert ebook.cover_content is None


def test_set_cover_with_content_clears_path():
    ebook = Ebook('Book')
    ebook.cover_path = '/old.jpg'
    returned = ebook.set_cover(cover_content=b'img-bytes')
    assert returned is ebook
    assert ebook.cover_content == b'img-bytes'
    assert ebook.cover_path is None


def test_set_cover_without_values_raises():
    ebook = Ebook('Book')
    with pytest.raises(ValueError):
        ebook.set_cover()


def test_add_chapter_rejects_subchapter_level():
    ebook = Ebook('Book')
    parent = ebook.create_chapter('Parent')
    sub = parent.create_subchapter('Sub')
    with pytest.raises(ValueError):
        ebook.add_chapter(sub)
    assert ebook.chapter_list == [parent]


def test_create_chapter_appends_to_list():
    ebook = Ebook('Book')
    chapter = ebook.create_chapter('Intro')
    assert isinstance(chapter, Chapter)
    assert ebook.chapter_list == [chapter]
    assert chapter.title == 'Intro'


def test_chapter_with_missing_file_raises(tmp_path):
    ebook = Ebook('Book')
    with pytest.raises(ValueError):
        Chapter('Missing', ebook, str(tmp_path / 'nope.html'))


def test_create_without_chapters_raises():
    ebook = Ebook('Book')
    with pytest.raises(ValueError):
        ebook._create()
    with pytest.raises(ValueError):
        ebook.save(str(pathlib.Path('unused.mobi')))


def test_save_copies_generated_mobi(tmp_path, fake_kindlegen):
    ebook = _ebook_with_chapter('Book')
    out = tmp_path / 'out.mobi'
    ebook.save(str(out))
    assert out.read_bytes() == b'MOBI'
    assert len(fake_kindlegen) == 1


def test_create_calls_kindlegen_with_opf(fake_kindlegen):
    ebook = _ebook_with_chapter('My Book')
    ebook._create()
    args = fake_kindlegen[0]
    assert args[0] is ebooklib.kindlegen
    assert args[1] == '-dont_append_source'
    opf_path = pathlib.Path(args[2])
    assert opf_path.name == 'MyBook.opf'
    assert opf_path.parent == ebook.tmpdir


def test_create_tolerates_kindlegen_failure(monkeypatch):
    monkeypatch.setattr(ebooklib.subprocess, 'call', lambda *a, **kw: 1)
    ebook = _ebook_with_chapter('Book')
    # A non-zero return code is currently ignored; create should not raise.
    ebook._create()


def test_headings_structure_and_play_order():
    ebook = Ebook('Book')
    first = ebook.create_chapter('A')
    first.create_subchapter('A1')
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


def test_headings_are_cached():
    ebook = Ebook('Book')
    ebook.create_chapter('A')
    util = EbookUtil(ebook)
    first = util.headings
    second = util.headings
    assert first is second


def test_render_toc_ncx_contains_titles_and_default_author():
    ebook = Ebook('Book')
    ebook.create_chapter('Chapter One')
    util = ebook._eu
    name = util._render_toc_ncx()
    assert name == 'toc.ncx'
    text = (ebook.tmpdir / name).read_text(encoding='utf-8')
    assert 'Chapter One' in text
    assert 'jachinlin.github.io' in text


def test_render_opf_writes_file_named_after_title():
    ebook = Ebook('My Book', author='Jane')
    ebook.create_chapter('Chapter One')
    name = ebook._eu._render_opf()
    assert name == 'MyBook.opf'
    text = (ebook.tmpdir / name).read_text(encoding='utf-8')
    assert 'My Book' in text
    assert 'Jane' in text


def test_save_cover_copies_user_cover(tmp_path):
    src = tmp_path / 'mycover.png'
    src.write_bytes(b'PNGDATA')
    ebook = Ebook('Book')
    ebook.set_cover(cover_path=str(src))
    ebook._eu._save_cover()
    assert (ebook.tmpdir / 'mycover.png').read_bytes() == b'PNGDATA'


def test_save_cover_content_not_implemented():
    ebook = Ebook('Book')
    ebook.set_cover(cover_content=b'raw-bytes')
    with pytest.raises(NotImplementedError):
        ebook._eu._save_cover()


def test_save_cover_default_template_is_copied():
    ebook = Ebook('Book')
    ebook._eu._save_cover()
    assert (ebook.tmpdir / 'cover.jpg').is_file()


def test_chapter_save_writes_content_and_static_files(tmp_path):
    static = tmp_path / 'img.png'
    static.write_bytes(b'IMG')
    ebook = Ebook('Book')
    chapter = ebook.create_chapter('Intro')
    chapter.add_static_files(str(static))
    chapter.set_content('<p>text</p>')
    chapter.save()
    assert (ebook.tmpdir / 'Intro.html').read_text(encoding='utf-8') == '<p>text</p>'
    assert (ebook.tmpdir / 'img.png').read_bytes() == b'IMG'


def test_chapter_content_reads_file(tmp_path):
    path = tmp_path / 'chap.html'
    path.write_text('<p>from file</p>', encoding='utf-8')
    ebook = Ebook('Book')
    chapter = Chapter('Chap', ebook, str(path))
    assert chapter.content == '<p>from file</p>'


def test_create_subchapter_respects_max_level():
    ebook = Ebook('Book')
    top = ebook.create_chapter('L1')
    sub = top.create_subchapter('L2')
    assert top.is_top_chapter is True
    assert sub.is_top_chapter is False
    with pytest.raises(Exception):
        sub.create_subchapter('L3')


def test_del_removes_temp_directory():
    ebook = Ebook('Bye')
    tmpdir = ebook.tmpdir
    assert tmpdir.exists()
    del ebook
    # EbookUtil holds a back-reference to the Ebook, so collect the cycle.
    gc.collect()
    assert not tmpdir.exists()
