import gc
import pathlib

import pytest
from jinja2 import DictLoader, Environment

from kindle_maker import ebooklib
from kindle_maker.ebooklib import Chapter, Ebook, format_file_name
from kindle_maker.kindlegen import kindlegen


@pytest.fixture(autouse=True)
def tmp_root(tmp_path, monkeypatch):
    """Redirect Ebook temp directories into pytest's tmp_path."""
    root = tmp_path / 'tmproot'
    root.mkdir()
    monkeypatch.setattr(ebooklib.tempfile, 'gettempdir', lambda: str(root))
    return root


@pytest.fixture
def kindlegen_calls(monkeypatch):
    """Replace subprocess.call; record calls and fake the kindlegen output."""
    calls = []

    def fake_call(args, *a, **kw):
        calls.append(list(args))
        if args[0] == kindlegen:
            opf = pathlib.Path(args[2])
            (opf.parent / (opf.stem + '.mobi')).write_bytes(b'MOBI')
        return 0

    monkeypatch.setattr(ebooklib.subprocess, 'call', fake_call)
    return calls


def make_ebook(title='My Book', author=None):
    return Ebook(title, author=author)


def test_format_file_name_removes_slash_and_spaces():
    assert format_file_name('a/b c+d') == 'abc-d'


def test_format_file_name_replaces_and_strips_illegal_characters():
    assert format_file_name('"x\\y:z|w"') == 'xy-z-w'


def test_ebook_init_sets_defaults():
    ebook = make_ebook('Title', author='Someone')

    assert ebook.title == 'Title'
    assert ebook.author == 'Someone'
    assert ebook.cover_path is None
    assert ebook.cover_content is None
    assert ebook.chapter_list == []


def test_ebook_init_creates_tmpdir_under_tempdir(tmp_root):
    ebook = make_ebook()

    assert ebook.tmpdir.is_dir()
    assert ebook.tmpdir.parent == tmp_root


def test_ebook_dest_file_path_is_sanitized_mobi():
    ebook = make_ebook('My Book: Vol/1')

    assert ebook.dest_file_path == pathlib.Path('MyBook-Vol1.mobi')
    assert ebook.full_dest_file_path == ebook.tmpdir / 'MyBook-Vol1.mobi'


def test_set_cover_path_then_content_replaces_previous_cover():
    ebook = make_ebook()

    assert ebook.set_cover(cover_path='cover.jpg') is ebook
    assert ebook.cover_path == 'cover.jpg'
    assert ebook.cover_content is None

    ebook.set_cover(cover_content=b'bytes')
    assert ebook.cover_content == b'bytes'
    assert ebook.cover_path is None


def test_set_cover_without_arguments_raises():
    ebook = make_ebook()

    with pytest.raises(ValueError):
        ebook.set_cover()


def test_add_chapter_rejects_subchapter():
    ebook = make_ebook()
    sub = Chapter('Sub', ebook, level=2)

    with pytest.raises(ValueError):
        ebook.add_chapter(sub)
    assert ebook.chapter_list == []


def test_chapter_with_missing_file_raises(tmp_path):
    ebook = make_ebook()

    with pytest.raises(ValueError):
        Chapter('X', ebook, str(tmp_path / 'missing.html'))


def test_chapter_content_is_read_from_utf8_file(tmp_path):
    ebook = make_ebook()
    src = tmp_path / 'chap.html'
    src.write_text('h\u00e9llo', encoding='utf-8')

    chapter = ebook.create_chapter('Intro', str(src))

    assert chapter.content == 'h\u00e9llo'


def test_create_subchapter_respects_max_level():
    ebook = make_ebook()
    top = ebook.create_chapter('Top')
    sub = top.create_subchapter('Sub')

    assert top.is_top_chapter is True
    assert sub.is_top_chapter is False
    with pytest.raises(Exception, match='max level'):
        sub.create_subchapter('SubSub')


def test_add_static_files_rejects_missing_file(tmp_path):
    ebook = make_ebook()
    chapter = ebook.create_chapter('Intro')

    with pytest.raises(AssertionError):
        chapter.add_static_files(str(tmp_path / 'nope.png'))


def test_chapter_save_writes_content_and_copies_static_files(tmp_path):
    ebook = make_ebook()
    img = tmp_path / 'img.png'
    img.write_bytes(b'PNG')
    chapter = ebook.create_chapter('Intro')
    chapter.set_content('<p>hi</p>')
    chapter.add_static_files(str(img))

    chapter.save()

    assert (ebook.tmpdir / 'Intro.html').read_text(encoding='utf-8') == '<p>hi</p>'
    assert (ebook.tmpdir / 'img.png').read_bytes() == b'PNG'


def test_headings_have_play_order_and_nested_sub_headings():
    ebook = make_ebook()
    intro = ebook.create_chapter('Intro')
    intro.create_subchapter('Part A')
    ebook.create_chapter('Outro')

    headings = ebook._eu.headings

    assert len(headings) == 2
    assert headings[0]['title'] == 'Intro'
    assert headings[0]['play_order'] == 2
    assert headings[0]['file_name'] == 'Intro.html'
    assert headings[0]['sub_headings'] == [
        {'title': 'Part A', 'play_order': 3, 'file_name': 'PartA.html'}
    ]
    assert headings[1]['title'] == 'Outro'
    assert headings[1]['play_order'] == 4
    assert headings[1]['sub_headings'] == []


def test_render_file_writes_rendered_context_as_utf8():
    ebook = make_ebook()
    ebook._eu._templates_env = Environment(
        loader=DictLoader({'greet.txt': 'Hi {{ name }}'}))

    ebook._eu._render_file('greet.txt', {'name': 'Zoe'}, 'out.txt')

    assert (ebook.tmpdir / 'out.txt').read_text(encoding='utf-8') == 'Hi Zoe'


def test_render_toc_ncx_uses_default_author_and_headings():
    ebook = make_ebook('Book')
    ebook.create_chapter('Intro')
    ebook._eu._templates_env = Environment(loader=DictLoader({
        'toc.xml': '{{ title }}|{{ author }}|'
                   '{% for h in headings %}{{ h.title }};{% endfor %}'
    }))

    name = ebook._eu._render_toc_ncx()

    assert name == 'toc.ncx'
    assert (ebook.tmpdir / 'toc.ncx').read_text(encoding='utf-8') == \
        'Book|jachinlin.github.io|Intro;'


def test_render_opf_uses_sanitized_file_name():
    ebook = make_ebook('My Book: Vol/1')
    ebook._eu._templates_env = Environment(
        loader=DictLoader({'opf.xml': 'opf'}))

    opf_name = ebook._eu._render_opf()

    assert opf_name == 'MyBook-Vol1.opf'
    assert (ebook.tmpdir / opf_name).is_file()


def test_save_cover_default_copies_packaged_cover():
    ebook = make_ebook()
    source = pathlib.Path(ebooklib.__file__).parent / 'templates' / 'cover.jpg'

    ebook._eu._save_cover()

    assert (ebook.tmpdir / 'cover.jpg').read_bytes() == source.read_bytes()


def test_save_cover_with_content_raises_not_implemented():
    ebook = make_ebook()
    ebook.set_cover(cover_content=b'raw')

    with pytest.raises(NotImplementedError):
        ebook._eu._save_cover()


def test_save_without_chapters_raises(tmp_path):
    ebook = make_ebook()

    with pytest.raises(ValueError, match='chapter list is empty'):
        ebook.save(str(tmp_path / 'out.mobi'))


def test_save_copies_generated_mobi_to_destination(tmp_path, kindlegen_calls):
    ebook = make_ebook('My Book')
    ebook.create_chapter('Intro').set_content('<p>hi</p>')
    target = tmp_path / 'out.mobi'

    ebook.save(str(target))

    assert target.read_bytes() == b'MOBI'


def test_create_invokes_kindlegen_with_opf_path(kindlegen_calls):
    ebook = make_ebook('My Book')
    ebook.create_chapter('Intro').set_content('<p>hi</p>')

    ebook._eu.create()

    assert kindlegen_calls[0][0] == kindlegen
    assert kindlegen_calls[0][1] == '-dont_append_source'
    assert kindlegen_calls[0][2] == str(ebook.tmpdir / 'MyBook.opf')


def test_create_does_not_raise_on_nonzero_kindlegen_exit(monkeypatch):
    ebook = make_ebook()
    ebook.create_chapter('Intro').set_content('<p>hi</p>')
    monkeypatch.setattr(ebooklib.subprocess, 'call', lambda args, *a, **kw: 1)

    assert ebook._eu.create() is None


def test_show_on_linux_uses_xdg_open(monkeypatch, kindlegen_calls):
    monkeypatch.setattr(ebooklib.platform, 'system', lambda: 'Linux')
    ebook = make_ebook('My Book')
    ebook.create_chapter('Intro').set_content('<p>hi</p>')

    assert ebook.show() == 0

    assert kindlegen_calls[-1][0] == 'xdg-open'
    assert kindlegen_calls[-1][1] == str(ebook.full_dest_file_path)


def test_del_removes_tmpdir():
    ebook = make_ebook('Gone')
    tmpdir = ebook.tmpdir
    assert tmpdir.is_dir()

    del ebook
    gc.collect()

    assert not tmpdir.exists()
