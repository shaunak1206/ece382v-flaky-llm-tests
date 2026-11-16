import gc
import itertools
import pathlib
import tempfile
import types

import pytest

from kindle_maker import ebooklib
from kindle_maker.ebooklib import Chapter, Ebook, EbookUtil, format_file_name


class FakeProcess:
    """Stands in for the subprocess module so kindlegen is never run."""

    def __init__(self):
        self.calls = []
        self.returncode = 0
        self.on_call = None

    def call(self, args):
        self.calls.append(list(args))
        if self.on_call is not None:
            self.on_call(args)
        return self.returncode


@pytest.fixture(autouse=True)
def scratch_dir(monkeypatch, tmp_path):
    """
    Point the temp directory at a per-test scratch folder and make the
    'time.time()' value used for Ebook temp dir names a deterministic counter.
    Both are module attributes replaced through monkeypatch, so they are
    restored after each test.
    """
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    monkeypatch.setattr(tempfile, 'gettempdir', lambda: str(scratch))
    counter = itertools.count(1)
    fake_time = types.SimpleNamespace(time=lambda: next(counter))
    monkeypatch.setattr(ebooklib, 'time', fake_time)
    return scratch


@pytest.fixture
def fake_subprocess(monkeypatch):
    fake = FakeProcess()
    monkeypatch.setattr(ebooklib, 'subprocess',
                        types.SimpleNamespace(call=fake.call))
    return fake


def test_format_file_name_removes_and_replaces_illegal_characters():
    assert format_file_name('a"b\\c:d|e/f g') == 'abc-d-efg'


def test_ebook_defaults_after_init():
    ebook = Ebook('Title')
    assert ebook.title == 'Title'
    assert ebook.author is None
    assert ebook.cover_path is None
    assert ebook.cover_content is None
    assert ebook.chapter_list == []
    assert isinstance(ebook._eu, EbookUtil)
    assert ebook.tmpdir.is_dir()


def test_each_ebook_gets_its_own_tmpdir_under_tempdir(scratch_dir):
    first = Ebook('A')
    second = Ebook('B')
    assert first.tmpdir != second.tmpdir
    assert first.tmpdir.parent == scratch_dir
    assert second.tmpdir.parent == scratch_dir


def test_dest_file_path_uses_sanitized_title():
    ebook = Ebook('A/B C+D')
    assert ebook.dest_file_path == pathlib.Path('ABC-D.mobi')
    assert ebook.full_dest_file_path == ebook.tmpdir / 'ABC-D.mobi'


def test_create_chapter_appends_and_returns_chapter():
    ebook = Ebook('Book')
    chapter = ebook.create_chapter('One')
    assert isinstance(chapter, Chapter)
    assert ebook.chapter_list == [chapter]
    assert chapter.is_top_chapter


def test_add_chapter_rejects_non_top_level_chapter():
    ebook = Ebook('Book')
    sub = Chapter('Sub', ebook, level=2)
    with pytest.raises(ValueError):
        ebook.add_chapter(sub)
    assert ebook.chapter_list == []


def test_create_chapter_rejects_missing_file(tmp_path):
    ebook = Ebook('Book')
    with pytest.raises(ValueError):
        ebook.create_chapter('X', str(tmp_path / 'missing.html'))
    assert ebook.chapter_list == []


def test_set_cover_with_path_clears_content():
    ebook = Ebook('Book')
    ebook.set_cover(cover_content=b'data')
    returned = ebook.set_cover(cover_path='/some/cover.jpg')
    assert returned is ebook
    assert ebook.cover_path == '/some/cover.jpg'
    assert ebook.cover_content is None


def test_set_cover_with_content_clears_path():
    ebook = Ebook('Book')
    ebook.set_cover(cover_path='/some/cover.jpg')
    returned = ebook.set_cover(cover_content=b'data')
    assert returned is ebook
    assert ebook.cover_content == b'data'
    assert ebook.cover_path is None


def test_set_cover_without_arguments_raises():
    ebook = Ebook('Book')
    with pytest.raises(ValueError):
        ebook.set_cover()


def test_save_without_chapters_raises(tmp_path):
    ebook = Ebook('Book')
    with pytest.raises(ValueError):
        ebook.save(str(tmp_path / 'out.mobi'))


def test_save_copies_generated_mobi_to_target(tmp_path, fake_subprocess):
    ebook = Ebook('Book')
    ebook.create_chapter('One').set_content('<p>one</p>')

    def write_fake_mobi(args):
        opf_path = pathlib.Path(args[2])
        (opf_path.parent / ebook.dest_file_path).write_bytes(b'MOBI')

    fake_subprocess.on_call = write_fake_mobi
    target = tmp_path / 'out.mobi'
    ebook.save(str(target))
    assert target.read_bytes() == b'MOBI'


def test_deleting_ebook_removes_its_tmpdir():
    ebook = Ebook('Book')
    tmpdir = ebook.tmpdir
    (tmpdir / 'leftover.txt').write_text('x', encoding='utf-8')
    assert tmpdir.is_dir()
    del ebook
    gc.collect()
    assert not tmpdir.exists()


def test_headings_empty_without_chapters():
    ebook = Ebook('Book')
    assert ebook._eu.headings == []


def test_headings_play_order_and_file_names():
    ebook = Ebook('Book')
    one = ebook.create_chapter('One')
    one.create_subchapter('One-A')
    two = ebook.create_chapter('Two')
    two.create_subchapter('Two-A')

    headings = ebook._eu.headings
    assert [h['play_order'] for h in headings] == [2, 4]
    assert headings[0]['file_name'] == 'One.html'
    assert headings[0]['sub_headings'][0]['play_order'] == 3
    assert headings[0]['sub_headings'][0]['file_name'] == 'One-A.html'
    assert headings[1]['sub_headings'][0]['play_order'] == 5
    assert headings[1]['sub_headings'][0]['file_name'] == 'Two-A.html'


def test_headings_are_cached_after_first_access():
    ebook = Ebook('Book')
    ebook.create_chapter('One')
    eu = ebook._eu
    first = eu.headings
    ebook.create_chapter('Two')
    second = eu.headings
    assert second is first
    assert len(second) == 1


def test_render_opf_returns_sanitized_name_and_writes_file():
    ebook = Ebook('My Book')
    ebook.create_chapter('One')
    name = ebook._eu._render_opf()
    assert name == 'MyBook.opf'
    opf = ebook.tmpdir / name
    assert opf.is_file()
    assert 'jachinlin.github.io' in opf.read_text(encoding='utf-8')


def test_render_toc_ncx_writes_file_with_chapter_title():
    ebook = Ebook('Book')
    ebook.create_chapter('Intro')
    name = ebook._eu._render_toc_ncx()
    assert name == 'toc.ncx'
    text = (ebook.tmpdir / name).read_text(encoding='utf-8')
    assert 'Intro' in text


def test_save_cover_copies_user_supplied_cover(tmp_path):
    source = tmp_path / 'cover.png'
    source.write_bytes(b'image-bytes')
    ebook = Ebook('Book')
    ebook.set_cover(cover_path=str(source))
    ebook._eu._save_cover()
    assert (ebook.tmpdir / 'cover.png').read_bytes() == b'image-bytes'


def test_save_cover_uses_bundled_cover_by_default():
    ebook = Ebook('Book')
    ebook._eu._save_cover()
    assert (ebook.tmpdir / 'cover.jpg').is_file()


def test_save_cover_with_content_is_not_implemented():
    ebook = Ebook('Book')
    ebook.set_cover(cover_content=b'raw')
    with pytest.raises(NotImplementedError):
        ebook._eu._save_cover()


def test_save_chapters_writes_chapter_and_subchapter_html():
    ebook = Ebook('Book')
    chapter = ebook.create_chapter('One').set_content('<p>one</p>')
    chapter.create_subchapter('One-A').set_content('<p>sub</p>')
    ebook._eu._save_chapters()
    assert (ebook.tmpdir / 'One.html').read_text(encoding='utf-8') == '<p>one</p>'
    assert (ebook.tmpdir / 'One-A.html').read_text(encoding='utf-8') == '<p>sub</p>'


def test_create_invokes_kindlegen_with_opf_path(fake_subprocess):
    ebook = Ebook('Book')
    ebook.create_chapter('One').set_content('<p>one</p>')
    ebook._eu.create()
    assert len(fake_subprocess.calls) == 1
    args = fake_subprocess.calls[0]
    assert args[0] == ebooklib.kindlegen
    assert args[1] == '-dont_append_source'
    assert args[2] == str(ebook.tmpdir / 'Book.opf')


def test_create_ignores_nonzero_kindlegen_exit_code(fake_subprocess):
    fake_subprocess.returncode = 1
    ebook = Ebook('Book')
    ebook.create_chapter('One').set_content('<p>one</p>')
    ebook._eu.create()
    assert len(fake_subprocess.calls) == 1
