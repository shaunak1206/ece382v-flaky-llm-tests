import os
import pathlib
import platform
import subprocess
import tempfile

import pytest

from kindle_maker.ebooklib import Chapter, Ebook, format_file_name, make_ebook


@pytest.fixture
def ebook_root(tmp_path, monkeypatch):
    """Point Ebook temp directories at a per-test folder."""
    root = tmp_path / "tmproot"
    root.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(root))
    return root


@pytest.fixture
def fake_kindlegen(monkeypatch):
    """Replace the kindlegen/open subprocess call with an in-process fake."""
    calls = []
    state = {"rc": 0}

    def fake_call(args, *a, **kw):
        calls.append(list(args))
        if args[0] in ("open", "xdg-open"):
            return 0
        opf = pathlib.Path(args[-1])
        opf.with_suffix(".mobi").write_bytes(b"MOBI-FAKE")
        return state["rc"]

    monkeypatch.setattr(subprocess, "call", fake_call)
    return calls, state


def test_format_file_name_strips_slashes_quotes_spaces_and_backslashes():
    assert format_file_name("a/b c/d") == "abcd"
    assert format_file_name('say "hi"\\now') == "sayhinow"


def test_format_file_name_maps_plus_colon_pipe_to_dash():
    assert format_file_name("a+b:c|d") == "a-b-c-d"


def test_ebook_init_creates_tmpdir_and_defaults(ebook_root):
    book = Ebook("My Book", author="Jane")
    assert book.title == "My Book"
    assert book.author == "Jane"
    assert book.tmpdir.is_dir()
    assert book.tmpdir.parent == ebook_root
    assert book.chapter_list == []
    assert book.cover_path is None
    assert book.cover_content is None


def test_ebook_dest_file_path_is_sanitized_mobi(ebook_root):
    book = Ebook("A: B/C")
    assert book.dest_file_path == pathlib.Path("A-BC.mobi")
    assert book.full_dest_file_path == book.tmpdir / "A-BC.mobi"


def test_set_cover_path_clears_previous_content(ebook_root):
    book = Ebook("T")
    book.set_cover(cover_content=b"abc")
    result = book.set_cover(cover_path="/some/cover.jpg")
    assert result is book
    assert book.cover_path == "/some/cover.jpg"
    assert book.cover_content is None


def test_set_cover_without_arguments_raises(ebook_root):
    book = Ebook("T")
    with pytest.raises(ValueError):
        book.set_cover()


def test_add_chapter_rejects_non_top_level_chapter(ebook_root):
    book = Ebook("T")
    sub = Chapter("Sub", book, level=2)
    with pytest.raises(ValueError):
        book.add_chapter(sub)
    assert book.chapter_list == []


def test_create_chapter_registers_and_returns_chapter(ebook_root):
    book = Ebook("T")
    ch = book.create_chapter("Intro")
    assert isinstance(ch, Chapter)
    assert book.chapter_list == [ch]
    assert ch.title == "Intro"
    assert ch.is_top_chapter


def test_create_chapter_missing_file_is_not_registered(ebook_root, tmp_path):
    book = Ebook("T")
    with pytest.raises(ValueError):
        book.create_chapter("Intro", str(tmp_path / "nope.html"))
    assert book.chapter_list == []


def test_create_subchapter_is_nested_at_level_two(ebook_root):
    book = Ebook("T")
    top = book.create_chapter("Top")
    sub = top.create_subchapter("Sub")
    assert top.sub_chapters == [sub]
    assert sub.title == "Sub"
    assert sub.is_top_chapter is False


def test_create_subchapter_beyond_max_level_raises(ebook_root):
    book = Ebook("T")
    sub = book.create_chapter("Top").create_subchapter("Sub")
    with pytest.raises(Exception, match="max level"):
        sub.create_subchapter("Too deep")


def test_chapter_content_reads_utf8_file(ebook_root, tmp_path):
    src = tmp_path / "ch.html"
    src.write_text("<p>héllo</p>", encoding="utf-8")
    book = Ebook("T")
    ch = book.create_chapter("Ch", str(src))
    assert ch.content == "<p>héllo</p>"


def test_chapter_save_writes_html_and_copies_static_files(ebook_root, tmp_path):
    img = tmp_path / "pic.png"
    img.write_bytes(b"\x89PNG")
    book = Ebook("T")
    ch = book.create_chapter("Ch One")
    ch.set_content("<p>body</p>").add_static_files(str(img))
    ch.save()
    assert (book.tmpdir / "ChOne.html").read_text(encoding="utf-8") == "<p>body</p>"
    assert (book.tmpdir / "pic.png").read_bytes() == b"\x89PNG"


def test_headings_number_chapters_and_subheadings_in_order(ebook_root):
    book = Ebook("T")
    a = book.create_chapter("Alpha Beta")
    a.create_subchapter("Sub One")
    a.create_subchapter("Sub Two")
    book.create_chapter("Gamma")
    hs = book._eu.headings
    assert hs[0] == {
        "title": "Alpha Beta",
        "play_order": 2,
        "file_name": "AlphaBeta.html",
        "sub_headings": [
            {"title": "Sub One", "play_order": 3, "file_name": "SubOne.html"},
            {"title": "Sub Two", "play_order": 4, "file_name": "SubTwo.html"},
        ],
    }
    assert hs[1] == {
        "title": "Gamma",
        "play_order": 5,
        "file_name": "Gamma.html",
        "sub_headings": [],
    }


def test_render_opf_names_file_and_defaults_author(ebook_root):
    book = Ebook("My: Book")
    book.create_chapter("Intro")
    name = book._eu._render_opf()
    assert name == "My-Book.opf"
    text = (book.tmpdir / name).read_text(encoding="utf-8")
    assert "<dc:title>My: Book</dc:title>" in text
    assert "jachinlin.github.io" in text
    assert 'href="Intro.html"' in text


def test_render_toc_ncx_uses_author_and_nested_play_order(ebook_root):
    book = Ebook("T", author="Jane Doe")
    ch = book.create_chapter("Intro")
    ch.create_subchapter("Part")
    name = book._eu._render_toc_ncx()
    assert name == "toc.ncx"
    text = (book.tmpdir / "toc.ncx").read_text(encoding="utf-8")
    assert "<docAuthor><text>Jane Doe</text></docAuthor>" in text
    assert 'playOrder="2"' in text
    assert 'playOrder="3"' in text
    assert 'src="Part.html"' in text


def test_save_cover_copies_bundled_cover_by_default(ebook_root):
    book = Ebook("T")
    book._eu._save_cover()
    assert (book.tmpdir / "cover.jpg").is_file()


def test_save_cover_copies_user_cover_file(ebook_root, tmp_path):
    cov = tmp_path / "front.jpg"
    cov.write_bytes(b"jpgdata")
    book = Ebook("T").set_cover(cover_path=str(cov))
    book._eu._save_cover()
    assert (book.tmpdir / "front.jpg").read_bytes() == b"jpgdata"


def test_save_cover_raw_content_is_not_implemented(ebook_root):
    book = Ebook("T").set_cover(cover_content=b"data")
    with pytest.raises(NotImplementedError):
        book._eu._save_cover()


def test_save_without_chapters_raises(ebook_root, tmp_path):
    book = Ebook("T")
    with pytest.raises(ValueError, match="empty"):
        book.save(str(tmp_path / "out.mobi"))


def test_save_builds_with_kindlegen_and_copies_mobi(ebook_root, fake_kindlegen, tmp_path):
    calls, _state = fake_kindlegen
    book = Ebook("My Book")
    book.create_chapter("Intro").set_content("<p>hi</p>")
    out = tmp_path / "out.mobi"
    book.save(str(out))
    assert out.read_bytes() == b"MOBI-FAKE"
    assert calls[-1][1] == "-dont_append_source"
    assert calls[-1][2].endswith("MyBook.opf")


def test_save_ignores_nonzero_kindlegen_exit_code(ebook_root, fake_kindlegen, tmp_path):
    _calls, state = fake_kindlegen
    state["rc"] = 1
    book = Ebook("T")
    book.create_chapter("C").set_content("x")
    out = tmp_path / "o.mobi"
    book.save(str(out))
    assert out.exists()


@pytest.mark.parametrize("system,opener", [("Linux", "xdg-open"), ("Darwin", "open")])
def test_show_opens_mobi_with_platform_opener(ebook_root, fake_kindlegen, monkeypatch,
                                              system, opener):
    calls, _state = fake_kindlegen
    monkeypatch.setattr(platform, "system", lambda: system)
    book = Ebook("T")
    book.create_chapter("C").set_content("x")
    assert book.show() == 0
    assert calls[-1] == [opener, str(book.full_dest_file_path)]


def test_make_ebook_builds_mobi_from_toc_md(ebook_root, fake_kindlegen, tmp_path):
    _calls, _state = fake_kindlegen
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    (src / "toc.md").write_text("\nMy Book\n# Intro\n## Part One\n# Ending\n",
                                encoding="utf-8")
    (src / "Intro.html").write_text("<p>intro</p>", encoding="utf-8")
    (src / "Part One.html").write_text("<p>part</p>", encoding="utf-8")
    (src / "Ending.html").write_text("<p>end</p>", encoding="utf-8")
    (src / "style.css").write_text("p{}", encoding="utf-8")
    with pytest.warns(UserWarning, match="deprecated"):
        result = make_ebook(str(src), str(out))
    assert result == os.path.join(str(out), "My Book.mobi")
    assert os.path.isfile(result)
