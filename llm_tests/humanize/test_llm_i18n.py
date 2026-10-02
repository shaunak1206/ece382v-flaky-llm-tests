"""Unit tests for humanize.i18n: activate/deactivate, thousands_separator,
decimal_separator, and the effect of an active locale on translated output
(ordinal, apnumber, naturaldelta) and on number separators (intcomma).

Based only on the public API in subjects/humanize/src/humanize/i18n.py.

The checked-out source ships uncompiled ``*.po`` translation catalogs (no
``*.mo`` files), so ``humanize.activate()`` cannot find a compiled catalog on
disk for locales such as "fr_FR" or "de_DE". To exercise real translated
output without adding any dependency beyond the standard library, this file
includes a small, self-contained ``*.po`` -> ``*.mo`` compiler and points
``activate()`` at a temporary directory holding the compiled catalog.
"""

from __future__ import annotations

import datetime as dt
import re
import struct
from pathlib import Path

import pytest

import humanize
from humanize import i18n

HUMANIZE_LOCALE_DIR = Path(i18n.__file__).parent / "locale"

_LINE_RE = re.compile(r'^(msgctxt|msgid_plural|msgid|msgstr)(\[(\d+)\])?\s+"(.*)"$')
_CONT_RE = re.compile(r'^"(.*)"$')


def _unescape(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s) and s[i + 1] in ('n', 't', '"', "\\"):
            out.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\"}[s[i + 1]])
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _parse_po(text: str) -> list[dict]:
    """Minimal PO parser: enough for humanize's own catalogs (no comments
    inside continued strings, one entry per blank-line-separated block).
    """
    entries: list[dict] = []
    cur: dict = {}
    cur_field = None

    def flush():
        nonlocal cur, cur_field
        if cur:
            entries.append(cur)
        cur = {}
        cur_field = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if m:
            key, _bracket, idx, val = m.groups()
            val = _unescape(val)
            if key == "msgstr" and idx is not None:
                cur.setdefault("msgstr_plural", {})[int(idx)] = val
                cur_field = ("msgstr_plural", int(idx))
            else:
                cur[key] = val
                cur_field = key
            continue
        m2 = _CONT_RE.match(line)
        if m2 and cur_field is not None:
            val = _unescape(m2.group(1))
            if isinstance(cur_field, tuple):
                cur["msgstr_plural"][cur_field[1]] += val
            else:
                cur[cur_field] += val
    flush()
    return entries


def _build_catalog(entries: list[dict]) -> dict[str, str]:
    catalog: dict[str, str] = {}
    for e in entries:
        if "msgid" not in e:
            continue
        msgid = e["msgid"]
        msgctxt = e.get("msgctxt")
        if "msgid_plural" in e:
            key = msgid + "\x00" + e["msgid_plural"]
            mp = e.get("msgstr_plural", {})
            value = "\x00".join(mp[i] for i in sorted(mp))
        else:
            key = msgid
            value = e.get("msgstr", "")
        if msgctxt is not None:
            key = msgctxt + "\x04" + key
        catalog[key] = value
    return catalog


def _compile_mo_bytes(catalog: dict[str, str]) -> bytes:
    """Build a minimal, valid GNU gettext ``.mo`` binary from a catalog dict."""
    keys = sorted(catalog.keys())
    n = len(keys)
    header_size = 7 * 4
    o_table_offset = header_size
    t_table_offset = o_table_offset + n * 8
    ids_start = t_table_offset + n * 8

    ids_blob = b""
    id_offsets = []
    for k in keys:
        kb = k.encode("utf-8")
        id_offsets.append((len(kb), len(ids_blob)))
        ids_blob += kb + b"\x00"

    strs_start = ids_start + len(ids_blob)
    strs_blob = b""
    str_offsets = []
    for k in keys:
        vb = catalog[k].encode("utf-8")
        str_offsets.append((len(vb), len(strs_blob)))
        strs_blob += vb + b"\x00"

    o_table = b"".join(
        struct.pack("<II", length, ids_start + off) for length, off in id_offsets
    )
    t_table = b"".join(
        struct.pack("<II", length, strs_start + off) for length, off in str_offsets
    )
    header = struct.pack("<7I", 0x950412DE, 0, n, o_table_offset, t_table_offset, 0, 0)
    return header + o_table + t_table + ids_blob + strs_blob


def _compile_locale(locale: str, out_dir: Path) -> str:
    po_path = HUMANIZE_LOCALE_DIR / locale / "LC_MESSAGES" / "humanize.po"
    entries = _parse_po(po_path.read_text(encoding="utf-8"))
    catalog = _build_catalog(entries)
    mo_dir = out_dir / locale / "LC_MESSAGES"
    mo_dir.mkdir(parents=True, exist_ok=True)
    (mo_dir / "humanize.mo").write_bytes(_compile_mo_bytes(catalog))
    return str(out_dir)


@pytest.fixture(autouse=True)
def _reset_locale():
    """Guarantee no test in this file leaks an active locale to another."""
    humanize.deactivate()
    try:
        yield
    finally:
        humanize.deactivate()


@pytest.fixture
def compiled_locales(tmp_path):
    """Compile the real fr_FR and de_DE catalogs into a temp directory and
    return its path, suitable for ``humanize.activate(locale, path=...)``.
    """
    _compile_locale("fr_FR", tmp_path)
    _compile_locale("de_DE", tmp_path)
    return str(tmp_path)


def test_default_thousands_and_decimal_separators():
    assert humanize.thousands_separator() == ","
    assert humanize.decimal_separator() == "."


def test_deactivate_restores_default_separators(compiled_locales):
    humanize.activate("fr_FR", path=compiled_locales)
    humanize.deactivate()
    assert humanize.thousands_separator() == ","
    assert humanize.decimal_separator() == "."


def test_activate_none_is_same_as_deactivate(compiled_locales):
    humanize.activate("fr_FR", path=compiled_locales)
    humanize.activate(None)
    assert humanize.thousands_separator() == ","
    assert humanize.decimal_separator() == "."


def test_activate_english_locale_is_a_no_op(compiled_locales):
    humanize.activate("en_US")
    assert humanize.thousands_separator() == ","
    assert humanize.decimal_separator() == "."


def test_activate_unknown_locale_without_catalog_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        humanize.activate("xx_XX", path=str(tmp_path))


def test_activate_fr_fr_sets_french_separators(compiled_locales):
    humanize.activate("fr_FR", path=compiled_locales)
    assert humanize.thousands_separator() == " "
    assert humanize.decimal_separator() == "."


def test_activate_de_de_sets_german_separators(compiled_locales):
    humanize.activate("de_DE", path=compiled_locales)
    assert humanize.thousands_separator() == "."
    assert humanize.decimal_separator() == ","


def test_intcomma_uses_active_locale_separators(compiled_locales):
    humanize.activate("fr_FR", path=compiled_locales)
    assert humanize.intcomma(1234567.89, 2) == "1 234 567.89"


def test_intcomma_german_separators(compiled_locales):
    humanize.activate("de_DE", path=compiled_locales)
    assert humanize.intcomma(1234567.89, 2) == "1.234.567,89"


def test_ordinal_uses_french_translation(compiled_locales):
    humanize.activate("fr_FR", path=compiled_locales)
    assert humanize.ordinal(1) == "1er"
    assert humanize.ordinal(2) == "2e"


def test_apnumber_uses_french_translation(compiled_locales):
    humanize.activate("fr_FR", path=compiled_locales)
    assert humanize.apnumber(0) == "zéro"
    assert humanize.apnumber(1) == "un"


def test_apnumber_uses_german_translation(compiled_locales):
    humanize.activate("de_DE", path=compiled_locales)
    assert humanize.apnumber(1) == "eins"


def test_naturaldelta_uses_french_translation(compiled_locales):
    humanize.activate("fr_FR", path=compiled_locales)
    assert humanize.naturaldelta(dt.timedelta(seconds=1)) == "une seconde"
    assert humanize.naturaldelta(dt.timedelta(seconds=30)) == "30 secondes"


def test_naturaldelta_uses_german_translation(compiled_locales):
    humanize.activate("de_DE", path=compiled_locales)
    assert humanize.naturaldelta(dt.timedelta(days=0, seconds=0)) == "ein Moment"
    assert humanize.naturaldelta(dt.timedelta(hours=1)) == "eine Stunde"


def test_naturalsize_uses_french_translation(compiled_locales):
    humanize.activate("fr_FR", path=compiled_locales)
    assert humanize.naturalsize(1) == "1 octet"


def test_translation_does_not_leak_across_activate_calls(compiled_locales):
    humanize.activate("fr_FR", path=compiled_locales)
    assert humanize.apnumber(1) == "un"
    humanize.activate("de_DE", path=compiled_locales)
    assert humanize.apnumber(1) == "eins"
    humanize.deactivate()
    assert humanize.apnumber(1) == "one"
