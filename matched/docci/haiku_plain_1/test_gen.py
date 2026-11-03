"""Unit tests for docci.xlsx: xlsx_to_bytes and xlsx_to_file."""

import datetime
import io
import zipfile

import pytest
from openpyxl import Workbook, load_workbook

from docci import xlsx as xlsx_mod
from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


def _load(data):
    """Load xlsx bytes with openpyxl directly (independent of the code under test)."""
    return load_workbook(io.BytesIO(data))


class _FakeWorkbook:
    """Duck-typed stand-in for openpyxl.Workbook that records the stream it receives."""

    def __init__(self, payload=b"payload"):
        self.payload = payload
        self.received = []

    def save(self, stream):
        self.received.append(stream)
        stream.write(self.payload)


class _ExplodingWorkbook:
    """Workbook whose save always fails."""

    def save(self, stream):
        raise RuntimeError("disk full")


# ---------------------------------------------------------------- xlsx_to_bytes


def test_xlsx_to_bytes_returns_bytes():
    data = xlsx_to_bytes(Workbook())
    assert type(data) is bytes
    assert len(data) > 0


def test_xlsx_to_bytes_produces_zip_archive_with_workbook_part():
    data = xlsx_to_bytes(Workbook())
    assert data.startswith(b"PK")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert "xl/workbook.xml" in archive.namelist()


def test_xlsx_to_bytes_round_trips_cell_value():
    wb = Workbook()
    wb.active["A1"] = "hello"
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active["A1"].value == "hello"


def test_xlsx_to_bytes_round_trips_numbers():
    wb = Workbook()
    wb.active["A1"] = 42
    wb.active["A2"] = 0.25
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active["A1"].value == 42
    assert loaded.active["A2"].value == 0.25


def test_xlsx_to_bytes_round_trips_datetime():
    when = datetime.datetime(2026, 11, 2)
    wb = Workbook()
    wb.active["A1"] = when
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active["A1"].value == when


def test_xlsx_to_bytes_preserves_formula_text():
    wb = Workbook()
    wb.active["A1"] = 1
    wb.active["A2"] = 2
    wb.active["A3"] = "=A1+A2"
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active["A3"].value == "=A1+A2"


def test_xlsx_to_bytes_preserves_unicode_text():
    text = "naïve café 日本語"
    wb = Workbook()
    wb.active["A1"] = text
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active["A1"].value == text


def test_xlsx_to_bytes_round_trips_multiple_sheets_in_order():
    wb = Workbook()
    wb.active.title = "Data"
    wb.active["A1"] = "first"
    notes = wb.create_sheet("Notes")
    notes["A1"] = "second"
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.sheetnames == ["Data", "Notes"]
    assert loaded["Data"]["A1"].value == "first"
    assert loaded["Notes"]["A1"].value == "second"


def test_xlsx_to_bytes_round_trips_empty_workbook():
    loaded = _load(xlsx_to_bytes(Workbook()))
    assert loaded.sheetnames == ["Sheet"]
    assert loaded.active["A1"].value is None


def test_xlsx_to_bytes_writes_all_rows():
    wb = Workbook()
    ws = wb.active
    for i in range(1, 1001):
        ws.append([i, i * 2])
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active.max_row == 1000
    assert loaded.active.cell(row=1000, column=2).value == 2000


def test_xlsx_to_bytes_does_not_mutate_workbook():
    wb = Workbook()
    wb.active["A1"] = "x"
    names_before = list(wb.sheetnames)
    xlsx_to_bytes(wb)
    assert wb.sheetnames == names_before
    assert wb.active["A1"].value == "x"


def test_xlsx_to_bytes_passes_bytesio_to_save():
    fake = _FakeWorkbook()
    xlsx_to_bytes(fake)
    assert len(fake.received) == 1
    assert isinstance(fake.received[0], io.BytesIO)


def test_xlsx_to_bytes_returns_exactly_what_save_wrote():
    fake = _FakeWorkbook(payload=b"\x00\x01binary\xff")
    assert xlsx_to_bytes(fake) == b"\x00\x01binary\xff"


def test_xlsx_to_bytes_uses_fresh_stream_per_call():
    fake = _FakeWorkbook(payload=b"abc")
    first = xlsx_to_bytes(fake)
    second = xlsx_to_bytes(fake)
    assert first == b"abc"
    assert second == b"abc"
    assert fake.received[0] is not fake.received[1]


def test_xlsx_to_bytes_propagates_save_errors():
    with pytest.raises(RuntimeError, match="disk full"):
        xlsx_to_bytes(_ExplodingWorkbook())


# ----------------------------------------------------------------- xlsx_to_file


def test_xlsx_to_file_returns_file_attachment_with_xlsx_bytes():
    att = xlsx_to_file(Workbook(), "report.xlsx")
    assert isinstance(att, FileAttachment)
    assert isinstance(att.content, bytes)
    assert att.content.startswith(b"PK")


def test_xlsx_to_file_keeps_plain_name():
    att = xlsx_to_file(Workbook(), "report.xlsx")
    assert att.name == "report.xlsx"
    assert att.extension == "xlsx"
    assert att.name_without_extension == "report"


def test_xlsx_to_file_strips_directory_from_name():
    att = xlsx_to_file(Workbook(), "out/reports/summary.xlsx")
    assert att.name == "summary.xlsx"


def test_xlsx_to_file_removes_restricted_characters_from_name():
    att = xlsx_to_file(Workbook(), "bad:name*?.xlsx")
    assert att.name == "badname.xlsx"


def test_xlsx_to_file_delegates_to_xlsx_to_bytes(monkeypatch):
    seen = []

    def fake_to_bytes(xlsx):
        seen.append(xlsx)
        return b"sentinel-bytes"

    monkeypatch.setattr(xlsx_mod, "xlsx_to_bytes", fake_to_bytes)
    wb = Workbook()
    att = xlsx_to_file(wb, "x.xlsx")
    assert len(seen) == 1 and seen[0] is wb
    assert att.content == b"sentinel-bytes"


def test_xlsx_to_file_propagates_save_errors():
    with pytest.raises(RuntimeError, match="disk full"):
        xlsx_to_file(_ExplodingWorkbook(), "x.xlsx")


def test_xlsx_to_file_save_writes_loadable_workbook(tmp_path):
    wb = Workbook()
    wb.active["A1"] = "hi"
    att = xlsx_to_file(wb, "data.xlsx")
    path = tmp_path / "data.xlsx"
    att.save(str(path))
    assert path.read_bytes() == att.content
    assert load_workbook(str(path)).active["A1"].value == "hi"
