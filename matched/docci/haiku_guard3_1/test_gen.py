import zipfile
from io import BytesIO

import pytest
from openpyxl import Workbook, load_workbook

import docci.xlsx as xlsx_module
from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


ZIP_MAGIC = b"PK\x03\x04"


class _FakeWorkbook:
    """Minimal stand-in exposing only the save() method used by xlsx_to_bytes."""

    def __init__(self, payload=b"payload", error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    def save(self, stream):
        self.calls.append(stream)
        if self.error is not None:
            raise self.error
        stream.write(self.payload)


def _make_workbook():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "hello"
    ws["B1"] = 42
    ws["A2"] = 3.5
    return wb


def test_xlsx_to_bytes_returns_bytes_instance():
    result = xlsx_to_bytes(_make_workbook())

    assert isinstance(result, bytes)
    assert len(result) > 0


def test_xlsx_to_bytes_output_is_zip_archive():
    result = xlsx_to_bytes(_make_workbook())

    assert result.startswith(ZIP_MAGIC)
    assert zipfile.is_zipfile(BytesIO(result))


def test_xlsx_to_bytes_archive_contains_workbook_part():
    result = xlsx_to_bytes(_make_workbook())

    with zipfile.ZipFile(BytesIO(result)) as archive:
        names = archive.namelist()

    assert "xl/workbook.xml" in names


def test_xlsx_to_bytes_round_trip_preserves_string_value():
    data = xlsx_to_bytes(_make_workbook())

    loaded = load_workbook(BytesIO(data))

    assert loaded.active["A1"].value == "hello"


def test_xlsx_to_bytes_round_trip_preserves_integer_value():
    data = xlsx_to_bytes(_make_workbook())

    loaded = load_workbook(BytesIO(data))

    assert loaded.active["B1"].value == 42


def test_xlsx_to_bytes_round_trip_preserves_float_value():
    data = xlsx_to_bytes(_make_workbook())

    loaded = load_workbook(BytesIO(data))

    assert loaded.active["A2"].value == pytest.approx(3.5, abs=1e-9)


def test_xlsx_to_bytes_round_trip_preserves_sheet_title():
    wb = Workbook()
    wb.active.title = "Summary"

    loaded = load_workbook(BytesIO(xlsx_to_bytes(wb)))

    assert loaded.sheetnames == ["Summary"]


def test_xlsx_to_bytes_round_trip_preserves_multiple_sheets_in_order():
    wb = Workbook()
    wb.active.title = "First"
    wb.create_sheet("Second")
    wb.create_sheet("Third")

    loaded = load_workbook(BytesIO(xlsx_to_bytes(wb)))

    assert loaded.sheetnames == ["First", "Second", "Third"]


def test_xlsx_to_bytes_empty_workbook_round_trip():
    wb = Workbook()

    loaded = load_workbook(BytesIO(xlsx_to_bytes(wb)))

    assert loaded.sheetnames == ["Sheet"]
    assert loaded.active["A1"].value is None


def test_xlsx_to_bytes_does_not_mutate_workbook():
    wb = _make_workbook()
    sheetnames_before = list(wb.sheetnames)

    xlsx_to_bytes(wb)

    assert wb.sheetnames == sheetnames_before
    assert wb.active["A1"].value == "hello"
    assert wb.active["B1"].value == 42
    assert wb.active["A2"].value == 3.5


def test_xlsx_to_bytes_calls_save_once_with_bytesio():
    fake = _FakeWorkbook(payload=b"abc")

    xlsx_to_bytes(fake)

    assert len(fake.calls) == 1
    assert isinstance(fake.calls[0], BytesIO)


def test_xlsx_to_bytes_returns_exactly_what_save_wrote():
    fake = _FakeWorkbook(payload=b"exact-content")

    assert xlsx_to_bytes(fake) == b"exact-content"


def test_xlsx_to_bytes_propagates_save_error():
    fake = _FakeWorkbook(error=ValueError("cannot save"))

    with pytest.raises(ValueError, match="cannot save"):
        xlsx_to_bytes(fake)


def test_xlsx_to_file_returns_file_attachment():
    result = xlsx_to_file(_make_workbook(), "report.xlsx")

    assert isinstance(result, FileAttachment)


def test_xlsx_to_file_sets_name():
    result = xlsx_to_file(_make_workbook(), "report.xlsx")

    assert result.name == "report.xlsx"


def test_xlsx_to_file_content_is_valid_xlsx():
    result = xlsx_to_file(_make_workbook(), "report.xlsx")

    assert isinstance(result.content, bytes)
    assert result.content.startswith(ZIP_MAGIC)

    loaded = load_workbook(BytesIO(result.content))
    assert loaded.active["A1"].value == "hello"


def test_xlsx_to_file_content_comes_from_xlsx_to_bytes(monkeypatch):
    sentinel = b"sentinel-bytes"
    monkeypatch.setattr(xlsx_module, "xlsx_to_bytes", lambda xlsx: sentinel)

    result = xlsx_to_file(_make_workbook(), "any.xlsx")

    assert result.content == sentinel


def test_xlsx_to_file_passes_same_workbook_to_xlsx_to_bytes(monkeypatch):
    received = []

    def fake_xlsx_to_bytes(xlsx):
        received.append(xlsx)
        return b""

    monkeypatch.setattr(xlsx_module, "xlsx_to_bytes", fake_xlsx_to_bytes)
    wb = _make_workbook()

    xlsx_to_file(wb, "name.xlsx")

    assert len(received) == 1
    assert received[0] is wb


def test_xlsx_to_file_propagates_error_from_xlsx_to_bytes(monkeypatch):
    def failing_xlsx_to_bytes(xlsx):
        raise RuntimeError("boom")

    monkeypatch.setattr(xlsx_module, "xlsx_to_bytes", failing_xlsx_to_bytes)

    with pytest.raises(RuntimeError, match="boom"):
        xlsx_to_file(_make_workbook(), "fail.xlsx")


def test_xlsx_to_file_accepts_empty_name():
    result = xlsx_to_file(Workbook(), "")

    assert result.name == ""
    assert result.content.startswith(ZIP_MAGIC)
