import zipfile
from io import BytesIO

import pytest
from openpyxl import Workbook, load_workbook

from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file, xlsx_from_bytes


def make_workbook_with_data():
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws["A1"] = "hello"
    ws["B1"] = 123
    ws["A2"] = "world"
    return wb


def test_xlsx_to_bytes_returns_bytes():
    wb = make_workbook_with_data()

    result = xlsx_to_bytes(wb)

    assert isinstance(result, bytes)


def test_xlsx_to_bytes_produces_nonempty_output():
    wb = make_workbook_with_data()

    result = xlsx_to_bytes(wb)

    assert len(result) > 0


def test_xlsx_to_bytes_output_is_a_valid_zip_archive():
    wb = make_workbook_with_data()

    result = xlsx_to_bytes(wb)

    assert zipfile.is_zipfile(BytesIO(result))


def test_xlsx_to_bytes_roundtrip_preserves_cell_values():
    wb = make_workbook_with_data()

    result = xlsx_to_bytes(wb)
    loaded = xlsx_from_bytes(result)

    ws = loaded.active
    assert ws["A1"].value == "hello"
    assert ws["B1"].value == 123
    assert ws["A2"].value == "world"


def test_xlsx_to_bytes_roundtrip_preserves_sheet_name():
    wb = make_workbook_with_data()

    result = xlsx_to_bytes(wb)
    loaded = load_workbook(BytesIO(result))

    assert loaded.active.title == "Sheet1"


def test_xlsx_to_bytes_calls_workbook_save_with_bytesio(monkeypatch):
    wb = Workbook()
    calls = {}

    def fake_save(stream):
        calls["stream_type"] = type(stream)
        stream.write(b"fake-xlsx-data")

    monkeypatch.setattr(wb, "save", fake_save)

    result = xlsx_to_bytes(wb)

    assert calls["stream_type"] is BytesIO
    assert result == b"fake-xlsx-data"


def test_xlsx_to_bytes_uses_new_buffer_each_call():
    wb = make_workbook_with_data()

    first = xlsx_to_bytes(wb)
    second = xlsx_to_bytes(wb)

    assert first == second
    assert first is not second


def test_xlsx_to_file_returns_fileattachment():
    wb = make_workbook_with_data()

    attachment = xlsx_to_file(wb, "report.xlsx")

    assert isinstance(attachment, FileAttachment)


def test_xlsx_to_file_sets_name():
    wb = make_workbook_with_data()

    attachment = xlsx_to_file(wb, "report.xlsx")

    assert attachment.name == "report.xlsx"


def test_xlsx_to_file_sets_content_to_bytes():
    wb = make_workbook_with_data()

    attachment = xlsx_to_file(wb, "report.xlsx")
    expected = xlsx_to_bytes(wb)

    assert attachment.content == expected


def test_xlsx_to_file_content_is_valid_xlsx_bytes():
    wb = make_workbook_with_data()

    attachment = xlsx_to_file(wb, "report.xlsx")
    loaded = xlsx_from_bytes(attachment.content)

    assert loaded.active["A1"].value == "hello"
    assert loaded.active["B1"].value == 123


def test_xlsx_to_file_roundtrip_from_attachment():
    wb = make_workbook_with_data()

    attachment = xlsx_to_file(wb, "report.xlsx")
    loaded = xlsx_from_bytes(attachment.content)

    assert loaded.active["A2"].value == "world"
    assert loaded.active.title == "Sheet1"
