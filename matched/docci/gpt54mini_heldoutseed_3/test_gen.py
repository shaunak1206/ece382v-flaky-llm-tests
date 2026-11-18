import io

import pytest
from openpyxl import Workbook, load_workbook

import docci.xlsx as xlsx_mod
from docci.file import FileAttachment


def make_workbook(values=None, sheet_name="Sheet"):
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name
    if values:
        for row in values:
            ws.append(row)
    return wb


def load_cells(xlsx_bytes, sheet_name=None):
    wb = load_workbook(io.BytesIO(xlsx_bytes))
    ws = wb[sheet_name] if sheet_name else wb.active
    return wb, ws


def test_xlsx_to_bytes_returns_bytes():
    wb = make_workbook([[1, 2], ["a", "b"]])

    result = xlsx_mod.xlsx_to_bytes(wb)

    assert isinstance(result, bytes)
    assert len(result) > 0


def test_xlsx_to_bytes_roundtrip_preserves_cell_values():
    wb = make_workbook([[1, "x"], [3.14, True]])

    data = xlsx_mod.xlsx_to_bytes(wb)
    _, ws = load_cells(data)

    assert ws["A1"].value == 1
    assert ws["B1"].value == "x"
    assert ws["A2"].value == 3.14
    assert ws["B2"].value is True


def test_xlsx_to_bytes_roundtrip_preserves_sheet_name():
    wb = make_workbook([[1]], sheet_name="Report")

    data = xlsx_mod.xlsx_to_bytes(wb)
    loaded_wb, ws = load_cells(data)

    assert loaded_wb.sheetnames == ["Report"]
    assert ws.title == "Report"


def test_xlsx_to_bytes_works_for_empty_workbook():
    wb = Workbook()

    data = xlsx_mod.xlsx_to_bytes(wb)

    assert isinstance(data, bytes)
    assert len(data) > 0
    loaded_wb, ws = load_cells(data)
    assert loaded_wb.sheetnames == ["Sheet"]
    assert ws.max_row == 1
    assert ws.max_column == 1


def test_xlsx_to_bytes_calls_workbook_save(monkeypatch):
    wb = Workbook()
    called = {}

    def fake_save(stream):
        called["stream_type"] = type(stream)
        stream.write(b"fake-xlsx")

    monkeypatch.setattr(wb, "save", fake_save)

    result = xlsx_mod.xlsx_to_bytes(wb)

    assert result == b"fake-xlsx"
    assert called["stream_type"] is io.BytesIO


def test_xlsx_to_bytes_uses_new_bytesio_stream_each_call():
    wb = make_workbook([[1]])

    result1 = xlsx_mod.xlsx_to_bytes(wb)
    result2 = xlsx_mod.xlsx_to_bytes(wb)

    assert result1 == result2
    assert result1 is not result2


def test_xlsx_to_bytes_return_is_independent_of_later_workbook_mutations():
    wb = make_workbook([[1, 2]])

    data_before = xlsx_mod.xlsx_to_bytes(wb)
    wb.active["A1"] = 99
    data_after = xlsx_mod.xlsx_to_bytes(wb)

    _, ws_before = load_cells(data_before)
    _, ws_after = load_cells(data_after)

    assert ws_before["A1"].value == 1
    assert ws_after["A1"].value == 99


def test_xlsx_to_file_returns_fileattachment():
    wb = make_workbook([["hello"]])

    attachment = xlsx_mod.xlsx_to_file(wb, "report.xlsx")

    assert isinstance(attachment, FileAttachment)
    assert attachment.name == "report.xlsx"
    assert isinstance(attachment.content, bytes)
    assert len(attachment.content) > 0


def test_xlsx_to_file_uses_xlsx_to_bytes(monkeypatch):
    wb = Workbook()
    seen = {}

    def fake_xlsx_to_bytes(xlsx):
        seen["workbook"] = xlsx
        return b"payload"

    monkeypatch.setattr(xlsx_mod, "xlsx_to_bytes", fake_xlsx_to_bytes)

    attachment = xlsx_mod.xlsx_to_file(wb, "name.xlsx")

    assert seen["workbook"] is wb
    assert attachment.name == "name.xlsx"
    assert attachment.content == b"payload"


def test_xlsx_to_file_produces_attachment_with_real_xlsx_bytes():
    wb = make_workbook([[1, 2], [3, 4]])

    attachment = xlsx_mod.xlsx_to_file(wb, "data.xlsx")

    loaded_wb, ws = load_cells(attachment.content)

    assert loaded_wb.sheetnames == ["Sheet"]
    assert ws["A1"].value == 1
    assert ws["B2"].value == 4


def test_xlsx_to_file_content_matches_xlsx_to_bytes():
    wb = make_workbook([["a", "b"]])

    bytes_result = xlsx_mod.xlsx_to_bytes(wb)
    attachment = xlsx_mod.xlsx_to_file(wb, "same.xlsx")

    assert attachment.content == bytes_result


def test_xlsx_to_file_name_is_preserved_verbatim():
    wb = make_workbook([[1]])

    attachment = xlsx_mod.xlsx_to_file(wb, "  spaced name .xlsx  ")

    assert attachment.name == "  spaced name .xlsx  "
