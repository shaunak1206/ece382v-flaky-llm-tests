import io

import pytest
from openpyxl import Workbook, load_workbook

from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


def make_workbook(values=None, extra_sheet=False):
    wb = Workbook()
    ws = wb.active
    ws.title = "Main"
    values = values or [["A1", "B1"], ["A2", "B2"]]
    for row in values:
        ws.append(row)
    if extra_sheet:
        ws2 = wb.create_sheet("Second")
        ws2["A1"] = "extra"
    return wb


def test_xlsx_to_bytes_returns_bytes():
    wb = make_workbook()
    data = xlsx_to_bytes(wb)
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_xlsx_to_bytes_produces_valid_xlsx_stream():
    wb = make_workbook()
    data = xlsx_to_bytes(wb)

    loaded = load_workbook(io.BytesIO(data))
    assert loaded.sheetnames == ["Main"]
    assert loaded["Main"]["A1"].value == "A1"
    assert loaded["Main"]["B2"].value == "B2"


def test_xlsx_to_bytes_preserves_multiple_sheets():
    wb = make_workbook(extra_sheet=True)
    data = xlsx_to_bytes(wb)

    loaded = load_workbook(io.BytesIO(data))
    assert loaded.sheetnames == ["Main", "Second"]
    assert loaded["Second"]["A1"].value == "extra"


def test_xlsx_to_bytes_can_be_called_multiple_times_with_same_result_length():
    wb = make_workbook()
    data1 = xlsx_to_bytes(wb)
    data2 = xlsx_to_bytes(wb)

    assert data1 == data2


def test_xlsx_to_bytes_on_empty_workbook_round_trips():
    wb = Workbook()
    data = xlsx_to_bytes(wb)

    loaded = load_workbook(io.BytesIO(data))
    assert loaded.active.title == wb.active.title
    assert loaded.active.max_row == 1
    assert loaded.active.max_column == 1


def test_xlsx_to_bytes_does_not_modify_source_workbook():
    wb = make_workbook()
    before = wb["Main"]["A1"].value

    _ = xlsx_to_bytes(wb)

    assert wb["Main"]["A1"].value == before
    assert wb.sheetnames == ["Main"]


def test_xlsx_to_file_returns_fileattachment():
    wb = make_workbook()
    attachment = xlsx_to_file(wb, "report.xlsx")

    assert isinstance(attachment, FileAttachment)
    assert attachment.name == "report.xlsx"
    assert isinstance(attachment.content, bytes)
    assert len(attachment.content) > 0


def test_xlsx_to_file_uses_xlsx_to_bytes_content_round_trip():
    wb = make_workbook()
    attachment = xlsx_to_file(wb, "report.xlsx")

    loaded = load_workbook(io.BytesIO(attachment.content))
    assert loaded["Main"]["A2"].value == "A2"
    assert loaded["Main"]["B1"].value == "B1"


def test_xlsx_to_file_content_matches_xlsx_to_bytes():
    wb = make_workbook()
    expected = xlsx_to_bytes(wb)
    attachment = xlsx_to_file(wb, "report.xlsx")

    assert attachment.content == expected


def test_xlsx_to_file_accepts_custom_name():
    wb = make_workbook()
    attachment = xlsx_to_file(wb, "custom_name.xlsx")

    assert attachment.name == "custom_name.xlsx"


def test_xlsx_to_file_with_unicode_name():
    wb = make_workbook()
    attachment = xlsx_to_file(wb, "résumé.xlsx")

    assert attachment.name == "résumé.xlsx"
    assert attachment.content[:2] == b"PK"


def test_xlsx_to_bytes_writes_zip_signature():
    wb = make_workbook()
    data = xlsx_to_bytes(wb)

    assert data[:2] == b"PK"
