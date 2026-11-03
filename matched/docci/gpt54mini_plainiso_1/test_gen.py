import pytest
from openpyxl import Workbook, load_workbook
from io import BytesIO

from docci.xlsx import xlsx_to_bytes, xlsx_to_file


def make_workbook(values=None):
    wb = Workbook()
    ws = wb.active
    values = values or []
    for row in values:
        ws.append(row)
    return wb


def workbook_values(wb):
    ws = wb.active
    return [tuple(row) for row in ws.iter_rows(values_only=True)]


def test_xlsx_to_bytes_returns_bytes():
    wb = make_workbook([["a", "b"], [1, 2]])
    result = xlsx_to_bytes(wb)
    assert isinstance(result, bytes)


def test_xlsx_to_bytes_non_empty_for_workbook():
    wb = make_workbook([["hello"]])
    result = xlsx_to_bytes(wb)
    assert len(result) > 0


def test_xlsx_to_bytes_roundtrip_preserves_cell_values():
    wb = make_workbook([["name", "age"], ["Alice", 30], ["Bob", 40]])
    result = xlsx_to_bytes(wb)

    loaded = load_workbook(BytesIO(result))
    assert workbook_values(loaded) == [("name", "age"), ("Alice", 30), ("Bob", 40)]


def test_xlsx_to_bytes_roundtrip_preserves_multiple_sheets():
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "First"
    ws1["A1"] = "alpha"

    ws2 = wb.create_sheet("Second")
    ws2["B2"] = "beta"

    result = xlsx_to_bytes(wb)
    loaded = load_workbook(BytesIO(result))

    assert loaded.sheetnames == ["First", "Second"]
    assert loaded["First"]["A1"].value == "alpha"
    assert loaded["Second"]["B2"].value == "beta"


def test_xlsx_to_bytes_handles_empty_workbook():
    wb = Workbook()
    result = xlsx_to_bytes(wb)
    loaded = load_workbook(BytesIO(result))

    assert loaded.active.title == wb.active.title
    assert workbook_values(loaded) == []


def test_xlsx_to_bytes_keeps_formula_cell_as_formula_string():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = 1
    ws["A2"] = 2
    ws["A3"] = "=SUM(A1:A2)"

    result = xlsx_to_bytes(wb)
    loaded = load_workbook(BytesIO(result), data_only=False)

    assert loaded.active["A3"].value == "=SUM(A1:A2)"


def test_xlsx_to_bytes_preserves_numeric_and_string_types():
    wb = make_workbook([[1, "1", 1.5, True, None]])
    result = xlsx_to_bytes(wb)
    loaded = load_workbook(BytesIO(result))
    row = next(loaded.active.iter_rows(values_only=True))

    assert row == (1, "1", 1.5, True, None)


def test_xlsx_to_file_returns_fileattachment_with_name_and_bytes(monkeypatch):
    wb = make_workbook([["x"]])

    sentinel = b"sentinel-bytes"

    def fake_xlsx_to_bytes(xlsx):
        assert xlsx is wb
        return sentinel

    monkeypatch.setattr("docci.xlsx.xlsx_to_bytes", fake_xlsx_to_bytes)

    attachment = xlsx_to_file(wb, "report.xlsx")

    assert attachment.name == "report.xlsx"
    assert attachment.content == sentinel


def test_xlsx_to_file_uses_xlsx_to_bytes(monkeypatch):
    wb = make_workbook([["x"]])
    calls = []

    def fake_xlsx_to_bytes(xlsx):
        calls.append(xlsx)
        return b"abc"

    monkeypatch.setattr("docci.xlsx.xlsx_to_bytes", fake_xlsx_to_bytes)

    attachment = xlsx_to_file(wb, "test.xlsx")

    assert calls == [wb]
    assert attachment.content == b"abc"


def test_xlsx_to_file_returns_fileattachment_instance():
    wb = make_workbook([["x"]])
    attachment = xlsx_to_file(wb, "file.xlsx")

    from docci.file import FileAttachment

    assert isinstance(attachment, FileAttachment)


def test_xlsx_to_file_with_unicode_filename():
    wb = make_workbook([["x"]])
    attachment = xlsx_to_file(wb, "rêport-✓.xlsx")

    assert attachment.name == "rêport-✓.xlsx"
    assert isinstance(attachment.content, bytes)


def test_xlsx_to_file_roundtrip_attachment_content_loads_as_workbook():
    wb = make_workbook([["header"], ["value"]])
    attachment = xlsx_to_file(wb, "data.xlsx")

    loaded = load_workbook(BytesIO(attachment.content))
    assert workbook_values(loaded) == [("header",), ("value",)]
