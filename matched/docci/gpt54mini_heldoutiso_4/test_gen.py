import pytest
from openpyxl import Workbook, load_workbook

from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


def make_workbook(values=None):
    wb = Workbook()
    ws = wb.active
    if values is not None:
        for row in values:
            ws.append(row)
    return wb


def workbook_values(workbook):
    ws = workbook.active
    return [tuple(row) for row in ws.iter_rows(values_only=True)]


def test_xlsx_to_bytes_returns_bytes():
    wb = make_workbook([["a", "b"], [1, 2]])
    result = xlsx_to_bytes(wb)
    assert isinstance(result, bytes)


def test_xlsx_to_bytes_produces_non_empty_output():
    wb = make_workbook([["hello"]])
    result = xlsx_to_bytes(wb)
    assert len(result) > 0


def test_xlsx_to_bytes_output_is_valid_xlsx():
    wb = make_workbook([["name", "age"], ["Alice", 30]])
    result = xlsx_to_bytes(wb)

    restored = load_workbook(filename=None, data_only=False)
    # load_workbook() cannot load from raw bytes directly, so use BytesIO-like handling via openpyxl API
    # by reusing xlsx_from_bytes semantics with BytesIO here.
    from io import BytesIO

    restored = load_workbook(BytesIO(result))
    assert workbook_values(restored) == [("name", "age"), ("Alice", 30)]


def test_xlsx_to_bytes_preserves_cell_values():
    wb = make_workbook([["x", 1, True], ["y", 2, False]])
    result = xlsx_to_bytes(wb)

    from io import BytesIO

    restored = load_workbook(BytesIO(result))
    assert workbook_values(restored) == [("x", 1, True), ("y", 2, False)]


def test_xlsx_to_bytes_can_be_called_multiple_times_on_same_workbook():
    wb = make_workbook([["a"], ["b"]])

    first = xlsx_to_bytes(wb)
    second = xlsx_to_bytes(wb)

    assert first == second


def test_xlsx_to_bytes_does_not_return_same_object_each_time():
    wb = make_workbook([["a"]])

    first = xlsx_to_bytes(wb)
    second = xlsx_to_bytes(wb)

    assert first is not second
    assert first == second


def test_xlsx_to_file_returns_fileattachment_instance():
    wb = make_workbook([["a", "b"]])

    attachment = xlsx_to_file(wb, "report.xlsx")

    assert isinstance(attachment, FileAttachment)


def test_xlsx_to_file_sets_name():
    wb = make_workbook([["a"]])

    attachment = xlsx_to_file(wb, "monthly.xlsx")

    assert attachment.name == "monthly.xlsx"


def test_xlsx_to_file_sets_content_to_xlsx_bytes():
    wb = make_workbook([["a", "b"], [1, 2]])

    attachment = xlsx_to_file(wb, "data.xlsx")
    expected = xlsx_to_bytes(wb)

    assert attachment.content == expected


def test_xlsx_to_file_content_is_valid_xlsx():
    wb = make_workbook([["alpha", "beta"], [10, 20]])

    attachment = xlsx_to_file(wb, "export.xlsx")

    from io import BytesIO

    restored = load_workbook(BytesIO(attachment.content))
    assert workbook_values(restored) == [("alpha", "beta"), (10, 20)]


def test_xlsx_to_file_uses_xlsx_to_bytes(monkeypatch):
    wb = make_workbook([["x"]])

    called = {}

    def fake_xlsx_to_bytes(arg):
        called["workbook"] = arg
        return b"fake-bytes"

    monkeypatch.setattr("docci.xlsx.xlsx_to_bytes", fake_xlsx_to_bytes)

    attachment = xlsx_to_file(wb, "fake.xlsx")

    assert called["workbook"] is wb
    assert attachment.name == "fake.xlsx"
    assert attachment.content == b"fake-bytes"


def test_xlsx_to_file_accepts_unicode_filename():
    wb = make_workbook([["value"]])

    attachment = xlsx_to_file(wb, "überblick.xlsx")

    assert attachment.name == "überblick.xlsx"
    assert isinstance(attachment.content, bytes)
