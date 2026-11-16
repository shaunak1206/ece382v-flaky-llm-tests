import datetime
from io import BytesIO
from zipfile import ZipFile

import pytest
from openpyxl import Workbook, load_workbook

from docci.file import FileAttachment
from docci.xlsx import (
    dicts_to_xlsx,
    xlsx_from_bytes,
    xlsx_from_file,
    xlsx_to_bytes,
    xlsx_to_file,
)


@pytest.fixture
def workbook():
    wb = Workbook()
    ws = wb.active
    ws.title = "Data"
    ws.append(["name", "age"])
    ws.append(["Alice", 30])
    ws.append(["Bob", 25])
    return wb


def test_xlsx_to_bytes_returns_bytes(workbook):
    result = xlsx_to_bytes(workbook)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_xlsx_to_bytes_is_zip_archive(workbook):
    result = xlsx_to_bytes(workbook)
    assert result[:2] == b"PK"
    with ZipFile(BytesIO(result)) as archive:
        assert "[Content_Types].xml" in archive.namelist()
        assert "xl/workbook.xml" in archive.namelist()


def test_xlsx_to_bytes_roundtrip_values(workbook):
    loaded = load_workbook(BytesIO(xlsx_to_bytes(workbook)))
    rows = [list(row) for row in loaded.active.iter_rows(values_only=True)]
    assert rows == [["name", "age"], ["Alice", 30], ["Bob", 25]]


def test_xlsx_to_bytes_preserves_sheet_title(workbook):
    loaded = load_workbook(BytesIO(xlsx_to_bytes(workbook)))
    assert loaded.sheetnames == ["Data"]


def test_xlsx_to_bytes_empty_workbook():
    result = xlsx_to_bytes(Workbook())
    loaded = load_workbook(BytesIO(result))
    assert len(loaded.sheetnames) == 1
    assert loaded.active.max_row == 1
    assert loaded.active["A1"].value is None


def test_xlsx_to_bytes_multiple_sheets():
    wb = Workbook()
    wb.active.title = "First"
    second = wb.create_sheet("Second")
    second["A1"] = "hello"
    loaded = load_workbook(BytesIO(xlsx_to_bytes(wb)))
    assert loaded.sheetnames == ["First", "Second"]
    assert loaded["Second"]["A1"].value == "hello"


def test_xlsx_to_bytes_unicode_values():
    wb = Workbook()
    wb.active["A1"] = "Привет, мир ✓"
    loaded = load_workbook(BytesIO(xlsx_to_bytes(wb)))
    assert loaded.active["A1"].value == "Привет, мир ✓"


def test_xlsx_to_bytes_preserves_formula():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = 1
    ws["A2"] = 2
    ws["A3"] = "=SUM(A1:A2)"
    loaded = load_workbook(BytesIO(xlsx_to_bytes(wb)))
    assert loaded.active["A3"].value == "=SUM(A1:A2)"


def test_xlsx_to_bytes_preserves_dates_and_floats():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = datetime.datetime(2020, 1, 2, 3, 4, 5)
    ws["A2"] = 1.5
    loaded = load_workbook(BytesIO(xlsx_to_bytes(wb)))
    assert loaded.active["A1"].value == datetime.datetime(2020, 1, 2, 3, 4, 5)
    assert loaded.active["A2"].value == 1.5


def test_xlsx_to_bytes_can_be_called_repeatedly(workbook):
    first = xlsx_to_bytes(workbook)
    second = xlsx_to_bytes(workbook)
    for data in (first, second):
        loaded = load_workbook(BytesIO(data))
        assert loaded.active["A2"].value == "Alice"


def test_xlsx_to_bytes_does_not_modify_workbook(workbook):
    xlsx_to_bytes(workbook)
    assert workbook.sheetnames == ["Data"]
    assert workbook.active["B3"].value == 25


def test_xlsx_to_bytes_reflects_changes_between_calls(workbook):
    xlsx_to_bytes(workbook)
    workbook.active["A2"] = "Changed"
    loaded = load_workbook(BytesIO(xlsx_to_bytes(workbook)))
    assert loaded.active["A2"].value == "Changed"


def test_xlsx_to_file_returns_file_attachment(workbook):
    result = xlsx_to_file(workbook, "report.xlsx")
    assert isinstance(result, FileAttachment)


def test_xlsx_to_file_sets_name(workbook):
    result = xlsx_to_file(workbook, "report.xlsx")
    assert result.name == "report.xlsx"


def test_xlsx_to_file_content_is_valid_xlsx(workbook):
    result = xlsx_to_file(workbook, "report.xlsx")
    assert isinstance(result.content, bytes)
    loaded = load_workbook(BytesIO(result.content))
    assert [list(r) for r in loaded.active.iter_rows(values_only=True)] == [
        ["name", "age"],
        ["Alice", 30],
        ["Bob", 25],
    ]


def test_xlsx_to_file_content_matches_loadable_bytes(workbook):
    result = xlsx_to_file(workbook, "a.xlsx")
    direct = load_workbook(BytesIO(xlsx_to_bytes(workbook)))
    from_file = load_workbook(BytesIO(result.content))
    assert direct.sheetnames == from_file.sheetnames
    assert direct.active["A2"].value == from_file.active["A2"].value


def test_xlsx_to_file_roundtrip_with_xlsx_from_file(workbook):
    attachment = xlsx_to_file(workbook, "round.xlsx")
    loaded = xlsx_from_file(attachment)
    assert loaded.active["A3"].value == "Bob"
    assert loaded.active["B3"].value == 25


def test_xlsx_to_bytes_roundtrip_with_dicts_and_from_bytes():
    wb = dicts_to_xlsx([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
    loaded = xlsx_from_bytes(xlsx_to_bytes(wb))
    rows = [tuple(r) for r in loaded.active.iter_rows(values_only=True)]
    assert rows == [("a", "b"), (1, "x"), (2, "y")]


def test_xlsx_to_bytes_invalid_input_raises():
    with pytest.raises(AttributeError):
        xlsx_to_bytes(None)
