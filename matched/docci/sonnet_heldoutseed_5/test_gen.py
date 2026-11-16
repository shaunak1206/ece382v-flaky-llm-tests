import datetime
import zipfile
from io import BytesIO

import pytest
from openpyxl import Workbook, load_workbook

from docci import xlsx as xlsx_module
from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


@pytest.fixture
def workbook():
    wb = Workbook()
    ws = wb.active
    ws.title = "Data"
    ws.append(["id", "name", "score"])
    ws.append([1, "Alice", 9.5])
    ws.append([2, "Bob", 7.25])
    return wb


def _load(content):
    return load_workbook(BytesIO(content))


def test_xlsx_to_bytes_returns_bytes(workbook):
    result = xlsx_to_bytes(workbook)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_xlsx_to_bytes_is_zip_archive(workbook):
    result = xlsx_to_bytes(workbook)
    assert result[:2] == b"PK"
    assert zipfile.is_zipfile(BytesIO(result))


def test_xlsx_to_bytes_contains_workbook_part(workbook):
    result = xlsx_to_bytes(workbook)
    with zipfile.ZipFile(BytesIO(result)) as archive:
        assert "xl/workbook.xml" in archive.namelist()


def test_xlsx_to_bytes_roundtrip_values(workbook):
    loaded = _load(xlsx_to_bytes(workbook))
    rows = [list(row) for row in loaded.active.iter_rows(values_only=True)]
    assert rows == [["id", "name", "score"], [1, "Alice", 9.5], [2, "Bob", 7.25]]


def test_xlsx_to_bytes_preserves_sheet_title(workbook):
    loaded = _load(xlsx_to_bytes(workbook))
    assert loaded.sheetnames == ["Data"]


def test_xlsx_to_bytes_multiple_sheets(workbook):
    other = workbook.create_sheet("Other")
    other["A1"] = "hello"
    loaded = _load(xlsx_to_bytes(workbook))
    assert loaded.sheetnames == ["Data", "Other"]
    assert loaded["Other"]["A1"].value == "hello"


def test_xlsx_to_bytes_empty_workbook():
    result = xlsx_to_bytes(Workbook())
    loaded = _load(result)
    assert len(loaded.sheetnames) == 1
    assert list(loaded.active.iter_rows(values_only=True)) == [(None,)]


def test_xlsx_to_bytes_unicode_values():
    wb = Workbook()
    wb.active["A1"] = "Привет, мир ✓"
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active["A1"].value == "Привет, мир ✓"


def test_xlsx_to_bytes_dates_roundtrip():
    wb = Workbook()
    moment = datetime.datetime(2020, 5, 17, 13, 45, 30)
    wb.active["A1"] = moment
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active["A1"].value == moment


def test_xlsx_to_bytes_formula_preserved():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = 1
    ws["A2"] = 2
    ws["A3"] = "=SUM(A1:A2)"
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active["A3"].value == "=SUM(A1:A2)"


def test_xlsx_to_bytes_can_be_called_repeatedly(workbook):
    first = xlsx_to_bytes(workbook)
    second = xlsx_to_bytes(workbook)
    rows1 = list(_load(first).active.iter_rows(values_only=True))
    rows2 = list(_load(second).active.iter_rows(values_only=True))
    assert rows1 == rows2


def test_xlsx_to_bytes_does_not_modify_workbook(workbook):
    before = list(workbook.active.iter_rows(values_only=True))
    xlsx_to_bytes(workbook)
    after = list(workbook.active.iter_rows(values_only=True))
    assert before == after
    assert workbook.sheetnames == ["Data"]


def test_xlsx_to_bytes_many_rows():
    wb = Workbook()
    ws = wb.active
    for i in range(500):
        ws.append([i, str(i)])
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active.max_row == 500
    assert loaded.active["A500"].value == 499
    assert loaded.active["B500"].value == "499"


def test_xlsx_to_file_returns_file_attachment(workbook):
    result = xlsx_to_file(workbook, "report.xlsx")
    assert isinstance(result, FileAttachment)


def test_xlsx_to_file_sets_name(workbook):
    result = xlsx_to_file(workbook, "report.xlsx")
    assert result.name == "report.xlsx"


def test_xlsx_to_file_content_is_valid_xlsx(workbook):
    result = xlsx_to_file(workbook, "report.xlsx")
    assert isinstance(result.content, bytes)
    loaded = _load(result.content)
    rows = list(loaded.active.iter_rows(values_only=True))
    assert rows[0] == ("id", "name", "score")
    assert rows[1] == (1, "Alice", 9.5)


def test_xlsx_to_file_uses_xlsx_to_bytes(workbook, monkeypatch):
    calls = []

    def fake_to_bytes(wb):
        calls.append(wb)
        return b"fake-content"

    monkeypatch.setattr(xlsx_module, "xlsx_to_bytes", fake_to_bytes)
    result = xlsx_module.xlsx_to_file(workbook, "x.xlsx")
    assert calls == [workbook]
    assert result.content == b"fake-content"
    assert result.name == "x.xlsx"


def test_xlsx_to_file_different_names(workbook):
    first = xlsx_to_file(workbook, "a.xlsx")
    second = xlsx_to_file(workbook, "b.xlsx")
    assert first.name == "a.xlsx"
    assert second.name == "b.xlsx"
