import zipfile
from datetime import datetime
from io import BytesIO

import pytest
from openpyxl import Workbook, load_workbook

import docci.xlsx as xlsx_module
from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


@pytest.fixture
def workbook():
    wb = Workbook()
    sheet = wb.active
    sheet.title = "Data"
    sheet.append(["name", "value"])
    sheet.append(["alpha", 1])
    sheet.append(["beta", 2])
    return wb


def _load(data: bytes) -> Workbook:
    return load_workbook(BytesIO(data))


# ---------------------------------------------------------------------------
# xlsx_to_bytes
# ---------------------------------------------------------------------------

def test_xlsx_to_bytes_returns_bytes(workbook):
    result = xlsx_to_bytes(workbook)
    assert isinstance(result, bytes)


def test_xlsx_to_bytes_returns_non_empty(workbook):
    assert len(xlsx_to_bytes(workbook)) > 0


def test_xlsx_to_bytes_produces_zip_container(workbook):
    result = xlsx_to_bytes(workbook)
    assert result[:2] == b"PK"
    assert zipfile.is_zipfile(BytesIO(result))


def test_xlsx_to_bytes_round_trip_preserves_values(workbook):
    loaded = _load(xlsx_to_bytes(workbook))
    sheet = loaded["Data"]
    assert sheet["A1"].value == "name"
    assert sheet["B1"].value == "value"
    assert sheet["A2"].value == "alpha"
    assert sheet["B2"].value == 1
    assert sheet["A3"].value == "beta"
    assert sheet["B3"].value == 2


def test_xlsx_to_bytes_preserves_sheet_names_and_order():
    wb = Workbook()
    wb.active.title = "First"
    wb.create_sheet("Second")
    wb.create_sheet("Third")

    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.sheetnames == ["First", "Second", "Third"]


def test_xlsx_to_bytes_empty_workbook_is_loadable():
    loaded = _load(xlsx_to_bytes(Workbook()))
    assert len(loaded.sheetnames) >= 1


def test_xlsx_to_bytes_does_not_mutate_workbook(workbook):
    before = [[c.value for c in row] for row in workbook.active.iter_rows()]
    xlsx_to_bytes(workbook)
    after = [[c.value for c in row] for row in workbook.active.iter_rows()]
    assert before == after


def test_xlsx_to_bytes_can_be_called_repeatedly(workbook):
    first = _load(xlsx_to_bytes(workbook))
    second = _load(xlsx_to_bytes(workbook))
    assert first["Data"]["A2"].value == second["Data"]["A2"].value == "alpha"


def test_xlsx_to_bytes_reflects_changes_between_calls(workbook):
    first = xlsx_to_bytes(workbook)
    workbook.active["A4"] = "gamma"
    second = xlsx_to_bytes(workbook)

    assert first != second
    assert _load(second)["Data"]["A4"].value == "gamma"
    assert _load(first)["Data"]["A4"].value is None


def test_xlsx_to_bytes_preserves_formulas():
    wb = Workbook()
    wb.active["A1"] = 2
    wb.active["A2"] = 3
    wb.active["A3"] = "=A1+A2"

    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active["A3"].value == "=A1+A2"


def test_xlsx_to_bytes_preserves_datetime_values():
    wb = Workbook()
    wb.active["A1"] = datetime(2024, 5, 17, 12, 30, 0)

    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active["A1"].value == datetime(2024, 5, 17, 12, 30, 0)


def test_xlsx_to_bytes_round_trip_large_sheet():
    wb = Workbook()
    sheet = wb.active
    for i in range(500):
        sheet.append([i, f"row-{i}"])

    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active.max_row == 500
    assert loaded.active["B500"].value == "row-499"


def test_xlsx_to_bytes_rejects_non_workbook():
    with pytest.raises(AttributeError):
        xlsx_to_bytes(object())


# ---------------------------------------------------------------------------
# xlsx_to_file
# ---------------------------------------------------------------------------

def test_xlsx_to_file_returns_file_attachment(workbook):
    result = xlsx_to_file(workbook, "report.xlsx")
    assert isinstance(result, FileAttachment)


def test_xlsx_to_file_sets_name(workbook):
    result = xlsx_to_file(workbook, "report.xlsx")
    assert result.name == "report.xlsx"


def test_xlsx_to_file_content_matches_xlsx_to_bytes(workbook):
    result = xlsx_to_file(workbook, "report.xlsx")
    assert isinstance(result.content, bytes)
    assert _load(result.content)["Data"]["A2"].value == "alpha"


def test_xlsx_to_file_delegates_to_xlsx_to_bytes(monkeypatch, workbook):
    calls = []

    def fake_xlsx_to_bytes(xlsx):
        calls.append(xlsx)
        return b"sentinel"

    monkeypatch.setattr(xlsx_module, "xlsx_to_bytes", fake_xlsx_to_bytes)

    result = xlsx_to_file(workbook, "data.xlsx")

    assert calls == [workbook]
    assert result.content == b"sentinel"
    assert result.name == "data.xlsx"


def test_xlsx_to_file_rejects_non_workbook():
    with pytest.raises(AttributeError):
        xlsx_to_file(object(), "bad.xlsx")
