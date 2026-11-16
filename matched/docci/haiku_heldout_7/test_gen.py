import io
import zipfile
import datetime

import pytest
from openpyxl import Workbook, load_workbook

from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


def _load(data):
    return load_workbook(io.BytesIO(data))


def _make_simple_workbook():
    wb = Workbook()
    ws = wb.active
    ws.append(["name", "age"])
    ws.append(["alice", 30])
    ws.append(["bob", 25])
    return wb


# ---------------------------------------------------------------------------
# xlsx_to_bytes
# ---------------------------------------------------------------------------

def test_xlsx_to_bytes_returns_bytes_instance():
    result = xlsx_to_bytes(_make_simple_workbook())
    assert isinstance(result, bytes)


def test_xlsx_to_bytes_returns_non_empty_bytes():
    result = xlsx_to_bytes(_make_simple_workbook())
    assert len(result) > 0


def test_xlsx_to_bytes_output_is_zip_container():
    # XLSX files are OOXML packages, which are zip archives.
    result = xlsx_to_bytes(_make_simple_workbook())
    assert result[:4] == b"PK\x03\x04"
    assert zipfile.is_zipfile(io.BytesIO(result))


def test_xlsx_to_bytes_round_trip_preserves_values():
    data = xlsx_to_bytes(_make_simple_workbook())
    ws = _load(data).active
    rows = [[cell.value for cell in row] for row in ws.iter_rows()]
    assert rows == [["name", "age"], ["alice", 30], ["bob", 25]]


def test_xlsx_to_bytes_preserves_sheet_names_and_order():
    wb = Workbook()
    wb.active.title = "First"
    wb.create_sheet("Second")
    wb.create_sheet("Third")
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.sheetnames == ["First", "Second", "Third"]


def test_xlsx_to_bytes_empty_default_workbook_is_loadable():
    loaded = _load(xlsx_to_bytes(Workbook()))
    assert loaded.sheetnames == ["Sheet"]
    assert loaded.active["A1"].value is None


def test_xlsx_to_bytes_preserves_mixed_cell_types():
    wb = Workbook()
    ws = wb.active
    when = datetime.datetime(2024, 1, 2, 3, 4, 5)
    ws["A1"] = "text"
    ws["B1"] = 42
    ws["C1"] = 3.5
    ws["D1"] = True
    ws["E1"] = when
    loaded = _load(xlsx_to_bytes(wb)).active
    assert loaded["A1"].value == "text"
    assert loaded["B1"].value == 42
    assert loaded["C1"].value == pytest.approx(3.5)
    assert loaded["D1"].value is True
    assert loaded["E1"].value == when


def test_xlsx_to_bytes_preserves_formulas_as_text():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = 1
    ws["A2"] = 2
    ws["A3"] = "=A1+A2"
    loaded = _load(xlsx_to_bytes(wb)).active
    assert loaded["A3"].value == "=A1+A2"


def test_xlsx_to_bytes_preserves_unicode_text():
    wb = Workbook()
    wb.active["A1"] = "héllo wörld 你好 🚀"
    loaded = _load(xlsx_to_bytes(wb)).active
    assert loaded["A1"].value == "héllo wörld 你好 🚀"


def test_xlsx_to_bytes_preserves_merged_cells():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "merged"
    ws.merge_cells("A1:C1")
    loaded = _load(xlsx_to_bytes(wb)).active
    assert [str(r) for r in loaded.merged_cells.ranges] == ["A1:C1"]


def test_xlsx_to_bytes_preserves_cell_font_style():
    from openpyxl.styles import Font

    wb = Workbook()
    ws = wb.active
    ws["A1"] = "bold"
    ws["A1"].font = Font(bold=True)
    loaded = _load(xlsx_to_bytes(wb)).active
    assert loaded["A1"].font.bold is True


def test_xlsx_to_bytes_handles_many_rows():
    wb = Workbook()
    ws = wb.active
    for i in range(1000):
        ws.append([i, i * 2])
    loaded = _load(xlsx_to_bytes(wb)).active
    assert loaded.max_row == 1000
    assert loaded["A1000"].value == 999
    assert loaded["B1000"].value == 1998


def test_xlsx_to_bytes_does_not_modify_workbook():
    wb = _make_simple_workbook()
    sheetnames_before = list(wb.sheetnames)
    value_before = wb.active["A2"].value
    xlsx_to_bytes(wb)
    assert wb.sheetnames == sheetnames_before
    assert wb.active["A2"].value == value_before


def test_xlsx_to_bytes_repeated_calls_are_independent():
    wb = _make_simple_workbook()
    first = xlsx_to_bytes(wb)
    second = xlsx_to_bytes(wb)
    assert first is not second
    assert _load(first).active["A2"].value == "alice"
    assert _load(second).active["A2"].value == "alice"


def test_xlsx_to_bytes_raises_for_non_workbook():
    with pytest.raises(AttributeError):
        xlsx_to_bytes(object())


# ---------------------------------------------------------------------------
# xlsx_to_file
# ---------------------------------------------------------------------------

def test_xlsx_to_file_returns_file_attachment():
    result = xlsx_to_file(_make_simple_workbook(), "report.xlsx")
    assert isinstance(result, FileAttachment)


def test_xlsx_to_file_stores_given_name():
    result = xlsx_to_file(_make_simple_workbook(), "report.xlsx")
    assert result.name == "report.xlsx"


def test_xlsx_to_file_content_is_xlsx_bytes():
    result = xlsx_to_file(_make_simple_workbook(), "report.xlsx")
    assert isinstance(result.content, bytes)
    assert result.content[:4] == b"PK\x03\x04"


def test_xlsx_to_file_content_round_trips_values():
    result = xlsx_to_file(_make_simple_workbook(), "report.xlsx")
    ws = _load(result.content).active
    assert ws["A1"].value == "name"
    assert ws["B3"].value == 25


def test_xlsx_to_file_content_matches_xlsx_to_bytes_values():
    wb = _make_simple_workbook()
    from_file = _load(xlsx_to_file(wb, "data.xlsx").content)
    from_bytes = _load(xlsx_to_bytes(wb))
    rows_file = [[c.value for c in r] for r in from_file.active.iter_rows()]
    rows_bytes = [[c.value for c in r] for r in from_bytes.active.iter_rows()]
    assert rows_file == rows_bytes


def test_xlsx_to_file_accepts_empty_name():
    result = xlsx_to_file(_make_simple_workbook(), "")
    assert result.name == ""
    assert result.content[:4] == b"PK\x03\x04"


def test_xlsx_to_file_accepts_unicode_name():
    result = xlsx_to_file(_make_simple_workbook(), "отчёт_2024.xlsx")
    assert result.name == "отчёт_2024.xlsx"


def test_xlsx_to_file_does_not_modify_workbook():
    wb = _make_simple_workbook()
    sheetnames_before = list(wb.sheetnames)
    xlsx_to_file(wb, "data.xlsx")
    assert wb.sheetnames == sheetnames_before
    assert wb.active["A2"].value == "alice"


def test_xlsx_to_file_raises_for_non_workbook():
    with pytest.raises(AttributeError):
        xlsx_to_file(None, "data.xlsx")
