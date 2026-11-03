"""Unit tests for docci.xlsx.xlsx_to_bytes and docci.xlsx.xlsx_to_file."""

from io import BytesIO

from openpyxl import Workbook, load_workbook

from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file

ZIP_MAGIC = b"PK\x03\x04"


def _make_workbook():
    wb = Workbook()
    ws = wb.active
    ws.title = "Data"
    ws.append(["name", "count", "ratio", "flag"])
    ws.append(["alpha", 3, 0.5, True])
    ws.append(["beta", 7, 1.25, False])
    return wb


def _load(data):
    return load_workbook(BytesIO(data))


def _rows(wb):
    return [tuple(row) for row in wb.active.iter_rows(values_only=True)]


def test_xlsx_to_bytes_returns_bytes_instance():
    data = xlsx_to_bytes(_make_workbook())
    assert isinstance(data, bytes)


def test_xlsx_to_bytes_output_is_zip_container():
    data = xlsx_to_bytes(_make_workbook())
    assert data.startswith(ZIP_MAGIC)


def test_xlsx_to_bytes_output_is_not_empty():
    assert len(xlsx_to_bytes(Workbook())) > 0


def test_xlsx_to_bytes_round_trip_preserves_sheet_title():
    loaded = _load(xlsx_to_bytes(_make_workbook()))
    assert loaded.sheetnames == ["Data"]


def test_xlsx_to_bytes_round_trip_preserves_cell_values():
    loaded = _load(xlsx_to_bytes(_make_workbook()))
    assert _rows(loaded) == [
        ("name", "count", "ratio", "flag"),
        ("alpha", 3, 0.5, True),
        ("beta", 7, 1.25, False),
    ]


def test_xlsx_to_bytes_default_workbook_has_single_sheet():
    loaded = _load(xlsx_to_bytes(Workbook()))
    assert loaded.sheetnames == ["Sheet"]


def test_xlsx_to_bytes_preserves_sheet_order_for_multiple_sheets():
    wb = Workbook()
    wb.active.title = "First"
    wb.create_sheet("Second")
    wb.create_sheet("Third")
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.sheetnames == ["First", "Second", "Third"]


def test_xlsx_to_bytes_preserves_values_on_each_sheet():
    wb = Workbook()
    wb.active.title = "A"
    wb["A"]["A1"] = "one"
    other = wb.create_sheet("B")
    other["B2"] = 42
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded["A"]["A1"].value == "one"
    assert loaded["B"]["B2"].value == 42


def test_xlsx_to_bytes_preserves_unicode_text():
    wb = Workbook()
    wb.active["A1"] = "héllo 世界 ✓"
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active["A1"].value == "héllo 世界 ✓"


def test_xlsx_to_bytes_keeps_formula_text():
    wb = Workbook()
    wb.active["A1"] = 2
    wb.active["A2"] = 3
    wb.active["A3"] = "=A1+A2"
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active["A3"].value == "=A1+A2"


def test_xlsx_to_bytes_leaves_empty_cells_as_none():
    wb = Workbook()
    wb.active["A1"] = "x"
    wb.active["C1"] = "z"
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active["B1"].value is None


def test_xlsx_to_bytes_does_not_modify_workbook():
    wb = _make_workbook()
    rows_before = _rows(wb)
    names_before = list(wb.sheetnames)
    xlsx_to_bytes(wb)
    assert _rows(wb) == rows_before
    assert wb.sheetnames == names_before


def test_xlsx_to_bytes_repeated_calls_give_same_values():
    wb = _make_workbook()
    first = _load(xlsx_to_bytes(wb))
    second = _load(xlsx_to_bytes(wb))
    assert _rows(first) == _rows(second)


def test_xlsx_to_bytes_snapshot_is_independent_of_later_edits():
    wb = _make_workbook()
    snapshot = xlsx_to_bytes(wb)
    wb.active["A2"] = "changed"
    loaded = _load(snapshot)
    assert loaded.active["A2"].value == "alpha"


def test_xlsx_to_bytes_round_trips_many_rows():
    wb = Workbook()
    ws = wb.active
    for i in range(1000):
        ws.append([i, str(i)])
    loaded = _load(xlsx_to_bytes(wb))
    rows = list(loaded.active.iter_rows(values_only=True))
    assert len(rows) == 1000
    assert rows[0] == (0, "0")
    assert rows[-1] == (999, "999")


def test_xlsx_to_file_returns_file_attachment():
    result = xlsx_to_file(_make_workbook(), "report.xlsx")
    assert isinstance(result, FileAttachment)


def test_xlsx_to_file_keeps_given_name():
    result = xlsx_to_file(_make_workbook(), "report.xlsx")
    assert result.name == "report.xlsx"


def test_xlsx_to_file_content_matches_xlsx_to_bytes_values():
    wb = _make_workbook()
    result = xlsx_to_file(wb, "report.xlsx")
    assert isinstance(result.content, bytes)
    assert _rows(_load(result.content)) == _rows(_load(xlsx_to_bytes(wb)))


def test_xlsx_to_file_content_is_zip_container():
    result = xlsx_to_file(Workbook(), "empty.xlsx")
    assert result.content.startswith(ZIP_MAGIC)


def test_xlsx_to_file_extension_is_xlsx():
    result = xlsx_to_file(Workbook(), "grades.xlsx")
    assert result.extension == "xlsx"
    assert result.name_without_extension == "grades"


def test_xlsx_to_file_strips_directory_from_name():
    result = xlsx_to_file(Workbook(), "exports/2026/summary.xlsx")
    assert result.name == "summary.xlsx"


def test_xlsx_to_file_removes_restricted_characters_from_name():
    result = xlsx_to_file(Workbook(), "bad:name*?.xlsx")
    assert result.name == "badname.xlsx"


def test_xlsx_to_file_save_then_load_workbook_round_trip(tmp_path):
    path = tmp_path / "out.xlsx"
    result = xlsx_to_file(_make_workbook(), "out.xlsx")
    result.save(str(path))
    loaded = load_workbook(str(path))
    assert _rows(loaded) == _rows(_make_workbook())
