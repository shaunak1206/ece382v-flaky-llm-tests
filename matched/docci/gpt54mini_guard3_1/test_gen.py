import pytest
from openpyxl import Workbook
from openpyxl import load_workbook

from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


def _make_workbook(values=None):
    wb = Workbook()
    ws = wb.active
    values = values or [["A1", "B1"], ["A2", "B2"]]
    for row in values:
        ws.append(row)
    return wb


def _load_bytes(xlsx_bytes):
    return load_workbook(filename=pytest.importorskip("io").BytesIO(xlsx_bytes))


def test_xlsx_to_bytes_returns_nonempty_bytes():
    wb = _make_workbook()
    data = xlsx_to_bytes(wb)

    assert isinstance(data, bytes)
    assert len(data) > 0


def test_xlsx_to_bytes_produces_loadable_workbook():
    wb = _make_workbook([["x", 1], ["y", 2]])
    data = xlsx_to_bytes(wb)

    loaded = load_workbook(filename=pytest.importorskip("io").BytesIO(data))
    ws = loaded.active

    assert ws["A1"].value == "x"
    assert ws["B1"].value == 1
    assert ws["A2"].value == "y"
    assert ws["B2"].value == 2


def test_xlsx_to_bytes_round_trip_preserves_dimensions():
    wb = _make_workbook([["a", "b", "c"], [1, 2, 3], [4, 5, 6]])
    data = xlsx_to_bytes(wb)
    loaded = load_workbook(filename=pytest.importorskip("io").BytesIO(data))
    ws = loaded.active

    assert ws.max_row == 3
    assert ws.max_column == 3


def test_xlsx_to_bytes_round_trip_preserves_cell_types():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = 123
    ws["B1"] = 4.5
    ws["C1"] = True
    ws["D1"] = None
    ws["E1"] = "text"

    data = xlsx_to_bytes(wb)
    loaded = load_workbook(filename=pytest.importorskip("io").BytesIO(data))
    ws2 = loaded.active

    assert ws2["A1"].value == 123
    assert ws2["B1"].value == 4.5
    assert ws2["C1"].value is True
    assert ws2["D1"].value is None
    assert ws2["E1"].value == "text"


def test_xlsx_to_bytes_can_be_called_multiple_times_on_same_workbook():
    wb = _make_workbook([["hello"]])

    data1 = xlsx_to_bytes(wb)
    data2 = xlsx_to_bytes(wb)

    assert data1 == data2


def test_xlsx_to_bytes_does_not_mutate_cell_values():
    wb = _make_workbook([["foo", "bar"]])
    before = wb.active["A1"].value, wb.active["B1"].value

    _ = xlsx_to_bytes(wb)

    after = wb.active["A1"].value, wb.active["B1"].value
    assert after == before


def test_xlsx_to_file_returns_fileattachment():
    wb = _make_workbook([["name"]])

    file_attachment = xlsx_to_file(wb, "report.xlsx")

    assert isinstance(file_attachment, FileAttachment)


def test_xlsx_to_file_uses_provided_name():
    wb = _make_workbook([["name"]])

    file_attachment = xlsx_to_file(wb, "custom_name.xlsx")

    assert file_attachment.name == "custom_name.xlsx"


def test_xlsx_to_file_content_matches_xlsx_to_bytes():
    wb = _make_workbook([["a", "b"]])

    file_attachment = xlsx_to_file(wb, "test.xlsx")
    data = xlsx_to_bytes(wb)

    assert file_attachment.content == data


def test_xlsx_to_file_content_is_loadable_workbook():
    wb = _make_workbook([["alpha", "beta"]])

    file_attachment = xlsx_to_file(wb, "test.xlsx")
    loaded = load_workbook(filename=pytest.importorskip("io").BytesIO(file_attachment.content))
    ws = loaded.active

    assert ws["A1"].value == "alpha"
    assert ws["B1"].value == "beta"


def test_xlsx_to_file_does_not_modify_workbook():
    wb = _make_workbook([["x", "y"]])
    original = wb.active["A1"].value, wb.active["B1"].value

    _ = xlsx_to_file(wb, "out.xlsx")

    assert (wb.active["A1"].value, wb.active["B1"].value) == original


def test_xlsx_to_file_and_bytes_are_independent_of_name():
    wb = _make_workbook([["same"]])

    file1 = xlsx_to_file(wb, "a.xlsx")
    file2 = xlsx_to_file(wb, "b.xlsx")

    assert file1.content == file2.content
    assert file1.name == "a.xlsx"
    assert file2.name == "b.xlsx"


def test_xlsx_to_bytes_empty_workbook_has_default_sheet():
    wb = Workbook()

    data = xlsx_to_bytes(wb)
    loaded = load_workbook(filename=pytest.importorskip("io").BytesIO(data))

    assert loaded.active.title == "Sheet"


def test_xlsx_to_file_empty_workbook_round_trip():
    wb = Workbook()

    file_attachment = xlsx_to_file(wb, "empty.xlsx")
    loaded = load_workbook(filename=pytest.importorskip("io").BytesIO(file_attachment.content))

    assert loaded.active.title == "Sheet"
    assert loaded.active.max_row == 1
    assert loaded.active.max_column == 1
