import io

from openpyxl import Workbook, load_workbook

from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


def _make_workbook(values):
    wb = Workbook()
    ws = wb.active
    for row in values:
        ws.append(row)
    return wb


def _load_values_from_bytes(data):
    wb = load_workbook(io.BytesIO(data))
    ws = wb.active
    return [tuple(row) for row in ws.iter_rows(values_only=True)]


def test_xlsx_to_bytes_returns_bytes():
    wb = _make_workbook([["a", "b"], [1, 2]])
    result = xlsx_to_bytes(wb)

    assert isinstance(result, bytes)
    assert len(result) > 0


def test_xlsx_to_bytes_round_trip_preserves_cell_values():
    wb = _make_workbook([["name", "age"], ["alice", 30], ["bob", 25]])
    data = xlsx_to_bytes(wb)

    assert _load_values_from_bytes(data) == [
        ("name", "age"),
        ("alice", 30),
        ("bob", 25),
    ]


def test_xlsx_to_bytes_does_not_mutate_original_workbook():
    wb = _make_workbook([["x", "y"], [1, 2]])
    before = list(wb.active.iter_rows(values_only=True))

    _ = xlsx_to_bytes(wb)

    after = list(wb.active.iter_rows(values_only=True))
    assert after == before


def test_xlsx_to_bytes_is_deterministic_for_same_workbook_state():
    wb = _make_workbook([["h1", "h2"], ["v1", "v2"]])

    first = xlsx_to_bytes(wb)
    second = xlsx_to_bytes(wb)

    assert first == second


def test_xlsx_to_bytes_changes_when_workbook_changes():
    wb = _make_workbook([["h1", "h2"], ["v1", "v2"]])
    first = xlsx_to_bytes(wb)

    wb.active.append(["v3", "v4"])
    second = xlsx_to_bytes(wb)

    assert first != second
    assert _load_values_from_bytes(second) == [
        ("h1", "h2"),
        ("v1", "v2"),
        ("v3", "v4"),
    ]


def test_xlsx_to_bytes_accepts_empty_workbook():
    wb = Workbook()

    data = xlsx_to_bytes(wb)

    assert isinstance(data, bytes)
    assert _load_values_from_bytes(data) == []


def test_xlsx_to_bytes_handles_unicode_content():
    wb = _make_workbook([["café", "Δ"], ["naïve", "東京"]])

    data = xlsx_to_bytes(wb)

    assert _load_values_from_bytes(data) == [
        ("café", "Δ"),
        ("naïve", "東京"),
    ]


def test_xlsx_to_file_returns_fileattachment_with_given_name():
    wb = _make_workbook([["a"], [1]])

    attachment = xlsx_to_file(wb, "report.xlsx")

    assert isinstance(attachment, FileAttachment)
    assert attachment.name == "report.xlsx"


def test_xlsx_to_file_uses_same_bytes_as_xlsx_to_bytes():
    wb = _make_workbook([["a", "b"], [10, 20]])

    attachment = xlsx_to_file(wb, "data.xlsx")
    expected = xlsx_to_bytes(wb)

    assert attachment.content == expected


def test_xlsx_to_file_round_trip_preserves_content():
    wb = _make_workbook([["x", "y"], [3, 4]])

    attachment = xlsx_to_file(wb, "roundtrip.xlsx")
    loaded = load_workbook(io.BytesIO(attachment.content))

    assert [tuple(row) for row in loaded.active.iter_rows(values_only=True)] == [
        ("x", "y"),
        ("3", "4"),
    ]


def test_xlsx_to_file_does_not_mutate_original_workbook():
    wb = _make_workbook([["col"], ["value"]])
    before = list(wb.active.iter_rows(values_only=True))

    _ = xlsx_to_file(wb, "name.xlsx")

    after = list(wb.active.iter_rows(values_only=True))
    assert after == before


def test_xlsx_to_file_returns_distinct_attachments_for_distinct_calls():
    wb = _make_workbook([["a"], [1]])

    first = xlsx_to_file(wb, "same.xlsx")
    second = xlsx_to_file(wb, "same.xlsx")

    assert first is not second
    assert first.name == second.name == "same.xlsx"
    assert first.content == second.content


def test_xlsx_to_file_content_updates_after_workbook_change():
    wb = _make_workbook([["a"], [1]])
    first = xlsx_to_file(wb, "file.xlsx").content

    wb.active.append([2])
    second = xlsx_to_file(wb, "file.xlsx").content

    assert first != second
    assert [tuple(row) for row in load_workbook(io.BytesIO(second)).active.iter_rows(values_only=True)] == [
        ("a",),
        (1,),
        (2,),
    ]


def test_xlsx_to_bytes_with_multiple_sheets_preserves_active_sheet_only_in_round_trip():
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Sheet1"
    ws1.append(["active"])
    ws2 = wb.create_sheet("Sheet2")
    ws2.append(["inactive"])

    data = xlsx_to_bytes(wb)
    loaded = load_workbook(io.BytesIO(data))

    assert loaded.sheetnames == ["Sheet1", "Sheet2"]
    assert loaded["Sheet1"]["A1"].value == "active"
    assert loaded["Sheet2"]["A1"].value == "inactive"
