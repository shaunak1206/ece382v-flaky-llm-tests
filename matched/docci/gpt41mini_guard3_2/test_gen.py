import random
import io
import pytest
from openpyxl import Workbook, load_workbook

from docci.xlsx import xlsx_to_bytes, xlsx_to_file, dicts_to_xlsx, xlsx_from_bytes, xlsx_from_file
from docci.file import FileAttachment


def test_xlsx_to_bytes_empty_workbook():
    random.seed(42)
    wb = Workbook()
    result = xlsx_to_bytes(wb)
    assert isinstance(result, bytes)
    # The bytes must correspond to a valid xlsx file
    wb2 = load_workbook(filename=io.BytesIO(result))
    assert wb2.active.title == wb.active.title
    # Default sheet is named "Sheet"
    assert wb2.active.title == "Sheet"


def test_xlsx_to_file_returns_fileattachment():
    random.seed(42)
    wb = Workbook()
    filename = "test.xlsx"
    file_attachment = xlsx_to_file(wb, filename)
    assert isinstance(file_attachment, FileAttachment)
    assert file_attachment.name == filename
    # content is bytes and can be loaded via load_workbook
    assert isinstance(file_attachment.content, bytes)
    wb_loaded = load_workbook(io.BytesIO(file_attachment.content))
    assert wb_loaded.active.title == wb.active.title


def test_xlsx_to_bytes_and_back_roundtrip():
    random.seed(42)
    wb = Workbook()
    sheet = wb.active
    sheet.title = "Data"
    sheet.append(["A", "B", "C"])
    sheet.append([1, 2, 3])
    sheet.append([4, 5, 6])
    b = xlsx_to_bytes(wb)
    wb2 = xlsx_from_bytes(b)
    sheet2 = wb2.active
    assert sheet2.title == "Data"
    expected_rows = [
        ("A", "B", "C"),
        (1, 2, 3),
        (4, 5, 6),
    ]
    actual_rows = [tuple(cell.value for cell in row) for row in sheet2.iter_rows()]
    assert actual_rows == expected_rows


def test_xlsx_to_file_and_back_roundtrip():
    random.seed(42)
    wb = Workbook()
    sheet = wb.active
    sheet.title = "Roundtrip"
    sheet.append(["X", "Y"])
    sheet.append([7, 8])
    fa = xlsx_to_file(wb, "roundtrip.xlsx")
    wb2 = xlsx_from_file(fa)
    sheet2 = wb2.active
    assert sheet2.title == "Roundtrip"
    actual_rows = [tuple(cell.value for cell in row) for row in sheet2.iter_rows()]
    expected_rows = [
        ("X", "Y"),
        (7, 8),
    ]
    assert actual_rows == expected_rows


def test_dicts_to_xlsx_with_default_headers():
    random.seed(42)
    dicts = [
        {"one": 1, "two": 2},
        {"one": 10, "two": 20},
    ]
    wb = dicts_to_xlsx(dicts)
    sheet = wb.active
    # The headers are keys of the first dict, order may vary since dict keys in Python >=3.7 keep insertion order reliably.
    # But to be safe, check set equality of headers
    headers = tuple(cell.value for cell in sheet[1])
    assert set(headers) == set(dicts[0].keys())
    # Rows should have values from dicts in the order of headers
    rows = []
    for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row):
        rows.append(tuple(cell.value for cell in row))
    for row_vals, d in zip(rows, dicts):
        # Values correspond to those in the dict under the header keys order
        expected = tuple(d[h] for h in headers)
        assert row_vals == expected


def test_dicts_to_xlsx_with_given_headers():
    random.seed(42)
    dicts = [
        {"a": 1, "b": 2, "c": 3},
        {"a": 4, "b": 5, "c": 6},
    ]
    headers = ("c", "b")
    wb = dicts_to_xlsx(dicts, headers)
    sheet = wb.active
    actual_headers = tuple(cell.value for cell in sheet[1])
    assert actual_headers == headers
    rows = []
    for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row):
        rows.append(tuple(cell.value for cell in row))
    for row_vals, d in zip(rows, dicts):
        expected = tuple(d[h] for h in headers)
        assert row_vals == expected


def test_dicts_to_xlsx_empty_headers_raises_index_error():
    # An empty dicts list should raise IndexError when headers=None because dicts[0] access fails
    random.seed(42)
    with pytest.raises(IndexError):
        dicts_to_xlsx([])


def test_dicts_to_xlsx_empty_with_headers_produces_only_header_row():
    random.seed(42)
    headers = ("x", "y")
    wb = dicts_to_xlsx([], headers)
    sheet = wb.active
    # Sheet should have exactly one row with headers
    rows = list(sheet.iter_rows(values_only=True))
    assert rows == [headers]


def test_xlsx_from_bytes_invalid_data_raises_exception():
    random.seed(42)
    with pytest.raises(Exception):
        # random bytes that are not a valid XLSX file
        xlsx_from_bytes(b"notanxlsx")


def test_xlsx_from_file_loads_fileattachment_content(monkeypatch):
    random.seed(42)
    wb = Workbook()
    wb.active.append(["abc", "def"])
    content_bytes = xlsx_to_bytes(wb)
    fa = FileAttachment("file.xlsx", content_bytes)

    # Monkeypatch xlsx_from_bytes to verify it receives fa.content
    called = {}
    def fake_xlsx_from_bytes(b):
        called["content"] = b
        return wb
    monkeypatch.setattr("docci.xlsx.xlsx_from_bytes", fake_xlsx_from_bytes)

    result = xlsx_from_file(fa)
    assert called["content"] is fa.content
    assert result is wb


def test_xlsx_to_bytes_content_is_consistent_across_invocations():
    random.seed(42)
    wb = Workbook()
    sheet = wb.active
    sheet.append(["header"])
    sheet.append([123])
    b1 = xlsx_to_bytes(wb)
    b2 = xlsx_to_bytes(wb)
    assert b1 == b2


def test_xlsx_to_file_different_names_have_different_name_attribute():
    random.seed(42)
    wb = Workbook()
    f1 = xlsx_to_file(wb, "a.xlsx")
    f2 = xlsx_to_file(wb, "b.xlsx")
    assert f1.name == "a.xlsx"
    assert f2.name == "b.xlsx"
    assert f1.name != f2.name


def test_xlsx_to_bytes_large_data():
    random.seed(42)
    wb = Workbook()
    sheet = wb.active
    # Append 100 rows with 10 columns each
    for i in range(100):
        row = list(range(i, i+10))
        sheet.append(row)
    b = xlsx_to_bytes(wb)
    assert isinstance(b, bytes)
    # The output is not empty (arbitrary but safe)
    assert len(b) > 1000
    # Loading back yields the same number of rows and values
    wb2 = load_workbook(io.BytesIO(b))
    sheet2 = wb2.active
    assert sheet2.max_row == 101  # 1 header row + 100 rows
    # Check a few cells values (first data row values)
    first_data_row = [cell.value for cell in sheet2[2]]
    assert first_data_row == list(range(0, 10))


def test_dicts_to_xlsx_values_with_none(monkeypatch):
    random.seed(42)
    dicts = [
        {"a": None, "b": 2},
        {"a": 3, "b": None},
    ]
    # Confirm None values are handled and stored as None, not string "None"
    wb = dicts_to_xlsx(dicts)
    sheet = wb.active
    headers = tuple(cell.value for cell in sheet[1])
    rows = []
    for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row):
        rows.append(tuple(cell.value for cell in row))
    for row_vals, d in zip(rows, dicts):
        expected = tuple(d[h] for h in headers)
        assert row_vals == expected


def test_xlsx_to_bytes_is_readable_by_load_workbook(monkeypatch):
    random.seed(42)
    wb = Workbook()
    wb.active.append(["A1"])
    b = xlsx_to_bytes(wb)

    # Monkeypatch openpyxl.load_workbook to a version that checks input is BytesIO or raises
    original_load_workbook = load_workbook
    def fake_load_workbook(fp, *args, **kwargs):
        assert isinstance(fp, (io.BytesIO, io.BufferedReader))
        return original_load_workbook(fp, *args, **kwargs)
    monkeypatch.setattr("docci.xlsx.load_workbook", fake_load_workbook)

    wb2 = xlsx_from_bytes(b)
    assert wb2.active["A1"].value == "A1"


def test_xlsx_from_file_uses_xlsx_from_bytes_indirectly(monkeypatch):
    random.seed(42)
    wb = Workbook()
    content = xlsx_to_bytes(wb)
    fa = FileAttachment("file.xlsx", content)
    called = {}
    def fake_xlsx_from_bytes(b):
        called["called"] = True
        assert b == content
        return wb
    monkeypatch.setattr("docci.xlsx.xlsx_from_bytes", fake_xlsx_from_bytes)
    result = xlsx_from_file(fa)
    assert called["called"]
    assert result is wb


def test_xlsx_to_file_content_matches_xlsx_to_bytes(monkeypatch):
    random.seed(42)
    wb = Workbook()
    b1 = xlsx_to_bytes(wb)
    fa = xlsx_to_file(wb, "file.xlsx")
    # FileAttachment content matches the bytes output of xlsx_to_bytes exactly
    assert fa.content == b1


def test_dicts_to_xlsx_headers_must_be_sequence():
    random.seed(42)
    dicts = [{"k":1}]
    headers = ["k"]
    wb1 = dicts_to_xlsx(dicts, headers)
    headers_tuple = ("k",)
    wb2 = dicts_to_xlsx(dicts, headers_tuple)
    # Both should produce same sheet values
    data1 = [[cell.value for cell in row] for row in wb1.active.iter_rows()]
    data2 = [[cell.value for cell in row] for row in wb2.active.iter_rows()]
    assert data1 == data2
