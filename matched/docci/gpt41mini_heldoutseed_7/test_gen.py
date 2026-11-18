import io
import pytest
from openpyxl import Workbook, load_workbook

from docci.xlsx import xlsx_to_bytes, xlsx_to_file, dicts_to_xlsx, xlsx_from_bytes, xlsx_from_file
from docci.file import FileAttachment


def test_xlsx_to_bytes_returns_bytes():
    wb = Workbook()
    wb.active.append(["Header1", "Header2"])
    wb.active.append(["Value1", "Value2"])

    result = xlsx_to_bytes(wb)
    assert isinstance(result, bytes)
    assert len(result) > 0

    # The bytes should be a valid xlsx file - try to load it back
    wb_loaded = load_workbook(io.BytesIO(result))
    assert wb_loaded.active["A1"].value == "Header1"
    assert wb_loaded.active["B2"].value == "Value2"


def test_xlsx_to_file_returns_fileattachment_with_correct_name_and_content():
    wb = Workbook()
    wb.active.append([1, 2, 3])
    file_name = "testfile.xlsx"

    file_attachment = xlsx_to_file(wb, file_name)
    assert isinstance(file_attachment, FileAttachment)
    assert file_attachment.name == file_name
    assert isinstance(file_attachment.content, bytes)
    assert len(file_attachment.content) > 0

    # Loaded workbook from content matches the original content
    wb_loaded = load_workbook(io.BytesIO(file_attachment.content))
    assert wb_loaded.active["A1"].value == 1


def test_xlsx_to_file_and_bytes_are_consistent():
    wb = Workbook()
    wb.active.append(["a", "b", "c"])
    file_attachment = xlsx_to_file(wb, "file.xlsx")
    bytes_data = xlsx_to_bytes(wb)

    # The content in file_attachment should equal bytes_data
    assert file_attachment.content == bytes_data


def test_dicts_to_xlsx_creates_workbook_with_headers_and_rows():
    data = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]
    headers = ["a", "b"]
    wb = dicts_to_xlsx(data, headers)

    ws = wb.active
    # Header row
    assert tuple(cell.value for cell in ws[1]) == tuple(headers)
    # Data rows
    assert tuple(cell.value for cell in ws[2]) == (1, 2)
    assert tuple(cell.value for cell in ws[3]) == (3, 4)


def test_dicts_to_xlsx_defaults_headers_to_first_dict_keys():
    data = [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
    ]
    wb = dicts_to_xlsx(data)

    ws = wb.active
    # Headers are keys of the first dict
    assert set(cell.value for cell in ws[1]) == {"name", "age"}
    # Data rows
    row2 = tuple(cell.value for cell in ws[2])
    row3 = tuple(cell.value for cell in ws[3])
    # Values should correspond to dicts values but order depends on dict keys order
    assert ("Alice" in row2 and 30 in row2)
    assert ("Bob" in row3 and 25 in row3)


def test_dicts_to_xlsx_with_empty_list_raises_indexerror():
    with pytest.raises(IndexError):
        dicts_to_xlsx([])


def test_xlsx_from_bytes_returns_workbook():
    wb_orig = Workbook()
    ws = wb_orig.active
    ws.append(["x", "y"])
    ws.append([10, 20])

    bytes_data = xlsx_to_bytes(wb_orig)
    wb_loaded = xlsx_from_bytes(bytes_data)

    assert isinstance(wb_loaded, Workbook)
    assert wb_loaded.active["A1"].value == "x"
    assert wb_loaded.active["B2"].value == 20


def test_xlsx_from_bytes_with_invalid_bytes_raises_exception():
    with pytest.raises(Exception):
        # invalid bytes not representing xlsx
        xlsx_from_bytes(b"not a valid xlsx")


def test_xlsx_from_file_returns_workbook():
    wb_orig = Workbook()
    wb_orig.active.append(["header"])
    bytes_data = xlsx_to_bytes(wb_orig)
    file = FileAttachment("file.xlsx", bytes_data)

    wb_loaded = xlsx_from_file(file)
    assert isinstance(wb_loaded, Workbook)
    assert wb_loaded.active["A1"].value == "header"


def test_xlsx_from_file_with_corrupted_content_raises_exception():
    file = FileAttachment("badfile.xlsx", b"corrupted content")
    with pytest.raises(Exception):
        xlsx_from_file(file)


def test_xlsx_to_bytes_creates_nonempty_output_for_empty_workbook():
    wb = Workbook()
    bts = xlsx_to_bytes(wb)
    assert isinstance(bts, bytes)
    assert len(bts) > 0


def test_xlsx_to_file_with_empty_workbook_and_empty_name():
    wb = Workbook()
    file = xlsx_to_file(wb, "")
    assert isinstance(file, FileAttachment)
    assert file.name == ""
    assert isinstance(file.content, bytes)
    assert len(file.content) > 0


def test_dicts_to_xlsx_with_headers_different_order_than_dict_keys():
    data = [{"a": 1, "b": 2}]
    headers = ["b", "a"]
    wb = dicts_to_xlsx(data, headers)

    ws = wb.active
    assert tuple(cell.value for cell in ws[1]) == tuple(headers)
    # The appended values use dict_.values(), which are in dict key order ("a", then "b")
    # So they won't match headers order
    row2 = tuple(cell.value for cell in ws[2])
    assert row2 == (1, 2)  # corresponds to 'a':1, 'b':2


def test_dicts_to_xlsx_with_multiple_dicts_each_having_extra_keys():
    data = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4, "c": 5},
        {"a": 6}
    ]
    headers = ["a", "b", "c"]
    wb = dicts_to_xlsx(data, headers)

    ws = wb.active
    # Headers
    assert tuple(cell.value for cell in ws[1]) == tuple(headers)
    # Rows: values are dict_.values() - missing keys won't appear, so data columns may shift
    # This mismatch is a quirk of current dicts_to_xlsx implementation
    assert tuple(cell.value for cell in ws[2]) == (1, 2)
    assert tuple(cell.value for cell in ws[3]) == (3, 4, 5)
    assert tuple(cell.value for cell in ws[4]) == (6,)


def test_xlsx_to_bytes_can_handle_multiple_sheets():
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Sheet1"
    ws1.append(["A1"])
    ws2 = wb.create_sheet("Sheet2")
    ws2.append(["B1"])

    result = xlsx_to_bytes(wb)
    wb2 = load_workbook(io.BytesIO(result))
    assert "Sheet1" in wb2.sheetnames
    assert "Sheet2" in wb2.sheetnames
    assert wb2["Sheet1"]["A1"].value == "A1"
    assert wb2["Sheet2"]["A1"].value == "B1"


def test_xlsx_to_file_preserves_multiple_sheets():
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "First"
    ws1.append([1])
    ws2 = wb.create_sheet("Second")
    ws2.append([2])

    file = xlsx_to_file(wb, "multisheet.xlsx")
    wb_loaded = xlsx_from_bytes(file.content)

    assert wb_loaded.sheetnames == ["First", "Second"]
    assert wb_loaded["First"]["A1"].value == 1
    assert wb_loaded["Second"]["A1"].value == 2
