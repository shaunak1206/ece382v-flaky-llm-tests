import pytest
from openpyxl import Workbook, load_workbook
from io import BytesIO
from docci.xlsx import xlsx_to_bytes, xlsx_to_file, dicts_to_xlsx, xlsx_from_bytes, xlsx_from_file
from docci.file import FileAttachment


def create_sample_workbook():
    wb = Workbook()
    ws = wb.active
    ws.append(["A", "B", "C"])
    ws.append([1, 2, 3])
    ws.append([4, 5, 6])
    return wb


def test_xlsx_to_bytes_returns_bytes():
    wb = create_sample_workbook()
    result = xlsx_to_bytes(wb)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_xlsx_to_bytes_produces_valid_excel():
    wb = create_sample_workbook()
    bytes_ = xlsx_to_bytes(wb)
    wb2 = load_workbook(filename=BytesIO(bytes_))
    ws2 = wb2.active
    rows = list(ws2.values)
    assert rows[0] == ("A", "B", "C")
    assert rows[1] == (1, 2, 3)
    assert rows[2] == (4, 5, 6)


def test_xlsx_to_file_returns_fileattachment():
    wb = create_sample_workbook()
    filename = "testfile.xlsx"
    file_attachment = xlsx_to_file(wb, filename)
    assert isinstance(file_attachment, FileAttachment)
    assert file_attachment.name == filename
    assert isinstance(file_attachment.content, bytes)
    assert len(file_attachment.content) > 0


def test_xlsx_to_file_content_matches_xlsx_to_bytes():
    wb = create_sample_workbook()
    file_attachment = xlsx_to_file(wb, "test.xlsx")
    bytes_data = xlsx_to_bytes(wb)
    assert file_attachment.content == bytes_data


def test_dicts_to_xlsx_creates_workbook_with_headers_and_values():
    data = [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
    ]
    wb = dicts_to_xlsx(data)
    ws = wb.active
    rows = list(ws.values)
    # Check headers are keys of first dict by default (order not guaranteed, but openpyxl preserves dict.keys() order in later pythons)
    assert set(rows[0]) == {"name", "age"}
    # Check rows contain the data values
    assert tuple(rows[1]) == ("Alice", 30)
    assert tuple(rows[2]) == ("Bob", 25)


def test_dicts_to_xlsx_with_custom_headers_overrides_default():
    data = [
        {"name": "Charlie", "score": 90},
        {"name": "Dana", "score": 85},
    ]
    headers = ["name", "score"]
    wb = dicts_to_xlsx(data, headers)
    ws = wb.active
    rows = list(ws.values)
    assert rows[0] == tuple(headers)
    assert tuple(rows[1]) == ("Charlie", 90)
    assert tuple(rows[2]) == ("Dana", 85)


def test_dicts_to_xlsx_with_empty_list_raises_index_error():
    with pytest.raises(IndexError):
        dicts_to_xlsx([])


def test_dicts_to_xlsx_with_headers_and_dicts_length_mismatch():
    # headers longer than the dict keys, the sheet will have header columns without corresponding data
    data = [{"a": 1}]
    headers = ["a", "b", "c"]
    wb = dicts_to_xlsx(data, headers)
    ws = wb.active
    rows = list(ws.values)
    # Headers are preserved
    assert rows[0] == tuple(headers)
    # The data row fills only first cell, others are None because dict_.values() tuple length is shorter
    # Because dict values only 1 element, tuple(dict_.values()) is length 1, appending to row leaves other columns empty
    assert rows[1] == (1,)


def test_xlsx_from_bytes_returns_workbook():
    wb = create_sample_workbook()
    bytes_ = xlsx_to_bytes(wb)
    wb2 = xlsx_from_bytes(bytes_)
    assert isinstance(wb2, Workbook)
    ws2 = wb2.active
    # Check content matches original
    assert list(ws2.values) == list(wb.active.values)


def test_xlsx_from_bytes_with_invalid_bytes_raises_exception():
    with pytest.raises(Exception):
        xlsx_from_bytes(b"not an excel file")


def test_xlsx_from_file_returns_workbook_and_content_matches():
    wb = create_sample_workbook()
    file_attachment = xlsx_to_file(wb, "sample.xlsx")
    wb2 = xlsx_from_file(file_attachment)
    assert isinstance(wb2, Workbook)
    ws_orig = wb.active
    ws_loaded = wb2.active
    assert list(ws_orig.values) == list(ws_loaded.values)


def test_xlsx_to_bytes_different_workbooks_produce_different_bytes():
    wb1 = Workbook()
    ws1 = wb1.active
    ws1.append(["val1"])
    wb2 = Workbook()
    ws2 = wb2.active
    ws2.append(["val2"])
    b1 = xlsx_to_bytes(wb1)
    b2 = xlsx_to_bytes(wb2)
    assert b1 != b2


def test_xlsx_to_file_name_is_preserved():
    wb = create_sample_workbook()
    filename = "myfile.xlsx"
    file_attachment = xlsx_to_file(wb, filename)
    assert file_attachment.name == filename


def test_roundtrip_bytes_to_workbook_and_back():
    wb = create_sample_workbook()
    bytes1 = xlsx_to_bytes(wb)
    wb2 = xlsx_from_bytes(bytes1)
    bytes2 = xlsx_to_bytes(wb2)
    assert bytes1 == bytes2


def test_dicts_to_xlsx_with_non_string_headers():
    data = [{"one": 1, "two": 2}]
    headers = [1, 2, 3]  # headers can be non-strings (openpyxl supports this)
    wb = dicts_to_xlsx(data, headers)
    rows = list(wb.active.values)
    assert rows[0] == tuple(headers)
    # The data row is shorter, only one value, so the rest columns should be missing (not padded)
    assert rows[1] == (1,)


def test_dicts_to_xlsx_preserves_order_of_headers():
    data = [{"first": 1, "second": 2, "third": 3}]
    headers = ["third", "first", "second"]
    wb = dicts_to_xlsx(data, headers)
    rows = list(wb.active.values)
    assert rows[0] == tuple(headers)


def test_xlsx_to_bytes_blank_workbook_not_empty_bytes():
    wb = Workbook()  # Blank workbook with nothing added
    bytes_ = xlsx_to_bytes(wb)
    assert isinstance(bytes_, bytes)
    assert len(bytes_) > 0


def test_xlsx_to_file_blank_workbook():
    wb = Workbook()
    file_attachment = xlsx_to_file(wb, "blank.xlsx")
    assert isinstance(file_attachment.content, bytes)
    assert len(file_attachment.content) > 0


def test_xlsx_from_file_empty_file_content_raises():
    # Create a FileAttachment with empty content
    empty_file = FileAttachment("empty.xlsx", b"")
    with pytest.raises(Exception):
        xlsx_from_file(empty_file)
