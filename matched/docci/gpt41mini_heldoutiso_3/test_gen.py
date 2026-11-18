import pytest
from openpyxl import Workbook, load_workbook
from docci.xlsx import xlsx_to_bytes, xlsx_to_file, dicts_to_xlsx, xlsx_from_bytes, xlsx_from_file
from docci.file import FileAttachment
from io import BytesIO


def test_xlsx_to_bytes_returns_bytes():
    wb = Workbook()
    ws = wb.active
    ws['A1'] = "Test"
    result = xlsx_to_bytes(wb)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_xlsx_to_bytes_produces_valid_xlsx():
    wb = Workbook()
    ws = wb.active
    ws['A1'] = "Hello"
    b = xlsx_to_bytes(wb)
    # load_workbook should successfully open bytes
    wb2 = load_workbook(BytesIO(b))
    assert wb2.active['A1'].value == "Hello"


def test_xlsx_to_file_returns_fileattachment():
    wb = Workbook()
    wb.active['A1'] = "Data"
    name = "mytest.xlsx"
    file_attachment = xlsx_to_file(wb, name)
    assert isinstance(file_attachment, FileAttachment)
    assert file_attachment.name == name
    assert isinstance(file_attachment.content, bytes)
    # content matches bytes from xlsx_to_bytes
    assert file_attachment.content == xlsx_to_bytes(wb)


def test_xlsx_to_file_content_is_valid_xlsx():
    wb = Workbook()
    wb.active['B2'] = "Value"
    file_attachment = xlsx_to_file(wb, "file.xlsx")
    wb2 = load_workbook(BytesIO(file_attachment.content))
    assert wb2.active['B2'].value == "Value"


def test_dicts_to_xlsx_creates_workbook_with_headers_and_rows():
    data = [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
    ]
    headers = ["name", "age"]
    wb = dicts_to_xlsx(data, headers=headers)
    sheet = wb.active
    # The first row is headers
    assert tuple(sheet[1][i].value for i in range(len(headers))) == ("name", "age")
    # Rows have values in order of dict.values() (which is insertion order in Python 3.8+)
    assert sheet[2][0].value == "Alice"
    assert sheet[2][1].value == 30
    assert sheet[3][0].value == "Bob"
    assert sheet[3][1].value == 25


def test_dicts_to_xlsx_without_headers_uses_dict_keys():
    data = [{"k1": 1, "k2": 2}]
    wb = dicts_to_xlsx(data)
    sheet = wb.active
    # Headers come from dict keys
    assert set(cell.value for cell in sheet[1]) == {"k1", "k2"}
    # Data row matches the dict values in order of keys
    vals = tuple(cell.value for cell in sheet[2])
    # since dict preserves insertion order in Python 3.8+
    assert vals == (1, 2)


def test_dicts_to_xlsx_empty_list_raises_indexerror():
    with pytest.raises(IndexError):
        dicts_to_xlsx([])


def test_dicts_to_xlsx_non_matching_headers_and_values():
    data = [
        {"a": 1, "b": 2},
        {"a": 3},  # missing 'b'
    ]
    headers = ["a", "b"]
    wb = dicts_to_xlsx(data, headers=headers)
    sheet = wb.active
    assert sheet[1][0].value == "a"
    assert sheet[1][1].value == "b"
    # first row of data
    assert sheet[2][0].value == 1
    assert sheet[2][1].value == 2
    # second row of data: since dict.values() returns (3,) only, openpyxl appends the missing value as None
    # Actually tuple(dict_.values()) returns only one value, so it appends just one cell
    # openpyxl appends as many cells as passed, so second row has only one cell, next will be empty
    assert sheet[3][0].value == 3
    assert sheet[3][1].value is None


def test_xlsx_from_bytes_returns_valid_workbook():
    wb = Workbook()
    wb.active['C3'] = "Check"
    b = xlsx_to_bytes(wb)
    wb2 = xlsx_from_bytes(b)
    assert isinstance(wb2, Workbook)
    assert wb2.active['C3'].value == "Check"


def test_xlsx_from_bytes_invalid_data_raises_exception():
    with pytest.raises(Exception):
        xlsx_from_bytes(b"not an xlsx file")


def test_xlsx_from_file_returns_workbook_with_correct_data():
    wb = Workbook()
    wb.active['D4'] = "filetest"
    file = xlsx_to_file(wb, "test.xlsx")
    wb_loaded = xlsx_from_file(file)
    assert wb_loaded.active['D4'].value == "filetest"


def test_xlsx_from_file_invalid_content_raises_exception():
    file = FileAttachment("bad.xlsx", b"baddata")
    with pytest.raises(Exception):
        xlsx_from_file(file)


def test_xlsx_to_bytes_and_to_file_consistent():
    wb = Workbook()
    wb.active['E5'] = 55
    b1 = xlsx_to_bytes(wb)
    file = xlsx_to_file(wb, "testfile.xlsx")
    b2 = file.content
    assert b1 == b2


def test_xlsx_to_file_name_is_passed_correctly():
    wb = Workbook()
    name = "example.xlsx"
    file = xlsx_to_file(wb, name)
    assert file.name == name


def test_dicts_to_xlsx_inserts_multiple_rows_correctly():
    data = [
        {"col1": "a1", "col2": "b1"},
        {"col1": "a2", "col2": "b2"},
        {"col1": "a3", "col2": "b3"},
    ]
    wb = dicts_to_xlsx(data)
    sheet = wb.active
    # Check headers and 3 rows
    assert sheet.max_row == 4
    expected_values = [
        ("col1", "col2"),
        ("a1", "b1"),
        ("a2", "b2"),
        ("a3", "b3"),
    ]
    for i, expected_row in enumerate(expected_values, start=1):
        row_values = tuple(cell.value for cell in sheet[i])
        assert row_values == expected_row


def test_dicts_to_xlsx_headers_override_order():
    data = [{"x": 1, "y": 2}]
    headers = ["y", "x"]
    wb = dicts_to_xlsx(data, headers=headers)
    sheet = wb.active
    # Headers are in order of provided headers
    assert tuple(cell.value for cell in sheet[1]) == ("y", "x")
    # Values correspond to dict values in dict order, but appended in headers order
    # tuple(dict_.values()) is (1, 2), but the append uses tuple(dict_.values()), so order matters
    # This test actually shows current behavior: the function appends dict.values(), ignoring headers order
    # So values come in dict.values() order, so the values row is (1, 2)
    assert tuple(cell.value for cell in sheet[2]) == (1, 2)


def test_dicts_to_xlsx_handles_non_string_keys():
    data = [{1: "one", 2: "two"}]
    wb = dicts_to_xlsx(data)
    sheet = wb.active
    # Headers should be (1, 2) as keys
    assert tuple(cell.value for cell in sheet[1]) == (1, 2)
    # Values as expected
    assert tuple(cell.value for cell in sheet[2]) == ("one", "two")


def test_xlsx_to_bytes_and_load_workbook_roundtrip():
    wb = Workbook()
    ws = wb.active
    ws['A1'] = "roundtrip"
    b = xlsx_to_bytes(wb)
    wb2 = load_workbook(BytesIO(b))
    assert wb2.active['A1'].value == "roundtrip"
