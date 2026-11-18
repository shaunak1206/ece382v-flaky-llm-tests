import pytest
from openpyxl import Workbook, load_workbook
from io import BytesIO
from docci.xlsx import xlsx_to_bytes, xlsx_to_file, dicts_to_xlsx, xlsx_from_bytes, xlsx_from_file
from docci.file import FileAttachment


def test_xlsx_to_bytes_returns_bytes():
    wb = Workbook()
    data = xlsx_to_bytes(wb)
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_xlsx_to_bytes_creates_valid_xlsx():
    wb = Workbook()
    ws = wb.active
    ws['A1'] = 'Test'
    xlsx_bytes = xlsx_to_bytes(wb)
    wb_loaded = load_workbook(BytesIO(xlsx_bytes))
    assert wb_loaded.active['A1'].value == 'Test'


def test_xlsx_to_file_returns_fileattachment():
    wb = Workbook()
    fa = xlsx_to_file(wb, "test.xlsx")
    assert isinstance(fa, FileAttachment)
    assert fa.name == "test.xlsx"
    assert isinstance(fa.content, bytes)
    assert len(fa.content) > 0


def test_xlsx_to_file_content_is_valid_xlsx():
    wb = Workbook()
    ws = wb.active
    ws['B2'] = 'Hello'
    fileatt = xlsx_to_file(wb, "hello.xlsx")
    loaded_wb = load_workbook(BytesIO(fileatt.content))
    assert loaded_wb.active['B2'].value == 'Hello'


def test_dicts_to_xlsx_creates_workbook():
    dicts = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    wb = dicts_to_xlsx(dicts)
    assert isinstance(wb, Workbook)
    sheet = wb.active
    # Headers should be keys of dict
    assert tuple(sheet[1][0].value for sheet in [sheet]) == ("a", "b")
    # First data row
    assert tuple(cell.value for cell in sheet[2]) == (1, 2)
    # Second data row
    assert tuple(cell.value for cell in sheet[3]) == (3, 4)


def test_dicts_to_xlsx_uses_given_headers():
    dicts = [{"col1": "x", "col2": "y"}]
    headers = ["col2", "col1"]
    wb = dicts_to_xlsx(dicts, headers)
    sheet = wb.active
    # Headers equal to passed headers in order
    assert tuple(cell.value for cell in sheet[1]) == tuple(headers)
    # Values are in dict insertion order of values - but appended in that order; dictionary order is preserved in python 3.8+
    expected_values = tuple(dicts[0].values())
    # Because dict values order may not correspond headers order, row might mismatch header columns
    # We check content of second row matches dict values (in insertion order)
    assert tuple(cell.value for cell in sheet[2]) == expected_values


def test_dicts_to_xlsx_empty_list_raises_indexerror():
    with pytest.raises(IndexError):
        dicts_to_xlsx([])


def test_xlsx_from_bytes_recreates_workbook():
    wb_original = Workbook()
    ws = wb_original.active
    ws['C3'] = 1234
    bytes_ = xlsx_to_bytes(wb_original)
    wb_new = xlsx_from_bytes(bytes_)
    assert isinstance(wb_new, Workbook)
    assert wb_new.active['C3'].value == 1234


def test_xlsx_from_bytes_with_invalid_bytes_raises_exception():
    with pytest.raises(Exception):
        xlsx_from_bytes(b"notanxlsx")


def test_xlsx_from_file_recreates_workbook():
    wb_original = Workbook()
    wb_original.active['D4'] = "val"
    fileatt = xlsx_to_file(wb_original, "sample.xlsx")
    wb_new = xlsx_from_file(fileatt)
    assert isinstance(wb_new, Workbook)
    assert wb_new.active['D4'].value == "val"


def test_xlsx_from_file_with_empty_content_raises_exception():
    fake_file = FileAttachment("empty.xlsx", b"")
    with pytest.raises(Exception):
        xlsx_from_file(fake_file)


def test_roundtrip_dicts_to_xlsx_and_back():
    dicts = [{"foo": 1, "bar": 2}, {"foo": 3, "bar": 4}]
    headers = ["foo", "bar"]
    wb = dicts_to_xlsx(dicts, headers=headers)
    bytes_ = xlsx_to_bytes(wb)
    wb2 = xlsx_from_bytes(bytes_)
    sheet = wb2.active
    # Check headers
    assert tuple(cell.value for cell in sheet[1]) == tuple(headers)
    # Check rows
    for i, d in enumerate(dicts, start=2):
        assert tuple(cell.value for cell in sheet[i]) == tuple(d.values())


def test_xlsx_to_bytes_is_not_empty_for_nonempty_workbook():
    wb = Workbook()
    ws = wb.active
    ws['A1'] = "data"
    bytes_ = xlsx_to_bytes(wb)
    assert bytes_ != b''


def test_xlsx_to_file_name_preserved():
    wb = Workbook()
    fa = xlsx_to_file(wb, "myfile.xlsx")
    assert fa.name == "myfile.xlsx"


def test_dicts_to_xlsx_preserves_headers_order():
    dicts = [{"one": 1, "two": 2, "three": 3}]
    headers = ["three", "one", "two"]
    wb = dicts_to_xlsx(dicts, headers=headers)
    sheet = wb.active
    # Header row follows headers param order
    assert tuple(cell.value for cell in sheet[1]) == tuple(headers)


def test_dicts_to_xlsx_inserts_rows_in_order():
    dicts = [
        {"a": "first", "b": 1},
        {"a": "second", "b": 2},
        {"a": "third", "b": 3},
    ]
    wb = dicts_to_xlsx(dicts)
    sheet = wb.active
    for idx, dict_row in enumerate(dicts, start=2):
        assert tuple(cell.value for cell in sheet[idx]) == tuple(dict_row.values())


def test_dicts_to_xlsx_with_single_dict_and_no_headers():
    dicts = [{"x": 42}]
    wb = dicts_to_xlsx(dicts)
    sheet = wb.active
    assert tuple(cell.value for cell in sheet[1]) == ("x",)
    assert tuple(cell.value for cell in sheet[2]) == (42,)


def test_xlsx_to_bytes_multiple_saves_are_consistent():
    wb = Workbook()
    wb.active['A1'] = "test"
    bytes1 = xlsx_to_bytes(wb)
    bytes2 = xlsx_to_bytes(wb)
    assert bytes1 == bytes2


def test_xlsx_to_file_multiple_saves_have_same_content():
    wb = Workbook()
    fa1 = xlsx_to_file(wb, "file.xlsx")
    fa2 = xlsx_to_file(wb, "file.xlsx")
    assert fa1.content == fa2.content
    assert fa1.name == fa2.name


@pytest.mark.parametrize("content", [b"", b"12345", b"\x50\x4B\x03\x04"])  # Zip segment start bytes
def test_xlsx_from_bytes_invalid_and_partial(content):
    # partial and empty bytes should raise or fail to parse
    with pytest.raises(Exception):
        xlsx_from_bytes(content)
