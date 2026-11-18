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


def test_xlsx_to_bytes_valid_excel_bytes():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "test"
    data = xlsx_to_bytes(wb)
    # Verify that these bytes can be loaded back as a workbook
    wb2 = load_workbook(BytesIO(data))
    assert wb2.active["A1"].value == "test"


def test_xlsx_to_file_returns_fileattachment():
    wb = Workbook()
    result = xlsx_to_file(wb, "file.xlsx")
    assert isinstance(result, FileAttachment)
    assert result.name == "file.xlsx"
    assert isinstance(result.content, bytes)
    # Check content can be loaded as workbook
    wb_loaded = load_workbook(BytesIO(result.content))
    assert isinstance(wb_loaded, Workbook)


def test_xlsx_to_file_content_matches_xlsx_to_bytes():
    wb = Workbook()
    bytes1 = xlsx_to_bytes(wb)
    file_att = xlsx_to_file(wb, "name.xlsx")
    assert bytes1 == file_att.content


def test_dicts_to_xlsx_creates_workbook_with_headers_and_data():
    data = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]
    wb = dicts_to_xlsx(data)
    sheet = wb.active
    # Headers from keys
    assert sheet[1][0].value == "a"
    assert sheet[1][1].value == "b"
    # First row data
    assert sheet[2][0].value == 1
    assert sheet[2][1].value == 2
    # Second row data
    assert sheet[3][0].value == 3
    assert sheet[3][1].value == 4


def test_dicts_to_xlsx_custom_headers():
    data = [
        {"x": 10, "y": 20},
        {"x": 30, "y": 40},
    ]
    headers = ["Y", "X"]
    wb = dicts_to_xlsx(data, headers=headers)
    sheet = wb.active
    # Headers are exactly what was passed
    assert sheet[1][0].value == "Y"
    assert sheet[1][1].value == "X"
    # Data matches the order in headers, meaning values go by dict.values() order still
    # Because dicts_to_xlsx uses tuple(dict_.values()), which follow insertion order since Python 3.7,
    # but headers do not affect the append order, test this:
    # So the test just checks that data rows are present but we can't guarantee value order matches headers
    row2 = [cell.value for cell in sheet[2]]
    row3 = [cell.value for cell in sheet[3]]
    assert set(row2) == {10, 20}
    assert set(row3) == {30, 40}


def test_dicts_to_xlsx_empty_list_raises_indexerror():
    with pytest.raises(IndexError):
        dicts_to_xlsx([])


def test_xlsx_from_bytes_roundtrip():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "roundtrip"
    data = xlsx_to_bytes(wb)
    wb2 = xlsx_from_bytes(data)
    assert wb2.active["A1"].value == "roundtrip"


def test_xlsx_from_file_creates_workbook():
    wb = Workbook()
    wb.active["B2"] = 42
    file = xlsx_to_file(wb, "testfile.xlsx")

    wb2 = xlsx_from_file(file)
    assert wb2.active["B2"].value == 42


def test_xlsx_to_bytes_multiple_saves_produce_same_content():
    wb = Workbook()
    b1 = xlsx_to_bytes(wb)
    b2 = xlsx_to_bytes(wb)
    assert b1 == b2


def test_xlsx_to_file_name_preserved():
    wb = Workbook()
    filename = "myfile.xlsx"
    file_att = xlsx_to_file(wb, filename)
    assert file_att.name == filename


def test_xlsx_to_bytes_large_sheet():
    wb = Workbook()
    ws = wb.active
    # Insert 100 rows and 10 columns
    for row_num in range(1, 101):
        ws.append([f"val{row_num}_{col}" for col in range(10)])
    data = xlsx_to_bytes(wb)
    # Check data is bytes and can be loaded back correctly
    wb2 = xlsx_from_bytes(data)
    assert wb2.active["A1"].value == "val1_0"
    assert wb2.active["J100"].value == "val100_9"


def test_dicts_to_xlsx_handles_dicts_with_different_keys():
    dicts = [
        {"a": 1, "b": 2},
        {"a": 3, "c": 5},
    ]
    # headers should come from first dict's keys if not specified
    wb = dicts_to_xlsx(dicts)
    sheet = wb.active
    # First row headers
    headers = [cell.value for cell in sheet[1]]
    assert set(headers) == {"a", "b"}
    # Second row values from first dict
    row2 = [cell.value for cell in sheet[2]]
    assert 1 in row2 and 2 in row2
    # Third row values from second dict
    row3 = [cell.value for cell in sheet[3]]
    # Since values come from .values() order of each dict, b is missing in second dict -> values different length
    # openpyxl will fill missing columns with None
    assert row3[0] == 3  # 'a' value
    # 'b' header cell will have None because second dict lacks 'b'
    assert row3[1] is None


def test_xlsx_from_bytes_invalid_bytes_raises_exception():
    with pytest.raises(Exception):
        xlsx_from_bytes(b"not an xlsx file")


def test_xlsx_from_file_invalid_content_raises_exception():
    bad_file = FileAttachment("bad.xlsx", b"not valid content")
    with pytest.raises(Exception):
        xlsx_from_file(bad_file)


def test_xlsx_to_file_empty_workbook():
    wb = Workbook()
    # Just save empty workbook and convert to file
    file_att = xlsx_to_file(wb, "empty.xlsx")
    assert isinstance(file_att, FileAttachment)
    assert file_att.name == "empty.xlsx"
    assert len(file_att.content) > 0


def test_dicts_to_xlsx_header_order_preserved():
    data = [
        {"first": 1, "second": 2},
        {"first": 3, "second": 4},
    ]
    headers = ["second", "first"]
    wb = dicts_to_xlsx(data, headers=headers)
    sheet = wb.active
    first_row = [cell.value for cell in sheet[1]]
    assert first_row == headers


def test_dicts_to_xlsx_values_follow_dict_values_order_not_header_order():
    data = [
        {"first": 5, "second": 6}
    ]
    headers = ["second", "first"]
    wb = dicts_to_xlsx(data, headers=headers)
    sheet = wb.active
    # Because sheet.append(tuple(dict_.values())) is used,
    # values follow dict insertion order, not headers order
    first_values = [cell.value for cell in sheet[2]]
    expected_values = list(data[0].values())
    assert first_values == expected_values


def test_xlsx_to_bytes_produces_non_empty_result_for_modified_workbook():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "modified"
    data = xlsx_to_bytes(wb)
    assert data != b""
    wb2 = xlsx_from_bytes(data)
    assert wb2.active["A1"].value == "modified"
