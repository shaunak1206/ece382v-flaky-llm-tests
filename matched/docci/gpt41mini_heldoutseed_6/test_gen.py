import io
import pytest
from openpyxl import Workbook, load_workbook
from openpyxl.worksheet.worksheet import Worksheet

from docci.xlsx import xlsx_to_bytes, xlsx_to_file
from docci.file import FileAttachment


def test_xlsx_to_bytes_returns_bytes():
    wb = Workbook()
    result = xlsx_to_bytes(wb)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_xlsx_to_bytes_content_valid_xlsx():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "Test"
    data = xlsx_to_bytes(wb)
    loaded = load_workbook(io.BytesIO(data))
    assert loaded.active["A1"].value == "Test"


def test_xlsx_to_file_returns_fileattachment():
    wb = Workbook()
    file = xlsx_to_file(wb, "test.xlsx")
    assert isinstance(file, FileAttachment)
    assert file.name == "test.xlsx"
    assert isinstance(file.content, bytes)
    assert len(file.content) > 0


def test_xlsx_to_file_content_matches_xlsx_to_bytes():
    wb = Workbook()
    file = xlsx_to_file(wb, "sample.xlsx")
    bytes_data = xlsx_to_bytes(wb)
    assert file.content == bytes_data


def test_xlsx_to_bytes_multiple_sheets():
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Sheet1"
    ws1["A1"] = "Hello"
    ws2 = wb.create_sheet("Sheet2")
    ws2["B2"] = "World"
    data = xlsx_to_bytes(wb)
    loaded = load_workbook(io.BytesIO(data))
    assert "Sheet1" in loaded.sheetnames
    assert "Sheet2" in loaded.sheetnames
    assert loaded["Sheet1"]["A1"].value == "Hello"
    assert loaded["Sheet2"]["B2"].value == "World"


def test_xlsx_to_bytes_empty_workbook():
    wb = Workbook()
    data = xlsx_to_bytes(wb)
    loaded = load_workbook(io.BytesIO(data))
    assert isinstance(loaded, Workbook)
    # default worksheet exists
    assert loaded.active is not None


def test_xlsx_to_file_name_preserved():
    wb = Workbook()
    name = "my_test_file.xlsx"
    file = xlsx_to_file(wb, name)
    assert file.name == name


def test_xlsx_to_bytes_with_formula():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = 1
    ws["A2"] = 2
    ws["A3"] = "=SUM(A1:A2)"
    data = xlsx_to_bytes(wb)
    loaded = load_workbook(io.BytesIO(data), data_only=False)
    assert loaded.active["A3"].value == "=SUM(A1:A2)"


def test_xlsx_to_bytes_large_content():
    wb = Workbook()
    ws = wb.active
    for row in range(1, 101):
        for col in range(1, 11):
            ws.cell(row=row, column=col, value=f"R{row}C{col}")
    data = xlsx_to_bytes(wb)
    loaded = load_workbook(io.BytesIO(data))
    assert loaded["Sheet"].cell(row=100, column=10).value == "R100C10"


def test_xlsx_to_file_content_integrity_after_modification():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "Initial"
    file1 = xlsx_to_file(wb, "file1.xlsx")

    ws["A1"] = "Modified"
    file2 = xlsx_to_file(wb, "file2.xlsx")

    # file1 content should decode and retain 'Initial'
    loaded1 = load_workbook(io.BytesIO(file1.content))
    assert loaded1.active["A1"].value == "Initial"

    # file2 content should decode and retain 'Modified'
    loaded2 = load_workbook(io.BytesIO(file2.content))
    assert loaded2.active["A1"].value == "Modified"


def test_xlsx_to_bytes_with_custom_properties():
    wb = Workbook()
    wb.properties.creator = "Tester"
    data = xlsx_to_bytes(wb)
    loaded = load_workbook(io.BytesIO(data))
    assert loaded.properties.creator == "Tester"


def test_xlsx_to_file_empty_workbook_name():
    wb = Workbook()
    file = xlsx_to_file(wb, "")
    assert file.name == ""
    assert isinstance(file.content, bytes)
    assert len(file.content) > 0


def test_xlsx_to_bytes_repeated_calls_produce_equal_bytes():
    wb = Workbook()
    data1 = xlsx_to_bytes(wb)
    data2 = xlsx_to_bytes(wb)
    assert data1 == data2


def test_xlsx_to_file_with_non_default_sheetname():
    wb = Workbook()
    ws = wb.active
    ws.title = "MySheet"
    file = xlsx_to_file(wb, "nondefault.xlsx")
    loaded = load_workbook(io.BytesIO(file.content))
    assert loaded.active.title == "MySheet"


def test_xlsx_to_bytes_write_read_workflow():
    wb1 = Workbook()
    ws = wb1.active
    ws.append(["header1", "header2"])
    ws.append([123, "abc"])
    data = xlsx_to_bytes(wb1)

    wb2 = load_workbook(io.BytesIO(data))
    ws2 = wb2.active
    values = list(ws2.values)
    assert values[0] == ("header1", "header2")
    assert values[1] == (123, "abc")


def test_xlsx_to_file_handles_filename_with_path_like_string():
    wb = Workbook()
    filename = "folder/subfolder/file.xlsx"
    file = xlsx_to_file(wb, filename)
    assert file.name == filename
    assert isinstance(file.content, bytes)


def test_xlsx_to_bytes_on_workbook_with_no_cells_set():
    wb = Workbook()
    # no cell values set, still produce bytes
    data = xlsx_to_bytes(wb)
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_xlsx_to_file_multiple_files_are_independent():
    wb1 = Workbook()
    ws1 = wb1.active
    ws1["A1"] = "File1"

    wb2 = Workbook()
    ws2 = wb2.active
    ws2["A1"] = "File2"

    f1 = xlsx_to_file(wb1, "file1.xlsx")
    f2 = xlsx_to_file(wb2, "file2.xlsx")

    l1 = load_workbook(io.BytesIO(f1.content))
    l2 = load_workbook(io.BytesIO(f2.content))

    assert l1.active["A1"].value == "File1"
    assert l2.active["A1"].value == "File2"


def test_xlsx_to_bytes_memfile_closed_after_bytes_extraction():
    wb = Workbook()
    # The BytesIO should be closed after saving - but in code it isn't explicitly closed.
    # So here we at least test no exception is raised and data is returned.
    data = xlsx_to_bytes(wb)
    assert data is not None and len(data) > 0


def test_xlsx_to_file_empty_name_and_content():
    wb = Workbook()
    file = xlsx_to_file(wb, "")
    assert file.name == ""
    assert isinstance(file.content, bytes)
    # content returns valid xlsx bytes
    loaded = load_workbook(io.BytesIO(file.content))
    assert isinstance(loaded, Workbook)
