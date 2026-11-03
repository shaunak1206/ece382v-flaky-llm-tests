import io
import pytest
from openpyxl import Workbook, load_workbook
from docci.xlsx import xlsx_to_bytes, xlsx_to_file
from docci.file import FileAttachment


def create_sample_workbook():
    wb = Workbook()
    ws = wb.active
    ws.append(["header1", "header2"])
    ws.append([1, 2])
    return wb


def test_xlsx_to_bytes_returns_bytes():
    wb = create_sample_workbook()
    result = xlsx_to_bytes(wb)
    assert isinstance(result, bytes)
    # Basic check that bytes start with expected XLSX signature (PK...)
    assert result[:2] == b"PK"


def test_xlsx_to_bytes_empty_workbook():
    wb = Workbook()
    b = xlsx_to_bytes(wb)
    assert isinstance(b, bytes)
    assert len(b) > 0


def test_xlsx_to_bytes_produces_valid_xlsx():
    wb = create_sample_workbook()
    b = xlsx_to_bytes(wb)
    # Try reading it back with openpyxl to confirm validity
    wb2 = load_workbook(io.BytesIO(b))
    assert wb2.active.max_row == 2
    assert wb2.active.max_column == 2
    assert wb2.active["A1"].value == "header1"
    assert wb2.active["B2"].value == 2


def test_xlsx_to_file_returns_fileattachment():
    wb = create_sample_workbook()
    file = xlsx_to_file(wb, "test.xlsx")
    assert isinstance(file, FileAttachment)
    assert file.name == "test.xlsx"
    assert isinstance(file.content, bytes)
    assert len(file.content) > 0


def test_xlsx_to_file_content_is_valid_xlsx():
    wb = create_sample_workbook()
    file = xlsx_to_file(wb, "export.xlsx")
    wb2 = load_workbook(io.BytesIO(file.content))
    assert wb2.active["A1"].value == "header1"
    assert wb2.active["A2"].value == 1


def test_xlsx_to_file_with_empty_workbook():
    wb = Workbook()
    file = xlsx_to_file(wb, "empty.xlsx")
    assert file.name == "empty.xlsx"
    assert file.content.startswith(b"PK")


def test_xlsx_to_bytes_multiple_sheets():
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Sheet1"
    ws1.append(["A", "B"])
    ws1.append([10, 20])
    ws2 = wb.create_sheet("Sheet2")
    ws2.append(["X", "Y"])
    ws2.append([100, 200])

    b = xlsx_to_bytes(wb)
    wb2 = load_workbook(io.BytesIO(b))
    assert sorted(wb2.sheetnames) == ["Sheet1", "Sheet2"]
    assert wb2["Sheet1"]["A2"].value == 10
    assert wb2["Sheet2"]["B2"].value == 200


def test_xlsx_to_bytes_with_formula_cell():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = 10
    ws["A2"] = 20
    ws["A3"] = "=SUM(A1:A2)"
    b = xlsx_to_bytes(wb)
    wb2 = load_workbook(io.BytesIO(b), data_only=False)
    assert wb2.active["A3"].value == "=SUM(A1:A2)"


def test_xlsx_to_file_different_filename():
    wb = create_sample_workbook()
    filename = "myfile_test_123.xlsx"
    file = xlsx_to_file(wb, filename)
    assert file.name == filename


def test_xlsx_to_file_filename_not_empty():
    wb = create_sample_workbook()
    file = xlsx_to_file(wb, "")
    # It should accept empty string, but name is empty then
    assert file.name == ""


def test_xlsx_to_bytes_same_workbook_consistency():
    wb = create_sample_workbook()
    b1 = xlsx_to_bytes(wb)
    b2 = xlsx_to_bytes(wb)
    assert b1 == b2


def test_xlsx_to_bytes_changing_workbook_changes_output():
    wb = create_sample_workbook()
    b1 = xlsx_to_bytes(wb)
    ws = wb.active
    ws.append([3, 4])
    b2 = xlsx_to_bytes(wb)
    assert b1 != b2


def test_xlsx_to_file_multiple_calls_generate_different_objects():
    wb = create_sample_workbook()
    file1 = xlsx_to_file(wb, "file1.xlsx")
    file2 = xlsx_to_file(wb, "file2.xlsx")
    assert file1 is not file2
    assert file1.content == file2.content
    assert file1.name == "file1.xlsx"
    assert file2.name == "file2.xlsx"


def test_xlsx_to_bytes_handles_large_data():
    wb = Workbook()
    ws = wb.active
    # Insert 1000 rows of 10 columns
    for i in range(1, 1001):
        ws.append([j * i for j in range(1, 11)])
    b = xlsx_to_bytes(wb)
    assert isinstance(b, bytes)
    assert len(b) > 0


def test_xlsx_to_file_handles_large_data():
    wb = Workbook()
    ws = wb.active
    for r in range(500):
        ws.append([r] * 5)
    file = xlsx_to_file(wb, "large.xlsx")
    assert isinstance(file, FileAttachment)
    assert file.name == "large.xlsx"
    assert len(file.content) > 0


def test_xlsx_to_bytes_retains_data_types():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = 123
    ws["B1"] = 12.34
    ws["C1"] = "text"
    ws["D1"] = True
    b = xlsx_to_bytes(wb)
    wb2 = load_workbook(io.BytesIO(b))
    ws2 = wb2.active
    assert ws2["A1"].value == 123
    assert ws2["B1"].value == 12.34
    assert ws2["C1"].value == "text"
    assert ws2["D1"].value is True


def test_xlsx_to_file_filecontent_is_bytes():
    wb = create_sample_workbook()
    file = xlsx_to_file(wb, "sample.xlsx")
    assert isinstance(file.content, bytes)


def test_xlsx_to_bytes_with_no_active_sheet():
    wb = Workbook()
    # Remove active sheet forcibly (hack, openpyxl allows deleting sheets)
    for sheet in wb.sheetnames[:]:
        std = wb[sheet]
        wb.remove(std)
    # openpyxl requires at least one sheet so create new now:
    wb.create_sheet("Sheet1")
    b = xlsx_to_bytes(wb)
    assert isinstance(b, bytes)


def test_xlsx_to_bytes_with_non_ascii_characters():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "Привет"
    ws["A2"] = "你好"
    ws["A3"] = "こんにちは"
    b = xlsx_to_bytes(wb)
    wb2 = load_workbook(io.BytesIO(b))
    ws2 = wb2.active
    assert ws2["A1"].value == "Привет"
    assert ws2["A2"].value == "你好"
    assert ws2["A3"].value == "こんにちは"
