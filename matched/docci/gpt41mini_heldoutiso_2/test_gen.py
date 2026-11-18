import pytest
from openpyxl import Workbook, load_workbook
from io import BytesIO
from docci.xlsx import xlsx_to_bytes, xlsx_to_file
from docci.file import FileAttachment


def create_sample_workbook():
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["Header1", "Header2"])
    ws.append([1, 2])
    ws.append([3, 4])
    return wb


def test_xlsx_to_bytes_returns_bytes():
    wb = create_sample_workbook()
    data = xlsx_to_bytes(wb)
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_xlsx_to_bytes_returns_valid_xlsx():
    wb = create_sample_workbook()
    data = xlsx_to_bytes(wb)
    # Try loading bytes as workbook
    loaded_wb = load_workbook(BytesIO(data))
    assert loaded_wb.active.title == "Sheet1"
    rows = list(loaded_wb.active.iter_rows(values_only=True))
    assert rows[0] == ("Header1", "Header2")
    assert rows[1] == (1, 2)
    assert rows[2] == (3, 4)


def test_xlsx_to_file_returns_fileattachment():
    wb = create_sample_workbook()
    name = "test.xlsx"
    file_attach = xlsx_to_file(wb, name)
    assert isinstance(file_attach, FileAttachment)
    assert file_attach.name == name
    assert isinstance(file_attach.content, bytes)
    assert len(file_attach.content) > 0


def test_xlsx_to_file_content_is_valid_xlsx():
    wb = create_sample_workbook()
    name = "filetest.xlsx"
    file_attach = xlsx_to_file(wb, name)
    loaded_wb = load_workbook(BytesIO(file_attach.content))
    assert loaded_wb.active.title == "Sheet1"


def test_xlsx_to_bytes_and_xlsx_to_file_content_equality():
    wb = create_sample_workbook()
    bytes_data = xlsx_to_bytes(wb)
    file_attach = xlsx_to_file(wb, "name.xlsx")
    assert bytes_data == file_attach.content


def test_xlsx_to_bytes_empty_workbook():
    wb = Workbook()
    data = xlsx_to_bytes(wb)
    assert isinstance(data, bytes)
    loaded_wb = load_workbook(BytesIO(data))
    # There is at least one default sheet
    assert loaded_wb.active.title == wb.active.title


def test_xlsx_to_file_empty_workbook():
    wb = Workbook()
    file_attach = xlsx_to_file(wb, "empty.xlsx")
    assert file_attach.name == "empty.xlsx"
    loaded_wb = load_workbook(BytesIO(file_attach.content))
    assert loaded_wb.active.title == wb.active.title


def test_xlsx_to_bytes_multiple_sheets():
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Sheet1"
    ws1.append([1])
    ws2 = wb.create_sheet("Sheet2")
    ws2.append([2])
    data = xlsx_to_bytes(wb)
    loaded_wb = load_workbook(BytesIO(data))
    assert "Sheet1" in loaded_wb.sheetnames
    assert "Sheet2" in loaded_wb.sheetnames
    assert list(loaded_wb["Sheet1"].values)[1][0] == 1
    assert list(loaded_wb["Sheet2"].values)[0][0] == 2


def test_xlsx_to_file_multiple_sheets():
    wb = Workbook()
    wb.active.title = "Main"
    wb.active.append(["A"])
    wb.create_sheet("Second").append(["B"])
    file_attach = xlsx_to_file(wb, "multisheet.xlsx")
    loaded = load_workbook(BytesIO(file_attach.content))
    assert set(loaded.sheetnames) == {"Main", "Second"}
    assert loaded["Main"].cell(row=1, column=1).value == "A"
    assert loaded["Second"].cell(row=1, column=1).value == "B"


def test_xlsx_to_bytes_unicode_data():
    wb = Workbook()
    ws = wb.active
    ws.append(["Üñîçødë", "测试", "тест"])
    ws.append([1, 2, 3])
    data = xlsx_to_bytes(wb)
    loaded = load_workbook(BytesIO(data))
    values = list(loaded.active.values)
    assert values[0] == ("Üñîçødë", "测试", "тест")
    assert values[1] == (1, 2, 3)


def test_xlsx_to_file_unicode_data():
    wb = Workbook()
    ws = wb.active
    ws.append(["ÄÖÜ", "漢字", "😀"])
    ws.append([44, 55, 66])
    file_attach = xlsx_to_file(wb, "unicode.xlsx")
    loaded = load_workbook(BytesIO(file_attach.content))
    values = list(loaded.active.values)
    assert values[0] == ("ÄÖÜ", "漢字", "😀")
    assert values[1] == (44, 55, 66)


def test_xlsx_to_file_name_preserved():
    wb = create_sample_workbook()
    name = "myfile123.xlsx"
    file_attach = xlsx_to_file(wb, name)
    assert file_attach.name == name


def test_xlsx_to_bytes_produces_different_bytes_for_different_wbs():
    wb1 = Workbook()
    wb1.active.append(["A"])
    wb2 = Workbook()
    wb2.active.append(["B"])
    b1 = xlsx_to_bytes(wb1)
    b2 = xlsx_to_bytes(wb2)
    assert b1 != b2


def test_xlsx_to_file_produces_different_content_for_different_wbs():
    wb1 = Workbook()
    wb1.active.append(["AAA"])
    wb2 = Workbook()
    wb2.active.append(["BBB"])
    f1 = xlsx_to_file(wb1, "a.xlsx")
    f2 = xlsx_to_file(wb2, "b.xlsx")
    assert f1.content != f2.content


def test_xlsx_to_bytes_closed_after_saving():
    # This indirectly checks that BytesIO is closed or doesn't cause issues by repeated calls
    wb = create_sample_workbook()
    for _ in range(3):
        data = xlsx_to_bytes(wb)
        assert data is not None


def test_xlsx_to_file_multiple_calls_return_independent_attachments():
    wb = create_sample_workbook()
    f1 = xlsx_to_file(wb, "file1.xlsx")
    f2 = xlsx_to_file(wb, "file2.xlsx")
    assert f1.name == "file1.xlsx"
    assert f2.name == "file2.xlsx"
    assert f1.content == f2.content  # Same content as same wb provided, name differs
    # Modify wb and test different content
    wb.active.append(["new"])
    f3 = xlsx_to_file(wb, "file3.xlsx")
    assert f3.content != f1.content


def test_xlsx_to_bytes_workbook_with_formulas():
    wb = Workbook()
    ws = wb.active
    ws.append([1, 2])
    ws["C1"] = "=SUM(A1:B1)"
    data = xlsx_to_bytes(wb)
    loaded = load_workbook(BytesIO(data), data_only=False)
    val = loaded.active["C1"].value
    assert val == "=SUM(A1:B1)"


def test_xlsx_to_file_workbook_with_formulas():
    wb = Workbook()
    ws = wb.active
    ws.append([10, 20])
    ws["C1"] = "=A1*B1"
    file_attach = xlsx_to_file(wb, "formula.xlsx")
    loaded = load_workbook(BytesIO(file_attach.content), data_only=False)
    assert loaded.active["C1"].value == "=A1*B1"
