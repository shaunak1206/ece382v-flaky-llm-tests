import io
import pytest
from openpyxl import Workbook, load_workbook
from openpyxl.worksheet.worksheet import Worksheet

from docci.xlsx import xlsx_to_bytes, xlsx_to_file
from docci.file import FileAttachment


def test_xlsx_to_bytes_returns_bytes_and_contains_valid_xlsx_data():
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws["A1"] = "Test"
    bts = xlsx_to_bytes(wb)
    assert isinstance(bts, bytes)
    # Confirm bytes can be loaded back as workbook
    wb2 = load_workbook(io.BytesIO(bts))
    assert wb2.active["A1"].value == "Test"
    assert wb2.active.title == "Sheet1"


def test_xlsx_to_bytes_empty_workbook():
    wb = Workbook()
    bts = xlsx_to_bytes(wb)
    assert isinstance(bts, bytes)
    wb2 = load_workbook(io.BytesIO(bts))
    # Default sheet name is 'Sheet'
    assert wb2.sheetnames == ["Sheet"]


def test_xlsx_to_file_returns_fileattachment_with_correct_name_and_content():
    wb = Workbook()
    wb.active["A1"] = "Hello"
    filename = "testfile.xlsx"
    file_att = xlsx_to_file(wb, filename)
    assert isinstance(file_att, FileAttachment)
    assert file_att.name == filename
    assert isinstance(file_att.content, bytes)
    # content should match bytes from xlsx_to_bytes
    assert file_att.content == xlsx_to_bytes(wb)
    # Also confirm can reload workbook from content
    wb2 = load_workbook(io.BytesIO(file_att.content))
    assert wb2.active["A1"].value == "Hello"


def test_xlsx_to_file_preserves_workbook_data():
    wb = Workbook()
    ws = wb.active
    ws.append(["Name", "Age"])
    ws.append(["Alice", 30])
    fa = xlsx_to_file(wb, "people.xlsx")
    wb2 = load_workbook(io.BytesIO(fa.content))
    ws2 = wb2.active
    assert ws2["A1"].value == "Name"
    assert ws2["B1"].value == "Age"
    assert ws2["A2"].value == "Alice"
    assert ws2["B2"].value == 30


def test_xlsx_to_bytes_with_multiple_sheets_preserved():
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Sheet1"
    ws1["A1"] = 123
    ws2 = wb.create_sheet("Second")
    ws2["B2"] = "data"
    bts = xlsx_to_bytes(wb)
    wb2 = load_workbook(io.BytesIO(bts))
    assert "Sheet1" in wb2.sheetnames
    assert "Second" in wb2.sheetnames
    assert wb2["Sheet1"]["A1"].value == 123
    assert wb2["Second"]["B2"].value == "data"


def test_xlsx_to_file_with_empty_workbook():
    wb = Workbook()
    fa = xlsx_to_file(wb, "empty.xlsx")
    assert fa.name == "empty.xlsx"
    # Content bytes nonempty
    assert isinstance(fa.content, bytes)
    assert len(fa.content) > 0
    wb2 = load_workbook(io.BytesIO(fa.content))
    assert wb2.sheetnames == ["Sheet"]


def test_xlsx_to_bytes_object_is_not_modified_after_call():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "unchanged"
    before = [(row[0].value, row[1].value if len(row) > 1 else None)
              for row in ws.iter_rows(min_row=1, max_row=1)]
    xlsx_to_bytes(wb)
    after = [(row[0].value, row[1].value if len(row) > 1 else None)
             for row in ws.iter_rows(min_row=1, max_row=1)]
    assert before == after


def test_xlsx_to_file_reads_back_identical_workbook(tmp_path):
    # Saves to file via xlsx_to_file returns a FileAttachment that contains bytes exactly matching saving natively
    wb = Workbook()
    wb.active["C3"] = "value"
    fa = xlsx_to_file(wb, "file.xlsx")
    path = tmp_path / fa.name
    path.write_bytes(fa.content)
    wb2 = load_workbook(str(path))
    assert wb2.active["C3"].value == "value"


def test_xlsx_to_bytes_file_roundtrip_equivalence():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "roundtrip"
    bytes1 = xlsx_to_bytes(wb)
    fa = xlsx_to_file(wb, "test.xlsx")
    bytes2 = fa.content
    assert bytes1 == bytes2


def test_xlsx_to_file_custom_sheet_title_and_data():
    wb = Workbook()
    wb.active.title = "DataSheet"
    wb.active["B2"] = 42
    fa = xlsx_to_file(wb, "custom.xlsx")
    wb2 = load_workbook(io.BytesIO(fa.content))
    assert wb2.active.title == "DataSheet"
    assert wb2.active["B2"].value == 42


def test_xlsx_to_file_name_nonempty_string():
    wb = Workbook()
    fa = xlsx_to_file(wb, "filename.xlsx")
    assert isinstance(fa.name, str)
    assert fa.name == "filename.xlsx"


def test_xlsx_to_bytes_produces_valid_openxml_structure():
    wb = Workbook()
    bts = xlsx_to_bytes(wb)
    # XLSX files are zip archives with specific entry names
    import zipfile
    with zipfile.ZipFile(io.BytesIO(bts)) as zf:
        # Confirm expected files inside xlsx zip
        names = zf.namelist()
        assert "[Content_Types].xml" in names
        assert any(name.startswith("xl/") for name in names)
        assert any(name.startswith("docProps/") for name in names)


def test_xlsx_to_file_accepts_different_workbook_states():
    wb = Workbook()
    # Add multiple rows
    sheet = wb.active
    sheet.append(["head1", "head2"])
    sheet.append([1, 2])
    sheet.append([3, 4])
    fa = xlsx_to_file(wb, "multirows.xlsx")
    wb2 = load_workbook(io.BytesIO(fa.content))
    values = [[cell.value for cell in row] for row in wb2.active.iter_rows(min_row=1, max_row=3)]
    assert values[0] == ("head1", "head2")
    assert values[1] == (1, 2)
    assert values[2] == (3, 4)


def test_xlsx_to_file_preserves_number_formats():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = 1234.5
    ws["A1"].number_format = "#,##0.00"
    fa = xlsx_to_file(wb, "numfmt.xlsx")
    wb2 = load_workbook(io.BytesIO(fa.content))
    assert wb2.active["A1"].value == 1234.5
    assert wb2.active["A1"].number_format == "#,##0.00"


def test_xlsx_to_file_preserves_formula_cells():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = 1
    ws["A2"] = 2
    ws["A3"] = "=SUM(A1:A2)"
    fa = xlsx_to_file(wb, "formula.xlsx")
    wb2 = load_workbook(io.BytesIO(fa.content), data_only=False)
    assert wb2.active["A3"].value == "=SUM(A1:A2)"


def test_xlsx_to_bytes_multiple_times_produces_same_bytes():
    wb = Workbook()
    wb.active["A1"] = "val"
    bts1 = xlsx_to_bytes(wb)
    bts2 = xlsx_to_bytes(wb)
    assert bts1 == bts2


def test_xlsx_to_file_with_unicode_characters():
    wb = Workbook()
    wb.active["A1"] = "测试"  # chinese chars for test
    wb.active["B1"] = "emoji 😊"
    fa = xlsx_to_file(wb, "unicode.xlsx")
    wb2 = load_workbook(io.BytesIO(fa.content))
    assert wb2.active["A1"].value == "测试"
    assert wb2.active["B1"].value == "emoji 😊"
