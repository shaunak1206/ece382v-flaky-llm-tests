import pytest
from openpyxl import Workbook, load_workbook
from io import BytesIO
from docci.xlsx import xlsx_to_bytes, xlsx_to_file
from docci.file import FileAttachment

@pytest.fixture
def sample_workbook():
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["Header1", "Header2"])
    ws.append([1, "a"])
    ws.append([2, "b"])
    return wb

def test_xlsx_to_bytes_returns_bytes(sample_workbook):
    result = xlsx_to_bytes(sample_workbook)
    assert isinstance(result, bytes)
    assert len(result) > 0

def test_xlsx_to_bytes_produces_valid_excel(sample_workbook):
    data_bytes = xlsx_to_bytes(sample_workbook)
    # Try to load from the bytes using openpyxl to confirm validity
    wb = load_workbook(BytesIO(data_bytes))
    ws = wb.active
    assert ws.title == "Sheet1"
    rows = list(ws.values)
    assert rows[0] == ("Header1", "Header2")
    assert rows[1] == (1, "a")
    assert rows[2] == (2, "b")

def test_xlsx_to_file_returns_fileattachment(sample_workbook):
    file_attachment = xlsx_to_file(sample_workbook, name="test.xlsx")
    assert isinstance(file_attachment, FileAttachment)
    assert file_attachment.name == "test.xlsx"
    # content attribute should be bytes
    assert isinstance(file_attachment.content, bytes)
    assert len(file_attachment.content) > 0

def test_xlsx_to_file_content_is_valid_excel(sample_workbook):
    file_attachment = xlsx_to_file(sample_workbook, "myfile.xlsx")
    wb = load_workbook(BytesIO(file_attachment.content))
    ws = wb.active
    assert ws.title == "Sheet1"
    rows = list(ws.values)
    assert rows[0] == ("Header1", "Header2")

def test_xlsx_to_bytes_and_xlsx_to_file_consistent_content(sample_workbook):
    bytes_direct = xlsx_to_bytes(sample_workbook)
    file_attachment = xlsx_to_file(sample_workbook, "demo.xlsx")
    # The content from file_attachment should equal bytes_direct
    assert file_attachment.content == bytes_direct

def test_xlsx_to_bytes_empty_workbook():
    wb = Workbook()
    bytes_data = xlsx_to_bytes(wb)
    assert isinstance(bytes_data, bytes)
    assert len(bytes_data) > 0
    # Loading back must succeed
    loaded_wb = load_workbook(BytesIO(bytes_data))
    assert loaded_wb is not None
    # It should have one sheet by default
    assert len(loaded_wb.sheetnames) == 1

def test_xlsx_to_file_empty_workbook():
    wb = Workbook()
    file_attachment = xlsx_to_file(wb, "empty.xlsx")
    assert file_attachment.name == "empty.xlsx"
    assert isinstance(file_attachment.content, bytes)
    assert len(file_attachment.content) > 0

def test_xlsx_to_file_filename_types():
    wb = Workbook()
    # Accept string filename
    file_attachment = xlsx_to_file(wb, "file1.xlsx")
    assert file_attachment.name == "file1.xlsx"

    # Accept different string names, including with special chars
    names = ["a.xlsx", "b file.xlsx", "c-file.xlsx", "d_file.xlsx"]
    for n in names:
        fa = xlsx_to_file(wb, n)
        assert fa.name == n
        assert isinstance(fa.content, bytes)

def test_xlsx_to_bytes_saves_content_consistently(sample_workbook):
    bytes1 = xlsx_to_bytes(sample_workbook)
    bytes2 = xlsx_to_bytes(sample_workbook)
    # Both calls produce the same bytes content for the same workbook state
    assert bytes1 == bytes2

def test_xlsx_to_file_then_load_workbook_roundtrip(sample_workbook):
    file_attachment = xlsx_to_file(sample_workbook, "roundtrip.xlsx")
    loaded_wb = load_workbook(BytesIO(file_attachment.content))
    wi = loaded_wb.active
    original_rows = list(sample_workbook.active.values)
    loaded_rows = list(wi.values)
    assert original_rows == loaded_rows

def test_xlsx_to_bytes_does_not_modify_workbook(sample_workbook):
    before_vals = list(sample_workbook.active.values)
    _ = xlsx_to_bytes(sample_workbook)
    after_vals = list(sample_workbook.active.values)
    assert before_vals == after_vals

def test_xlsx_to_file_does_not_modify_workbook(sample_workbook):
    before_vals = list(sample_workbook.active.values)
    _ = xlsx_to_file(sample_workbook, "unchanged.xlsx")
    after_vals = list(sample_workbook.active.values)
    assert before_vals == after_vals

def test_xlsx_to_bytes_large_data():
    wb = Workbook()
    ws = wb.active
    ws.append(["Number", "Square"])
    for i in range(1000):
        ws.append([i, i*i])
    data = xlsx_to_bytes(wb)
    assert isinstance(data, bytes)
    assert len(data) > 1000  # definitely larger than 1kb
    # Load back and check count
    wb2 = load_workbook(BytesIO(data))
    rows = list(wb2.active.values)
    assert len(rows) == 1001  # header + 1000 rows
    assert rows[0] == ("Number", "Square")
    assert rows[1000] == (999, 999*999)

def test_xlsx_to_file_large_data():
    wb = Workbook()
    ws = wb.active
    ws.append(["Letter", "ASCII"])
    for i in range(65, 91):
        ws.append([chr(i), i])
    file_attachment = xlsx_to_file(wb, "letters.xlsx")
    assert file_attachment.name == "letters.xlsx"
    content = file_attachment.content
    assert isinstance(content, bytes)
    # Load back and validate rows
    wb2 = load_workbook(BytesIO(content))
    rows = list(wb2.active.values)
    assert rows[0] == ("Letter", "ASCII")
    assert rows[-1] == ("Z", 90)
    assert len(rows) == 27  # 1 header + 26 letters

# Additional sanity test that the generated bytes are not empty for normal Workbook
def test_xlsx_to_bytes_normal_workbook_not_empty():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "Test"
    out_bytes = xlsx_to_bytes(wb)
    assert isinstance(out_bytes, bytes)
    assert len(out_bytes) > 0

# Test that xlsx_to_file uses xlsx_to_bytes internally consistently (by patching)
def test_xlsx_to_file_uses_xlsx_to_bytes(monkeypatch, sample_workbook):
    called = {}

    def fake_xlsx_to_bytes(xlsx):
        called["called"] = True
        return b"fake bytes"
    monkeypatch.setattr("docci.xlsx.xlsx_to_bytes", fake_xlsx_to_bytes)

    fa = xlsx_to_file(sample_workbook, "fakename.xlsx")
    assert called.get("called") is True
    assert fa.name == "fakename.xlsx"
    assert fa.content == b"fake bytes"
