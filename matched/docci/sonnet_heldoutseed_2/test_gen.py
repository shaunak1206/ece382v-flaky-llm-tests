import zipfile
from datetime import datetime
from io import BytesIO

import pytest
from openpyxl import Workbook, load_workbook

from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


def _make_workbook(rows=(("a", "b"), (1, 2))):
    wb = Workbook()
    ws = wb.active
    for row in rows:
        ws.append(row)
    return wb


def _load(content):
    return load_workbook(BytesIO(content))


def test_xlsx_to_bytes_returns_bytes():
    result = xlsx_to_bytes(_make_workbook())
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_xlsx_to_bytes_is_zip_archive():
    result = xlsx_to_bytes(_make_workbook())
    assert result[:2] == b"PK"
    assert zipfile.is_zipfile(BytesIO(result))


def test_xlsx_to_bytes_contains_workbook_part():
    result = xlsx_to_bytes(_make_workbook())
    with zipfile.ZipFile(BytesIO(result)) as zf:
        assert "xl/workbook.xml" in zf.namelist()
        assert "[Content_Types].xml" in zf.namelist()


def test_xlsx_to_bytes_roundtrip_values():
    result = xlsx_to_bytes(_make_workbook([("name", "age"), ("Bob", 30), ("Ann", 25)]))
    ws = _load(result).active
    assert [list(r) for r in ws.iter_rows(values_only=True)] == [
        ["name", "age"],
        ["Bob", 30],
        ["Ann", 25],
    ]


def test_xlsx_to_bytes_empty_workbook():
    result = xlsx_to_bytes(Workbook())
    assert zipfile.is_zipfile(BytesIO(result))
    wb = _load(result)
    assert len(wb.sheetnames) == 1
    assert wb.active.max_row == 1
    assert wb.active["A1"].value is None


def test_xlsx_to_bytes_multiple_sheets_and_titles():
    wb = Workbook()
    wb.active.title = "First"
    wb.active["A1"] = "one"
    second = wb.create_sheet("Second")
    second["B2"] = "two"
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.sheetnames == ["First", "Second"]
    assert loaded["First"]["A1"].value == "one"
    assert loaded["Second"]["B2"].value == "two"


def test_xlsx_to_bytes_unicode_values():
    wb = _make_workbook([("привет", "日本語", "é")])
    ws = _load(xlsx_to_bytes(wb)).active
    assert [c.value for c in ws[1]] == ["привет", "日本語", "é"]


def test_xlsx_to_bytes_preserves_datetime():
    moment = datetime(2020, 1, 2, 3, 4, 5)
    wb = _make_workbook([(moment,)])
    ws = _load(xlsx_to_bytes(wb)).active
    assert ws["A1"].value == moment


def test_xlsx_to_bytes_preserves_formula():
    wb = _make_workbook([(1, 2, "=A1+B1")])
    ws = _load(xlsx_to_bytes(wb)).active
    assert ws["C1"].value == "=A1+B1"


def test_xlsx_to_bytes_does_not_modify_workbook():
    wb = _make_workbook()
    xlsx_to_bytes(wb)
    assert wb.active["A1"].value == "a"
    assert wb.active["B2"].value == 2
    # Can be saved again with the same result content
    ws = _load(xlsx_to_bytes(wb)).active
    assert ws["B2"].value == 2


def test_xlsx_to_bytes_uses_workbook_save():
    class FakeWorkbook:
        def __init__(self):
            self.saved_to = None

        def save(self, stream):
            self.saved_to = stream
            stream.write(b"fake-content")

    fake = FakeWorkbook()
    assert xlsx_to_bytes(fake) == b"fake-content"
    assert fake.saved_to is not None


def test_xlsx_to_file_returns_file_attachment():
    result = xlsx_to_file(_make_workbook(), "report.xlsx")
    assert isinstance(result, FileAttachment)


def test_xlsx_to_file_sets_name():
    result = xlsx_to_file(_make_workbook(), "report.xlsx")
    assert result.name == "report.xlsx"


def test_xlsx_to_file_content_is_valid_xlsx():
    result = xlsx_to_file(_make_workbook([("x", 5)]), "data.xlsx")
    assert zipfile.is_zipfile(BytesIO(result.content))
    ws = _load(result.content).active
    assert ws["A1"].value == "x"
    assert ws["B1"].value == 5


def test_xlsx_to_file_content_matches_xlsx_to_bytes_structure():
    wb = _make_workbook()
    file = xlsx_to_file(wb, "a.xlsx")
    raw = xlsx_to_bytes(wb)
    with zipfile.ZipFile(BytesIO(file.content)) as z1, zipfile.ZipFile(BytesIO(raw)) as z2:
        assert sorted(z1.namelist()) == sorted(z2.namelist())


@pytest.mark.parametrize("name", ["a.xlsx", "my report.xlsx", "отчёт.xlsx", "noext"])
def test_xlsx_to_file_various_names(name):
    result = xlsx_to_file(Workbook(), name)
    assert result.name == name
    assert isinstance(result.content, bytes)
