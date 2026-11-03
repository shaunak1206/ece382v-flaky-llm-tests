import zipfile
from io import BytesIO

import pytest
from openpyxl import Workbook, load_workbook

from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


def make_workbook(rows=None, title=None):
    wb = Workbook()
    ws = wb.active
    if title is not None:
        ws.title = title
    for row in rows or []:
        ws.append(row)
    return wb


def roundtrip(data):
    return load_workbook(BytesIO(data))


def test_xlsx_to_bytes_returns_bytes():
    assert isinstance(xlsx_to_bytes(make_workbook()), bytes)


def test_xlsx_to_bytes_not_empty():
    assert len(xlsx_to_bytes(make_workbook())) > 0


def test_xlsx_to_bytes_is_zip_archive():
    data = xlsx_to_bytes(make_workbook())
    assert data[:2] == b"PK"
    assert zipfile.is_zipfile(BytesIO(data))


def test_xlsx_to_bytes_contains_workbook_part():
    data = xlsx_to_bytes(make_workbook())
    with zipfile.ZipFile(BytesIO(data)) as archive:
        names = archive.namelist()
    assert "xl/workbook.xml" in names
    assert "[Content_Types].xml" in names


def test_xlsx_to_bytes_roundtrip_values():
    rows = [("a", "b"), (1, 2), (3.5, "x")]
    loaded = roundtrip(xlsx_to_bytes(make_workbook(rows)))
    result = [tuple(c.value for c in row) for row in loaded.active.iter_rows()]
    assert result == rows


def test_xlsx_to_bytes_roundtrip_sheet_title():
    loaded = roundtrip(xlsx_to_bytes(make_workbook([("x",)], title="MySheet")))
    assert loaded.sheetnames == ["MySheet"]


def test_xlsx_to_bytes_multiple_sheets():
    wb = make_workbook([("first",)], title="One")
    ws2 = wb.create_sheet("Two")
    ws2.append(("second",))
    loaded = roundtrip(xlsx_to_bytes(wb))
    assert loaded.sheetnames == ["One", "Two"]
    assert loaded["One"]["A1"].value == "first"
    assert loaded["Two"]["A1"].value == "second"


def test_xlsx_to_bytes_empty_workbook_roundtrip():
    loaded = roundtrip(xlsx_to_bytes(Workbook()))
    assert len(loaded.sheetnames) == 1
    assert loaded.active.max_row == 1
    assert loaded.active["A1"].value is None


def test_xlsx_to_bytes_unicode_values():
    loaded = roundtrip(xlsx_to_bytes(make_workbook([("привет", "日本語", "café")])))
    assert [c.value for c in loaded.active[1]] == ["привет", "日本語", "café"]


def test_xlsx_to_bytes_formula_preserved():
    wb = make_workbook([(1, 2)])
    wb.active["C1"] = "=A1+B1"
    loaded = roundtrip(xlsx_to_bytes(wb))
    assert loaded.active["C1"].value == "=A1+B1"


def test_xlsx_to_bytes_does_not_modify_workbook():
    wb = make_workbook([("a", 1)])
    xlsx_to_bytes(wb)
    assert wb.active["A1"].value == "a"
    assert wb.active["B1"].value == 1
    assert wb.sheetnames == ["Sheet"]


def test_xlsx_to_bytes_can_be_called_repeatedly():
    wb = make_workbook([("a", 1)])
    first = roundtrip(xlsx_to_bytes(wb))
    second = roundtrip(xlsx_to_bytes(wb))
    assert first.active["A1"].value == second.active["A1"].value == "a"
    assert first.active["B1"].value == second.active["B1"].value == 1


def test_xlsx_to_file_returns_file_attachment():
    result = xlsx_to_file(make_workbook(), "report.xlsx")
    assert isinstance(result, FileAttachment)


def test_xlsx_to_file_name_and_extension():
    result = xlsx_to_file(make_workbook(), "report.xlsx")
    assert result.name == "report.xlsx"
    assert result.extension == "xlsx"
    assert result.name_without_extension == "report"


def test_xlsx_to_file_content_is_valid_xlsx():
    rows = [("h1", "h2"), ("v1", 2)]
    result = xlsx_to_file(make_workbook(rows), "data.xlsx")
    assert isinstance(result.content, bytes)
    loaded = roundtrip(result.content)
    assert [tuple(c.value for c in row) for row in loaded.active.iter_rows()] == rows


def test_xlsx_to_file_normalizes_restricted_name_chars():
    result = xlsx_to_file(make_workbook(), 'a*b?"c".xlsx')
    assert result.name == "abc.xlsx"


def test_xlsx_to_file_strips_directory_from_name():
    result = xlsx_to_file(make_workbook(), "/opt/data/out.xlsx")
    assert result.name == "out.xlsx"


def test_xlsx_to_file_content_stream_is_loadable():
    result = xlsx_to_file(make_workbook([("z", 9)]), "s.xlsx")
    loaded = load_workbook(result.content_stream)
    assert loaded.active["A1"].value == "z"
    assert loaded.active["B1"].value == 9


def test_xlsx_to_file_save_writes_loadable_file(tmp_path):
    result = xlsx_to_file(make_workbook([("k", 7)]), "saved.xlsx")
    target = tmp_path / "saved.xlsx"
    result.save(str(target))
    loaded = load_workbook(str(target))
    assert loaded.active["A1"].value == "k"
    assert loaded.active["B1"].value == 7


def test_xlsx_to_file_content_matches_xlsx_to_bytes_structure():
    wb = make_workbook([("a", 1), ("b", 2)])
    from_bytes = roundtrip(xlsx_to_bytes(wb))
    from_file = roundtrip(xlsx_to_file(wb, "f.xlsx").content)
    assert from_bytes.sheetnames == from_file.sheetnames
    assert [[c.value for c in r] for r in from_bytes.active.iter_rows()] == [
        [c.value for c in r] for r in from_file.active.iter_rows()
    ]


def test_xlsx_to_bytes_rejects_non_workbook():
    with pytest.raises(AttributeError):
        xlsx_to_bytes(object())


def test_xlsx_to_file_rejects_non_workbook():
    with pytest.raises(AttributeError):
        xlsx_to_file(None, "x.xlsx")
