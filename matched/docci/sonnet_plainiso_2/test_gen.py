import zipfile
from io import BytesIO
from unittest import mock

import pytest
from openpyxl import Workbook, load_workbook

from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


def make_workbook(rows=(("a", "b"), (1, 2))):
    wb = Workbook()
    ws = wb.active
    for row in rows:
        ws.append(row)
    return wb


def read_back(data):
    return load_workbook(BytesIO(data))


def test_xlsx_to_bytes_returns_bytes():
    assert isinstance(xlsx_to_bytes(make_workbook()), bytes)


def test_xlsx_to_bytes_non_empty():
    assert len(xlsx_to_bytes(make_workbook())) > 0


def test_xlsx_to_bytes_is_zip_archive():
    data = xlsx_to_bytes(make_workbook())
    assert data[:2] == b"PK"
    assert zipfile.is_zipfile(BytesIO(data))


def test_xlsx_to_bytes_contains_workbook_part():
    data = xlsx_to_bytes(make_workbook())
    with zipfile.ZipFile(BytesIO(data)) as archive:
        assert "xl/workbook.xml" in archive.namelist()


def test_xlsx_to_bytes_roundtrip_values():
    data = xlsx_to_bytes(make_workbook([("name", "age"), ("bob", 30), ("alice", 25)]))
    sheet = read_back(data).active
    rows = [tuple(cell for cell in row) for row in sheet.iter_rows(values_only=True)]
    assert rows == [("name", "age"), ("bob", 30), ("alice", 25)]


def test_xlsx_to_bytes_empty_workbook_is_valid():
    data = xlsx_to_bytes(Workbook())
    wb = read_back(data)
    assert len(wb.sheetnames) == 1
    assert wb.active.max_row == 1
    assert wb.active["A1"].value is None


def test_xlsx_to_bytes_preserves_sheet_title():
    wb = make_workbook()
    wb.active.title = "Report"
    assert read_back(xlsx_to_bytes(wb)).sheetnames == ["Report"]


def test_xlsx_to_bytes_preserves_multiple_sheets():
    wb = make_workbook()
    wb.active.title = "First"
    second = wb.create_sheet("Second")
    second["A1"] = "x"
    loaded = read_back(xlsx_to_bytes(wb))
    assert loaded.sheetnames == ["First", "Second"]
    assert loaded["Second"]["A1"].value == "x"


def test_xlsx_to_bytes_unicode_values():
    wb = make_workbook([("привет", "日本語")])
    sheet = read_back(xlsx_to_bytes(wb)).active
    assert sheet["A1"].value == "привет"
    assert sheet["B1"].value == "日本語"


def test_xlsx_to_bytes_calls_save_with_bytesio():
    wb = mock.Mock()
    xlsx_to_bytes(wb)
    wb.save.assert_called_once()
    (stream,), _ = wb.save.call_args
    assert isinstance(stream, BytesIO)


def test_xlsx_to_bytes_returns_what_save_wrote():
    wb = mock.Mock()
    wb.save.side_effect = lambda stream: stream.write(b"fake-content")
    assert xlsx_to_bytes(wb) == b"fake-content"


def test_xlsx_to_bytes_propagates_save_errors():
    wb = mock.Mock()
    wb.save.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        xlsx_to_bytes(wb)


def test_xlsx_to_file_returns_file_attachment():
    result = xlsx_to_file(make_workbook(), "report.xlsx")
    assert isinstance(result, FileAttachment)


def test_xlsx_to_file_sets_name():
    assert xlsx_to_file(make_workbook(), "report.xlsx").name == "report.xlsx"


def test_xlsx_to_file_normalizes_name():
    result = xlsx_to_file(make_workbook(), "/opt/data/re*port?.xlsx")
    assert result.name == "report.xlsx"
    assert result.extension == "xlsx"
    assert result.name_without_extension == "report"


def test_xlsx_to_file_content_is_readable_workbook():
    result = xlsx_to_file(make_workbook([("k", "v"), ("x", 5)]), "data.xlsx")
    sheet = load_workbook(result.content_stream).active
    assert [row for row in sheet.iter_rows(values_only=True)] == [("k", "v"), ("x", 5)]


def test_xlsx_to_file_uses_xlsx_to_bytes():
    wb = mock.Mock()
    wb.save.side_effect = lambda stream: stream.write(b"payload")
    result = xlsx_to_file(wb, "out.xlsx")
    assert result.content == b"payload"
    assert result.name == "out.xlsx"


def test_xlsx_to_file_empty_workbook():
    result = xlsx_to_file(Workbook(), "empty.xlsx")
    assert isinstance(result.content, bytes)
    assert zipfile.is_zipfile(result.content_stream)


def test_xlsx_to_file_save_writes_to_disk(tmp_path):
    result = xlsx_to_file(make_workbook(), "disk.xlsx")
    target = tmp_path / "disk.xlsx"
    result.save(str(target))
    assert target.read_bytes() == result.content
    assert load_workbook(str(target)).active["A1"].value == "a"
