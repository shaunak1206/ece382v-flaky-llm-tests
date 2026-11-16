import io
import zipfile
from unittest import mock

import pytest
from openpyxl import Workbook, load_workbook

import docci.xlsx as xlsx_module
from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


def _simple_workbook():
    wb = Workbook()
    ws = wb.active
    ws.append(["a", "b"])
    ws.append([1, 2])
    return wb


def _load(data):
    return load_workbook(io.BytesIO(data))


def test_xlsx_to_bytes_returns_bytes():
    result = xlsx_to_bytes(_simple_workbook())
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_xlsx_to_bytes_is_zip_archive():
    result = xlsx_to_bytes(_simple_workbook())
    assert result[:2] == b"PK"
    assert zipfile.is_zipfile(io.BytesIO(result))


def test_xlsx_to_bytes_contains_workbook_part():
    result = xlsx_to_bytes(_simple_workbook())
    with zipfile.ZipFile(io.BytesIO(result)) as zf:
        assert "xl/workbook.xml" in zf.namelist()
        assert "[Content_Types].xml" in zf.namelist()


def test_xlsx_to_bytes_roundtrip_values():
    result = xlsx_to_bytes(_simple_workbook())
    ws = _load(result).active
    assert [list(r) for r in ws.iter_rows(values_only=True)] == [["a", "b"], [1, 2]]


def test_xlsx_to_bytes_empty_workbook():
    result = xlsx_to_bytes(Workbook())
    wb = _load(result)
    assert len(wb.sheetnames) == 1
    assert wb.active.max_row == 1
    assert wb.active["A1"].value is None


def test_xlsx_to_bytes_preserves_multiple_sheets_and_titles():
    wb = Workbook()
    wb.active.title = "First"
    second = wb.create_sheet("Second")
    second["A1"] = "hello"
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.sheetnames == ["First", "Second"]
    assert loaded["Second"]["A1"].value == "hello"


def test_xlsx_to_bytes_unicode_and_types():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "Привет, 世界"
    ws["A2"] = 3.5
    ws["A3"] = True
    loaded = _load(xlsx_to_bytes(wb)).active
    assert loaded["A1"].value == "Привет, 世界"
    assert loaded["A2"].value == 3.5
    assert loaded["A3"].value is True


def test_xlsx_to_bytes_preserves_formula():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = 1
    ws["A2"] = 2
    ws["A3"] = "=SUM(A1:A2)"
    loaded = _load(xlsx_to_bytes(wb)).active
    assert loaded["A3"].value == "=SUM(A1:A2)"


def test_xlsx_to_bytes_can_be_called_repeatedly():
    wb = _simple_workbook()
    first = xlsx_to_bytes(wb)
    wb.active.append([3, 4])
    second = xlsx_to_bytes(wb)
    assert _load(first).active.max_row == 2
    assert _load(second).active.max_row == 3


def test_xlsx_to_bytes_uses_workbook_save_with_bytesio():
    fake = mock.Mock()

    def save(stream):
        stream.write(b"fake-content")

    fake.save.side_effect = save
    assert xlsx_to_bytes(fake) == b"fake-content"
    fake.save.assert_called_once()
    assert isinstance(fake.save.call_args[0][0], io.BytesIO)


def test_xlsx_to_bytes_propagates_save_errors():
    fake = mock.Mock()
    fake.save.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        xlsx_to_bytes(fake)


def test_xlsx_to_file_returns_file_attachment():
    result = xlsx_to_file(_simple_workbook(), "report.xlsx")
    assert isinstance(result, FileAttachment)


def test_xlsx_to_file_content_is_valid_xlsx():
    result = xlsx_to_file(_simple_workbook(), "report.xlsx")
    ws = _load(result.content).active
    assert [list(r) for r in ws.iter_rows(values_only=True)] == [["a", "b"], [1, 2]]


def test_xlsx_to_file_preserves_name():
    result = xlsx_to_file(_simple_workbook(), "my file.xlsx")
    assert result.name == "my file.xlsx"


def test_xlsx_to_file_passes_name_and_bytes_to_file_attachment(monkeypatch):
    monkeypatch.setattr(xlsx_module, "xlsx_to_bytes", lambda wb: b"DATA")
    fake_cls = mock.Mock(return_value="sentinel")
    monkeypatch.setattr(xlsx_module, "FileAttachment", fake_cls)

    wb = Workbook()
    result = xlsx_to_file(wb, "x.xlsx")

    assert result == "sentinel"
    fake_cls.assert_called_once_with("x.xlsx", b"DATA")


def test_xlsx_to_file_content_matches_xlsx_to_bytes_structure():
    wb = _simple_workbook()
    file_ = xlsx_to_file(wb, "a.xlsx")
    direct = xlsx_to_bytes(wb)
    with zipfile.ZipFile(io.BytesIO(file_.content)) as z1, zipfile.ZipFile(io.BytesIO(direct)) as z2:
        assert sorted(z1.namelist()) == sorted(z2.namelist())
