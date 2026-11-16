from datetime import datetime
from io import BytesIO
from unittest.mock import MagicMock

import pytest
from openpyxl import Workbook, load_workbook

from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


def _simple_workbook():
    wb = Workbook()
    ws = wb.active
    ws.append(["a", "b"])
    ws.append([1, 2])
    return wb


def _load(data):
    return load_workbook(BytesIO(data))


def test_xlsx_to_bytes_returns_bytes():
    assert isinstance(xlsx_to_bytes(_simple_workbook()), bytes)


def test_xlsx_to_bytes_is_zip_archive():
    data = xlsx_to_bytes(_simple_workbook())
    assert data.startswith(b"PK")


def test_xlsx_to_bytes_not_empty_for_empty_workbook():
    data = xlsx_to_bytes(Workbook())
    assert len(data) > 0
    loaded = _load(data)
    assert loaded.active.max_row == 1
    assert loaded.active["A1"].value is None


def test_xlsx_to_bytes_roundtrip_values():
    loaded = _load(xlsx_to_bytes(_simple_workbook()))
    rows = list(loaded.active.values)
    assert rows == [("a", "b"), (1, 2)]


def test_xlsx_to_bytes_preserves_sheet_titles():
    wb = Workbook()
    wb.active.title = "First"
    wb.create_sheet("Second")
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.sheetnames == ["First", "Second"]


def test_xlsx_to_bytes_multiple_sheets_content():
    wb = Workbook()
    wb.active.title = "One"
    wb["One"]["A1"] = "x"
    ws2 = wb.create_sheet("Two")
    ws2["B2"] = 42
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded["One"]["A1"].value == "x"
    assert loaded["Two"]["B2"].value == 42


def test_xlsx_to_bytes_unicode_values():
    wb = Workbook()
    wb.active["A1"] = "Привет, мир ✓"
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active["A1"].value == "Привет, мир ✓"


def test_xlsx_to_bytes_preserves_types():
    wb = Workbook()
    ws = wb.active
    moment = datetime(2020, 1, 2, 3, 4, 5)
    ws.append([1, 2.5, "text", True, moment])
    loaded = _load(xlsx_to_bytes(wb))
    assert list(loaded.active.values)[0] == (1, 2.5, "text", True, moment)


def test_xlsx_to_bytes_preserves_formula():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = 1
    ws["A2"] = 2
    ws["A3"] = "=SUM(A1:A2)"
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active["A3"].value == "=SUM(A1:A2)"


def test_xlsx_to_bytes_many_rows():
    wb = Workbook()
    ws = wb.active
    for i in range(500):
        ws.append([i, str(i)])
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active.max_row == 500
    assert loaded.active["A500"].value == 499
    assert loaded.active["B500"].value == "499"


def test_xlsx_to_bytes_can_be_called_repeatedly():
    wb = _simple_workbook()
    first = _load(xlsx_to_bytes(wb))
    second = _load(xlsx_to_bytes(wb))
    assert list(first.active.values) == list(second.active.values)


def test_xlsx_to_bytes_does_not_modify_workbook():
    wb = _simple_workbook()
    xlsx_to_bytes(wb)
    assert list(wb.active.values) == [("a", "b"), (1, 2)]


def test_xlsx_to_bytes_uses_save_with_stream():
    wb = MagicMock()

    def fake_save(stream):
        stream.write(b"fake-content")

    wb.save.side_effect = fake_save
    assert xlsx_to_bytes(wb) == b"fake-content"
    wb.save.assert_called_once()


def test_xlsx_to_file_returns_file_attachment():
    result = xlsx_to_file(_simple_workbook(), "report.xlsx")
    assert isinstance(result, FileAttachment)


def test_xlsx_to_file_content_is_loadable_workbook():
    result = xlsx_to_file(_simple_workbook(), "report.xlsx")
    assert isinstance(result.content, bytes)
    loaded = _load(result.content)
    assert list(loaded.active.values) == [("a", "b"), (1, 2)]


def test_xlsx_to_file_keeps_name():
    result = xlsx_to_file(_simple_workbook(), "my file.xlsx")
    assert result.name == "my file.xlsx"


def test_xlsx_to_file_content_matches_loaded_structure_of_bytes():
    wb = _simple_workbook()
    file_ = xlsx_to_file(wb, "x.xlsx")
    from_bytes = _load(xlsx_to_bytes(wb))
    from_file = _load(file_.content)
    assert list(from_bytes.active.values) == list(from_file.active.values)


def test_xlsx_to_file_propagates_save_errors():
    wb = MagicMock()
    wb.save.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        xlsx_to_file(wb, "x.xlsx")
