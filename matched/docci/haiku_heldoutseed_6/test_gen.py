"""Unit tests for docci.xlsx: xlsx_to_bytes and xlsx_to_file."""

import zipfile
from io import BytesIO

import pytest
from openpyxl import Workbook, load_workbook

from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file

ZIP_MAGIC = b"PK\x03\x04"


def _load(data):
    """Load xlsx bytes back into an openpyxl Workbook."""
    return load_workbook(BytesIO(data))


def _simple_workbook():
    wb = Workbook()
    ws = wb.active
    ws.title = "Data"
    ws.append(["name", "value"])
    ws.append(["alpha", 1])
    ws.append(["beta", 2])
    return wb


# ---------------------------------------------------------------------------
# xlsx_to_bytes
# ---------------------------------------------------------------------------

def test_xlsx_to_bytes_returns_bytes_instance():
    result = xlsx_to_bytes(_simple_workbook())
    assert isinstance(result, bytes)


def test_xlsx_to_bytes_is_non_empty():
    assert len(xlsx_to_bytes(Workbook())) > 0


def test_xlsx_to_bytes_output_is_zip_container():
    # .xlsx files are OPC/zip packages
    data = xlsx_to_bytes(_simple_workbook())
    assert data.startswith(ZIP_MAGIC)
    assert zipfile.is_zipfile(BytesIO(data))


def test_xlsx_to_bytes_output_is_loadable_by_openpyxl():
    data = xlsx_to_bytes(_simple_workbook())
    wb = _load(data)
    assert wb.active["A1"].value == "name"


def test_xlsx_to_bytes_preserves_cell_values():
    data = xlsx_to_bytes(_simple_workbook())
    ws = _load(data).active
    assert ws["A2"].value == "alpha"
    assert ws["B2"].value == 1
    assert ws["A3"].value == "beta"
    assert ws["B3"].value == 2


def test_xlsx_to_bytes_preserves_sheet_title():
    data = xlsx_to_bytes(_simple_workbook())
    assert _load(data).active.title == "Data"


def test_xlsx_to_bytes_preserves_multiple_sheets_in_order():
    wb = Workbook()
    first = wb.active
    first.title = "First"
    first["A1"] = "one"
    second = wb.create_sheet("Second")
    second["A1"] = "two"

    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.sheetnames == ["First", "Second"]
    assert loaded["First"]["A1"].value == "one"
    assert loaded["Second"]["A1"].value == "two"


def test_xlsx_to_bytes_empty_workbook_round_trips():
    loaded = _load(xlsx_to_bytes(Workbook()))
    assert len(loaded.sheetnames) == 1


def test_xlsx_to_bytes_preserves_numeric_types():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = 42
    ws["A2"] = 3.5
    ws["A3"] = -7

    ws2 = _load(xlsx_to_bytes(wb)).active
    assert ws2["A1"].value == 42
    assert isinstance(ws2["A1"].value, int)
    assert ws2["A2"].value == pytest.approx(3.5)
    assert ws2["A3"].value == -7


def test_xlsx_to_bytes_preserves_boolean_values():
    wb = Workbook()
    wb.active["A1"] = True
    wb.active["A2"] = False

    ws = _load(xlsx_to_bytes(wb)).active
    assert ws["A1"].value is True
    assert ws["A2"].value is False


def test_xlsx_to_bytes_preserves_formula_text():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = 1
    ws["A2"] = 2
    ws["A3"] = "=A1+A2"

    loaded = _load(xlsx_to_bytes(wb)).active
    assert loaded["A3"].value == "=A1+A2"


def test_xlsx_to_bytes_preserves_unicode_strings():
    wb = Workbook()
    wb.active["A1"] = "héllo 世界 🚀"

    loaded = _load(xlsx_to_bytes(wb)).active
    assert loaded["A1"].value == "héllo 世界 🚀"


def test_xlsx_to_bytes_different_content_produces_different_bytes():
    wb1 = Workbook()
    wb1.active["A1"] = "first"
    wb2 = Workbook()
    wb2.active["A1"] = "second"

    assert xlsx_to_bytes(wb1) != xlsx_to_bytes(wb2)


def test_xlsx_to_bytes_can_be_called_repeatedly_on_same_workbook():
    wb = _simple_workbook()
    first = xlsx_to_bytes(wb)
    second = xlsx_to_bytes(wb)

    assert _load(first).active["A2"].value == "alpha"
    assert _load(second).active["A2"].value == "alpha"


def test_xlsx_to_bytes_does_not_alter_workbook_contents():
    wb = _simple_workbook()
    xlsx_to_bytes(wb)

    assert wb.active.title == "Data"
    assert wb.active["A2"].value == "alpha"
    assert wb.active["B3"].value == 2


def test_xlsx_to_bytes_output_can_be_written_to_disk(tmp_path):
    path = tmp_path / "out.xlsx"
    path.write_bytes(xlsx_to_bytes(_simple_workbook()))

    loaded = load_workbook(path)
    assert loaded.active["A3"].value == "beta"


# ---------------------------------------------------------------------------
# xlsx_to_file
# ---------------------------------------------------------------------------

def test_xlsx_to_file_returns_file_attachment():
    result = xlsx_to_file(_simple_workbook(), "report.xlsx")
    assert isinstance(result, FileAttachment)


def test_xlsx_to_file_sets_name():
    result = xlsx_to_file(_simple_workbook(), "report.xlsx")
    assert result.name == "report.xlsx"


def test_xlsx_to_file_content_matches_xlsx_to_bytes_output_in_size():
    wb = _simple_workbook()
    attachment = xlsx_to_file(wb, "report.xlsx")
    assert isinstance(attachment.content, bytes)
    assert attachment.content.startswith(ZIP_MAGIC)


def test_xlsx_to_file_content_round_trips_values():
    attachment = xlsx_to_file(_simple_workbook(), "report.xlsx")
    ws = _load(attachment.content).active
    assert ws["A1"].value == "name"
    assert ws["B2"].value == 1
    assert ws["A3"].value == "beta"


def test_xlsx_to_file_content_preserves_sheet_title():
    attachment = xlsx_to_file(_simple_workbook(), "report.xlsx")
    assert _load(attachment.content).active.title == "Data"


def test_xlsx_to_file_keeps_name_unchanged_for_arbitrary_strings():
    name = "my report (final) v2.xlsx"
    attachment = xlsx_to_file(Workbook(), name)
    assert attachment.name == name


def test_xlsx_to_file_same_workbook_yields_independent_attachments():
    wb = _simple_workbook()
    a = xlsx_to_file(wb, "a.xlsx")
    b = xlsx_to_file(wb, "b.xlsx")

    assert a.name == "a.xlsx"
    assert b.name == "b.xlsx"
    assert _load(a.content).active["A2"].value == _load(b.content).active["A2"].value
