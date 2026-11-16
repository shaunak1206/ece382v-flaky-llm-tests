import zipfile
from io import BytesIO

import pytest
from openpyxl import Workbook, load_workbook

from docci.file import FileAttachment
from docci import xlsx as xlsx_module
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


def _make_workbook():
    wb = Workbook()
    ws = wb.active
    ws.title = "Data"
    ws["A1"] = "name"
    ws["B1"] = "value"
    ws["A2"] = "alpha"
    ws["B2"] = 42
    ws["A3"] = "beta"
    ws["B3"] = 3.5
    return wb


def _load(data):
    return load_workbook(BytesIO(data))


class _RecordingSaver:
    """Minimal stand-in for openpyxl.Workbook that records the save target."""

    def __init__(self, exc=None):
        self.received = []
        self._exc = exc

    def save(self, target):
        self.received.append(target)
        if self._exc is not None:
            raise self._exc
        target.write(b"fake-xlsx-bytes")


# ---------------------------------------------------------------------------
# xlsx_to_bytes
# ---------------------------------------------------------------------------

def test_xlsx_to_bytes_returns_bytes_instance():
    result = xlsx_to_bytes(Workbook())
    assert isinstance(result, bytes)


def test_xlsx_to_bytes_returns_non_empty_output():
    result = xlsx_to_bytes(Workbook())
    assert len(result) > 0


def test_xlsx_to_bytes_output_is_zip_archive():
    # xlsx is an OOXML package, which is a zip container.
    result = xlsx_to_bytes(_make_workbook())
    assert zipfile.is_zipfile(BytesIO(result))


def test_xlsx_to_bytes_output_starts_with_zip_magic():
    result = xlsx_to_bytes(_make_workbook())
    assert result[:4] == b"PK\x03\x04"


def test_xlsx_to_bytes_round_trip_preserves_cell_values():
    result = xlsx_to_bytes(_make_workbook())
    ws = _load(result).active
    assert ws["A1"].value == "name"
    assert ws["B1"].value == "value"
    assert ws["A2"].value == "alpha"
    assert ws["B2"].value == 42
    assert ws["A3"].value == "beta"
    assert ws["B3"].value == 3.5


def test_xlsx_to_bytes_round_trip_preserves_sheet_title():
    result = xlsx_to_bytes(_make_workbook())
    assert _load(result).sheetnames == ["Data"]


def test_xlsx_to_bytes_preserves_multiple_sheets_in_order():
    wb = Workbook()
    first = wb.active
    first.title = "First"
    first["A1"] = 1
    second = wb.create_sheet("Second")
    second["A1"] = 2
    third = wb.create_sheet("Third")
    third["A1"] = 3

    loaded = _load(xlsx_to_bytes(wb))

    assert loaded.sheetnames == ["First", "Second", "Third"]
    assert loaded["First"]["A1"].value == 1
    assert loaded["Second"]["A1"].value == 2
    assert loaded["Third"]["A1"].value == 3


def test_xlsx_to_bytes_empty_workbook_is_loadable():
    loaded = _load(xlsx_to_bytes(Workbook()))
    assert loaded.sheetnames == ["Sheet"]


def test_xlsx_to_bytes_preserves_formula_text():
    wb = Workbook()
    wb.active["A1"] = 1
    wb.active["A2"] = 2
    wb.active["A3"] = "=A1+A2"

    loaded = _load(xlsx_to_bytes(wb))

    assert loaded.active["A3"].value == "=A1+A2"


def test_xlsx_to_bytes_preserves_bool_and_none_values():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = True
    ws["A2"] = False
    ws["A3"] = None

    loaded = _load(xlsx_to_bytes(wb)).active

    assert loaded["A1"].value is True
    assert loaded["A2"].value is False
    assert loaded["A3"].value is None


def test_xlsx_to_bytes_does_not_modify_workbook():
    wb = _make_workbook()
    before = [[c.value for c in row] for row in wb.active.iter_rows()]

    xlsx_to_bytes(wb)

    after = [[c.value for c in row] for row in wb.active.iter_rows()]
    assert before == after
    assert wb.sheetnames == ["Data"]


def test_xlsx_to_bytes_independent_outputs_reflect_state_at_call_time():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "original"
    first = xlsx_to_bytes(wb)

    ws["A1"] = "changed"
    second = xlsx_to_bytes(wb)

    assert _load(first).active["A1"].value == "original"
    assert _load(second).active["A1"].value == "changed"


def test_xlsx_to_bytes_passes_bytesio_to_save(monkeypatch):
    saver = _RecordingSaver()

    result = xlsx_to_bytes(saver)

    assert len(saver.received) == 1
    assert isinstance(saver.received[0], BytesIO)
    assert result == b"fake-xlsx-bytes"


def test_xlsx_to_bytes_propagates_save_errors():
    saver = _RecordingSaver(exc=RuntimeError("disk full"))

    with pytest.raises(RuntimeError, match="disk full"):
        xlsx_to_bytes(saver)


# ---------------------------------------------------------------------------
# xlsx_to_file
# ---------------------------------------------------------------------------

def test_xlsx_to_file_returns_file_attachment():
    result = xlsx_to_file(Workbook(), "report.xlsx")
    assert isinstance(result, FileAttachment)


def test_xlsx_to_file_sets_name():
    result = xlsx_to_file(Workbook(), "report.xlsx")
    assert result.name == "report.xlsx"


def test_xlsx_to_file_name_is_not_altered():
    name = "My Report 2024 (final) - ünïcode.xlsx"
    result = xlsx_to_file(Workbook(), name)
    assert result.name == name


def test_xlsx_to_file_empty_name_is_kept():
    result = xlsx_to_file(Workbook(), "")
    assert result.name == ""


def test_xlsx_to_file_content_is_valid_xlsx_with_workbook_values():
    result = xlsx_to_file(_make_workbook(), "data.xlsx")

    assert isinstance(result.content, bytes)
    ws = _load(result.content).active
    assert ws["A2"].value == "alpha"
    assert ws["B2"].value == 42


def test_xlsx_to_file_content_matches_xlsx_to_bytes_values():
    wb = _make_workbook()

    attachment = xlsx_to_file(wb, "data.xlsx")
    direct = xlsx_to_bytes(wb)

    loaded_attachment = _load(attachment.content)
    loaded_direct = _load(direct)
    assert loaded_attachment.sheetnames == loaded_direct.sheetnames
    assert [[c.value for c in r] for r in loaded_attachment.active.iter_rows()] == [
        [c.value for c in r] for r in loaded_direct.active.iter_rows()
    ]


def test_xlsx_to_file_delegates_to_xlsx_to_bytes(monkeypatch):
    calls = []
    sentinel_wb = object()

    def fake_xlsx_to_bytes(xlsx):
        calls.append(xlsx)
        return b"delegated"

    monkeypatch.setattr(xlsx_module, "xlsx_to_bytes", fake_xlsx_to_bytes)

    result = xlsx_to_file(sentinel_wb, "x.xlsx")

    assert calls == [sentinel_wb]
    assert result.content == b"delegated"
    assert result.name == "x.xlsx"
