import pytest
from openpyxl import Workbook, load_workbook

from docci.xlsx import xlsx_to_bytes, xlsx_to_file


def make_workbook(values=None):
    wb = Workbook()
    ws = wb.active
    values = values or [["A1", "B1"], ["A2", "B2"]]
    for row in values:
        ws.append(row)
    return wb


def load_bytes_xlsx(data):
    return load_workbook(filename=__import__("io").BytesIO(data))


def test_xlsx_to_bytes_returns_bytes():
    wb = make_workbook()
    data = xlsx_to_bytes(wb)

    assert isinstance(data, bytes)
    assert len(data) > 0


def test_xlsx_to_bytes_produces_loadable_workbook():
    wb = make_workbook()
    data = xlsx_to_bytes(wb)

    loaded = load_workbook(__import__("io").BytesIO(data))
    assert loaded.active["A1"].value == "A1"
    assert loaded.active["B2"].value == "B2"


def test_xlsx_to_bytes_preserves_cell_values():
    wb = make_workbook([["x", 1, None], ["y", 2, 3]])
    data = xlsx_to_bytes(wb)

    loaded = load_workbook(__import__("io").BytesIO(data))
    ws = loaded.active
    assert ws["A1"].value == "x"
    assert ws["B1"].value == 1
    assert ws["C1"].value is None
    assert ws["A2"].value == "y"
    assert ws["B2"].value == 2
    assert ws["C2"].value == 3


def test_xlsx_to_bytes_roundtrip_with_multiple_rows_and_columns():
    wb = make_workbook([["h1", "h2", "h3"], ["r1c1", "r1c2", "r1c3"], ["r2c1", "r2c2", "r2c3"]])
    data = xlsx_to_bytes(wb)

    loaded = load_workbook(__import__("io").BytesIO(data))
    ws = loaded.active
    assert ws.max_row == 3
    assert ws.max_column == 3
    assert ws["C3"].value == "r2c3"


def test_xlsx_to_bytes_can_be_called_multiple_times_on_same_workbook():
    wb = make_workbook()
    first = xlsx_to_bytes(wb)
    second = xlsx_to_bytes(wb)

    assert isinstance(first, bytes)
    assert isinstance(second, bytes)
    assert first == second


def test_xlsx_to_bytes_reflects_workbook_changes_after_first_serialization():
    wb = make_workbook([["before"]])
    first = xlsx_to_bytes(wb)

    wb.active["A1"] = "after"
    second = xlsx_to_bytes(wb)

    assert first != second
    loaded = load_workbook(__import__("io").BytesIO(second))
    assert loaded.active["A1"].value == "after"


def test_xlsx_to_bytes_handles_empty_workbook():
    wb = Workbook()
    data = xlsx_to_bytes(wb)

    loaded = load_workbook(__import__("io").BytesIO(data))
    assert loaded.active.title == wb.active.title
    assert loaded.active.max_row == 1
    assert loaded.active.max_column == 1


def test_xlsx_to_bytes_uses_save_on_workbook(monkeypatch):
    wb = Workbook()
    called = {}

    def fake_save(stream):
        called["stream_type"] = type(stream).__name__
        stream.write(b"fake-bytes")

    monkeypatch.setattr(wb, "save", fake_save)

    data = xlsx_to_bytes(wb)

    assert data == b"fake-bytes"
    assert called["stream_type"] == "BytesIO"


def test_xlsx_to_bytes_propagates_save_exceptions(monkeypatch):
    wb = Workbook()

    def fake_save(stream):
        raise RuntimeError("boom")

    monkeypatch.setattr(wb, "save", fake_save)

    with pytest.raises(RuntimeError, match="boom"):
        xlsx_to_bytes(wb)


def test_xlsx_to_file_returns_fileattachment_with_name_and_content():
    wb = make_workbook()
    attachment = xlsx_to_file(wb, "report.xlsx")

    assert attachment.name == "report.xlsx"
    assert isinstance(attachment.content, bytes)
    assert len(attachment.content) > 0


def test_xlsx_to_file_content_matches_xlsx_to_bytes():
    wb = make_workbook([["hello", "world"]])

    attachment = xlsx_to_file(wb, "data.xlsx")
    data = xlsx_to_bytes(wb)

    assert attachment.content == data


def test_xlsx_to_file_result_can_be_loaded_as_workbook():
    wb = make_workbook([["alpha", "beta"]])
    attachment = xlsx_to_file(wb, "test.xlsx")

    loaded = load_workbook(__import__("io").BytesIO(attachment.content))
    assert loaded.active["A1"].value == "alpha"
    assert loaded.active["B1"].value == "beta"


def test_xlsx_to_file_uses_xlsx_to_bytes(monkeypatch):
    wb = Workbook()
    monkeypatch.setattr("docci.xlsx.xlsx_to_bytes", lambda x: b"patched-bytes")

    attachment = xlsx_to_file(wb, "patched.xlsx")

    assert attachment.name == "patched.xlsx"
    assert attachment.content == b"patched-bytes"
