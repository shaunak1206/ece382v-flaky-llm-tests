import io

import pytest
from openpyxl import Workbook, load_workbook

import docci.xlsx as xlsx_mod


def make_workbook(values=None):
    wb = Workbook()
    ws = wb.active
    if values is None:
        values = [["A1", "B1"], ["A2", "B2"]]
    for row in values:
        ws.append(row)
    return wb


def test_xlsx_to_bytes_returns_bytes():
    wb = make_workbook()
    result = xlsx_mod.xlsx_to_bytes(wb)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_xlsx_to_bytes_produces_valid_xlsx():
    wb = make_workbook([["hello", 123], ["world", 456]])
    payload = xlsx_mod.xlsx_to_bytes(wb)

    loaded = load_workbook(io.BytesIO(payload))
    ws = loaded.active
    assert ws["A1"].value == "hello"
    assert ws["B1"].value == 123
    assert ws["A2"].value == "world"
    assert ws["B2"].value == 456


def test_xlsx_to_bytes_empty_workbook_roundtrip():
    wb = Workbook()
    payload = xlsx_mod.xlsx_to_bytes(wb)

    loaded = load_workbook(io.BytesIO(payload))
    assert loaded.active.max_row == 1
    assert loaded.active.max_column == 1


def test_xlsx_to_bytes_does_not_mutate_workbook_data():
    wb = make_workbook([["x", "y"]])
    before = wb.active["A1"].value, wb.active["B1"].value

    _ = xlsx_mod.xlsx_to_bytes(wb)

    after = wb.active["A1"].value, wb.active["B1"].value
    assert after == before


def test_xlsx_to_bytes_can_be_called_multiple_times_with_same_result():
    wb = make_workbook([["repeat", 1]])
    first = xlsx_mod.xlsx_to_bytes(wb)
    second = xlsx_mod.xlsx_to_bytes(wb)
    assert first == second


def test_xlsx_to_bytes_result_is_independent_from_original_stream():
    wb = make_workbook([["a"]])
    payload = xlsx_mod.xlsx_to_bytes(wb)
    # mutate workbook after serialization
    wb.active["A1"] = "changed"

    loaded = load_workbook(io.BytesIO(payload))
    assert loaded.active["A1"].value == "a"


def test_xlsx_to_file_returns_fileattachment_with_name_and_content():
    wb = make_workbook([["name", "content"]])
    attachment = xlsx_mod.xlsx_to_file(wb, "report.xlsx")

    assert attachment.name == "report.xlsx"
    assert isinstance(attachment.content, bytes)
    assert len(attachment.content) > 0


def test_xlsx_to_file_content_matches_xlsx_to_bytes():
    wb = make_workbook([["same", "bytes"]])

    expected = xlsx_mod.xlsx_to_bytes(wb)
    attachment = xlsx_mod.xlsx_to_file(wb, "same.xlsx")

    assert attachment.content == expected


def test_xlsx_to_file_roundtrip_loads_workbook():
    wb = make_workbook([["r1c1", "r1c2"], ["r2c1", "r2c2"]])
    attachment = xlsx_mod.xlsx_to_file(wb, "data.xlsx")

    loaded = load_workbook(io.BytesIO(attachment.content))
    ws = loaded.active
    assert ws["A1"].value == "r1c1"
    assert ws["B1"].value == "r1c2"
    assert ws["A2"].value == "r2c1"
    assert ws["B2"].value == "r2c2"


def test_xlsx_to_file_delegates_to_xlsx_to_bytes(monkeypatch):
    wb = make_workbook([["delegated"]])
    calls = []

    def fake_xlsx_to_bytes(xlsx):
        calls.append(xlsx)
        return b"fake-xlsx-bytes"

    monkeypatch.setattr(xlsx_mod, "xlsx_to_bytes", fake_xlsx_to_bytes)

    attachment = xlsx_mod.xlsx_to_file(wb, "delegated.xlsx")

    assert calls == [wb]
    assert attachment.name == "delegated.xlsx"
    assert attachment.content == b"fake-xlsx-bytes"


def test_xlsx_to_bytes_accepts_workbook_with_multiple_sheets():
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "First"
    ws1["A1"] = "sheet1"
    ws2 = wb.create_sheet("Second")
    ws2["B2"] = "sheet2"

    payload = xlsx_mod.xlsx_to_bytes(wb)
    loaded = load_workbook(io.BytesIO(payload))

    assert loaded["First"]["A1"].value == "sheet1"
    assert loaded["Second"]["B2"].value == "sheet2"


def test_xlsx_to_file_with_unicode_name():
    wb = make_workbook([["unicode"]])
    attachment = xlsx_mod.xlsx_to_file(wb, "réport-✓.xlsx")

    assert attachment.name == "réport-✓.xlsx"
    assert isinstance(attachment.content, bytes)
    loaded = load_workbook(io.BytesIO(attachment.content))
    assert loaded.active["A1"].value == "unicode"


def test_xlsx_to_bytes_result_can_be_read_from_bytesio_without_errors():
    wb = make_workbook([["ok"]])
    payload = xlsx_mod.xlsx_to_bytes(wb)

    stream = io.BytesIO(payload)
    loaded = load_workbook(stream)
    assert loaded.active["A1"].value == "ok"
