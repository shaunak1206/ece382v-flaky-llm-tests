import io

import pytest
from openpyxl import Workbook, load_workbook

import docci.xlsx as xlsx_mod
from docci.file import FileAttachment


def make_workbook(values=None):
    wb = Workbook()
    ws = wb.active
    values = values or [["A", "B"], [1, 2]]
    for row in values:
        ws.append(row)
    return wb


def test_xlsx_to_bytes_returns_nonempty_bytes():
    wb = make_workbook()
    data = xlsx_mod.xlsx_to_bytes(wb)

    assert isinstance(data, bytes)
    assert len(data) > 0


def test_xlsx_to_bytes_roundtrip_preserves_cell_values():
    wb = make_workbook([["name", "age"], ["Alice", 30], ["Bob", 40]])
    data = xlsx_mod.xlsx_to_bytes(wb)

    loaded = load_workbook(io.BytesIO(data))
    ws = loaded.active

    assert ws["A1"].value == "name"
    assert ws["B1"].value == "age"
    assert ws["A2"].value == "Alice"
    assert ws["B2"].value == 30
    assert ws["A3"].value == "Bob"
    assert ws["B3"].value == 40


def test_xlsx_to_bytes_can_be_loaded_multiple_times():
    wb = make_workbook([["x"], [1], [2]])
    data = xlsx_mod.xlsx_to_bytes(wb)

    loaded1 = load_workbook(io.BytesIO(data))
    loaded2 = load_workbook(io.BytesIO(data))

    assert loaded1.active["A2"].value == 1
    assert loaded2.active["A3"].value == 2


def test_xlsx_to_bytes_does_not_modify_original_workbook():
    wb = make_workbook([["col"], ["before"]])

    before = wb.active["A2"].value
    data = xlsx_mod.xlsx_to_bytes(wb)
    after = wb.active["A2"].value

    assert before == "before"
    assert after == "before"
    assert isinstance(data, bytes)


def test_xlsx_to_bytes_preserves_empty_cell_structure():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "hello"
    ws["C3"] = "world"

    data = xlsx_mod.xlsx_to_bytes(wb)
    loaded = load_workbook(io.BytesIO(data))
    ws2 = loaded.active

    assert ws2["A1"].value == "hello"
    assert ws2["B2"].value is None
    assert ws2["C3"].value == "world"


def test_xlsx_to_bytes_supports_unicode_content():
    wb = make_workbook([["ключ", "值"], ["café", "naïve"]])
    data = xlsx_mod.xlsx_to_bytes(wb)

    loaded = load_workbook(io.BytesIO(data))
    ws = loaded.active

    assert ws["A1"].value == "ключ"
    assert ws["B1"].value == "值"
    assert ws["A2"].value == "café"
    assert ws["B2"].value == "naïve"


def test_xlsx_to_bytes_preserves_formula_cells_as_formula_strings():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = 1
    ws["A2"] = 2
    ws["A3"] = "=SUM(A1:A2)"

    data = xlsx_mod.xlsx_to_bytes(wb)
    loaded = load_workbook(io.BytesIO(data), data_only=False)
    ws2 = loaded.active

    assert ws2["A3"].value == "=SUM(A1:A2)"


def test_xlsx_to_file_returns_fileattachment():
    wb = make_workbook()
    attachment = xlsx_mod.xlsx_to_file(wb, "report.xlsx")

    assert isinstance(attachment, FileAttachment)


def test_xlsx_to_file_uses_provided_name():
    wb = make_workbook()
    attachment = xlsx_mod.xlsx_to_file(wb, "monthly-report.xlsx")

    assert attachment.name == "monthly-report.xlsx"


def test_xlsx_to_file_contains_same_bytes_as_xlsx_to_bytes():
    wb = make_workbook([["a"], [1]])
    attachment = xlsx_mod.xlsx_to_file(wb, "data.xlsx")
    raw = xlsx_mod.xlsx_to_bytes(wb)

    assert attachment.content == raw


def test_xlsx_to_file_roundtrip_loads_same_workbook():
    wb = make_workbook([["item", "qty"], ["pen", 3]])
    attachment = xlsx_mod.xlsx_to_file(wb, "inventory.xlsx")

    loaded = xlsx_mod.xlsx_from_file(attachment)
    ws = loaded.active

    assert ws["A1"].value == "item"
    assert ws["B1"].value == "qty"
    assert ws["A2"].value == "pen"
    assert ws["B2"].value == 3


def test_xlsx_to_file_calls_xlsx_to_bytes(monkeypatch):
    wb = make_workbook()
    called = {}

    def fake_xlsx_to_bytes(arg):
        called["workbook"] = arg
        return b"fake-bytes"

    monkeypatch.setattr(xlsx_mod, "xlsx_to_bytes", fake_xlsx_to_bytes)

    attachment = xlsx_mod.xlsx_to_file(wb, "fake.xlsx")

    assert called["workbook"] is wb
    assert attachment.name == "fake.xlsx"
    assert attachment.content == b"fake-bytes"


def test_xlsx_to_file_with_empty_workbook_roundtrips():
    wb = Workbook()
    attachment = xlsx_mod.xlsx_to_file(wb, "empty.xlsx")

    loaded = load_workbook(io.BytesIO(attachment.content))
    assert loaded.active.title == wb.active.title


def test_xlsx_to_bytes_handles_workbook_with_custom_sheet_title():
    wb = Workbook()
    ws = wb.active
    ws.title = "CustomSheet"
    ws["A1"] = "value"

    data = xlsx_mod.xlsx_to_bytes(wb)
    loaded = load_workbook(io.BytesIO(data))

    assert loaded.active.title == "CustomSheet"
    assert loaded["CustomSheet"]["A1"].value == "value"


def test_xlsx_to_file_content_is_bytes_and_not_bytearray():
    wb = make_workbook()
    attachment = xlsx_mod.xlsx_to_file(wb, "bytes.xlsx")

    assert isinstance(attachment.content, bytes)
    assert not isinstance(attachment.content, bytearray)


def test_xlsx_to_bytes_can_be_written_to_stream():
    wb = make_workbook([["n"], [10]])
    data = xlsx_mod.xlsx_to_bytes(wb)

    stream = io.BytesIO()
    stream.write(data)
    stream.seek(0)

    loaded = load_workbook(stream)
    assert loaded.active["A2"].value == 10
