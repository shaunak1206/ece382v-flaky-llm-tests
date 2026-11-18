import pytest
from openpyxl import Workbook, load_workbook

import docci.xlsx as xlsx_mod
from docci.xlsx import xlsx_to_bytes, xlsx_to_file
from docci.file import FileAttachment


def make_workbook():
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws["A1"] = "hello"
    ws["B1"] = 123
    ws.append(["row2a", "row2b"])
    return wb


def test_xlsx_to_bytes_returns_bytes():
    wb = make_workbook()
    data = xlsx_to_bytes(wb)
    assert isinstance(data, bytes)


def test_xlsx_to_bytes_produces_non_empty_payload():
    wb = make_workbook()
    data = xlsx_to_bytes(wb)
    assert len(data) > 0


def test_xlsx_to_bytes_roundtrips_with_openpyxl():
    wb = make_workbook()
    data = xlsx_to_bytes(wb)
    loaded = load_workbook(filename=__import__("io").BytesIO(data))
    ws = loaded.active
    assert ws["A1"].value == "hello"
    assert ws["B1"].value == 123
    assert ws["A2"].value == "row2a"
    assert ws["B2"].value == "row2b"


def test_xlsx_to_bytes_uses_workbook_save(monkeypatch):
    wb = make_workbook()
    called = {}

    def fake_save(target):
        called["target"] = target
        target.write(b"fake-xlsx-bytes")

    monkeypatch.setattr(wb, "save", fake_save)

    data = xlsx_to_bytes(wb)

    assert data == b"fake-xlsx-bytes"
    assert "target" in called


def test_xlsx_to_bytes_passes_a_bytesio_to_save(monkeypatch):
    wb = make_workbook()
    seen = {}

    def fake_save(target):
        seen["type"] = type(target)
        target.write(b"x")

    monkeypatch.setattr(wb, "save", fake_save)

    xlsx_to_bytes(wb)

    assert seen["type"].__name__ == "BytesIO"


def test_xlsx_to_bytes_does_not_return_same_object_as_internal_buffer():
    wb = make_workbook()
    data = xlsx_to_bytes(wb)
    assert isinstance(data, bytes)
    assert data is not None


def test_xlsx_to_file_returns_fileattachment():
    wb = make_workbook()
    attachment = xlsx_to_file(wb, "report.xlsx")
    assert isinstance(attachment, FileAttachment)


def test_xlsx_to_file_sets_name_and_content():
    wb = make_workbook()
    attachment = xlsx_to_file(wb, "report.xlsx")
    assert attachment.name == "report.xlsx"
    assert isinstance(attachment.content, bytes)
    assert len(attachment.content) > 0


def test_xlsx_to_file_content_matches_xlsx_to_bytes(monkeypatch):
    wb = make_workbook()
    expected = b"expected-content"
    monkeypatch.setattr(xlsx_mod, "xlsx_to_bytes", lambda x: expected)

    attachment = xlsx_to_file(wb, "foo.xlsx")

    assert attachment.name == "foo.xlsx"
    assert attachment.content == expected


def test_xlsx_to_file_calls_xlsx_to_bytes_with_same_workbook(monkeypatch):
    wb = make_workbook()
    seen = {}

    def fake_xlsx_to_bytes(arg):
        seen["arg"] = arg
        return b"bytes"

    monkeypatch.setattr(xlsx_mod, "xlsx_to_bytes", fake_xlsx_to_bytes)

    attachment = xlsx_to_file(wb, "foo.xlsx")

    assert seen["arg"] is wb
    assert attachment.content == b"bytes"


def test_xlsx_to_file_works_with_different_filenames():
    wb = make_workbook()
    attachment1 = xlsx_to_file(wb, "a.xlsx")
    attachment2 = xlsx_to_file(wb, "nested/path/b.xlsx")
    assert attachment1.name == "a.xlsx"
    assert attachment2.name == "nested/path/b.xlsx"
    assert attachment1.content != b""
    assert attachment2.content != b""


def test_xlsx_to_file_content_roundtrips_when_loaded():
    wb = make_workbook()
    attachment = xlsx_to_file(wb, "report.xlsx")
    loaded = load_workbook(filename=__import__("io").BytesIO(attachment.content))
    ws = loaded.active
    assert ws["A1"].value == "hello"
    assert ws["B1"].value == 123
