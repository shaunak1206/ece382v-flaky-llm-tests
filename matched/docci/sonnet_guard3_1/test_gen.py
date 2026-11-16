import zipfile
from io import BytesIO

import pytest
from openpyxl import Workbook, load_workbook

import docci.xlsx as xlsx_module
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


def _make_workbook(rows=(("a", "b"), (1, 2))):
    wb = Workbook()
    ws = wb.active
    for row in rows:
        ws.append(row)
    return wb


def _load(data):
    return load_workbook(BytesIO(data))


class _FakeAttachment:
    def __init__(self, name, content):
        self.name = name
        self.content = content


class _FakeWorkbook:
    """Workbook stand-in that writes fixed bytes to the stream."""

    def __init__(self, payload=b"fake-payload"):
        self.payload = payload
        self.saved_to = []

    def save(self, stream):
        self.saved_to.append(stream)
        stream.write(self.payload)


class _FailingWorkbook:
    def save(self, stream):
        raise RuntimeError("cannot save")


def test_xlsx_to_bytes_returns_bytes():
    assert isinstance(xlsx_to_bytes(_make_workbook()), bytes)


def test_xlsx_to_bytes_is_nonempty_zip_signature():
    data = xlsx_to_bytes(_make_workbook())
    assert len(data) > 0
    assert data.startswith(b"PK")


def test_xlsx_to_bytes_is_valid_zip_with_workbook_part():
    data = xlsx_to_bytes(_make_workbook())
    with zipfile.ZipFile(BytesIO(data)) as zf:
        assert zf.testzip() is None
        assert "xl/workbook.xml" in zf.namelist()


def test_xlsx_to_bytes_roundtrip_values():
    data = xlsx_to_bytes(_make_workbook((("name", "age"), ("Bob", 30), ("Ann", 25))))
    ws = _load(data).active
    assert [list(r) for r in ws.iter_rows(values_only=True)] == [
        ["name", "age"],
        ["Bob", 30],
        ["Ann", 25],
    ]


def test_xlsx_to_bytes_empty_workbook_loadable():
    data = xlsx_to_bytes(Workbook())
    wb = _load(data)
    assert len(wb.sheetnames) == 1
    assert wb.active.max_row == 1
    assert wb.active["A1"].value is None


def test_xlsx_to_bytes_preserves_multiple_sheet_titles():
    wb = Workbook()
    wb.active.title = "First"
    wb.create_sheet("Second")
    wb.create_sheet("Third")
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.sheetnames == ["First", "Second", "Third"]


def test_xlsx_to_bytes_preserves_floats_and_unicode():
    wb = _make_workbook((("текст", "日本語"), (1.5, -0.25)))
    ws = _load(xlsx_to_bytes(wb)).active
    assert ws["A1"].value == "текст"
    assert ws["B1"].value == "日本語"
    assert ws["A2"].value == pytest.approx(1.5)
    assert ws["B2"].value == pytest.approx(-0.25)


def test_xlsx_to_bytes_preserves_formula_text():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = 2
    ws["A2"] = 3
    ws["A3"] = "=SUM(A1:A2)"
    loaded = _load(xlsx_to_bytes(wb)).active
    assert loaded["A3"].value == "=SUM(A1:A2)"


def test_xlsx_to_bytes_can_be_called_twice_on_same_workbook():
    wb = _make_workbook()
    first = xlsx_to_bytes(wb)
    second = xlsx_to_bytes(wb)
    for data in (first, second):
        ws = _load(data).active
        assert [list(r) for r in ws.iter_rows(values_only=True)] == [["a", "b"], [1, 2]]


def test_xlsx_to_bytes_returns_what_save_wrote():
    fake = _FakeWorkbook(b"hello bytes")
    assert xlsx_to_bytes(fake) == b"hello bytes"
    assert len(fake.saved_to) == 1
    assert isinstance(fake.saved_to[0], BytesIO)


def test_xlsx_to_bytes_propagates_save_errors():
    with pytest.raises(RuntimeError, match="cannot save"):
        xlsx_to_bytes(_FailingWorkbook())


def test_xlsx_to_file_passes_name_and_bytes_to_file_attachment(monkeypatch):
    monkeypatch.setattr(xlsx_module, "FileAttachment", _FakeAttachment)
    result = xlsx_to_file(_FakeWorkbook(b"abc123"), "report.xlsx")
    assert isinstance(result, _FakeAttachment)
    assert result.name == "report.xlsx"
    assert result.content == b"abc123"


def test_xlsx_to_file_uses_xlsx_to_bytes(monkeypatch):
    calls = []

    def fake_to_bytes(wb):
        calls.append(wb)
        return b"from-fake"

    monkeypatch.setattr(xlsx_module, "xlsx_to_bytes", fake_to_bytes)
    monkeypatch.setattr(xlsx_module, "FileAttachment", _FakeAttachment)
    wb = object()
    result = xlsx_to_file(wb, "x.xlsx")
    assert calls == [wb]
    assert result.content == b"from-fake"
    assert result.name == "x.xlsx"


def test_xlsx_to_file_real_attachment_content_roundtrip():
    wb = _make_workbook((("k", "v"), ("x", 10)))
    attachment = xlsx_to_file(wb, "data.xlsx")
    assert isinstance(attachment.content, bytes)
    ws = _load(attachment.content).active
    assert [list(r) for r in ws.iter_rows(values_only=True)] == [["k", "v"], ["x", 10]]


def test_xlsx_to_file_propagates_save_errors(monkeypatch):
    monkeypatch.setattr(xlsx_module, "FileAttachment", _FakeAttachment)
    with pytest.raises(RuntimeError, match="cannot save"):
        xlsx_to_file(_FailingWorkbook(), "bad.xlsx")
