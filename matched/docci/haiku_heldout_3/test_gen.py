from io import BytesIO

import pytest
from openpyxl import Workbook, load_workbook

from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


def _load(data: bytes) -> Workbook:
    return load_workbook(BytesIO(data))


def _make_workbook() -> Workbook:
    wb = Workbook()
    ws = wb.active
    ws.title = "Data"
    ws.append(["name", "count", "ratio", "flag"])
    ws.append(["alpha", 1, 0.5, True])
    ws.append(["beta", 2, 1.25, False])
    return wb


# ---------- xlsx_to_bytes ----------

def test_xlsx_to_bytes_returns_bytes():
    result = xlsx_to_bytes(Workbook())
    assert isinstance(result, bytes)


def test_xlsx_to_bytes_returns_non_empty_output():
    result = xlsx_to_bytes(_make_workbook())
    assert len(result) > 0


def test_xlsx_to_bytes_output_is_zip_container():
    # xlsx files are zip archives and therefore start with the "PK" signature
    result = xlsx_to_bytes(_make_workbook())
    assert result[:2] == b"PK"


def test_xlsx_to_bytes_output_is_loadable_by_openpyxl():
    result = xlsx_to_bytes(_make_workbook())
    wb = _load(result)
    assert wb.active.title == "Data"


def test_xlsx_to_bytes_preserves_string_values():
    wb = _load(xlsx_to_bytes(_make_workbook()))
    assert wb["Data"]["A2"].value == "alpha"
    assert wb["Data"]["A3"].value == "beta"


def test_xlsx_to_bytes_preserves_numeric_values():
    wb = _load(xlsx_to_bytes(_make_workbook()))
    assert wb["Data"]["B2"].value == 1
    assert wb["Data"]["C3"].value == pytest.approx(1.25)


def test_xlsx_to_bytes_preserves_boolean_values():
    wb = _load(xlsx_to_bytes(_make_workbook()))
    assert wb["Data"]["D2"].value is True
    assert wb["Data"]["D3"].value is False


def test_xlsx_to_bytes_preserves_multiple_sheets_and_titles():
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


def test_xlsx_to_bytes_preserves_formula_text():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = 2
    ws["A2"] = 3
    ws["A3"] = "=A1+A2"

    loaded = _load(xlsx_to_bytes(wb))

    assert loaded.active["A3"].value == "=A1+A2"


def test_xlsx_to_bytes_preserves_unicode_text():
    wb = Workbook()
    wb.active["A1"] = "héllo wörld 你好 🚀"

    loaded = _load(xlsx_to_bytes(wb))

    assert loaded.active["A1"].value == "héllo wörld 你好 🚀"


def test_xlsx_to_bytes_handles_empty_workbook():
    loaded = _load(xlsx_to_bytes(Workbook()))
    assert len(loaded.sheetnames) == 1
    assert loaded.active["A1"].value is None


def test_xlsx_to_bytes_does_not_alter_workbook():
    wb = _make_workbook()
    before = [[c.value for c in row] for row in wb.active.iter_rows()]

    xlsx_to_bytes(wb)

    after = [[c.value for c in row] for row in wb.active.iter_rows()]
    assert before == after


def test_xlsx_to_bytes_workbook_remains_usable_after_call():
    wb = _make_workbook()
    xlsx_to_bytes(wb)

    wb.active["A1"] = "changed"
    loaded = _load(xlsx_to_bytes(wb))

    assert loaded.active["A1"].value == "changed"


def test_xlsx_to_bytes_handles_many_rows():
    wb = Workbook()
    ws = wb.active
    for i in range(1000):
        ws.append([i, str(i)])

    loaded = _load(xlsx_to_bytes(wb))

    assert loaded.active.max_row == 1000
    assert loaded.active["A1000"].value == 999


def test_xlsx_to_bytes_raises_for_non_workbook_input():
    with pytest.raises(AttributeError):
        xlsx_to_bytes("not a workbook")


# ---------- xlsx_to_file ----------

def test_xlsx_to_file_returns_file_attachment():
    result = xlsx_to_file(Workbook(), "report.xlsx")
    assert isinstance(result, FileAttachment)


def test_xlsx_to_file_sets_name():
    result = xlsx_to_file(Workbook(), "report.xlsx")
    assert result.name == "report.xlsx"


def test_xlsx_to_file_content_is_bytes():
    result = xlsx_to_file(_make_workbook(), "report.xlsx")
    assert isinstance(result.content, bytes)


def test_xlsx_to_file_content_matches_xlsx_to_bytes_values():
    wb = _make_workbook()
    file_att = xlsx_to_file(wb, "report.xlsx")

    loaded = _load(file_att.content)

    assert loaded.active.title == "Data"
    assert loaded["Data"]["A2"].value == "alpha"
    assert loaded["Data"]["B3"].value == 2


def test_xlsx_to_file_content_is_zip_container():
    result = xlsx_to_file(_make_workbook(), "report.xlsx")
    assert result.content[:2] == b"PK"


def test_xlsx_to_file_preserves_unicode_name():
    result = xlsx_to_file(Workbook(), "résumé_你好.xlsx")
    assert result.name == "résumé_你好.xlsx"


def test_xlsx_to_file_preserves_empty_name():
    result = xlsx_to_file(Workbook(), "")
    assert result.name == ""


def test_xlsx_to_file_raises_for_non_workbook_input():
    with pytest.raises(AttributeError):
        xlsx_to_file(None, "report.xlsx")
