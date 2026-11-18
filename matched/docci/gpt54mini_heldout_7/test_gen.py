import io

from openpyxl import Workbook, load_workbook

from docci.xlsx import xlsx_to_bytes, xlsx_to_file


def make_workbook(values=None):
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    values = values or []
    for row in values:
        ws.append(row)
    return wb


def test_xlsx_to_bytes_returns_bytes():
    wb = make_workbook()
    data = xlsx_to_bytes(wb)

    assert isinstance(data, bytes)
    assert len(data) > 0


def test_xlsx_to_bytes_produces_valid_xlsx_that_can_be_loaded():
    wb = make_workbook([["a", "b"], [1, 2]])
    data = xlsx_to_bytes(wb)

    loaded = load_workbook(io.BytesIO(data))
    ws = loaded.active

    assert ws["A1"].value == "a"
    assert ws["B1"].value == "b"
    assert ws["A2"].value == 1
    assert ws["B2"].value == 2


def test_xlsx_to_bytes_preserves_sheet_title():
    wb = make_workbook()
    wb.active.title = "CustomTitle"

    data = xlsx_to_bytes(wb)
    loaded = load_workbook(io.BytesIO(data))

    assert loaded.active.title == "CustomTitle"


def test_xlsx_to_bytes_is_deterministic_for_same_workbook_content():
    wb = make_workbook([["x", "y"], [3, 4]])

    data1 = xlsx_to_bytes(wb)
    data2 = xlsx_to_bytes(wb)

    assert data1 == data2


def test_xlsx_to_bytes_returns_different_bytes_for_different_workbooks():
    wb1 = make_workbook([["a"]])
    wb2 = make_workbook([["b"]])

    data1 = xlsx_to_bytes(wb1)
    data2 = xlsx_to_bytes(wb2)

    assert data1 != data2


def test_xlsx_to_bytes_does_not_modify_workbook_contents():
    wb = make_workbook([["hello", "world"]])
    before = wb.active["A1"].value, wb.active["B1"].value

    _ = xlsx_to_bytes(wb)

    after = wb.active["A1"].value, wb.active["B1"].value
    assert before == after


def test_xlsx_to_bytes_can_handle_unicode_values():
    wb = make_workbook([["ä", "東京", "🙂"]])

    data = xlsx_to_bytes(wb)
    loaded = load_workbook(io.BytesIO(data))
    ws = loaded.active

    assert ws["A1"].value == "ä"
    assert ws["B1"].value == "東京"
    assert ws["C1"].value == "🙂"


def test_xlsx_to_file_returns_fileattachment():
    wb = make_workbook([["value"]])

    attachment = xlsx_to_file(wb, "report.xlsx")

    assert attachment.__class__.__name__ == "FileAttachment"
    assert attachment.name == "report.xlsx"
    assert isinstance(attachment.content, bytes)
    assert len(attachment.content) > 0


def test_xlsx_to_file_uses_xlsx_to_bytes_content():
    wb = make_workbook([["a", 1]])
    expected = xlsx_to_bytes(wb)

    attachment = xlsx_to_file(wb, "data.xlsx")

    assert attachment.content == expected


def test_xlsx_to_file_content_can_be_loaded_as_workbook():
    wb = make_workbook([["row1", 123]])
    attachment = xlsx_to_file(wb, "sample.xlsx")

    loaded = load_workbook(io.BytesIO(attachment.content))
    ws = loaded.active

    assert ws["A1"].value == "row1"
    assert ws["B1"].value == 123


def test_xlsx_to_file_allows_unicode_filename():
    wb = make_workbook()
    attachment = xlsx_to_file(wb, "résumé.xlsx")

    assert attachment.name == "résumé.xlsx"


def test_xlsx_to_file_returns_independent_bytes_for_separate_calls():
    wb = make_workbook([["alpha"]])

    attachment1 = xlsx_to_file(wb, "first.xlsx")
    attachment2 = xlsx_to_file(wb, "second.xlsx")

    assert attachment1.content == attachment2.content
    assert attachment1 is not attachment2
    assert attachment1.name == "first.xlsx"
    assert attachment2.name == "second.xlsx"
