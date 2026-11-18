import pytest

from r_map.RegisterMap import RegisterMap, get_dimensions
from r_map.Register import Register
from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.ValidationError import ValidationError


def make_register(name="REG", address=0x10, width=16, fields=None):
    reg = Register(name=name, address=address, width=width)
    if fields:
        for ref, bf in fields:
            ref._add(bf)
            reg._add(ref)
    return reg


def make_bitfield(name="BF", width=8, reset_val=0, access="RW"):
    return BitField(name=name, width=width, reset_val=reset_val, access=access)


def test_get_dimensions_basic():
    assert get_dimensions(0x100, 0x104, 16) == (4, 6, 2)


def test_bitfield_rejects_invalid_width_and_reset():
    with pytest.raises(ValueError):
        BitField(name="A", width=0)
    with pytest.raises(ValueError):
        BitField(name="B", width=1, reset_val=-1)


def test_bitfield_masks_reset_value_to_width():
    bf = BitField(name="A", width=3, reset_val=0xFF)
    assert bf.reset_val == 0x7
    assert bf.value == 0x7
    assert bf.mask == 0x7


def test_bitfield_value_assignment_supports_int_and_resets_mask():
    bf = BitField(name="A", width=4, reset_val=0)
    bf.value = 0x1F
    assert bf.value == 0xF
    bf.reset()
    assert bf.value == 0


def test_bitfield_value_assignment_supports_string_and_raises_for_unknown_name():
    bf = BitField(name="A", width=4, reset_val=0)
    bf._children["ZERO"] = type("E", (), {"name": "ZERO", "value": 0})()
    bf._children["ONE"] = type("E", (), {"name": "ONE", "value": 1})()
    bf.value = "ONE"
    assert bf.value == 1

    with pytest.raises(ValueError):
        bf.value = "MISSING"


def test_bitfield_validate_detects_duplicate_enumeration_values():
    bf = BitField(name="A", width=4, reset_val=0)
    e1 = type("E", (), {"name": "E1", "value": 1, "__str__": lambda self: self.name})()
    e2 = type("E", (), {"name": "E2", "value": 1, "__str__": lambda self: self.name})()
    bf._children["E1"] = e1
    bf._children["E2"] = e2

    errors = list(bf.validate())
    assert any(isinstance(err, ValidationError) for err in errors)
    assert any("same value" in str(err) for err in errors)


def test_bitfieldref_requires_non_negative_offsets():
    bf = BitField(name="A", width=8, reset_val=0)
    with pytest.raises(ValueError):
        BitFieldRef(name="R", reg_offset=-1)
    with pytest.raises(ValueError):
        BitFieldRef(name="R", reg_offset=0, field_offset=-1)

    ref = BitFieldRef(name="R", reg_offset=0)
    ref._add(bf)
    assert ref.slice_width == bf.width


def test_bitfieldref_value_roundtrip_through_parent_bitfield():
    bf = BitField(name="A", width=8, reset_val=0)
    ref = BitFieldRef(name="R", reg_offset=2, slice_width=3, field_offset=1)
    ref._add(bf)

    ref.value = 0b101 << 2
    assert bf.value == 0b1010
    assert ref.value == 0b101 << 2


def test_bitfieldref_validate_without_child_bitfield_reports_error():
    ref = BitFieldRef(name="R", reg_offset=0, slice_width=1)
    errors = list(ref.validate())
    assert any("No child bitfield present" in str(err) for err in errors)


def test_register_iterates_children_sorted_by_reg_offset():
    bf1 = BitField(name="BF1", width=8, reset_val=0)
    bf2 = BitField(name="BF2", width=8, reset_val=0)
    r1 = BitFieldRef(name="R1", reg_offset=8, slice_width=8)
    r2 = BitFieldRef(name="R2", reg_offset=0, slice_width=8)
    r1._add(bf1)
    r2._add(bf2)

    reg = Register(name="REG", address=0x10, width=16)
    reg._add(r1)
    reg._add(r2)

    assert [c.name for c in reg] == ["R2", "R1"]


def test_register_value_get_set_and_reset():
    bf1 = BitField(name="BF1", width=8, reset_val=0x12)
    bf2 = BitField(name="BF2", width=8, reset_val=0x34)
    r1 = BitFieldRef(name="R1", reg_offset=0, slice_width=8)
    r2 = BitFieldRef(name="R2", reg_offset=8, slice_width=8)
    r1._add(bf1)
    r2._add(bf2)

    reg = Register(name="REG", address=0x10, width=16)
    reg._add(r1)
    reg._add(r2)

    assert reg.reset_val == 0x3412
    assert reg.value == 0x3412

    reg.value = 0xABCD
    assert bf1.value == 0xCD
    assert bf2.value == 0xAB
    assert reg.value == 0xABCD

    reg.reset()
    assert reg.value == 0x3412


def test_register_validate_rejects_non_bitfieldref_child():
    reg = Register(name="REG", address=0x10, width=8)
    reg._children["bad"] = object()

    errors = list(reg.validate())
    assert any("not of typeBitFieldRef" in str(err) for err in errors)


def test_register_validate_detects_overlapping_bitfieldrefs():
    bf1 = BitField(name="BF1", width=8, reset_val=0)
    bf2 = BitField(name="BF2", width=8, reset_val=0)
    r1 = BitFieldRef(name="R1", reg_offset=0, slice_width=8)
    r2 = BitFieldRef(name="R2", reg_offset=4, slice_width=8)
    r1._add(bf1)
    r2._add(bf2)

    reg = Register(name="REG", address=0x10, width=16)
    reg._add(r1)
    reg._add(r2)

    errors = list(reg.validate())
    assert any("overlaps" in str(err) for err in errors)


def test_registermap_read_updates_register_values_and_returns_block_data():
    reg1 = Register(name="R1", address=0x100, width=16)
    reg2 = Register(name="R2", address=0x102, width=16)

    bf1 = BitField(name="BF1", width=16, reset_val=0)
    bf2 = BitField(name="BF2", width=16, reset_val=0)
    ref1 = BitFieldRef(name="REF1", reg_offset=0, slice_width=16)
    ref2 = BitFieldRef(name="REF2", reg_offset=0, slice_width=16)
    ref1._add(bf1)
    ref2._add(bf2)
    reg1._add(ref1)
    reg2._add(ref2)

    rm = RegisterMap(name="MAP")
    rm._add(reg1)
    rm._add(reg2)

    expected = bytes([0x34, 0x12, 0x78, 0x56])

    def fake_read(address, size):
        assert address == 0x100
        assert size == 4
        return expected

    rm._block_read_func = fake_read

    data = rm.read()
    assert data == expected
    assert reg1.value == 0x1234
    assert reg2.value == 0x5678


def test_registermap_write_without_data_calls_block_writer_with_composed_bytes():
    reg1 = Register(name="R1", address=0x100, width=16)
    reg2 = Register(name="R2", address=0x102, width=16)

    bf1 = BitField(name="BF1", width=16, reset_val=0)
    bf2 = BitField(name="BF2", width=16, reset_val=0)
    ref1 = BitFieldRef(name="REF1", reg_offset=0, slice_width=16)
    ref2 = BitFieldRef(name="REF2", reg_offset=0, slice_width=16)
    ref1._add(bf1)
    ref2._add(bf2)
    reg1._add(ref1)
    reg2._add(ref2)

    reg1.value = 0x1234
    reg2.value = 0xABCD

    rm = RegisterMap(name="MAP")
    rm._add(reg1)
    rm._add(reg2)

    calls = []

    def fake_write(address, data):
        calls.append((address, bytes(data)))

    rm._block_write_func = fake_write
    rm.write()

    assert calls == [(0x100, bytes([0x34, 0x12, 0xCD, 0xAB]))]


def test_registermap_write_with_partial_data_updates_only_covered_registers():
    reg1 = Register(name="R1", address=0x100, width=16)
    reg2 = Register(name="R2", address=0x102, width=16)

    bf1 = BitField(name="BF1", width=16, reset_val=0)
    bf2 = BitField(name="BF2", width=16, reset_val=0)
    ref1 = BitFieldRef(name="REF1", reg_offset=0, slice_width=16)
    ref2 = BitFieldRef(name="REF2", reg_offset=0, slice_width=16)
    ref1._add(bf1)
    ref2._add(bf2)
    reg1._add(ref1)
    reg2._add(ref2)

    rm = RegisterMap(name="MAP")
    rm._add(reg1)
    rm._add(reg2)

    calls = []

    def fake_write(address, data):
        calls.append((address, bytes(data)))

    rm._block_write_func = fake_write

    rm.write(bytearray([0xFE, 0xCA]))  # only enough for first register
    assert reg1.value == 0xCAFE
    assert reg2.value == 0x0000
    assert calls == [(0x100, bytes([0xFE, 0xCA]))]
