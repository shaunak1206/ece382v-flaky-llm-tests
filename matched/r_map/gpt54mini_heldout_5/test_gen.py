import pytest

from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.Register import Register
from r_map.RegisterMap import RegisterMap, get_dimensions


def _add_child(parent, child):
    """
    Helper to attach children without relying on a higher-level public API.
    """
    if hasattr(parent, "_add"):
        parent._add(child)
    else:
        parent._children[child.name] = child
        child._parent = parent


def _make_register_with_refs():
    bf = BitField(name="bf", width=8, reset_val=0x12, access="RW")
    ref = BitFieldRef(name="ref", reg_offset=0, slice_width=8, field_offset=0)
    ref._add(bf)
    reg = Register(name="reg", width=8)
    _add_child(reg, ref)
    return reg, ref, bf


def test_bitfield_rejects_invalid_width_and_reset_val():
    with pytest.raises(ValueError):
        BitField(name="bf0", width=0)
    with pytest.raises(ValueError):
        BitField(name="bf1", width=1, reset_val=-1)


def test_bitfield_masks_reset_value_and_initial_value():
    bf = BitField(name="bf", width=3, reset_val=0xFF, access="RW")
    assert bf.mask == 0b111
    assert bf.reset_val == 0b111
    assert bf.value == 0b111


def test_bitfield_value_accepts_int_and_masks():
    bf = BitField(name="bf", width=4, reset_val=0, access="RW")
    bf.value = 0x3F
    assert bf.value == 0xF


def test_bitfield_value_accepts_enumeration_name_and_object():
    from r_map.Enumeration import Enumeration

    bf = BitField(name="bf", width=2, reset_val=0, access="RW")
    e0 = Enumeration(name="ZERO", value=0)
    e1 = Enumeration(name="ONE", value=1)
    bf._children = {"ZERO": e0, "ONE": e1}

    bf.value = "ONE"
    assert bf.value == 1
    bf.value = e0
    assert bf.value == 0


def test_bitfield_value_rejects_unknown_name_and_unsupported_type():
    bf = BitField(name="bf", width=2, reset_val=0, access="RW")
    with pytest.raises(ValueError):
        bf.value = "MISSING"
    with pytest.raises(NotImplementedError):
        bf.value = 3.14


def test_bitfield_annotation_uses_matching_enumeration_name():
    from r_map.Enumeration import Enumeration

    bf = BitField(name="bf", width=2, reset_val=0, access="RW")
    e0 = Enumeration(name="ZERO", value=0)
    e1 = Enumeration(name="ONE", value=1)
    bf._children = {"ZERO": e0, "ONE": e1}

    bf.value = 1
    assert bf.annotation == "ONE"
    bf.value = 2
    assert bf.annotation == "0x2"


def test_bitfield_validate_detects_duplicate_enumeration_values():
    from r_map.Enumeration import Enumeration

    bf = BitField(name="bf", width=2, reset_val=0, access="RW")
    e0 = Enumeration(name="A", value=0)
    e1 = Enumeration(name="B", value=0)
    bf._children = {"A": e0, "B": e1}

    errors = list(bf.validate())
    assert any("same value" in str(e) for e in errors)


def test_bitfieldref_defaults_slice_width_to_bitfield_width():
    bf = BitField(name="bf", width=5, reset_val=0, access="RW")
    ref = BitFieldRef(name="ref", reg_offset=2, slice_width=0, field_offset=0)
    ref._add(bf)
    assert ref.slice_width == 5
    assert ref.mask == 0b11111


def test_bitfieldref_value_maps_between_ref_and_bitfield():
    bf = BitField(name="bf", width=8, reset_val=0x00, access="RW")
    ref = BitFieldRef(name="ref", reg_offset=2, slice_width=3, field_offset=1)
    ref._add(bf)

    bf.value = 0
    ref.value = 0b101 << 2
    assert bf.value == 0b1010
    assert ref.value == (0b101 << 2)


def test_bitfieldref_validate_missing_child_and_slice_too_wide():
    ref = BitFieldRef(name="ref", reg_offset=0, slice_width=3, field_offset=0)
    errors = list(ref.validate())
    assert any("No child bitfield present" in str(e) for e in errors)

    bf = BitField(name="bf", width=2, reset_val=0, access="RW")
    ref._add(bf)
    errors = list(ref.validate())
    assert any("Slice width is larger" in str(e) for e in errors)


def test_register_value_roundtrip_and_reset():
    reg, ref, bf = _make_register_with_refs()
    reg.value = 0xAB
    assert reg.value == 0xAB

    bf.value = 0x00
    reg.reset()
    assert bf.value == bf.reset_val


def test_register_validate_detects_empty_and_non_bitfieldref_children():
    reg = Register(name="reg", width=8)
    errors = list(reg.validate())
    assert any("No bitfieldrefs present" in str(e) for e in errors)

    reg._children = {"x": object()}
    errors = list(reg.validate())
    assert any("not of typeBitFieldRef" in str(e) or "not of type BitFieldRef" in str(e) for e in errors)


def test_register_iterates_bitfieldrefs_sorted_by_reg_offset():
    bf1 = BitField(name="bf1", width=8, reset_val=0, access="RW")
    bf2 = BitField(name="bf2", width=8, reset_val=0, access="RW")
    r1 = BitFieldRef(name="r1", reg_offset=8, slice_width=8, field_offset=0)
    r2 = BitFieldRef(name="r2", reg_offset=0, slice_width=8, field_offset=0)
    r1._add(bf1)
    r2._add(bf2)

    reg = Register(name="reg", width=16)
    _add_child(reg, r1)
    _add_child(reg, r2)

    assert [c.name for c in reg] == ["r2", "r1"]


def test_registermap_get_dimensions_and_block_dimensions():
    assert get_dimensions(0x1000, 0x1004, 16) == (4, 6, 2)

    rm = RegisterMap(name="rm")
    r1 = Register(name="r1", width=8)
    r2 = Register(name="r2", width=16)
    r1.address = 0x1000
    r2.address = 0x1004
    _add_child(rm, r1)
    _add_child(rm, r2)

    assert rm._block_dimensions == (0x1000, 6)


def test_registermap_read_updates_register_values_from_block_io():
    rm = RegisterMap(name="rm")
    r1 = Register(name="r1", width=8)
    r2 = Register(name="r2", width=16)
    r1.address = 0x1000
    r2.address = 0x1002
    _add_child(rm, r1)
    _add_child(rm, r2)

    def block_read(address, size):
        assert address == 0x1000
        assert size == 4
        return bytes([0x11, 0x22, 0x33, 0x44])

    rm._block_read_func = block_read
    data = rm.read()

    assert data == bytes([0x11, 0x22, 0x33, 0x44])
    assert r1.value == 0x11
    assert r2.value == 0x4433


def test_registermap_write_builds_bytearray_when_no_data_provided():
    rm = RegisterMap(name="rm")
    r1 = Register(name="r1", width=8)
    r2 = Register(name="r2", width=16)
    r1.address = 0x1000
    r2.address = 0x1002
    r1.value = 0xAA
    r2.value = 0xBEEF
    _add_child(rm, r1)
    _add_child(rm, r2)

    captured = {}

    def block_write(address, data):
        captured["address"] = address
        captured["data"] = bytes(data)

    rm._block_write_func = block_write
    rm.write()

    assert captured["address"] == 0x1000
    assert captured["data"] == bytes([0xAA, 0x00, 0xEF, 0xBE])


def test_registermap_write_with_partial_input_updates_registers_until_data_exhausted():
    rm = RegisterMap(name="rm")
    r1 = Register(name="r1", width=8)
    r2 = Register(name="r2", width=16)
    r1.address = 0x1000
    r2.address = 0x1002
    _add_child(rm, r1)
    _add_child(rm, r2)

    captured = {}

    def block_write(address, data):
        captured["address"] = address
        captured["data"] = bytes(data)

    rm._block_write_func = block_write

    rm.write(data=bytes([0x12, 0x34]))

    assert r1.value == 0x12
    assert r2.value == 0
    assert captured["address"] == 0x1000
    assert captured["data"] == b"\x12\x34"
