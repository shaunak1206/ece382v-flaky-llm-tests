import pytest

from r_map.Register import Register
from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.RegisterMap import RegisterMap, get_dimensions
from r_map.ValidationError import ValidationError
from r_map.Enumeration import Enumeration


def make_bitfield(name="bf", width=8, reset_val=0, access="RW", value=None):
    bf = BitField(width=width, reset_val=reset_val, access=access, name=name)
    if value is not None:
        bf.value = value
    return bf


def make_register(name="reg", address=0, width=32):
    return Register(name=name, address=address, width=width)


def attach_ref_to_bf(ref, bf):
    ref._add(bf)
    return ref


def test_get_dimensions_basic():
    assert get_dimensions(100, 104, 16) == (4, 6, 2)
    assert get_dimensions(0, 0, 1) == (0, 1, 1)


def test_bitfield_init_masks_reset_value():
    bf = BitField(width=4, reset_val=0x1F, access="RW", name="bf")
    assert bf.width == 4
    assert bf.reset_val == 0xF
    assert bf.value == 0xF
    assert bf.mask == 0xF


def test_bitfield_rejects_invalid_width_and_reset_val():
    with pytest.raises(ValueError):
        BitField(width=0, reset_val=0, name="bf")
    with pytest.raises(ValueError):
        BitField(width=1, reset_val=-1, name="bf")


def test_bitfield_value_int_masks_to_width():
    bf = BitField(width=5, reset_val=0, name="bf")
    bf.value = 0x3F
    assert bf.value == 0x1F


def test_bitfield_value_enum_and_string_assignment():
    bf = BitField(width=3, reset_val=0, name="bf")
    e0 = Enumeration(name="ZERO", value=0)
    e5 = Enumeration(name="FIVE", value=5)
    bf._children = {"ZERO": e0, "FIVE": e5}

    bf.value = e5
    assert bf.value == 5

    bf.value = "ZERO"
    assert bf.value == 0


def test_bitfield_value_string_unknown_raises():
    bf = BitField(width=3, reset_val=0, name="bf")
    bf._children = {}
    with pytest.raises(ValueError):
        bf.value = "MISSING"


def test_bitfield_annotation_uses_matching_enumeration_name():
    bf = BitField(width=3, reset_val=0, name="bf")
    e1 = Enumeration(name="ONE", value=1)
    bf._children = {"ONE": e1}
    bf.value = 1
    assert bf.annotation == "ONE"


def test_bitfield_validate_detects_duplicate_enumeration_values():
    bf = BitField(width=3, reset_val=0, name="bf")
    e1 = Enumeration(name="A", value=1)
    e2 = Enumeration(name="B", value=1)
    bf._children = {"A": e1, "B": e2}

    errors = list(bf.validate())
    assert any(isinstance(err, ValidationError) for err in errors)
    assert any("same value" in str(err) for err in errors)


def test_bitfieldref_init_rejects_negative_offsets():
    with pytest.raises(ValueError):
        BitFieldRef(reg_offset=-1, slice_width=1, field_offset=0, name="ref")
    with pytest.raises(ValueError):
        BitFieldRef(reg_offset=0, slice_width=1, field_offset=-1, name="ref")


def test_bitfieldref_add_sets_reference_and_defaults_slice_width():
    bf = BitField(width=8, reset_val=0, name="bf")
    ref = BitFieldRef(reg_offset=2, slice_width=0, field_offset=1, name="ref")

    ref._add(bf)

    assert ref.bf is bf
    assert ref.slice_width == 8
    assert ref.mask == 0xFF
    assert ref in bf._references


def test_bitfieldref_value_get_and_set_round_trip():
    bf = BitField(width=8, reset_val=0xAA, name="bf")
    ref = BitFieldRef(reg_offset=2, slice_width=3, field_offset=1, name="ref")
    ref._add(bf)

    assert ref.value == ((0xAA >> 1) & 0x7) << 2

    ref.value = 0b11100
    expected = (0xAA & ~(0x7 << 1)) | (0x7 << 1)
    assert bf.value == expected


def test_bitfieldref_validate_without_child_reports_error():
    ref = BitFieldRef(reg_offset=0, slice_width=1, field_offset=0, name="ref")
    errors = list(ref.validate())
    assert any("No child bitfield present" in str(err) for err in errors)


def test_register_value_and_reset_use_child_bitfieldrefs():
    bf = BitField(width=8, reset_val=0x12, name="bf")
    ref = BitFieldRef(reg_offset=0, slice_width=8, field_offset=0, name="ref")
    ref._add(bf)

    reg = make_register(address=0x10, width=8)
    reg._children = {"ref": ref}
    ref._parent = reg

    assert reg.value == 0x12
    reg.value = 0xAB
    assert bf.value == 0xAB

    bf.value = 0x34
    reg.reset()
    assert bf.value == 0x12


def test_register_access_combines_unique_child_access_modes():
    bf1 = BitField(width=4, reset_val=0, access="RW", name="bf1")
    bf2 = BitField(width=4, reset_val=0, access="RO", name="bf2")
    ref1 = BitFieldRef(reg_offset=0, slice_width=4, field_offset=0, name="ref1")
    ref2 = BitFieldRef(reg_offset=4, slice_width=4, field_offset=0, name="ref2")
    ref1._add(bf1)
    ref2._add(bf2)

    reg = make_register(address=0x20, width=8)
    reg._children = {"ref1": ref1, "ref2": ref2}
    ref1._parent = reg
    ref2._parent = reg

    assert reg.access == "RO|RW"


def test_register_validate_detects_overlap_and_non_bitfieldref_child():
    bf = BitField(width=8, reset_val=0, name="bf")
    ref1 = BitFieldRef(reg_offset=0, slice_width=4, field_offset=0, name="ref1")
    ref2 = BitFieldRef(reg_offset=2, slice_width=4, field_offset=0, name="ref2")
    ref1._add(bf)
    ref2._add(bf)

    reg = make_register(address=0x30, width=8)
    reg._children = {"ref1": ref1, "ref2": ref2, "bad": object()}
    ref1._parent = reg
    ref2._parent = reg

    errors = list(reg.validate())
    assert any("not of typeBitFieldRef" in str(err) for err in errors)
    assert any("overlaps" in str(err) for err in errors)


def test_registermap_block_dimensions_and_read_write_round_trip(monkeypatch):
    rm = RegisterMap(name="map")
    r1 = make_register(name="r1", address=0x100, width=8)
    r2 = make_register(name="r2", address=0x102, width=16)
    rm._children = {"r1": r1, "r2": r2}

    # Monkeypatch IO functions on the instance.
    read_calls = []
    write_calls = []

    def block_read(addr, size):
        read_calls.append((addr, size))
        # r1 at offset 0 => 0x11, r2 at offset 2 => 0x2233 little-endian
        return bytes([0x11, 0x00, 0x33, 0x22])

    def block_write(addr, data):
        write_calls.append((addr, bytes(data)))

    rm._block_read_func = block_read
    rm._block_write_func = block_write

    assert rm._block_dimensions == (0x100, 4)

    data = rm.read()
    assert data == bytes([0x11, 0x00, 0x33, 0x22])
    assert r1.value == 0x11
    assert r2.value == 0x2233
    assert read_calls == [(0x100, 4)]

    r1.value = 0x44
    r2.value = 0x5566
    rm.write()

    assert write_calls == [(0x100, bytes([0x44, 0x00, 0x66, 0x55]))]


def test_registermap_write_partial_data_updates_prefix_only():
    rm = RegisterMap(name="map")
    r1 = make_register(name="r1", address=0x10, width=8)
    r2 = make_register(name="r2", address=0x12, width=16)
    rm._children = {"r1": r1, "r2": r2}

    captured = {}

    def block_write(addr, data):
        captured["addr"] = addr
        captured["data"] = bytes(data)

    rm._block_write_func = block_write

    # Partial write: only the first register should be updated from the data.
    r1.value = 0xAA
    r2.value = 0xBBBB
    rm.write(data=bytes([0x55, 0x00, 0x00, 0x00]))

    assert captured["addr"] == 0x10
    assert captured["data"] == bytes([0x55, 0x00, 0x00, 0x00])
    assert r1.value == 0x55
    assert r2.value == 0xBBBB


def test_registermap_write_raises_when_empty():
    rm = RegisterMap(name="empty")
    rm._children = {}
    rm._block_dimensions = (0x0, 0)
    rm._block_write_func = lambda addr, data: None

    with pytest.raises(ValueError):
        rm.write()
