import pytest

from r_map.Register import Register
from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.RegisterMap import RegisterMap, get_dimensions
from r_map.ValidationError import ValidationError


def make_register(name="reg", address=0, width=8):
    return Register(name=name, address=address, width=width)


def make_bitfield(name="bf", width=8, reset_val=0, access="RW"):
    return BitField(name=name, width=width, reset_val=reset_val, access=access)


def make_ref(name="ref", reg_offset=0, slice_width=0, field_offset=0):
    return BitFieldRef(
        name=name,
        reg_offset=reg_offset,
        slice_width=slice_width,
        field_offset=field_offset,
    )


def test_get_dimensions_basic():
    assert get_dimensions(0x100, 0x104, 16) == (4, 6, 2)


def test_bitfield_constructor_masks_reset_value():
    bf = make_bitfield(width=4, reset_val=0x1F)
    assert bf.mask == 0xF
    assert bf.reset_val == 0xF
    assert bf.value == 0xF


def test_bitfield_rejects_invalid_width_and_reset_val():
    with pytest.raises(ValueError):
        make_bitfield(width=0)
    with pytest.raises(ValueError):
        make_bitfield(width=1, reset_val=-1)


def test_bitfield_reset_restores_reset_value():
    bf = make_bitfield(width=8, reset_val=0x12)
    bf.value = 0xAB
    bf.reset()
    assert bf.value == 0x12


def test_bitfield_value_masks_int_assignment():
    bf = make_bitfield(width=4)
    bf.value = 0x3F
    assert bf.value == 0xF


def test_bitfield_validate_detects_duplicate_enumeration_values():
    bf = make_bitfield(width=2)
    # Add two enumerations with the same value; enumeration API is part of project.
    from r_map.Enumeration import Enumeration

    e1 = Enumeration(name="A", value=1)
    e2 = Enumeration(name="B", value=1)
    bf._add(e1)
    bf._add(e2)

    errors = list(bf.validate())
    assert any(isinstance(err, ValidationError) for err in errors)
    assert any("same value" in str(err) for err in errors)


def test_bitfieldref_constructor_sets_mask_and_defaults():
    bf = make_bitfield(width=8)
    ref = make_ref(reg_offset=2, slice_width=3, field_offset=1)
    ref._add(bf)

    assert ref.slice_width == 3
    assert ref.mask == 0b111
    assert ref.bf is bf


def test_bitfieldref_value_reads_and_writes_subfield():
    bf = make_bitfield(width=8, reset_val=0)
    ref = make_ref(reg_offset=2, slice_width=3, field_offset=1)
    ref._add(bf)

    ref.value = 0b101 << 2
    # Written into bits [3:1] of bf, so underlying value is 0b1010
    assert bf.value == 0b1010
    assert ref.value == 0b101 << 2


def test_bitfieldref_validate_without_child_reports_error():
    ref = make_ref(reg_offset=0, slice_width=1, field_offset=0)
    errors = list(ref.validate())
    assert any("No child bitfield present" in str(err) for err in errors)


def test_bitfieldref_validate_slice_too_wide_reports_error():
    bf = make_bitfield(width=4)
    ref = make_ref(reg_offset=0, slice_width=8, field_offset=0)
    ref._add(bf)
    errors = list(ref.validate())
    assert any("larger than bitfield's width" in str(err) for err in errors)


def test_register_value_property_aggregates_and_distributes_to_refs():
    bf = make_bitfield(width=8, reset_val=0)
    low = make_ref(name="low", reg_offset=0, slice_width=4, field_offset=0)
    high = make_ref(name="high", reg_offset=4, slice_width=4, field_offset=4)
    low._add(bf)
    high._add(bf)

    reg = make_register(width=8)
    reg._add(low)
    reg._add(high)

    reg.value = 0xA5
    assert bf.value == 0xA5
    assert reg.value == 0xA5


def test_register_iter_returns_refs_sorted_by_reg_offset():
    bf = make_bitfield(width=8)
    r1 = make_ref(name="r1", reg_offset=4, slice_width=4, field_offset=4)
    r0 = make_ref(name="r0", reg_offset=0, slice_width=4, field_offset=0)
    r1._add(bf)
    r0._add(bf)

    reg = make_register(width=8)
    reg._add(r1)
    reg._add(r0)

    assert [c.name for c in reg] == ["r0", "r1"]


def test_register_validate_reports_missing_children():
    reg = make_register(width=8)
    errors = list(reg.validate())
    assert any("No bitfieldrefs present" in str(err) for err in errors)


def test_register_validate_reports_non_bitfieldref_child():
    reg = make_register(width=8)
    bf = make_bitfield(width=8)
    reg._add(bf)

    errors = list(reg.validate())
    assert any("is not of type" in str(err) for err in errors)


def test_register_validate_reports_overlapping_refs():
    bf = make_bitfield(width=16)
    r0 = make_ref(name="r0", reg_offset=0, slice_width=8, field_offset=0)
    r1 = make_ref(name="r1", reg_offset=4, slice_width=8, field_offset=8)
    r0._add(bf)
    r1._add(bf)

    reg = make_register(width=16)
    reg._add(r0)
    reg._add(r1)

    errors = list(reg.validate())
    assert any("overlaps" in str(err) for err in errors)


def test_register_map_read_and_write_round_trip_partial_block():
    reg0 = make_register(name="r0", address=0x100, width=8)
    reg1 = make_register(name="r1", address=0x101, width=8)
    bf0 = make_bitfield(name="bf0", width=8, reset_val=0)
    bf1 = make_bitfield(name="bf1", width=8, reset_val=0)
    ref0 = make_ref(name="ref0", reg_offset=0, slice_width=8, field_offset=0)
    ref1 = make_ref(name="ref1", reg_offset=0, slice_width=8, field_offset=0)
    ref0._add(bf0)
    ref1._add(bf1)
    reg0._add(ref0)
    reg1._add(ref1)

    rm = RegisterMap(name="map")
    rm._add(reg0)
    rm._add(reg1)

    reads = []
    writes = []

    def block_read(addr, size):
        reads.append((addr, size))
        return bytes([0x11, 0x22])

    def block_write(addr, data):
        writes.append((addr, bytes(data)))

    rm._block_read_func = block_read
    rm._block_write_func = block_write

    data = rm.read()
    assert data == bytes([0x11, 0x22])
    assert reg0.value == 0x11
    assert reg1.value == 0x22

    reg0.value = 0x33
    reg1.value = 0x44
    rm.write()

    assert reads == [(0x100, 2)]
    assert writes == [(0x100, bytes([0x33, 0x44]))]


def test_register_map_write_with_supplied_data_updates_registers():
    reg0 = make_register(name="r0", address=0x200, width=8)
    reg1 = make_register(name="r1", address=0x201, width=8)
    bf0 = make_bitfield(name="bf0", width=8, reset_val=0)
    bf1 = make_bitfield(name="bf1", width=8, reset_val=0)
    ref0 = make_ref(name="ref0", reg_offset=0, slice_width=8, field_offset=0)
    ref1 = make_ref(name="ref1", reg_offset=0, slice_width=8, field_offset=0)
    ref0._add(bf0)
    ref1._add(bf1)
    reg0._add(ref0)
    reg1._add(ref1)

    rm = RegisterMap(name="map")
    rm._add(reg0)
    rm._add(reg1)

    writes = []

    def block_write(addr, data):
        writes.append((addr, bytes(data)))

    rm._block_write_func = block_write

    rm.write(bytearray([0xAA, 0xBB]))
    assert reg0.value == 0xAA
    assert reg1.value == 0xBB
    assert writes == [(0x200, bytes([0xAA, 0xBB]))]
