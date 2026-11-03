import pytest

from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.Register import Register
from r_map.RegisterMap import RegisterMap, get_dimensions
from r_map.ValidationError import ValidationError


def make_bitfield(name="bf", width=8, reset_val=0x12, access="RW"):
    return BitField(name=name, width=width, reset_val=reset_val, access=access)


def make_register_with_bitfield(
    reg_name="reg",
    bf_name="bf",
    bf_width=8,
    bf_reset=0x12,
    reg_address=0x1000,
    reg_offset=0,
    slice_width=0,
    field_offset=0,
):
    reg = Register(name=reg_name, address=reg_address, width=bf_width)
    ref = BitFieldRef(
        name=f"{bf_name}_ref",
        reg_offset=reg_offset,
        slice_width=slice_width,
        field_offset=field_offset,
    )
    bf = make_bitfield(name=bf_name, width=bf_width, reset_val=bf_reset)
    reg._add(ref)
    ref._add(bf)
    return reg, ref, bf


def make_register_map(regs, name="map"):
    rm = RegisterMap(name=name)
    for reg in regs:
        rm._add(reg)
    return rm


def test_bitfield_rejects_invalid_width_and_reset_val():
    with pytest.raises(ValueError):
        BitField(width=0)
    with pytest.raises(ValueError):
        BitField(width=1, reset_val=-1)


def test_bitfield_masks_reset_value_to_width():
    bf = BitField(name="bf", width=4, reset_val=0x1F)
    assert bf.reset_val == 0xF
    assert bf.value == 0xF


def test_bitfield_value_accepts_int_str_and_enumeration():
    bf = BitField(name="bf", width=8, reset_val=0)
    bf.value = 0x123
    assert bf.value == 0x23

    enum = type("EnumLike", (), {"name": "MODE_A", "value": 7})()
    bf._children[enum.name] = enum
    bf.value = "MODE_A"
    assert bf.value == 7

    import r_map

    real_enum = r_map.Enumeration
    bf.value = real_enum("MODE_B", 3)
    assert bf.value == 3


def test_bitfield_value_rejects_unknown_string():
    bf = BitField(name="bf", width=8, reset_val=0)
    with pytest.raises(ValueError):
        bf.value = "DOES_NOT_EXIST"


def test_bitfield_annotation_uses_matching_enum_name_or_hex():
    bf = BitField(name="bf", width=8, reset_val=0)
    e0 = type("EnumLike", (), {"name": "ZERO", "value": 0})()
    bf._children[e0.name] = e0
    assert bf.annotation == "ZERO"
    bf.value = 5
    assert bf.annotation == hex(5)


def test_bitfield_validate_detects_duplicate_enum_values():
    bf = BitField(name="bf", width=8, reset_val=0)
    e1 = type("EnumLike", (), {"name": "A", "value": 1})()
    e2 = type("EnumLike", (), {"name": "B", "value": 1})()
    bf._children["A"] = e1
    bf._children["B"] = e2
    errs = list(bf.validate())
    assert any(isinstance(e, ValidationError) for e in errs)
    assert any("same value" in str(e) for e in errs)


def test_bitfieldref_rejects_invalid_slice_and_field_offsets():
    with pytest.raises(ValueError):
        BitFieldRef(name="r", reg_offset=0, slice_width=-1)
    with pytest.raises(ValueError):
        BitFieldRef(name="r", reg_offset=0, field_offset=-1)


def test_bitfieldref_value_roundtrip_with_partial_slice():
    reg = Register(name="reg", address=0x0, width=16)
    ref = BitFieldRef(name="ref", reg_offset=4, slice_width=4, field_offset=8)
    bf = BitField(name="bf", width=16, reset_val=0xABCD)
    reg._add(ref)
    ref._add(bf)

    assert ref.value == ((0xABCD >> 8) & 0xF) << 4

    ref.value = 0xA0
    assert bf.value == (0xABCD & ~(0xF << 8)) | (0xA << 8)


def test_bitfieldref_validate_without_bitfield_reports_error():
    ref = BitFieldRef(name="ref", reg_offset=0, slice_width=4, field_offset=0)
    errs = list(ref.validate())
    assert any("No child bitfield present" in str(e) for e in errs)


def test_register_value_and_reset_propagate_to_bitfieldrefs():
    reg, ref, bf = make_register_with_bitfield(bf_width=8, bf_reset=0x12)
    assert reg.value == 0x12
    reg.value = 0x34
    assert bf.value == 0x34
    reg.reset()
    assert bf.value == 0x12


def test_register_iteration_sorts_by_reg_offset():
    reg = Register(name="reg", address=0x0, width=16)
    ref_hi = BitFieldRef(name="hi", reg_offset=8, slice_width=8, field_offset=0)
    ref_lo = BitFieldRef(name="lo", reg_offset=0, slice_width=8, field_offset=0)
    bf_hi = BitField(name="bf_hi", width=8, reset_val=0x11)
    bf_lo = BitField(name="bf_lo", width=8, reset_val=0x22)
    reg._add(ref_hi)
    reg._add(ref_lo)
    ref_hi._add(bf_hi)
    ref_lo._add(bf_lo)

    assert [c.name for c in reg] == ["lo", "hi"]


def test_register_validate_detects_no_children_and_overlaps():
    empty = Register(name="empty", address=0x0, width=8)
    errs = list(empty.validate())
    assert any("No bitfieldrefs present" in str(e) for e in errs)

    reg = Register(name="reg", address=0x0, width=16)
    r1 = BitFieldRef(name="r1", reg_offset=0, slice_width=8, field_offset=0)
    r2 = BitFieldRef(name="r2", reg_offset=4, slice_width=8, field_offset=0)
    bf1 = BitField(name="bf1", width=8, reset_val=0)
    bf2 = BitField(name="bf2", width=8, reset_val=0)
    reg._add(r1)
    reg._add(r2)
    r1._add(bf1)
    r2._add(bf2)

    errs = list(reg.validate())
    assert any("overlaps" in str(e) for e in errs)


def test_register_read_uses_io_function_and_updates_value():
    reg, ref, bf = make_register_with_bitfield(bf_width=16, bf_reset=0x0000)
    calls = []

    def read_func(addr):
        calls.append(addr)
        return 0xBEEF

    reg._reg_read_func = read_func
    val = reg.read()
    assert val == 0xBEEF
    assert reg.value == 0xBEEF
    assert calls == [reg.address]


def test_register_write_uses_io_function_and_default_value():
    reg, ref, bf = make_register_with_bitfield(bf_width=16, bf_reset=0x1234)
    calls = []

    def write_func(addr, val):
        calls.append((addr, val))

    reg._reg_write_func = write_func
    reg.write()
    assert calls == [(reg.address, 0x1234)]

    reg.write(0xABCD)
    assert bf.value == 0xABCD
    assert calls[-1] == (reg.address, 0xABCD)


def test_registermap_get_dimensions_and_block_read_write_roundtrip():
    reg1 = Register(name="r1", address=0x1000, width=16)
    ref1 = BitFieldRef(name="r1ref", reg_offset=0, slice_width=16, field_offset=0)
    bf1 = BitField(name="bf1", width=16, reset_val=0x1122)
    reg1._add(ref1)
    ref1._add(bf1)

    reg2 = Register(name="r2", address=0x1002, width=16)
    ref2 = BitFieldRef(name="r2ref", reg_offset=0, slice_width=16, field_offset=0)
    bf2 = BitField(name="bf2", width=16, reset_val=0x3344)
    reg2._add(ref2)
    ref2._add(bf2)

    rm = make_register_map([reg1, reg2])

    assert get_dimensions(0x1000, 0x1002, 16) == (2, 4, 2)

    read_calls = []
    write_calls = []

    def block_read(addr, size):
        read_calls.append((addr, size))
        return (0x33441122).to_bytes(4, byteorder="little")

    def block_write(addr, data):
        write_calls.append((addr, bytes(data)))

    rm._block_read_func = block_read
    rm._block_write_func = block_write

    data = rm.read()
    assert read_calls == [(0x1000, 4)]
    assert data == (0x33441122).to_bytes(4, byteorder="little")
    assert reg1.value == 0x1122
    assert reg2.value == 0x3344

    reg1.value = 0xA1A2
    reg2.value = 0xB1B2
    rm.write()
    assert write_calls == [(0x1000, bytes([0xA2, 0xA1, 0xB2, 0xB1]))]


def test_registermap_write_partial_data_updates_only_within_bounds():
    reg1 = Register(name="r1", address=0x2000, width=16)
    ref1 = BitFieldRef(name="r1ref", reg_offset=0, slice_width=16, field_offset=0)
    bf1 = BitField(name="bf1", width=16, reset_val=0x1111)
    reg1._add(ref1)
    ref1._add(bf1)

    reg2 = Register(name="r2", address=0x2002, width=16)
    ref2 = BitFieldRef(name="r2ref", reg_offset=0, slice_width=16, field_offset=0)
    bf2 = BitField(name="bf2", width=16, reset_val=0x2222)
    reg2._add(ref2)
    ref2._add(bf2)

    rm = make_register_map([reg1, reg2])

    calls = []

    def block_write(addr, data):
        calls.append((addr, bytes(data)))

    rm._block_write_func = block_write

    rm.write(bytearray([0xAA, 0xBB]))
    assert bf1.value == 0xBBAA
    assert bf2.value == 0x2222
    assert calls == [(0x2000, bytes([0xAA, 0xBB]))]
