import pytest
from math import ceil

from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.Register import Register
from r_map.RegisterMap import RegisterMap, get_dimensions
from r_map.ValidationError import ValidationError


def _attach(parent, child):
    """Attach a child node to a parent, using the public API if available."""
    if hasattr(parent, 'add_child'):
        parent.add_child(child)
    else:
        parent._children[child.name] = child
        child._parent = parent


def _make_split_reg(lo_reset=0, hi_reset=0):
    """8-bit register with two 4-bit fields: lo at bits [3:0], hi at bits [7:4]."""
    reg = Register(name='reg', address=0x0, width=8)
    lo_bf = BitField(name='lo_bf', width=4, reset_val=lo_reset)
    hi_bf = BitField(name='hi_bf', width=4, reset_val=hi_reset)
    lo = BitFieldRef(name='lo', reg_offset=0, slice_width=4)
    hi = BitFieldRef(name='hi', reg_offset=4, slice_width=4)
    lo._add(lo_bf)
    hi._add(hi_bf)
    _attach(reg, lo)
    _attach(reg, hi)
    return reg, lo_bf, hi_bf


def _make_full_reg(name, address, width, value=0):
    """Register with a single full-width BitFieldRef, set to value."""
    reg = Register(name=name, address=address, width=width)
    bf = BitField(name=name + '_bf', width=width)
    ref = BitFieldRef(name=name + '_ref', reg_offset=0, slice_width=width)
    ref._add(bf)
    _attach(reg, ref)
    reg.value = value
    return reg


# ---------------------------------------------------------------- BitField

def test_bitfield_defaults():
    bf = BitField(name='b')
    assert bf.width == 1
    assert bf.reset_val == 0
    assert bf.value == 0
    assert bf.mask == 1


def test_bitfield_rejects_zero_width():
    with pytest.raises(ValueError):
        BitField(name='b', width=0)


def test_bitfield_value_masked_to_width():
    bf = BitField(name='b', width=4)
    bf.value = 0x1F
    assert bf.value == 0xF


def test_bitfield_reset_restores_reset_val():
    bf = BitField(name='b', width=4, reset_val=5)
    bf.value = 2
    bf.reset()
    assert bf.value == 5


def test_bitfield_unknown_string_raises():
    bf = BitField(name='b', width=4)
    with pytest.raises(ValueError):
        bf.value = 'NOPE'


def test_bitfield_annotation_falls_back_to_hex():
    bf = BitField(name='b', width=8)
    bf.value = 0xAB
    assert bf.annotation == '0xab'


def test_bitfield_write_without_parent_raises():
    bf = BitField(name='b', width=4)
    with pytest.raises(ValueError):
        bf.write(1)


def test_bitfield_validate_clean_without_enumerations():
    bf = BitField(name='b', width=4)
    assert list(bf.validate()) == []


# ---------------------------------------------------------- BitFieldRef

def test_bitfieldref_negative_slice_width_raises():
    with pytest.raises(ValueError):
        BitFieldRef(name='r', reg_offset=0, slice_width=-1)


def test_bitfieldref_add_takes_width_from_bitfield():
    ref = BitFieldRef(name='r', reg_offset=0)
    bf = BitField(name='b', width=6)
    ref._add(bf)
    assert ref.slice_width == 6
    assert ref.mask == 0x3F


def test_bitfieldref_add_rejects_non_bitfield():
    ref = BitFieldRef(name='r', reg_offset=0, slice_width=4)
    with pytest.raises(ValueError):
        ref._add(object())


def test_bitfieldref_value_getter_extracts_slice():
    bf = BitField(name='b', width=8)
    bf.value = 0xAB
    ref = BitFieldRef(name='r', reg_offset=4, slice_width=4, field_offset=4)
    ref._add(bf)
    # bits [7:4] of 0xAB are 0xA, shifted into reg_offset 4 -> 0xA0
    assert ref.value == 0xA0


def test_bitfieldref_value_setter_preserves_other_bits():
    bf = BitField(name='b', width=8)
    bf.value = 0xF0
    ref = BitFieldRef(name='r', reg_offset=0, slice_width=4, field_offset=0)
    ref._add(bf)
    ref.value = 0x05
    assert bf.value == 0xF5


def test_bitfieldref_reset_val_uses_reset_value():
    bf = BitField(name='b', width=8, reset_val=0xC3)
    ref = BitFieldRef(name='r', reg_offset=0, slice_width=4, field_offset=4)
    ref._add(bf)
    assert ref.reset_val == 0xC


def test_bitfieldref_validate_flags_oversized_slice():
    bf = BitField(name='b', width=4)
    ref = BitFieldRef(name='r', reg_offset=0, slice_width=8)
    ref._add(bf)
    errors = [e for e in ref.validate() if isinstance(e, ValidationError)]
    assert len(errors) >= 1


# -------------------------------------------------------------- Register

def test_register_value_combines_fields():
    reg, lo_bf, hi_bf = _make_split_reg()
    lo_bf.value = 0x5
    hi_bf.value = 0xA
    assert reg.value == 0xA5


def test_register_value_setter_distributes():
    reg, lo_bf, hi_bf = _make_split_reg()
    reg.value = 0xA5
    assert lo_bf.value == 0x5
    assert hi_bf.value == 0xA


def test_register_reset_val_combines():
    reg, _, _ = _make_split_reg(lo_reset=3, hi_reset=2)
    assert reg.reset_val == 0x23


def test_register_reset_restores_fields():
    reg, lo_bf, hi_bf = _make_split_reg(lo_reset=3, hi_reset=2)
    lo_bf.value = 0
    hi_bf.value = 0xF
    reg.reset()
    assert lo_bf.value == 3
    assert hi_bf.value == 2


def test_register_validate_flags_overlap():
    reg = Register(name='reg', address=0x0, width=8)
    a_bf = BitField(name='a_bf', width=4)
    b_bf = BitField(name='b_bf', width=4)
    a = BitFieldRef(name='a', reg_offset=0, slice_width=4)
    b = BitFieldRef(name='b', reg_offset=2, slice_width=4)
    a._add(a_bf)
    b._add(b_bf)
    _attach(reg, a)
    _attach(reg, b)
    errors = [e for e in reg.validate() if isinstance(e, ValidationError)]
    assert len(errors) >= 1


def test_register_access_joins_unique_sorted():
    reg = Register(name='reg', address=0x0, width=8)
    a_bf = BitField(name='a_bf', width=4, access='RW')
    b_bf = BitField(name='b_bf', width=4, access='RO')
    a = BitFieldRef(name='a', reg_offset=0, slice_width=4)
    b = BitFieldRef(name='b', reg_offset=4, slice_width=4)
    a._add(a_bf)
    b._add(b_bf)
    _attach(reg, a)
    _attach(reg, b)
    assert reg.access == 'RO|RW'


# ----------------------------------------------------------- RegisterMap

def test_get_dimensions():
    start_index, end_index, num_bytes = get_dimensions(0x100, 0x104, 32)
    assert start_index == 4
    assert end_index == 8
    assert num_bytes == 4


def test_register_map_block_dimensions_no_registers_raises():
    rm = RegisterMap(name='empty')
    with pytest.raises(ValueError):
        _ = rm._block_dimensions


def test_register_map_block_dimensions_span():
    rm = RegisterMap(name='rm')
    _attach(rm, _make_full_reg('a', 0, 8))
    _attach(rm, _make_full_reg('b', 4, 32))
    assert rm._block_dimensions == (0, 8)


def test_register_map_write_builds_block():
    rm = RegisterMap(name='rm')
    _attach(rm, _make_full_reg('r0', 0, 8, 0x12))
    _attach(rm, _make_full_reg('r1', 2, 16, 0x3456))

    calls = []
    rm._block_write_func = lambda addr, data: calls.append((addr, bytes(data)))
    rm.write()

    assert calls == [(0, b'\x12\x00\x56\x34')]


def test_register_map_read_populates_registers():
    rm = RegisterMap(name='rm')
    r0 = _make_full_reg('r0', 0, 8)
    r1 = _make_full_reg('r1', 2, 16)
    _attach(rm, r0)
    _attach(rm, r1)

    data = b'\x12\x00\x56\x34'
    rm._block_read_func = lambda address, size: data

    result = rm.read()

    assert result == data
    assert r0.value == 0x12
    assert r1.value == 0x3456


def test_register_map_write_explicit_data():
    rm = RegisterMap(name='rm')
    r0 = _make_full_reg('r0', 0, 8)
    r1 = _make_full_reg('r1', 2, 16)
    _attach(rm, r0)
    _attach(rm, r1)

    calls = []
    rm._block_write_func = lambda addr, data: calls.append((addr, bytes(data)))

    data = b'\x01\x02\x03\x04'
    rm.write(data)

    assert r0.value == 0x01
    assert r1.value == 0x0403
    assert calls == [(0, data)]
