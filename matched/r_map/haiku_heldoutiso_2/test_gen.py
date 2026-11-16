import pytest

import r_map
from r_map.RegisterMap import get_dimensions
from r_map.ValidationError import ValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_reg(address=0, width=8, name='reg'):
    return r_map.Register(name=name, address=address, width=width)


def add_field(reg, name, width, reg_offset, field_offset=0,
              slice_width=0, reset_val=0):
    """Create a BitField, wrap it in a BitFieldRef, and attach to reg."""
    bf = r_map.BitField(name=name, width=width, reset_val=reset_val)
    ref = r_map.BitFieldRef(name=name + '_ref', reg_offset=reg_offset,
                            slice_width=slice_width,
                            field_offset=field_offset)
    ref.add_child(bf)
    reg.add_child(ref)
    return bf, ref


def make_simple_reg(name, address, width=8):
    """Register with one full-width bitfield reference."""
    reg = r_map.Register(name=name, address=address, width=width)
    add_field(reg, name + '_bf', width, reg_offset=0)
    return reg


# ---------------------------------------------------------------------------
# BitField
# ---------------------------------------------------------------------------

def test_bitfield_rejects_zero_width():
    with pytest.raises(ValueError):
        r_map.BitField(name='bf', width=0)


def test_bitfield_rejects_negative_reset_val():
    with pytest.raises(ValueError):
        r_map.BitField(name='bf', width=4, reset_val=-1)


def test_bitfield_int_assignment_is_masked_to_width():
    bf = r_map.BitField(name='bf', width=4)
    bf.value = 0x1F
    assert bf.value == 0xF


def test_bitfield_reset_restores_reset_val():
    bf = r_map.BitField(name='bf', width=8, reset_val=0x5)
    bf.value = 0xAA
    bf.reset()
    assert bf.value == 0x5


def test_bitfield_set_by_enumeration_name():
    bf = r_map.BitField(name='bf', width=4)
    bf.add_child(r_map.Enumeration(name='ON', value=1))
    bf.add_child(r_map.Enumeration(name='OFF', value=0))
    bf.value = 'ON'
    assert bf.value == 1


def test_bitfield_unknown_enumeration_name_raises():
    bf = r_map.BitField(name='bf', width=4)
    bf.add_child(r_map.Enumeration(name='ON', value=1))
    with pytest.raises(ValueError):
        bf.value = 'NOPE'


def test_bitfield_unsupported_type_raises_not_implemented():
    bf = r_map.BitField(name='bf', width=4)
    with pytest.raises(NotImplementedError):
        bf.value = 1.5


def test_bitfield_annotation_uses_enum_name_or_hex():
    bf = r_map.BitField(name='bf', width=4)
    bf.add_child(r_map.Enumeration(name='ON', value=1))
    bf.value = 1
    assert bf.annotation == 'ON'
    bf.value = 7
    assert bf.annotation == hex(7)


def test_bitfield_validate_flags_duplicate_enum_values():
    bf = r_map.BitField(name='bf', width=4)
    bf.add_child(r_map.Enumeration(name='A', value=2))
    bf.add_child(r_map.Enumeration(name='B', value=2))
    errors = list(bf.validate())
    assert any(isinstance(e, ValidationError) for e in errors)


def test_bitfield_write_without_parent_register_raises():
    bf = r_map.BitField(name='orphan', width=4)
    with pytest.raises(ValueError):
        bf.write(1)


# ---------------------------------------------------------------------------
# BitFieldRef
# ---------------------------------------------------------------------------

def test_bitfieldref_rejects_negative_slice_width():
    with pytest.raises(ValueError):
        r_map.BitFieldRef(name='r', reg_offset=0, slice_width=-1)


def test_bitfieldref_rejects_negative_field_offset():
    with pytest.raises(ValueError):
        r_map.BitFieldRef(name='r', reg_offset=0, field_offset=-1)


def test_bitfieldref_zero_slice_width_takes_bitfield_width():
    bf = r_map.BitField(name='bf', width=6)
    ref = r_map.BitFieldRef(name='ref', reg_offset=0)
    ref.add_child(bf)
    assert ref.slice_width == 6


def test_bitfieldref_value_applies_field_and_reg_offset():
    bf = r_map.BitField(name='bf', width=8)
    ref = r_map.BitFieldRef(name='ref', reg_offset=8, slice_width=4,
                            field_offset=4)
    ref.add_child(bf)
    bf.value = 0xAB
    # bits [7:4] of 0xAB are 0xA, shifted to reg offset 8
    assert ref.value == 0xA00


def test_bitfieldref_validate_slice_wider_than_bitfield():
    bf = r_map.BitField(name='bf', width=4)
    ref = r_map.BitFieldRef(name='ref', reg_offset=0, slice_width=8)
    ref.add_child(bf)
    errors = list(ref.validate())
    assert any(isinstance(e, ValidationError) for e in errors)


def test_bitfieldref_validate_without_bitfield_reports_error():
    ref = r_map.BitFieldRef(name='ref', reg_offset=0, slice_width=4)
    errors = list(ref.validate())
    assert len(errors) >= 1
    assert any(isinstance(e, ValidationError) for e in errors)


def test_bitfieldref_second_reference_to_same_bitfield_is_alias():
    bf = r_map.BitField(name='shared', width=4)
    ref1 = r_map.BitFieldRef(name='ref1', reg_offset=0)
    ref2 = r_map.BitFieldRef(name='ref2', reg_offset=4)
    ref1.add_child(bf)
    ref2.add_child(bf)
    assert ref2._alias is True
    assert len(bf._references) == 2


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

def test_register_value_combines_bitfield_refs():
    reg = make_reg(width=8)
    bf_lo, _ = add_field(reg, 'lo', 4, reg_offset=0)
    bf_hi, _ = add_field(reg, 'hi', 4, reg_offset=4)
    bf_lo.value = 0x3
    bf_hi.value = 0x5
    assert reg.value == 0x53


def test_register_value_setter_distributes_to_fields():
    reg = make_reg(width=8)
    bf_lo, _ = add_field(reg, 'lo', 4, reg_offset=0)
    bf_hi, _ = add_field(reg, 'hi', 4, reg_offset=4)
    reg.value = 0xAB
    assert bf_lo.value == 0xB
    assert bf_hi.value == 0xA
    assert reg.value == 0xAB


def test_register_reset_val_combines_field_reset_values():
    reg = make_reg(width=8)
    add_field(reg, 'lo', 4, reg_offset=0, reset_val=1)
    add_field(reg, 'hi', 4, reg_offset=4, reset_val=2)
    assert reg.reset_val == 0x21


def test_register_reset_restores_all_fields():
    reg = make_reg(width=8)
    bf_lo, _ = add_field(reg, 'lo', 4, reg_offset=0, reset_val=1)
    bf_hi, _ = add_field(reg, 'hi', 4, reg_offset=4, reset_val=2)
    reg.value = 0xFF
    reg.reset()
    assert reg.value == 0x21


def test_register_iterates_refs_sorted_by_reg_offset():
    reg = make_reg(width=16)
    add_field(reg, 'c', 4, reg_offset=8)
    add_field(reg, 'a', 4, reg_offset=0)
    add_field(reg, 'b', 4, reg_offset=4)
    offsets = [child.reg_offset for child in reg]
    assert offsets == sorted(offsets)


def test_register_validate_no_children_reports_error():
    reg = make_reg()
    errors = list(reg.validate())
    assert any(isinstance(e, ValidationError) for e in errors)


def test_register_validate_detects_overlapping_refs():
    reg = make_reg(width=8)
    add_field(reg, 'a', 4, reg_offset=0, slice_width=4)
    add_field(reg, 'b', 4, reg_offset=2, slice_width=4)
    errors = list(reg.validate())
    assert any('overlaps' in str(e) for e in errors)


def test_register_validate_accepts_non_overlapping_refs():
    reg = make_reg(width=8)
    add_field(reg, 'a', 4, reg_offset=0, slice_width=4)
    add_field(reg, 'b', 4, reg_offset=4, slice_width=4)
    errors = list(reg.validate())
    assert not any('overlaps' in str(e) for e in errors)


def test_register_read_uses_read_func_and_returns_value():
    reg = make_simple_reg('r', address=0x10, width=8)
    seen = []

    def read_func(addr):
        seen.append(addr)
        return 0x5A

    reg._reg_read_func = read_func
    assert reg.read() == 0x5A
    assert reg.value == 0x5A
    assert seen == [0x10]


def test_register_write_with_value_sets_field_and_calls_write_func():
    reg = make_simple_reg('r', address=0x20, width=8)
    calls = []
    reg._reg_write_func = lambda addr, val: calls.append((addr, val))
    reg.write(0x34)
    assert reg.value == 0x34
    assert calls == [(0x20, 0x34)]


# ---------------------------------------------------------------------------
# RegisterMap
# ---------------------------------------------------------------------------

def test_get_dimensions_computes_slice_indices():
    start, end, nbytes = get_dimensions(0x100, 0x104, 32)
    assert (start, end, nbytes) == (4, 8, 4)


def test_registermap_block_dimensions_span_registers():
    rmap = r_map.RegisterMap(name='map')
    rmap.add_child(make_reg(address=0, width=32, name='r0'))
    rmap.add_child(make_reg(address=4, width=32, name='r1'))
    assert rmap._block_dimensions == (0, 8)


def test_registermap_block_dimensions_empty_raises():
    rmap = r_map.RegisterMap(name='empty')
    with pytest.raises(ValueError):
        _ = rmap._block_dimensions


def test_registermap_read_populates_registers():
    rmap = r_map.RegisterMap(name='map')
    r0 = make_simple_reg('r0', address=0, width=8)
    r1 = make_simple_reg('r1', address=1, width=8)
    rmap.add_child(r0)
    rmap.add_child(r1)
    rmap._block_read_func = lambda addr, size: bytes([0x11, 0x22])
    data = rmap.read()
    assert data == bytes([0x11, 0x22])
    assert r0.value == 0x11
    assert r1.value == 0x22


def test_registermap_write_without_data_packs_register_values():
    rmap = r_map.RegisterMap(name='map')
    r0 = make_simple_reg('r0', address=0, width=8)
    r1 = make_simple_reg('r1', address=1, width=8)
    rmap.add_child(r0)
    rmap.add_child(r1)
    r0.value = 0x11
    r1.value = 0x22
    calls = []
    rmap._block_write_func = lambda addr, data: calls.append((addr, bytes(data)))
    rmap.write()
    assert calls == [(0, bytes([0x11, 0x22]))]


def test_registermap_write_with_data_updates_registers():
    rmap = r_map.RegisterMap(name='map')
    r0 = make_simple_reg('r0', address=0, width=8)
    r1 = make_simple_reg('r1', address=1, width=8)
    rmap.add_child(r0)
    rmap.add_child(r1)
    calls = []
    rmap._block_write_func = lambda addr, data: calls.append((addr, bytes(data)))
    rmap.write(bytes([0xAA, 0xBB]))
    assert r0.value == 0xAA
    assert r1.value == 0xBB
    assert calls == [(0, bytes([0xAA, 0xBB]))]
