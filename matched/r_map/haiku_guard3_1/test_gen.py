import pytest

import r_map
from r_map.RegisterMap import get_dimensions
from r_map.ValidationError import ValidationError


def _add_field(reg, name, width, reg_offset=0, field_offset=0, slice_width=0,
               reset_val=0, access='RW'):
    """Create a BitField and a BitFieldRef to it, and attach the ref to reg.

    Uses only the constructors and the BitFieldRef._add method shown in the
    source, and sets the parent link the same way the library does.
    """
    bf = r_map.BitField(name=name, width=width, reset_val=reset_val,
                        access=access)
    ref = r_map.BitFieldRef(name=name + '_ref', reg_offset=reg_offset,
                            slice_width=slice_width, field_offset=field_offset)
    ref._add(bf)
    ref._parent = reg
    reg._children[ref.name] = ref
    return bf, ref


def _count_validation_errors(node):
    return sum(1 for e in node.validate() if isinstance(e, ValidationError))


# ---------------------------------------------------------------- BitField

def test_bitfield_default_value_is_reset_value_and_mask_is_width_bits():
    bf = r_map.BitField(name='f', width=4, reset_val=0x5)
    assert bf.value == 0x5
    assert bf.reset_val == 0x5
    assert bf.mask == 0xF


def test_bitfield_int_assignment_is_masked_to_width():
    bf = r_map.BitField(name='f', width=4)
    bf.value = 0x1F
    assert bf.value == 0xF
    bf.value = 0x3
    assert bf.value == 0x3


def test_bitfield_rejects_zero_width():
    with pytest.raises(ValueError):
        r_map.BitField(name='f', width=0)


def test_bitfield_rejects_negative_reset_value():
    with pytest.raises(ValueError):
        r_map.BitField(name='f', width=4, reset_val=-1)


def test_bitfield_reset_restores_reset_value():
    bf = r_map.BitField(name='f', width=4, reset_val=0x5)
    bf.value = 0xF
    assert bf.value == 0xF
    bf.reset()
    assert bf.value == 0x5


def test_bitfield_rejects_unsupported_value_type():
    bf = r_map.BitField(name='f', width=4)
    with pytest.raises(NotImplementedError):
        bf.value = 1.5


def test_bitfield_read_and_write_without_parent_raise():
    bf = r_map.BitField(name='f', width=4)
    with pytest.raises(ValueError):
        bf.read()
    with pytest.raises(ValueError):
        bf.write(1)


# ------------------------------------------------------------- BitFieldRef

def test_bitfieldref_getter_extracts_slice_and_shifts_to_reg_offset():
    bf = r_map.BitField(name='f', width=8)
    bf.value = 0xAB
    ref = r_map.BitFieldRef(name='r', reg_offset=4, slice_width=4,
                            field_offset=0)
    ref._add(bf)
    # low nibble 0xB shifted up by reg_offset 4
    assert ref.value == 0xB0


def test_bitfieldref_setter_updates_only_its_slice():
    bf = r_map.BitField(name='f', width=8)
    bf.value = 0xFF
    ref = r_map.BitFieldRef(name='r', reg_offset=0, slice_width=4,
                            field_offset=4)
    ref._add(bf)
    ref.value = 0x3
    # upper nibble of the field replaced by 0x3, lower nibble preserved
    assert bf.value == 0x3F


def test_bitfieldref_zero_slice_width_inherits_bitfield_width():
    bf = r_map.BitField(name='f', width=6)
    ref = r_map.BitFieldRef(name='r', reg_offset=0)
    ref._add(bf)
    assert ref.slice_width == 6
    assert ref.mask == 0x3F


def test_bitfieldref_validate_flags_slice_wider_than_bitfield():
    bf = r_map.BitField(name='f', width=4)
    too_wide = r_map.BitFieldRef(name='bad', reg_offset=0, slice_width=8)
    too_wide._add(bf)
    fits = r_map.BitFieldRef(name='ok', reg_offset=0, slice_width=4)
    fits._add(bf)

    assert _count_validation_errors(too_wide) > _count_validation_errors(fits)


# ----------------------------------------------------------------- Register

def test_register_value_is_or_of_field_values_at_their_offsets():
    reg = r_map.Register(name='reg', width=8, address=0x0)
    a, _ = _add_field(reg, 'a', 4, reg_offset=0)
    b, _ = _add_field(reg, 'b', 4, reg_offset=4)
    a.value = 0x5
    b.value = 0xA
    assert reg.value == 0xA5


def test_register_value_setter_distributes_to_fields():
    reg = r_map.Register(name='reg', width=8, address=0x0)
    a, _ = _add_field(reg, 'a', 4, reg_offset=0)
    b, _ = _add_field(reg, 'b', 4, reg_offset=4)
    reg.value = 0xA5
    assert a.value == 0x5
    assert b.value == 0xA


def test_register_iterates_children_sorted_by_reg_offset():
    reg = r_map.Register(name='reg', width=8, address=0x0)
    # insert in reverse offset order
    _add_field(reg, 'b', 4, reg_offset=4)
    _add_field(reg, 'a', 4, reg_offset=0)
    assert [r.reg_offset for r in reg] == [0, 4]


def test_register_validate_reports_overlapping_bitfieldrefs():
    overlapping = r_map.Register(name='overlap', width=8, address=0x0)
    _add_field(overlapping, 'a', 4, reg_offset=0, slice_width=4)
    _add_field(overlapping, 'b', 4, reg_offset=2, slice_width=4)

    adjacent = r_map.Register(name='adjacent', width=8, address=0x0)
    _add_field(adjacent, 'a', 4, reg_offset=0, slice_width=4)
    _add_field(adjacent, 'b', 4, reg_offset=4, slice_width=4)

    assert _count_validation_errors(overlapping) > \
        _count_validation_errors(adjacent)


def test_register_validate_reports_register_with_no_bitfieldrefs():
    empty = r_map.Register(name='empty', width=8, address=0x0)
    populated = r_map.Register(name='populated', width=8, address=0x0)
    _add_field(populated, 'a', 8)

    assert _count_validation_errors(empty) > \
        _count_validation_errors(populated)


def test_register_access_is_sorted_unique_join_of_field_access():
    reg = r_map.Register(name='reg', width=8, address=0x0)
    _add_field(reg, 'a', 4, reg_offset=0, access='RW')
    _add_field(reg, 'b', 4, reg_offset=4, access='RO')
    _add_field(reg, 'c', 0 + 1, reg_offset=0, field_offset=0, access='RW',
               reset_val=0)
    assert reg.access == 'RO|RW'


def test_register_reset_restores_each_field_reset_value():
    reg = r_map.Register(name='reg', width=8, address=0x0)
    a, _ = _add_field(reg, 'a', 4, reg_offset=0, reset_val=0x3)
    b, _ = _add_field(reg, 'b', 4, reg_offset=4, reset_val=0xC)
    a.value = 0
    b.value = 0
    reg.reset()
    assert a.value == 0x3
    assert b.value == 0xC


def test_register_read_stores_value_from_read_function_and_returns_it():
    reg = r_map.Register(name='reg', width=8, address=0x10)
    bf, _ = _add_field(reg, 'a', 8)
    calls = []

    def read_func(address):
        calls.append(address)
        return 0xA5

    reg._reg_read_func = read_func
    result = reg.read()

    assert result == 0xA5
    assert calls == [0x10]
    assert reg.value == 0xA5
    assert bf.value == 0xA5


# -------------------------------------------------------------- RegisterMap

def test_get_dimensions_returns_indices_and_byte_count():
    assert get_dimensions(0x10, 0x14, 32) == (4, 8, 4)
    # 12 bits round up to 2 bytes
    assert get_dimensions(0x0, 0x0, 12) == (0, 2, 2)


def test_registermap_read_splits_block_into_register_values():
    rmap = r_map.RegisterMap(name='map')
    r0 = r_map.Register(name='r0', width=8, address=0)
    _add_field(r0, 'f', 8)
    r1 = r_map.Register(name='r1', width=16, address=1)
    _add_field(r1, 'f', 16)
    rmap._children[r0.name] = r0
    rmap._children[r1.name] = r1

    block = bytes([0x12, 0x34, 0x56])
    captured = []

    def block_read(address, size):
        captured.append((address, size))
        return block

    rmap._block_read_func = block_read
    result = rmap.read()

    assert captured == [(0, 3)]
    assert result == block
    assert r0.value == 0x12
    assert r1.value == 0x5634


def test_registermap_write_without_data_builds_little_endian_block():
    rmap = r_map.RegisterMap(name='map')
    r0 = r_map.Register(name='r0', width=8, address=0)
    f0 = _add_field(r0, 'f', 8)[0]
    r1 = r_map.Register(name='r1', width=16, address=1)
    f1 = _add_field(r1, 'f', 16)[0]
    rmap._children[r0.name] = r0
    rmap._children[r1.name] = r1

    f0.value = 0x12
    f1.value = 0x5634

    captured = []

    def block_write(address, data):
        captured.append((address, bytes(data)))

    rmap._block_write_func = block_write
    rmap.write()

    assert captured == [(0, bytes([0x12, 0x34, 0x56]))]


def test_registermap_with_no_registers_raises_on_read():
    rmap = r_map.RegisterMap(name='empty_map')
    with pytest.raises(ValueError):
        rmap.read()
