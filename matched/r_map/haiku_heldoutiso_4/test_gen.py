import pytest

import r_map
from r_map.Register import Register
from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.RegisterMap import RegisterMap, get_dimensions
from r_map.ValidationError import ValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attach(parent, child):
    """Attach child to parent using the project's child-management hook when
    available, falling back to the internal children mapping."""
    add_child = getattr(parent, 'add_child', None)
    if callable(add_child):
        add_child(child)
        return
    parent._children[child.name] = child
    child._parent = parent


def _make_ref(reg, bf, reg_offset=0, slice_width=0, field_offset=0):
    ref = BitFieldRef(name=f'{bf.name}_ref', reg_offset=reg_offset,
                      slice_width=slice_width, field_offset=field_offset)
    ref._add(bf)
    _attach(reg, ref)
    return ref


def _recorder(return_value=None):
    calls = []

    def func(*args):
        calls.append(args)
        return return_value

    return func, calls


# ---------------------------------------------------------------------------
# BitField
# ---------------------------------------------------------------------------

def test_bitfield_rejects_zero_width():
    with pytest.raises(ValueError):
        BitField(name='bf', width=0)


def test_bitfield_rejects_negative_reset_val():
    with pytest.raises(ValueError):
        BitField(name='bf', width=4, reset_val=-1)


def test_bitfield_initial_value_equals_reset_val():
    bf = BitField(name='bf', width=4, reset_val=0xA)
    assert bf.value == 0xA


def test_bitfield_int_assignment_is_masked_to_width():
    bf = BitField(name='bf', width=4)
    bf.value = 0x1F
    assert bf.value == 0xF


def test_bitfield_reset_restores_reset_val():
    bf = BitField(name='bf', width=4, reset_val=0x3)
    bf.value = 0xC
    bf.reset()
    assert bf.value == 0x3


def test_bitfield_string_assignment_without_enumerations_raises():
    bf = BitField(name='bf', width=4)
    with pytest.raises(ValueError):
        bf.value = 'NOT_AN_ENUM'


def test_bitfield_unsupported_type_assignment_raises():
    bf = BitField(name='bf', width=4)
    with pytest.raises(NotImplementedError):
        bf.value = 1.5


def test_bitfield_annotation_falls_back_to_hex():
    bf = BitField(name='bf', width=8)
    bf.value = 0x0A
    assert bf.annotation == hex(0x0A)


def test_bitfield_write_without_parent_raises():
    bf = BitField(name='bf', width=4)
    with pytest.raises(ValueError):
        bf.write(3)


def test_bitfield_write_performs_read_modify_write_on_register():
    reg = Register(name='reg', address=0x0, width=8)
    read_func, _ = _recorder(return_value=0x03)
    write_func, write_calls = _recorder()
    reg._reg_read_func = read_func
    reg._reg_write_func = write_func

    bf = BitField(name='bf', width=4)
    _make_ref(reg, bf, reg_offset=4, slice_width=4)

    bf.write(0x9)

    # The bitfield occupies bits [7:4]; the register should now hold 0x90.
    assert write_calls == [(0x0, 0x90)]


def test_bitfield_read_returns_value_from_register():
    reg = Register(name='reg', address=0x0, width=8)
    reg._reg_read_func = lambda addr: 0xA0

    bf = BitField(name='bf', width=4)
    _make_ref(reg, bf, reg_offset=4, slice_width=4)

    assert bf.read() == 0xA


# ---------------------------------------------------------------------------
# BitFieldRef
# ---------------------------------------------------------------------------

def test_bitfieldref_rejects_negative_slice_width():
    with pytest.raises(ValueError):
        BitFieldRef(name='ref', reg_offset=0, slice_width=-1)


def test_bitfieldref_rejects_negative_field_offset():
    with pytest.raises(ValueError):
        BitFieldRef(name='ref', reg_offset=0, field_offset=-1)


def test_bitfieldref_zero_slice_width_takes_bitfield_width():
    reg = Register(name='reg', address=0x0, width=8)
    bf = BitField(name='bf', width=6)
    ref = _make_ref(reg, bf, reg_offset=0, slice_width=0)
    assert ref.slice_width == 6
    assert ref.mask == 0x3F


def test_bitfieldref_value_extracts_and_shifts_slice():
    reg = Register(name='reg', address=0x0, width=8)
    bf = BitField(name='bf', width=4)
    bf.value = 0b1011
    # field bits [2:1] of 0b1011 are 0b01, placed at register offset 4
    ref = _make_ref(reg, bf, reg_offset=4, slice_width=2, field_offset=1)
    assert ref.value == 0x10


def test_bitfieldref_value_setter_preserves_other_bits():
    reg = Register(name='reg', address=0x0, width=8)
    bf = BitField(name='bf', width=8)
    bf.value = 0xFF
    ref = _make_ref(reg, bf, reg_offset=0, slice_width=4, field_offset=0)

    ref.value = 0x00

    assert bf.value == 0xF0


def test_bitfieldref_reset_val_reflects_bitfield_reset_value():
    reg = Register(name='reg', address=0x0, width=8)
    bf = BitField(name='bf', width=4, reset_val=0b1010)
    ref = _make_ref(reg, bf, reg_offset=3, slice_width=2, field_offset=2)
    # (0b1010 >> 2) & 0b11 == 0b10, shifted by 3 -> 0b10000
    assert ref.reset_val == 16


def test_bitfieldref_validate_flags_slice_wider_than_bitfield():
    reg = Register(name='reg', address=0x0, width=8)
    bf = BitField(name='bf', width=4)
    ref = _make_ref(reg, bf, reg_offset=0, slice_width=8)

    errors = list(ref.validate())

    assert any(isinstance(e, ValidationError) for e in errors)


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

def test_register_value_combines_bitfield_refs():
    reg = Register(name='reg', address=0x0, width=8)
    bf_lo = BitField(name='lo', width=4)
    bf_hi = BitField(name='hi', width=4)
    bf_lo.value = 0x5
    bf_hi.value = 0xA
    _make_ref(reg, bf_lo, reg_offset=0, slice_width=4)
    _make_ref(reg, bf_hi, reg_offset=4, slice_width=4)

    assert reg.value == 0xA5


def test_register_value_setter_distributes_to_bitfields():
    reg = Register(name='reg', address=0x0, width=8)
    bf_lo = BitField(name='lo', width=4)
    bf_hi = BitField(name='hi', width=4)
    _make_ref(reg, bf_lo, reg_offset=0, slice_width=4)
    _make_ref(reg, bf_hi, reg_offset=4, slice_width=4)

    reg.value = 0xC3

    assert bf_lo.value == 0x3
    assert bf_hi.value == 0xC


def test_register_reset_val_combines_bitfield_reset_values():
    reg = Register(name='reg', address=0x0, width=8)
    bf_lo = BitField(name='lo', width=4, reset_val=0x2)
    bf_hi = BitField(name='hi', width=4, reset_val=0x7)
    _make_ref(reg, bf_lo, reg_offset=0, slice_width=4)
    _make_ref(reg, bf_hi, reg_offset=4, slice_width=4)

    assert reg.reset_val == 0x72


def test_register_reset_restores_bitfield_reset_values():
    reg = Register(name='reg', address=0x0, width=8)
    bf = BitField(name='bf', width=8, reset_val=0x42)
    _make_ref(reg, bf, reg_offset=0, slice_width=8)

    bf.value = 0x00
    reg.reset()

    assert bf.value == 0x42


def test_register_validate_reports_missing_bitfieldrefs():
    reg = Register(name='empty', address=0x0, width=8)
    errors = list(reg.validate())
    assert any(isinstance(e, ValidationError) for e in errors)


def test_register_validate_reports_overlapping_refs():
    reg = Register(name='reg', address=0x0, width=8)
    bf1 = BitField(name='a', width=4)
    bf2 = BitField(name='b', width=4)
    _make_ref(reg, bf1, reg_offset=0, slice_width=4)
    _make_ref(reg, bf2, reg_offset=2, slice_width=4)

    errors = list(reg.validate())

    assert any(isinstance(e, ValidationError) for e in errors)


def test_register_validate_clean_for_non_overlapping_refs():
    reg = Register(name='reg', address=0x0, width=8)
    bf1 = BitField(name='a', width=4)
    bf2 = BitField(name='b', width=4)
    _make_ref(reg, bf1, reg_offset=0, slice_width=4)
    _make_ref(reg, bf2, reg_offset=4, slice_width=4)

    errors = list(reg.validate())

    assert errors == []


def test_register_iterates_children_sorted_by_reg_offset():
    reg = Register(name='reg', address=0x0, width=8)
    bf_hi = BitField(name='hi', width=4)
    bf_lo = BitField(name='lo', width=4)
    # attach in reverse order of offset
    _make_ref(reg, bf_hi, reg_offset=4, slice_width=4)
    _make_ref(reg, bf_lo, reg_offset=0, slice_width=4)

    offsets = [child.reg_offset for child in reg]

    assert offsets == sorted(offsets)


def test_register_read_uses_read_function_and_updates_value():
    reg = Register(name='reg', address=0x20, width=8)
    read_func, read_calls = _recorder(return_value=0x5A)
    reg._reg_read_func = read_func

    bf = BitField(name='bf', width=8)
    _make_ref(reg, bf, reg_offset=0, slice_width=8)

    result = reg.read()

    assert read_calls == [(0x20,)]
    assert result == 0x5A
    assert reg.value == 0x5A


def test_register_write_with_value_updates_and_writes():
    reg = Register(name='reg', address=0x10, width=8)
    write_func, write_calls = _recorder()
    reg._reg_write_func = write_func

    bf = BitField(name='bf', width=8)
    _make_ref(reg, bf, reg_offset=0, slice_width=8)

    reg.write(0x3C)

    assert reg.value == 0x3C
    assert write_calls == [(0x10, 0x3C)]


def test_register_write_without_value_writes_current_value():
    reg = Register(name='reg', address=0x10, width=8)
    write_func, write_calls = _recorder()
    reg._reg_write_func = write_func

    bf = BitField(name='bf', width=8, reset_val=0x81)
    _make_ref(reg, bf, reg_offset=0, slice_width=8)

    reg.write()

    assert write_calls == [(0x10, 0x81)]


# ---------------------------------------------------------------------------
# RegisterMap
# ---------------------------------------------------------------------------

def test_get_dimensions_computes_indices_and_byte_count():
    start_index, end_index, num_bytes = get_dimensions(0x100, 0x102, 16)
    assert (start_index, end_index, num_bytes) == (2, 4, 2)


def test_registermap_block_dimensions_without_registers_raises():
    rm = RegisterMap(name='rm')
    rm._block_read_func = lambda addr, size: b''
    with pytest.raises(ValueError):
        rm.read()


def _build_two_byte_map():
    rm = RegisterMap(name='rm')
    reg0 = Register(name='r0', address=0x0, width=8)
    reg1 = Register(name='r1', address=0x1, width=8)
    _attach(rm, reg0)
    _attach(rm, reg1)
    _make_ref(reg0, BitField(name='b0', width=8), reg_offset=0, slice_width=8)
    _make_ref(reg1, BitField(name='b1', width=8), reg_offset=0, slice_width=8)
    return rm, reg0, reg1


def test_registermap_read_distributes_block_data_to_registers():
    rm, reg0, reg1 = _build_two_byte_map()
    read_func, read_calls = _recorder(return_value=None)
    rm._block_read_func = lambda addr, size: (read_calls.append((addr, size))
                                              or bytes([0x12, 0x34]))

    data = rm.read()

    assert read_calls == [(0x0, 2)]
    assert data == bytes([0x12, 0x34])
    assert reg0.value == 0x12
    assert reg1.value == 0x34


def test_registermap_write_without_data_serializes_register_values():
    rm, reg0, reg1 = _build_two_byte_map()
    reg0.value = 0x12
    reg1.value = 0x34
    write_func, write_calls = _recorder()
    rm._block_write_func = write_func

    rm.write()

    assert len(write_calls) == 1
    address, payload = write_calls[0]
    assert address == 0x0
    assert bytes(payload) == bytes([0x12, 0x34])


def test_registermap_write_with_data_updates_registers():
    rm, reg0, reg1 = _build_two_byte_map()
    write_func, write_calls = _recorder()
    rm._block_write_func = write_func

    rm.write(bytes([0xAB, 0xCD]))

    assert reg0.value == 0xAB
    assert reg1.value == 0xCD
    assert len(write_calls) == 1
    assert write_calls[0][0] == 0x0
    assert bytes(write_calls[0][1]) == bytes([0xAB, 0xCD])


def test_registermap_write_without_registers_raises():
    rm = RegisterMap(name='rm')
    rm._block_write_func = lambda addr, data: None
    rm._block_read_func = lambda addr, size: b''
    with pytest.raises(ValueError):
        rm.write()
