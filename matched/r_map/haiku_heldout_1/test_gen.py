import pytest

from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.Register import Register
from r_map.RegisterMap import RegisterMap, get_dimensions
from r_map.ValidationError import ValidationError


def _attach(parent, child):
    """Attach child to parent the same way BitFieldRef._add does."""
    parent._children[child.name] = child
    child._parent = parent


def _make_ref(name, reg_offset, slice_width, bf_width=8, field_offset=0,
              bf_value=0, reset_val=0):
    """Create a BitFieldRef bound to a freshly created BitField."""
    bf = BitField(name=name + '_bf', width=bf_width, reset_val=reset_val)
    bf.value = bf_value
    ref = BitFieldRef(name=name, reg_offset=reg_offset,
                      slice_width=slice_width, field_offset=field_offset)
    ref._add(bf)
    return ref


def _make_reg(name, address, width, refs):
    reg = Register(name=name, width=width, address=address)
    for ref in refs:
        _attach(reg, ref)
    return reg


# ---------------------------------------------------------------- BitField

def test_bitfield_rejects_zero_width():
    with pytest.raises(ValueError):
        BitField(name='bf', width=0)


def test_bitfield_rejects_negative_reset_val():
    with pytest.raises(ValueError):
        BitField(name='bf', width=4, reset_val=-1)


def test_bitfield_value_setter_masks_to_width():
    bf = BitField(name='bf', width=4)
    bf.value = 0x1F
    assert bf.value == 0xF


def test_bitfield_value_rejects_unsupported_type():
    bf = BitField(name='bf', width=4)
    with pytest.raises(NotImplementedError):
        bf.value = 1.5


def test_bitfield_reset_restores_reset_val():
    bf = BitField(name='bf', width=4, reset_val=5)
    bf.value = 9
    bf.reset()
    assert bf.value == 5


def test_bitfield_write_without_parent_raises():
    bf = BitField(name='bf', width=4)
    with pytest.raises(ValueError):
        bf.write(1)


# -------------------------------------------------------------- BitFieldRef

def test_bitfieldref_negative_slice_width_raises():
    with pytest.raises(ValueError):
        BitFieldRef(name='r', reg_offset=0, slice_width=-1)


def test_bitfieldref_negative_field_offset_raises():
    with pytest.raises(ValueError):
        BitFieldRef(name='r', reg_offset=0, slice_width=4, field_offset=-1)


def test_bitfieldref_getter_extracts_and_shifts_slice():
    ref = _make_ref('r', reg_offset=4, slice_width=4, bf_width=8, bf_value=0xAB)
    assert ref.value == 0xB0


def test_bitfieldref_setter_preserves_other_bitfield_bits():
    ref = _make_ref('r', reg_offset=4, slice_width=4, bf_width=8, bf_value=0xA0)
    ref.value = 0x50
    assert ref.bf.value == 0xA5


def test_bitfieldref_slice_width_defaults_to_bitfield_width():
    ref = BitFieldRef(name='r', reg_offset=0)
    bf = BitField(name='bf', width=8)
    ref._add(bf)
    assert ref.slice_width == 8
    assert ref.mask == 0xFF


def test_bitfieldref_add_rejects_non_bitfield():
    ref = BitFieldRef(name='r', reg_offset=0, slice_width=4)
    with pytest.raises(ValueError):
        ref._add("not a bitfield")


def test_bitfieldref_reset_val_is_shifted_slice_of_reset():
    ref = _make_ref('r', reg_offset=4, slice_width=4, bf_width=8, reset_val=0xC3)
    assert ref.reset_val == 0x30


def test_bitfieldref_validate_flags_slice_wider_than_bitfield():
    ref = BitFieldRef(name='r', reg_offset=0, slice_width=8)
    ref._add(BitField(name='bf', width=4))
    errors = list(ref.validate())
    assert any(isinstance(e, ValidationError) for e in errors)


# ----------------------------------------------------------------- Register

def test_register_value_combines_children():
    reg = _make_reg('reg', 0, 8, [
        _make_ref('lo', reg_offset=0, slice_width=4, bf_width=4, bf_value=0x3),
        _make_ref('hi', reg_offset=4, slice_width=4, bf_width=4, bf_value=0x2),
    ])
    assert reg.value == 0x23


def test_register_value_setter_distributes_to_children():
    lo = _make_ref('lo', reg_offset=0, slice_width=4, bf_width=4)
    hi = _make_ref('hi', reg_offset=4, slice_width=4, bf_width=4)
    reg = _make_reg('reg', 0, 8, [lo, hi])
    reg.value = 0x52
    assert lo.bf.value == 0x2
    assert hi.bf.value == 0x5


def test_register_reset_val_combines_children_reset_values():
    reg = _make_reg('reg', 0, 8, [
        _make_ref('lo', reg_offset=0, slice_width=4, bf_width=4, reset_val=0x3),
        _make_ref('hi', reg_offset=4, slice_width=4, bf_width=4, reset_val=0xA),
    ])
    assert reg.reset_val == 0xA3


def test_register_validate_without_children_reports_error():
    reg = Register(name='reg', width=8, address=0)
    errors = list(reg.validate())
    assert any(isinstance(e, ValidationError) for e in errors)


def test_register_validate_reports_overlapping_bitfieldrefs():
    clean = _make_reg('clean', 0, 8, [
        _make_ref('a', reg_offset=0, slice_width=4, bf_width=4),
        _make_ref('b', reg_offset=4, slice_width=4, bf_width=4),
    ])
    overlapping = _make_reg('overlap', 0, 8, [
        _make_ref('a', reg_offset=0, slice_width=4, bf_width=4),
        _make_ref('b', reg_offset=2, slice_width=4, bf_width=4),
    ])
    clean_errors = list(clean.validate())
    overlap_errors = list(overlapping.validate())
    assert len(overlap_errors) > len(clean_errors)


def test_register_read_updates_value_but_returns_raw_value():
    reg = _make_reg('reg', 0x20, 8, [
        _make_ref('lo', reg_offset=0, slice_width=4, bf_width=4),
    ])
    reg._reg_read_func = lambda address: 0xA5
    result = reg.read()
    assert result == 0xA5          # raw value returned
    assert reg.value == 0x5        # only the 4-bit field is stored


def test_register_write_uses_argument_then_current_value():
    reg = _make_reg('reg', 0x20, 8, [
        _make_ref('all', reg_offset=0, slice_width=8, bf_width=8),
    ])
    writes = []
    reg._reg_write_func = lambda address, val: writes.append((address, val))

    reg.write(0x3C)
    assert writes == [(0x20, 0x3C)]
    assert reg.value == 0x3C

    reg.write()
    assert writes[-1] == (0x20, 0x3C)


# ------------------------------------------------- BitField read-modify-write

def test_bitfield_write_performs_read_modify_write_on_parent_register():
    memory = {0x20: 0xF0}
    lo = _make_ref('lo', reg_offset=0, slice_width=4, bf_width=4)
    hi = _make_ref('hi', reg_offset=4, slice_width=4, bf_width=4)
    reg = _make_reg('reg', 0x20, 8, [lo, hi])
    reg._reg_read_func = lambda address: memory[address]
    reg._reg_write_func = lambda address, val: memory.__setitem__(address, val)

    lo.bf.write(0x3)

    # high nibble preserved from the read-back, low nibble updated
    assert memory[0x20] == 0xF3


# ---------------------------------------------------------------- RegisterMap

def test_get_dimensions_returns_indices_and_byte_count():
    assert get_dimensions(0x100, 0x104, 32) == (4, 8, 4)
    assert get_dimensions(0, 2, 12) == (2, 4, 2)


def test_registermap_read_distributes_bytes_to_registers():
    reg0 = _make_reg('r0', 0, 8, [_make_ref('f0', 0, 8, bf_width=8)])
    reg1 = _make_reg('r1', 1, 8, [_make_ref('f1', 0, 8, bf_width=8)])
    rm = RegisterMap(name='map')
    _attach(rm, reg0)
    _attach(rm, reg1)
    rm._block_read_func = lambda address, size: bytes([0x34, 0x12])

    rm.read()

    assert reg0.value == 0x34
    assert reg1.value == 0x12


def test_registermap_write_without_data_encodes_register_values():
    reg0 = _make_reg('r0', 0, 8, [_make_ref('f0', 0, 8, bf_width=8)])
    reg1 = _make_reg('r1', 1, 8, [_make_ref('f1', 0, 8, bf_width=8)])
    rm = RegisterMap(name='map')
    _attach(rm, reg0)
    _attach(rm, reg1)
    reg0.value = 0xAA
    reg1.value = 0x55
    calls = []
    rm._block_write_func = lambda address, data: calls.append((address, bytes(data)))

    rm.write()

    assert calls == [(0, b'\xaa\x55')]


def test_registermap_write_with_data_updates_registers():
    reg0 = _make_reg('r0', 0, 8, [_make_ref('f0', 0, 8, bf_width=8)])
    reg1 = _make_reg('r1', 1, 8, [_make_ref('f1', 0, 8, bf_width=8)])
    rm = RegisterMap(name='map')
    _attach(rm, reg0)
    _attach(rm, reg1)
    calls = []
    rm._block_write_func = lambda address, data: calls.append((address, bytes(data)))

    rm.write(bytes([1, 2]))

    assert reg0.value == 1
    assert reg1.value == 2
    assert calls == [(0, b'\x01\x02')]


def test_registermap_write_empty_map_raises():
    rm = RegisterMap(name='empty')
    with pytest.raises(ValueError):
        rm.write()
