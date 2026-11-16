import pytest

from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.Register import Register
from r_map.RegisterMap import RegisterMap, get_dimensions


def _attach(parent, child):
    """Attach a child node the same way BitFieldRef._add does."""
    parent._children[child.name] = child
    child._parent = parent


def _make_reg(name, address, width, fields):
    """
    Build a Register with BitFieldRef children.

    fields: list of (reg_offset, slice_width, bf_width, value, reset_val)
    Returns (register, [bitfields]).
    """
    reg = Register(name=name, address=address, width=width)
    bfs = []
    for i, (reg_offset, slice_width, bf_width, value, reset_val) in enumerate(fields):
        bf = BitField(name=f'{name}_bf{i}', width=bf_width, reset_val=reset_val)
        bf.value = value
        ref = BitFieldRef(name=f'{name}_ref{i}', reg_offset=reg_offset,
                          slice_width=slice_width)
        ref._add(bf)
        _attach(reg, ref)
        bfs.append(bf)
    return reg, bfs


# ---------------------------------------------------------------- BitField

def test_bitfield_default_init():
    bf = BitField(name='b', width=4)
    assert bf.value == 0
    assert bf.mask == 0xF


def test_bitfield_reset_restores_reset_val():
    bf = BitField(name='b', width=4, reset_val=3)
    bf.value = 0xF
    bf.reset()
    assert bf.value == 3


def test_bitfield_zero_width_raises():
    with pytest.raises(ValueError):
        BitField(name='b', width=0)


def test_bitfield_negative_reset_val_raises():
    with pytest.raises(ValueError):
        BitField(name='b', width=4, reset_val=-1)


def test_bitfield_int_value_is_masked_to_width():
    bf = BitField(name='b', width=4)
    bf.value = 0x1F
    assert bf.value == 0xF


def test_bitfield_non_int_value_raises_not_implemented():
    bf = BitField(name='b', width=4)
    with pytest.raises(NotImplementedError):
        bf.value = 1.5


def test_bitfield_write_without_parent_raises():
    bf = BitField(name='lonely', width=4)
    with pytest.raises(ValueError):
        bf.write(1)


# ---------------------------------------------------------------- BitFieldRef

def test_bitfieldref_negative_slice_width_raises():
    with pytest.raises(ValueError):
        BitFieldRef(name='r', reg_offset=0, slice_width=-1)


def test_bitfieldref_negative_field_offset_raises():
    with pytest.raises(ValueError):
        BitFieldRef(name='r', reg_offset=0, slice_width=4, field_offset=-1)


def test_bitfieldref_value_getter_applies_offsets():
    bf = BitField(name='b', width=8)
    bf.value = 0xB6
    ref = BitFieldRef(name='r', reg_offset=4, slice_width=4, field_offset=2)
    ref._add(bf)
    # (0xB6 >> 2) & 0xF == 0xD, then << 4 == 0xD0
    assert ref.value == 0xD0


def test_bitfieldref_value_setter_only_touches_its_slice():
    bf = BitField(name='b', width=8)
    bf.value = 0xFF
    ref = BitFieldRef(name='r', reg_offset=4, slice_width=4, field_offset=2)
    ref._add(bf)
    ref.value = 0x50  # slice bits taken from (0x50 >> 4) & 0xF == 5
    assert bf.value == 0xD7


def test_bitfieldref_reset_val_uses_reset_of_bitfield():
    bf = BitField(name='b', width=8, reset_val=0x3C)
    ref = BitFieldRef(name='r', reg_offset=0, slice_width=4, field_offset=2)
    ref._add(bf)
    assert ref.reset_val == 0xF


def test_bitfieldref_slice_width_defaults_to_bitfield_width():
    bf = BitField(name='b', width=6)
    ref = BitFieldRef(name='r', reg_offset=0)
    ref._add(bf)
    assert ref.slice_width == 6
    assert ref.mask == 0x3F


# ---------------------------------------------------------------- Register

def test_register_value_combines_bitfieldrefs():
    reg, _ = _make_reg('r', 0x0, 16, [
        (0, 8, 8, 0x34, 0),
        (8, 8, 8, 0x12, 0),
    ])
    assert reg.value == 0x1234


def test_register_value_setter_distributes_to_bitfields():
    reg, bfs = _make_reg('r', 0x0, 16, [
        (0, 8, 8, 0, 0),
        (8, 8, 8, 0, 0),
    ])
    reg.value = 0xABCD
    assert bfs[0].value == 0xCD
    assert bfs[1].value == 0xAB


def test_register_reset_and_reset_val():
    reg, bfs = _make_reg('r', 0x0, 16, [
        (0, 8, 8, 0xAA, 0x34),
        (8, 8, 8, 0x00, 0x12),
    ])
    assert reg.reset_val == 0x1234
    reg.reset()
    assert reg.value == 0x1234


def test_register_read_uses_read_func_and_updates_value():
    reg, bfs = _make_reg('r', 0x10, 8, [(0, 8, 8, 0, 0)])
    reg._reg_read_func = lambda addr: 0x5A
    result = reg.read()
    assert result == 0x5A
    assert bfs[0].value == 0x5A


def test_register_write_sends_value_to_write_func():
    reg, bfs = _make_reg('r', 0x10, 8, [(0, 8, 8, 0, 0)])
    calls = []
    reg._reg_write_func = lambda addr, val: calls.append((addr, val))
    reg.write(0x7F)
    assert bfs[0].value == 0x7F
    assert calls == [(0x10, 0x7F)]


def test_register_validate_reports_error_when_empty():
    reg = Register(name='empty', address=0, width=8)
    errors = list(reg.validate())
    assert len(errors) > 0


def test_register_validate_detects_overlapping_fields():
    ok_reg, _ = _make_reg('ok', 0, 16, [
        (0, 8, 8, 0, 0),
        (8, 8, 8, 0, 0),
    ])
    overlap_reg, _ = _make_reg('bad', 0, 16, [
        (0, 8, 8, 0, 0),
        (4, 8, 8, 0, 0),
    ])
    ok_errors = list(ok_reg.validate())
    overlap_errors = list(overlap_reg.validate())
    assert len(overlap_errors) > len(ok_errors)


def test_register_iterates_children_sorted_by_reg_offset():
    reg, _ = _make_reg('r', 0, 16, [
        (8, 4, 4, 0, 0),
        (0, 4, 4, 0, 0),
        (4, 4, 4, 0, 0),
    ])
    offsets = [child.reg_offset for child in reg]
    assert offsets == [0, 4, 8]


# ---------------------------------------------------------------- RegisterMap

def test_get_dimensions_computes_indices_and_size():
    start_index, end_index, num_bytes = get_dimensions(0x100, 0x104, 16)
    assert start_index == 4
    assert end_index == 6
    assert num_bytes == 2


def test_register_map_read_populates_registers_little_endian():
    rm = RegisterMap(name='map')
    reg0, bf0 = _make_reg('r0', 0x100, 32, [(0, 32, 32, 0, 0)])
    reg1, bf1 = _make_reg('r1', 0x104, 16, [(0, 16, 16, 0, 0)])
    _attach(rm, reg0)
    _attach(rm, reg1)

    seen = []
    data = b'\x78\x56\x34\x12\x02\x00'
    rm._block_read_func = lambda addr, size: seen.append((addr, size)) or data

    result = rm.read()
    assert result == data
    assert seen == [(0x100, 6)]
    assert bf0.value == 0x12345678
    assert bf1.value == 0x0002


def test_register_map_write_builds_little_endian_block():
    rm = RegisterMap(name='map')
    reg0, _ = _make_reg('r0', 0x100, 32, [(0, 32, 32, 0x12345678, 0)])
    reg1, _ = _make_reg('r1', 0x104, 16, [(0, 16, 16, 0x0002, 0)])
    _attach(rm, reg0)
    _attach(rm, reg1)

    captured = []
    rm._block_write_func = lambda addr, data: captured.append((addr, bytes(data)))

    rm.write()
    assert captured == [(0x100, b'\x78\x56\x34\x12\x02\x00')]
