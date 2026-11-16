import pytest

from r_map import BitField, BitFieldRef, Register, RegisterMap
from r_map.RegisterMap import get_dimensions
from r_map.ValidationError import ValidationError


def attach(parent, child):
    """Attach a child node to a parent node the way the library stores children."""
    parent._children[child.name] = child
    child._parent = parent


def make_reg(name, address=0, width=8, reset_val=0):
    """Create a register with a single bitfield spanning its full width."""
    reg = Register(name=name, address=address, width=width)
    bf = BitField(name=name + '_bf', width=width, reset_val=reset_val)
    ref = BitFieldRef(name=name + '_ref', reg_offset=0, field_offset=0)
    ref._add(bf)
    attach(reg, ref)
    return reg


def make_two_field_reg(name='r', address=0):
    """8-bit register: low nibble = 0x3 (reset), high nibble = 0xA (reset)."""
    reg = Register(name=name, address=address, width=8)
    lo = BitFieldRef(name='lo', reg_offset=0, slice_width=4, field_offset=0)
    lo._add(BitField(name='a', width=4, reset_val=0x3))
    attach(reg, lo)
    hi = BitFieldRef(name='hi', reg_offset=4, field_offset=0)
    hi._add(BitField(name='b', width=4, reset_val=0xA))
    attach(reg, hi)
    return reg


# ---------------------------------------------------------------- BitField

def test_bitfield_rejects_zero_width():
    with pytest.raises(ValueError):
        BitField(name='bad', width=0)


def test_bitfield_rejects_negative_reset_val():
    with pytest.raises(ValueError):
        BitField(name='bad', width=4, reset_val=-1)


def test_bitfield_mask_and_value_truncation():
    bf = BitField(name='b', width=4)
    assert bf.mask == 0xF
    bf.value = 0x1F
    assert bf.value == 0xF


def test_bitfield_rejects_unsupported_value_type():
    bf = BitField(name='b', width=4)
    with pytest.raises(NotImplementedError):
        bf.value = 1.5


def test_bitfield_reset_restores_reset_val():
    bf = BitField(name='b', width=8, reset_val=0x5A)
    bf.value = 0
    bf.reset()
    assert bf.value == 0x5A


def test_bitfield_annotation_falls_back_to_hex():
    bf = BitField(name='b', width=8, reset_val=0x0A)
    assert bf.annotation == hex(0x0A)


def test_bitfield_write_without_parent_raises():
    bf = BitField(name='orphan', width=4)
    with pytest.raises(ValueError):
        bf.write(1)


# ---------------------------------------------------------------- BitFieldRef

def test_bitfieldref_rejects_negative_slice_width():
    with pytest.raises(ValueError):
        BitFieldRef(name='r', reg_offset=0, slice_width=-1)


def test_bitfieldref_rejects_negative_field_offset():
    with pytest.raises(ValueError):
        BitFieldRef(name='r', reg_offset=0, field_offset=-1)


def test_bitfieldref_takes_slice_width_from_bitfield_when_zero():
    ref = BitFieldRef(name='r', reg_offset=0)
    ref._add(BitField(name='b', width=8))
    assert ref.slice_width == 8
    assert ref.mask == 0xFF


def test_bitfieldref_value_is_shifted_to_register_offset():
    ref = BitFieldRef(name='r', reg_offset=4, slice_width=4, field_offset=4)
    ref._add(BitField(name='b', width=8, reset_val=0xAB))
    assert ref.value == 0xA0


def test_bitfieldref_setter_writes_into_bitfield():
    bf = BitField(name='b', width=8)
    ref = BitFieldRef(name='r', reg_offset=4, slice_width=4, field_offset=0)
    ref._add(bf)
    ref.value = 0x50
    assert bf.value == 0x5
    assert ref.value == 0x50


# ---------------------------------------------------------------- Register

def test_register_value_combines_bitfield_refs():
    reg = make_two_field_reg()
    assert reg.value == 0xA3


def test_register_value_setter_distributes_to_fields():
    reg = make_two_field_reg()
    reg.value = 0x5C
    assert reg.value == 0x5C
    assert reg.reset_val == 0xA3


def test_register_reset_restores_reset_values():
    reg = make_two_field_reg()
    reg.value = 0x00
    reg.reset()
    assert reg.value == 0xA3


def test_register_validate_reports_missing_bitfieldrefs():
    reg = Register(name='empty', address=0, width=8)
    errors = list(reg.validate())
    assert any(isinstance(e, ValidationError) for e in errors)


def test_register_validate_reports_overlapping_refs():
    reg = Register(name='ov', address=0, width=8)
    first = BitFieldRef(name='first', reg_offset=0, slice_width=4)
    second = BitFieldRef(name='second', reg_offset=2, slice_width=4)
    attach(reg, first)
    attach(reg, second)
    errors = list(reg.validate())
    assert any(isinstance(e, ValidationError) for e in errors)


def test_register_iterates_children_sorted_by_reg_offset():
    reg = Register(name='sorted', address=0, width=8)
    hi = BitFieldRef(name='hi', reg_offset=4, slice_width=4)
    lo = BitFieldRef(name='lo', reg_offset=0, slice_width=4)
    attach(reg, hi)
    attach(reg, lo)
    assert [c.reg_offset for c in reg] == [0, 4]


def test_register_read_returns_and_stores_hardware_value():
    reg = make_two_field_reg(address=0x4)
    reg._reg_read_func = lambda addr: 0x3C
    assert reg.read() == 0x3C
    assert reg.value == 0x3C


def test_register_write_sends_value_to_hardware():
    reg = make_two_field_reg(address=0x4)
    calls = []
    reg._reg_write_func = lambda addr, val: calls.append((addr, val))
    reg.write(0x3C)
    assert reg.value == 0x3C
    assert calls == [(0x4, 0x3C)]


def test_bitfield_write_performs_read_modify_write_on_parent():
    reg = make_reg('r', address=0x8, width=8, reset_val=0)
    bf = list(reg)[0].bf
    reads = []
    writes = []
    reg._reg_read_func = lambda addr: (reads.append(addr), 0)[1]
    reg._reg_write_func = lambda addr, val: writes.append((addr, val))
    bf.write(0x7)
    assert reads == [0x8]
    assert writes == [(0x8, 0x7)]


# ---------------------------------------------------------------- RegisterMap

def test_get_dimensions_for_32_bit_register():
    assert get_dimensions(0x100, 0x104, 32) == (4, 8, 4)


def test_get_dimensions_rounds_partial_bytes_up():
    assert get_dimensions(0, 0x2, 12) == (2, 4, 2)


def test_registermap_read_splits_block_into_registers():
    rm = RegisterMap(name='map')
    attach(rm, make_reg('r0', address=0, width=8))
    attach(rm, make_reg('r1', address=1, width=16))
    rm._block_read_func = lambda addr, size: bytes(b'\x12\x34\x56')
    data = rm.read()
    assert data == b'\x12\x34\x56'
    regs = list(rm)
    assert regs[0].value == 0x12
    assert regs[1].value == 0x5634


def test_registermap_write_with_data_updates_registers_and_block():
    rm = RegisterMap(name='map')
    attach(rm, make_reg('r0', address=0, width=8))
    attach(rm, make_reg('r1', address=1, width=16))
    written = []
    rm._block_write_func = lambda addr, data: written.append((addr, bytes(data)))
    rm.write(bytearray(b'\x12\x34\x56'))
    assert written == [(0, b'\x12\x34\x56')]
    regs = list(rm)
    assert regs[0].value == 0x12
    assert regs[1].value == 0x5634


def test_registermap_write_without_data_uses_register_values():
    rm = RegisterMap(name='map')
    attach(rm, make_reg('r0', address=0, width=8, reset_val=0x12))
    attach(rm, make_reg('r1', address=1, width=16, reset_val=0x5634))
    written = []
    rm._block_write_func = lambda addr, data: written.append((addr, bytes(data)))
    rm.write()
    assert written == [(0, b'\x12\x34\x56')]


# ---------------------------------------------------------------- misc

def test_bitfield_annotation_when_value_changes():
    bf = BitField(name='b', width=4, reset_val=0)
    bf.value = 0xB
    assert bf.annotation == hex(0xB)
