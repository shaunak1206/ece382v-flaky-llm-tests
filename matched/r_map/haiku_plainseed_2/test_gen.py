"""Unit tests for r_map Register, BitField, BitFieldRef and RegisterMap."""
import pytest

from r_map import BitField, BitFieldRef, Enumeration, Register, RegisterMap


def make_map(base=0x1000, name='rm'):
    """RegisterMap backed by a dict register file. Returns (rm, store, writes)."""
    rm = RegisterMap(name=name, local_address=base)
    store = {}
    writes = []

    def reg_read(address):
        return store.get(address, 0)

    def reg_write(address, value):
        writes.append((address, value))
        store[address] = value

    rm._reg_read_func = reg_read
    rm._reg_write_func = reg_write
    return rm, store, writes


def make_register(parent, name, local_address, width=8):
    reg = Register(name=name, local_address=local_address, width=width)
    parent._add(reg)
    return reg


def add_field(reg, name, reg_offset, width, field_offset=0, slice_width=0,
              reset_val=0, access='RW'):
    bf = BitField(name=name, width=width, reset_val=reset_val, access=access)
    ref = BitFieldRef(name=name + '_ref', reg_offset=reg_offset,
                      slice_width=slice_width, field_offset=field_offset)
    reg._add(ref)
    ref._add(bf)
    return bf


# BitField, without any register attached

def test_bitfield_rejects_zero_width():
    with pytest.raises(ValueError):
        BitField(name='f', width=0)


def test_bitfield_int_assignment_is_masked_to_width():
    bf = BitField(name='f', width=4)
    bf.value = 0x1F
    assert bf.value == 0xF


def test_bitfield_assign_enumeration_by_name_and_object():
    bf = BitField(name='mode', width=2)
    bf._add(Enumeration(name='OFF', value=0))
    bf._add(Enumeration(name='ON', value=1))
    bf.value = 'ON'
    assert bf.value == 1
    bf.value = Enumeration(name='OTHER', value=3)
    assert bf.value == 3


def test_bitfield_unknown_enumeration_name_raises():
    bf = BitField(name='mode', width=2)
    bf._add(Enumeration(name='OFF', value=0))
    with pytest.raises(ValueError):
        bf.value = 'MISSING'


def test_bitfield_reset_restores_reset_value():
    bf = BitField(name='f', width=8, reset_val=0x5A)
    bf.value = 0x00
    bf.reset()
    assert bf.value == 0x5A


def test_bitfield_validate_reports_duplicate_enum_values():
    bf = BitField(name='f', width=4)
    bf._add(Enumeration(name='A', value=1))
    bf._add(Enumeration(name='B', value=1))
    errors = list(bf.validate())
    assert len(errors) == 1
    assert 'same value' in errors[0].error


def test_bitfield_read_without_register_raises():
    bf = BitField(name='orphan', width=4)
    with pytest.raises(ValueError):
        bf.read()


# BitField IO through its parent register

def test_bitfield_write_updates_register():
    rm, store, _ = make_map()
    reg = make_register(rm, 'r0', 0x0, width=8)
    bf = add_field(reg, 'f', reg_offset=4, width=4)
    bf.write(0xA)
    assert store[0x1000] == 0xA0
    assert bf.value == 0xA


def test_bitfield_write_preserves_sibling_fields():
    rm, store, _ = make_map()
    reg = make_register(rm, 'r0', 0x0, width=8)
    lo = add_field(reg, 'lo', reg_offset=0, width=4)
    hi = add_field(reg, 'hi', reg_offset=4, width=4)
    hi.write(0x9)
    assert store[0x1000] == 0x90
    lo.write(0x2)
    assert store[0x1000] == 0x92
    assert hi.value == 0x9
    assert lo.value == 0x2


def test_bitfield_write_respects_always_write_false():
    rm, store, writes = make_map()
    reg = make_register(rm, 'r0', 0x0, width=8)
    bf = add_field(reg, 'f', reg_offset=0, width=4)
    store[0x1000] = 0x05
    bf.write(0x5, always_write=False)
    assert writes == []
    bf.write(0x6, always_write=False)
    assert writes == [(0x1000, 0x6)]
    assert store[0x1000] == 0x6


# Register

def test_register_value_combines_fields():
    rm, _, _ = make_map()
    reg = make_register(rm, 'r0', 0x0)
    lo = add_field(reg, 'lo', reg_offset=0, width=4)
    hi = add_field(reg, 'hi', reg_offset=4, width=4)
    lo.value = 0x3
    hi.value = 0xC
    assert reg.value == 0xC3


def test_register_value_setter_fans_out_to_fields():
    rm, _, _ = make_map()
    reg = make_register(rm, 'r0', 0x0)
    lo = add_field(reg, 'lo', reg_offset=0, width=4)
    hi = add_field(reg, 'hi', reg_offset=4, width=4)
    reg.value = 0xA5
    assert lo.value == 0x5
    assert hi.value == 0xA


def test_register_reset_restores_all_fields():
    rm, _, _ = make_map()
    reg = make_register(rm, 'r0', 0x0)
    lo = add_field(reg, 'lo', reg_offset=0, width=4, reset_val=0x3)
    hi = add_field(reg, 'hi', reg_offset=4, width=4, reset_val=0x7)
    reg.value = 0x00
    reg.reset()
    assert lo.value == 0x3
    assert hi.value == 0x7
    assert reg.value == 0x73
    assert reg.reset_val == 0x73


def test_register_read_write_uses_absolute_address():
    rm, store, _ = make_map(base=0x1000)
    reg = make_register(rm, 'r0', 0x8)
    add_field(reg, 'f', reg_offset=0, width=8)
    assert reg.address == 0x1008
    reg.write(0x42)
    assert store == {0x1008: 0x42}
    store[0x1008] = 0x77
    assert reg.read() == 0x77
    assert reg.value == 0x77


def test_register_validate_reports_missing_bitfieldrefs():
    reg = Register(name='empty', local_address=0, width=8)
    errors = list(reg.validate())
    assert len(errors) == 1
    assert 'No bitfieldrefs' in errors[0].error


def test_register_validate_reports_overlapping_fields():
    reg = Register(name='r0', local_address=0, width=8)
    add_field(reg, 'a', reg_offset=0, width=4)
    add_field(reg, 'b', reg_offset=2, width=4)
    errors = list(reg.validate())
    assert any('overlaps' in e.error for e in errors)


# BitFieldRef

def test_bitfieldref_slice_width_defaults_to_field_width():
    ref = BitFieldRef(name='r', reg_offset=0)
    ref._add(BitField(name='f', width=6))
    assert ref.slice_width == 6
    assert ref.mask == 0x3F


def test_bitfieldref_setter_only_touches_its_slice():
    bf = BitField(name='f', width=8)
    bf.value = 0xFF
    ref = BitFieldRef(name='r', reg_offset=4, slice_width=4, field_offset=2)
    ref._add(bf)
    assert ref.value == 0xF0
    ref.value = 0x50
    assert bf.value == 0xD7
    assert ref.value == 0x50


def test_bitfieldref_second_reference_becomes_alias():
    bf = BitField(name='shared', width=4)
    first = BitFieldRef(name='first', reg_offset=0)
    second = BitFieldRef(name='second', reg_offset=4)
    first._add(bf)
    second._add(bf)
    assert second.bf is bf
    assert second._alias is True
    assert second._ref is first


def test_bitfieldref_add_rejects_non_bitfield():
    ref = BitFieldRef(name='ref', reg_offset=0)
    with pytest.raises(ValueError):
        ref._add(Enumeration(name='OFF', value=0))


# RegisterMap

def test_block_dimensions_span_registers():
    rm, _, _ = make_map(base=0x1000)
    make_register(rm, 'a', 0x0, width=8)
    make_register(rm, 'b', 0x8, width=32)
    assert rm._block_dimensions == (0x1000, 12)


def test_map_read_populates_registers():
    base = 0x1000
    rm, _, _ = make_map(base=base)
    a = make_register(rm, 'a', 0x0, width=8)
    b = make_register(rm, 'b', 0x4, width=16)
    add_field(a, 'f', reg_offset=0, width=8)
    add_field(b, 'f', reg_offset=0, width=16)
    memory = bytearray([0x11, 0x00, 0x00, 0x00, 0x34, 0x12])
    rm._block_read_func = lambda address, size: bytes(
        memory[address - base:address - base + size])
    data = rm.read()
    assert data == bytes([0x11, 0x00, 0x00, 0x00, 0x34, 0x12])
    assert a.value == 0x11
    assert b.value == 0x1234


def test_map_write_serialises_little_endian():
    rm, _, _ = make_map(base=0x1000)
    a = make_register(rm, 'a', 0x0, width=8)
    b = make_register(rm, 'b', 0x4, width=16)
    add_field(a, 'f', reg_offset=0, width=8)
    add_field(b, 'f', reg_offset=0, width=16)
    a.value = 0xAA
    b.value = 0xBEEF
    blocks = []
    rm._block_write_func = lambda address, data: blocks.append(
        (address, bytes(data)))
    rm.write()
    assert blocks == [(0x1000, bytes([0xAA, 0x00, 0x00, 0x00, 0xEF, 0xBE]))]


def test_map_partial_write_stops_at_short_buffer():
    rm, _, _ = make_map(base=0x1000)
    a = make_register(rm, 'a', 0x0, width=8)
    b = make_register(rm, 'b', 0x4, width=16)
    add_field(a, 'f', reg_offset=0, width=8)
    add_field(b, 'f', reg_offset=0, width=16)
    b.value = 0x1111
    blocks = []
    rm._block_write_func = lambda address, data: blocks.append(
        (address, bytes(data)))
    short = bytes([0x42, 0x99])
    rm.write(short)
    assert a.value == 0x42
    assert b.value == 0x1111
    assert blocks == [(0x1000, short)]
