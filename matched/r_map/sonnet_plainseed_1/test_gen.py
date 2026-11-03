import importlib

import pytest

from r_map import (BitField, BitFieldRef, Enumeration, Node, Register,
                   RegisterMap)


# ---------------------------------------------------------------- helpers

def make_ref(name, bf, reg_offset=0, slice_width=0, field_offset=0):
    ref = BitFieldRef(name=name, reg_offset=reg_offset, slice_width=slice_width,
                      field_offset=field_offset)
    ref._add(bf)
    return ref


def make_reg(name='reg', local_address=0, width=8, lo_reset=0, hi_reset=0):
    """Register with a 4 bit field 'lo' at bit 0 and 'hi' at bit 4."""
    reg = Register(name=name, local_address=local_address, width=width)
    lo = BitField(name='lo', width=4, reset_val=lo_reset, access='RW')
    hi = BitField(name='hi', width=4, reset_val=hi_reset, access='RW')
    reg._add(make_ref('hi_ref', hi, reg_offset=4))
    reg._add(make_ref('lo_ref', lo, reg_offset=0))
    return reg, lo, hi


def attach_memory(reg, store):
    writes = []

    def rd(addr):
        return store[addr]

    def wr(addr, val):
        writes.append((addr, val))
        store[addr] = val

    reg._reg_read_func = rd
    reg._reg_write_func = wr
    return writes


# --------------------------------------------------------------- BitField

def test_bitfield_defaults_and_reset():
    bf = BitField(name='f', width=4, reset_val=0x5)
    assert bf.width == 4
    assert bf.mask == 0xF
    assert bf.access == 'XX'
    assert bf.value == 0x5
    bf.value = 0xA
    assert bf.value == 0xA
    bf.reset()
    assert bf.value == 0x5


def test_bitfield_invalid_arguments():
    with pytest.raises(ValueError):
        BitField(name='f', width=0)
    with pytest.raises(ValueError):
        BitField(name='f', width=4, reset_val=-1)
    with pytest.raises(ValueError):
        BitField(width=4)  # name is required


def test_bitfield_value_masks_to_width():
    bf = BitField(name='f', width=4)
    bf.value = 0x1F
    assert bf.value == 0xF
    bf.value = 0x10
    assert bf.value == 0


def test_bitfield_value_from_enumeration():
    bf = BitField(name='f', width=4)
    on = Enumeration(name='ON', value=1)
    off = Enumeration(name='OFF', value=0)
    bf._add(off)
    bf._add(on)
    bf.value = 'ON'
    assert bf.value == 1
    bf.value = 'OFF'
    assert bf.value == 0
    bf.value = on
    assert bf.value == 1
    with pytest.raises(ValueError):
        bf.value = 'MISSING'
    with pytest.raises(NotImplementedError):
        bf.value = 1.5


def test_bitfield_annotation():
    bf = BitField(name='f', width=4)
    bf._add(Enumeration(name='OFF', value=0))
    bf._add(Enumeration(name='ON', value=1))
    bf.value = 1
    assert bf.annotation == 'ON'
    bf.value = 0
    assert bf.annotation == 'OFF'
    bf.value = 5
    assert bf.annotation == hex(5)


def test_bitfield_validate_duplicate_enumerations():
    bf = BitField(name='f', width=4)
    assert list(bf.validate()) == []
    bf._add(Enumeration(name='A', value=1))
    bf._add(Enumeration(name='B', value=1))
    bf._add(Enumeration(name='C', value=2))
    errors = list(bf.validate())
    assert len(errors) == 1
    assert errors[0].obj is bf
    assert 'same value' in errors[0].error


def test_bitfield_str_contains_summary():
    bf = BitField(name='myfield', width=4, reset_val=3)
    text = str(bf)
    assert text.startswith('BitField: myfield')
    assert 'width: 4' in text
    assert 'reset_val' in text


def test_bitfield_io_without_parent_raises():
    bf = BitField(name='f', width=4)
    with pytest.raises(ValueError):
        bf.read()
    with pytest.raises(ValueError):
        bf.write(1)


def test_bitfield_read_and_write_through_register():
    reg, lo, hi = make_reg(local_address=0x20)
    store = {0x20: 0xF0}
    writes = attach_memory(reg, store)

    # read-modify-write keeps the other field intact
    lo.write(0x3)
    assert store[0x20] == 0xF3
    assert writes == [(0x20, 0xF3)]
    assert hi.value == 0xF

    store[0x20] = 0xA7
    assert hi.read() == 0xA
    assert lo.value == 0x7


def test_bitfield_write_skips_when_value_unchanged():
    reg, lo, hi = make_reg(local_address=0)
    store = {0: 0x53}
    writes = attach_memory(reg, store)

    lo.write(0x3, always_write=False)
    assert writes == []

    lo.write(0x4, always_write=False)
    assert writes == [(0, 0x54)]
    assert store[0] == 0x54


# ------------------------------------------------------------ BitFieldRef

def test_bitfieldref_value_and_reset_mapping():
    bf = BitField(name='f', width=8, reset_val=0xB4)
    ref = make_ref('r', bf, reg_offset=4, slice_width=4, field_offset=2)
    assert ref.bf is bf
    assert ref.mask == 0xF
    assert ref.reset_val == 0xD0
    assert ref.value == 0xD0

    ref.value = 0x30
    assert bf.value == 0x8C
    assert ref.value == 0x30
    # bits of the field outside the slice are untouched
    assert bf.value & 0x3 == 0


def test_bitfieldref_default_slice_width_uses_bitfield_width():
    bf = BitField(name='f', width=6)
    ref = make_ref('r', bf, reg_offset=2)
    assert ref.slice_width == 6
    assert ref.mask == 0x3F


def test_bitfieldref_invalid_arguments():
    with pytest.raises(ValueError):
        BitFieldRef(name='r', reg_offset=0, slice_width=-1)
    with pytest.raises(ValueError):
        BitFieldRef(name='r', reg_offset=0, field_offset=-1)


def test_bitfieldref_add_rules():
    ref = BitFieldRef(name='r', reg_offset=0)
    with pytest.raises(ValueError):
        ref._add(Node(name='plain'))
    bf = BitField(name='f', width=4)
    ref._add(bf)
    ref._add(bf)  # adding twice is a no-op
    assert len(ref) == 1
    assert bf._references == {ref}
    assert bf.parent is ref


def test_bitfieldref_alias_when_bitfield_shared():
    bf = BitField(name='f', width=4)
    first = make_ref('first', bf)
    second = make_ref('second', bf, reg_offset=4)
    assert bf._references == {first, second}
    assert second._alias is True
    assert second._ref is first
    assert not first._alias


def test_bitfieldref_validate_slice_wider_than_field():
    bf = BitField(name='f', width=4)
    ref = make_ref('r', bf, slice_width=8)
    errors = list(ref.validate())
    assert len(errors) == 1
    assert 'Slice width' in errors[0].error

    ok = make_ref('ok', BitField(name='g', width=4), slice_width=4)
    assert list(ok.validate()) == []


# --------------------------------------------------------------- Register

def test_register_value_reset_val_and_reset():
    reg, lo, hi = make_reg(lo_reset=0x3, hi_reset=0xA)
    assert reg.reset_val == 0xA3
    assert reg.value == 0xA3

    reg.value = 0x5C
    assert lo.value == 0xC
    assert hi.value == 0x5
    assert reg.value == 0x5C

    reg.reset()
    assert reg.value == 0xA3


def test_register_iteration_order_access_and_str():
    reg = Register(name='r', local_address=0x10, width=8)
    top = BitField(name='top', width=4, access='RO')
    bottom = BitField(name='bottom', width=4, access='RW')
    reg._add(make_ref('top_ref', top, reg_offset=4))
    reg._add(make_ref('bottom_ref', bottom, reg_offset=0))
    assert [c.name for c in reg] == ['bottom_ref', 'top_ref']
    assert reg.access == 'RO|RW'

    reg.value = 0xA3
    text = str(reg)
    assert text.startswith('Register: r')
    assert '0x00000010' in text
    assert 'value: 0xa3' in text


def test_register_validate():
    reg, _, _ = make_reg()
    assert list(reg.validate()) == []

    empty = Register(name='empty', local_address=0)
    errors = list(empty.validate())
    assert len(errors) == 1
    assert 'No bitfieldrefs' in errors[0].error

    overlap = Register(name='ovl', local_address=0)
    overlap._add(make_ref('a', BitField(name='fa', width=4), reg_offset=0))
    overlap._add(make_ref('b', BitField(name='fb', width=4), reg_offset=2))
    errors = list(overlap.validate())
    assert len(errors) == 1
    assert 'overlaps' in errors[0].error

    wrong_child = Register(name='bad', local_address=0)
    wrong_child._add(Enumeration(name='e', value=1))
    errors = list(wrong_child.validate())
    assert len(errors) == 1
    assert 'is not of type' in errors[0].error


def test_register_read_and_write_use_io_functions():
    reg, lo, hi = make_reg(local_address=0x40)
    store = {0x40: 0x9E}
    writes = attach_memory(reg, store)

    assert reg.read() == 0x9E
    assert hi.value == 0x9
    assert lo.value == 0xE

    reg.write(0x21)
    assert reg.value == 0x21
    assert writes[-1] == (0x40, 0x21)

    lo.value = 0x7
    reg.write()
    assert writes[-1] == (0x40, 0x27)


def test_register_io_functions_inherited_from_parent():
    regmap = RegisterMap(name='m', local_address=0x100)
    reg, lo, hi = make_reg(local_address=0x4)
    regmap._add(reg)
    seen = []

    def rd(addr):
        seen.append(addr)
        return 0x5A

    regmap._reg_read_func = rd
    assert reg.address == 0x104
    assert reg.read() == 0x5A
    assert seen == [0x104]
    assert hi.value == 0x5
    assert lo.value == 0xA

    orphan, _, _ = make_reg(name='orphan')
    with pytest.raises(AttributeError):
        orphan.read()


# ------------------------------------------------------------ RegisterMap

def test_get_dimensions():
    get_dimensions = importlib.import_module('r_map.RegisterMap').get_dimensions
    assert get_dimensions(0x100, 0x102, 16) == (2, 4, 2)
    assert get_dimensions(0x100, 0x100, 8) == (0, 1, 1)
    assert get_dimensions(0x100, 0x104, 12) == (4, 6, 2)


def test_registermap_without_registers_raises():
    regmap = RegisterMap(name='empty', local_address=0)
    with pytest.raises(ValueError):
        regmap._block_dimensions
    with pytest.raises(ValueError):
        regmap.read()
    with pytest.raises(ValueError):
        regmap.write()


def make_map():
    regmap = RegisterMap(name='m', local_address=0x1000)
    wide = Register(name='wide', local_address=0, width=16)
    wide._add(make_ref('wide_ref', BitField(name='wide_f', width=16)))
    narrow = Register(name='narrow', local_address=2, width=8)
    narrow._add(make_ref('narrow_ref', BitField(name='narrow_f', width=8)))
    regmap._add(narrow)
    regmap._add(wide)
    return regmap, wide, narrow


def test_registermap_read_splits_block_little_endian():
    regmap, wide, narrow = make_map()
    calls = []

    def block_read(addr, size):
        calls.append((addr, size))
        return b'\x34\x12\xab'

    regmap._block_read_func = block_read
    assert regmap._block_dimensions == (0x1000, 3)
    data = regmap.read()
    assert data == b'\x34\x12\xab'
    assert calls == [(0x1000, 3)]
    assert wide.value == 0x1234
    assert narrow.value == 0xAB


def test_registermap_write_from_registers_and_from_data():
    regmap, wide, narrow = make_map()
    writes = []
    regmap._block_write_func = lambda addr, data: writes.append((addr, bytes(data)))

    wide.value = 0x1234
    narrow.value = 0xAB
    regmap.write()
    assert writes[-1] == (0x1000, b'\x34\x12\xab')

    regmap.write(b'\x78\x56\x99')
    assert writes[-1] == (0x1000, b'\x78\x56\x99')
    assert wide.value == 0x5678
    assert narrow.value == 0x99
