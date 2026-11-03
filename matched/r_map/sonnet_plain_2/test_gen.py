import pytest

from r_map import (BitField, BitFieldRef, Enumeration, Register, RegisterMap)


def make_ref(name, bf, reg_offset=0, **kwargs):
    ref = BitFieldRef(name=name, reg_offset=reg_offset, **kwargs)
    ref._add(bf)
    return ref


def make_reg(name, addr, specs, width=32, access='RW'):
    """specs: iterable of (field_name, reg_offset, field_width, reset_val)"""
    reg = Register(name=name, local_address=addr, width=width)
    for fname, offset, fwidth, reset in specs:
        bf = BitField(name=fname, width=fwidth, reset_val=reset, access=access)
        reg._add(make_ref(fname, bf, reg_offset=offset))
    return reg


def two_field_reg(addr=0x10):
    return make_reg('r', addr, [('lo', 0, 8, 0x12), ('hi', 8, 8, 0x34)])


# ---------------------------------------------------------------- BitField

def test_bitfield_defaults_and_initial_value():
    bf = BitField(name='bf')
    assert bf.width == 1
    assert bf.reset_val == 0
    assert bf.access == 'XX'
    assert bf.value == 0
    assert bf.mask == 1

    bf2 = BitField(name='bf2', width=4, reset_val=0x5)
    assert bf2.value == 0x5
    assert bf2.mask == 0xF


def test_bitfield_invalid_construction():
    with pytest.raises(ValueError):
        BitField(name='bad', width=0)
    with pytest.raises(ValueError):
        BitField(name='bad', width=4, reset_val=-1)
    with pytest.raises(ValueError):
        BitField(width=4)  # name is required
    with pytest.raises(ValueError):
        BitField(name='bad', width=4, bogus=1)


def test_bitfield_value_masked_to_width_and_reset():
    bf = BitField(name='bf', width=4, reset_val=0x3)
    bf.value = 0x1F
    assert bf.value == 0xF
    bf.reset()
    assert bf.value == 0x3


def test_bitfield_value_from_enumeration_name_and_object():
    bf = BitField(name='mode', width=4)
    on = Enumeration(name='ON', value=0x9)
    off = Enumeration(name='OFF', value=0x2)
    bf._add(on)
    bf._add(off)

    bf.value = 'ON'
    assert bf.value == 0x9
    bf.value = off
    assert bf.value == 0x2
    assert bf.annotation == 'OFF'

    bf.value = 0x7
    assert bf.annotation == hex(0x7)


def test_bitfield_value_rejects_bad_inputs():
    bf = BitField(name='mode', width=4)
    bf._add(Enumeration(name='ON', value=1))
    with pytest.raises(ValueError):
        bf.value = 'MISSING'
    with pytest.raises(NotImplementedError):
        bf.value = 1.5


def test_bitfield_validate_duplicate_enumerations():
    bf = BitField(name='mode', width=4)
    bf._add(Enumeration(name='A', value=1))
    bf._add(Enumeration(name='B', value=1))
    bf._add(Enumeration(name='C', value=2))
    errors = list(bf.validate())
    assert len(errors) == 1
    assert 'same value' in errors[0].error
    assert errors[0].obj is bf

    ok = BitField(name='ok', width=4)
    ok._add(Enumeration(name='A', value=1))
    ok._add(Enumeration(name='B', value=2))
    assert list(ok.validate()) == []


def test_bitfield_read_write_without_parent_raises():
    bf = BitField(name='bf', width=4)
    with pytest.raises(ValueError):
        bf.read()
    with pytest.raises(ValueError):
        bf.write(1)


# -------------------------------------------------------------- BitFieldRef

def test_bitfieldref_slice_width_defaults_to_field_width():
    bf = BitField(name='bf', width=6)
    ref = make_ref('ref', bf, reg_offset=2)
    assert ref.slice_width == 6
    assert ref.mask == 0x3F
    assert ref.bf is bf
    assert bf.parent is ref
    assert ref in bf._references


def test_bitfieldref_add_rejects_non_bitfield_and_is_idempotent():
    ref = BitFieldRef(name='r', reg_offset=0)
    with pytest.raises(ValueError):
        ref._add(Enumeration(name='E', value=1))

    bf = BitField(name='bf', width=4)
    ref._add(bf)
    ref._add(bf)
    assert len(ref) == 1
    assert len(bf._references) == 1


def test_bitfieldref_value_and_reset_val_with_offsets():
    bf = BitField(name='bf', width=8, reset_val=0xC3)
    bf.value = 0xA5
    ref = make_ref('ref', bf, reg_offset=8, slice_width=4, field_offset=4)
    assert ref.value == 0xA00
    assert ref.reset_val == 0xC00


def test_bitfieldref_value_setter_updates_only_its_slice():
    bf = BitField(name='bf', width=8)
    bf.value = 0xA5
    ref = make_ref('ref', bf, reg_offset=8, slice_width=4, field_offset=4)
    ref.value = 0x500
    assert bf.value == 0x55
    # bits of the register outside the slice are ignored
    ref.value = 0xFF00 | 0xFF
    assert bf.value == 0xF5


def test_bitfieldref_alias_shares_field():
    bf = BitField(name='bf', width=4)
    ref1 = make_ref('a', bf, reg_offset=0)
    ref2 = make_ref('b', bf, reg_offset=8)
    assert ref2._alias is True
    assert ref2._ref is ref1
    assert len(bf._references) == 2

    bf.value = 0x6
    assert ref1.value == 0x6
    assert ref2.value == 0x600


def test_bitfieldref_validate_slice_wider_than_field():
    bf = BitField(name='bf', width=4)
    bad = make_ref('bad', bf, slice_width=8)
    errors = list(bad.validate())
    assert any('Slice width' in e.error for e in errors)

    good = make_ref('good', BitField(name='g', width=4), slice_width=2)
    assert list(good.validate()) == []


# ----------------------------------------------------------------- Register

def test_register_defaults_and_required_args():
    reg = Register(name='r', local_address=0x4)
    assert reg.width == 32
    assert reg.address == 0x4
    with pytest.raises(TypeError):
        Register(name='r')
    with pytest.raises(ValueError):
        Register(name='r', local_address=0, bogus=1)


def test_register_value_composes_and_distributes_fields():
    reg = two_field_reg()
    assert reg.value == 0x3412
    assert reg.reset_val == 0x3412

    reg.value = 0xAB56
    assert reg['lo'].bf.value == 0x56
    assert reg['hi'].bf.value == 0xAB
    assert reg.value == 0xAB56

    reg.value = 0x1FFFF
    assert reg.value == 0xFFFF


def test_register_children_iterate_in_reg_offset_order():
    reg = Register(name='r', local_address=0)
    for name, offset in (('c', 16), ('a', 0), ('b', 8)):
        reg._add(make_ref(name, BitField(name=name + '_f', width=8),
                          reg_offset=offset))
    assert [c.name for c in reg] == ['a', 'b', 'c']


def test_register_access_and_str():
    reg = Register(name='r', local_address=0x10, width=16)
    reg._add(make_ref('x', BitField(name='x', width=4, access='RW'), 0))
    reg._add(make_ref('y', BitField(name='y', width=4, access='RO'), 4))
    reg._add(make_ref('z', BitField(name='z', width=4, access='RW'), 8))
    assert reg.access == 'RO|RW'

    reg.value = 0x321
    text = str(reg)
    assert text.startswith('Register: r (0x00000010)')
    assert text.endswith('value: 0x0321')


def test_register_validate_reports_empty_overlap_and_wrong_child_type():
    empty = Register(name='empty', local_address=0)
    assert any('No bitfieldrefs' in e.error for e in empty.validate())

    overlap = Register(name='ov', local_address=0)
    overlap._add(make_ref('a', BitField(name='a_f', width=4), reg_offset=0))
    overlap._add(make_ref('b', BitField(name='b_f', width=4), reg_offset=2))
    assert any('overlaps' in e.error for e in overlap.validate())

    wrong = Register(name='wrong', local_address=0)
    wrong._add(BitField(name='raw', width=4))
    assert any('BitFieldRef' in e.error for e in wrong.validate())

    assert list(two_field_reg().validate()) == []


def test_register_read_and_write_use_io_functions():
    reg = two_field_reg(addr=0x40)
    writes = []
    reg._reg_read_func = lambda addr: 0xFFFF1234 if addr == 0x40 else 0
    reg._reg_write_func = lambda addr, val: writes.append((addr, val))

    # read returns the raw value, even bits that no field covers
    assert reg.read() == 0xFFFF1234
    assert reg.value == 0x1234

    reg.write()
    assert writes == [(0x40, 0x1234)]

    reg.write(0xBEEF)
    assert reg.value == 0xBEEF
    assert writes[-1] == (0x40, 0xBEEF)


def test_bitfield_write_does_read_modify_write_of_parent_register():
    reg = two_field_reg(addr=0x20)
    writes = []
    reg._reg_read_func = lambda addr: 0xAB12
    reg._reg_write_func = lambda addr, val: writes.append((addr, val))

    hi = reg['hi'].bf
    hi.write(0x34)
    assert writes == [(0x20, 0x3412)]

    # writing the value already present is skipped if always_write is False
    del writes[:]
    reg['lo'].bf.write(0x12, always_write=False)
    assert writes == []
    reg['lo'].bf.write(0x12, always_write=True)
    assert writes == [(0x20, 0xAB12)]

    assert hi.read() == 0xAB


# -------------------------------------------------------------- RegisterMap

def build_map():
    rmap = RegisterMap(name='m', local_address=0x100)
    reg_b = make_reg('b', 0x4, [('f', 0, 16, 0)], width=16)
    reg_a = make_reg('a', 0x0, [('f', 0, 32, 0)], width=32)
    rmap._add(reg_b)  # intentionally out of address order
    rmap._add(reg_a)
    return rmap, reg_a, reg_b


def test_registermap_addresses_and_ordering():
    rmap, reg_a, reg_b = build_map()
    assert reg_a.address == 0x100
    assert reg_b.address == 0x104
    assert [r.name for r in rmap] == ['a', 'b']
    assert rmap._block_dimensions == (0x100, 6)


def test_registermap_read_splits_block_into_registers():
    rmap, reg_a, reg_b = build_map()
    calls = []

    def block_read(addr, size):
        calls.append((addr, size))
        return b'\x78\x56\x34\x12\xcd\xab'

    rmap._block_read_func = block_read
    data = rmap.read()
    assert calls == [(0x100, 6)]
    assert data == b'\x78\x56\x34\x12\xcd\xab'
    assert reg_a.value == 0x12345678
    assert reg_b.value == 0xABCD


def test_registermap_write_builds_block_from_register_values():
    rmap, reg_a, reg_b = build_map()
    written = []
    rmap._block_write_func = lambda addr, data: written.append((addr, bytes(data)))

    reg_a.value = 0x12345678
    reg_b.value = 0xABCD
    rmap.write()
    assert written == [(0x100, b'\x78\x56\x34\x12\xcd\xab')]


def test_registermap_write_with_data_updates_registers():
    rmap, reg_a, reg_b = build_map()
    written = []
    rmap._block_write_func = lambda addr, data: written.append((addr, bytes(data)))

    payload = bytearray(b'\x01\x00\x00\x00\x02\x00')
    rmap.write(payload)
    assert reg_a.value == 1
    assert reg_b.value == 2
    assert written == [(0x100, bytes(payload))]


def test_registermap_without_registers_raises():
    rmap = RegisterMap(name='empty', local_address=0)
    with pytest.raises(ValueError, match='no registers'):
        rmap.read()
    with pytest.raises(ValueError):
        rmap.write()
