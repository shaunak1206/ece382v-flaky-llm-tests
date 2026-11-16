from unittest import mock

import pytest

import r_map  # noqa: F401  (import package first to resolve circular imports)
from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.Register import Register
from r_map.RegisterMap import RegisterMap, get_dimensions


def make_reg(name, fields, width=32, local_address=0):
    """fields: iterable of (name, reg_offset, bf_width, reset_val, access)"""
    reg = Register(name=name, width=width, local_address=local_address)
    for fname, reg_offset, bf_width, reset_val, access in fields:
        bf = BitField(name=fname, width=bf_width, reset_val=reset_val,
                      access=access)
        ref = BitFieldRef(name=fname + '_ref', reg_offset=reg_offset)
        ref._add(bf)
        reg._add(ref)
    return reg


def make_full_reg(name, local_address, reset_val=0):
    return make_reg(name, [(name + '_f', 0, 32, reset_val, 'RW')],
                    local_address=local_address)


def patch_attr(cls, attr, func):
    return mock.patch.object(cls, attr, new=staticmethod(func), create=True)


# ---------------------------------------------------------------- BitField

def test_bitfield_defaults_and_mask():
    bf = BitField(name='bf', width=4, reset_val=0x5)
    assert bf.mask == 0xF
    assert bf.value == 0x5
    assert bf.width == 4


def test_bitfield_invalid_arguments():
    with pytest.raises(ValueError):
        BitField(name='bf', width=0)
    with pytest.raises(ValueError):
        BitField(name='bf', width=4, reset_val=-1)


def test_bitfield_value_setter_masks_and_rejects_bad_types():
    bf = BitField(name='bf', width=4)
    bf.value = 0x1F3
    assert bf.value == 0x3
    with pytest.raises(NotImplementedError):
        bf.value = 1.5


def test_bitfield_reset_restores_reset_val():
    bf = BitField(name='bf', width=8, reset_val=0xA5)
    bf.value = 0x11
    assert bf.value == 0x11
    bf.reset()
    assert bf.value == 0xA5


def test_bitfield_annotation_falls_back_to_hex():
    bf = BitField(name='bf', width=8, reset_val=0x2A)
    assert bf.annotation == hex(0x2A)


def test_bitfield_io_without_references_raises():
    bf = BitField(name='bf', width=4)
    with pytest.raises(ValueError):
        bf.read()
    with pytest.raises(ValueError):
        bf.write(1)


# -------------------------------------------------------------- BitFieldRef

def test_bitfieldref_value_and_reset_val_with_offsets():
    bf = BitField(name='bf', width=8, reset_val=0xAB)
    ref = BitFieldRef(name='ref', reg_offset=4, slice_width=4, field_offset=4)
    ref._add(bf)
    assert ref.bf is bf
    assert ref.value == 0xA0
    assert ref.reset_val == 0xA0


def test_bitfieldref_slice_width_defaults_to_bitfield_width():
    bf = BitField(name='bf', width=6)
    ref = BitFieldRef(name='ref', reg_offset=2)
    assert ref.slice_width == 0
    ref._add(bf)
    assert ref.slice_width == 6
    assert ref.mask == 0x3F


def test_bitfieldref_value_setter_preserves_other_bits():
    bf = BitField(name='bf', width=8, reset_val=0xFF)
    ref = BitFieldRef(name='ref', reg_offset=4, slice_width=4, field_offset=0)
    ref._add(bf)
    ref.value = 0x50
    assert bf.value == 0xF5


def test_bitfieldref_invalid_arguments():
    with pytest.raises(ValueError):
        BitFieldRef(name='ref', reg_offset=0, slice_width=-1)
    with pytest.raises(ValueError):
        BitFieldRef(name='ref', reg_offset=0, field_offset=-1)
    ref = BitFieldRef(name='ref', reg_offset=0)
    with pytest.raises(ValueError):
        ref._add(object())


def test_bitfieldref_validate_slice_wider_than_bitfield():
    good_bf = BitField(name='bf_good', width=8)
    good = BitFieldRef(name='good', reg_offset=0, slice_width=4)
    good._add(good_bf)

    bad_bf = BitField(name='bf_bad', width=4)
    bad = BitFieldRef(name='bad', reg_offset=0, slice_width=8)
    bad._add(bad_bf)

    assert len(list(bad.validate())) > len(list(good.validate()))


# ----------------------------------------------------------------- Register

def test_register_value_combines_fields():
    reg = make_reg('r', [('a', 0, 4, 0x3, 'RW'), ('b', 8, 4, 0xC, 'RW')])
    assert reg.value == 0xC03
    assert reg.reset_val == 0xC03


def test_register_value_setter_distributes_to_fields():
    reg = make_reg('r', [('a', 0, 4, 0, 'RW'), ('b', 8, 4, 0, 'RW')])
    reg.value = 0x5A7
    assert reg.value == 0x507
    reg.reset()
    assert reg.value == 0


def test_register_iterates_sorted_by_reg_offset():
    reg = make_reg('r', [('hi', 16, 4, 0, 'RW'),
                         ('lo', 0, 4, 0, 'RW'),
                         ('mid', 8, 4, 0, 'RW')])
    assert [c.reg_offset for c in reg] == [0, 8, 16]


def test_register_access_is_sorted_unique_join():
    reg = make_reg('r', [('a', 0, 1, 0, 'RW'),
                         ('b', 4, 1, 0, 'RO'),
                         ('c', 8, 1, 0, 'RW')])
    assert reg.access == 'RO|RW'


def test_register_str_contains_hex_value():
    reg = make_reg('r', [('a', 0, 4, 0x5, 'RW')])
    assert 'value: 0x00000005' in str(reg)


def test_register_validate_detects_empty_and_overlap():
    ok = make_reg('ok', [('a', 0, 8, 0, 'RW'), ('b', 8, 4, 0, 'RW')])
    empty = Register(name='empty')
    overlapping = make_reg('ov', [('a', 0, 8, 0, 'RW'), ('b', 4, 4, 0, 'RW')])

    n_ok = len(list(ok.validate()))
    assert len(list(empty.validate())) > 0
    assert len(list(overlapping.validate())) > n_ok


def test_register_read_and_write_use_io_functions():
    reg = make_reg('r', [('a', 0, 8, 0, 'RW')], local_address=0x10)
    reads, writes = [], []

    def rd(addr):
        reads.append(addr)
        return 0x1FF  # more bits than the field holds

    def wr(addr, val):
        writes.append((addr, val))

    with patch_attr(Register, '_reg_read_func', rd), \
            patch_attr(Register, '_reg_write_func', wr):
        ret = reg.read()
        assert ret == 0x1FF  # raw value is returned
        assert reg.value == 0xFF
        assert reads == [reg.address]

        reg.write(0x12)
        assert reg.value == 0x12
        assert writes[-1] == (reg.address, 0x12)

        reg.write()
        assert writes[-1] == (reg.address, 0x12)


def test_bitfield_write_does_read_modify_write():
    reg = make_reg('r', [('a', 0, 4, 0, 'RW')], local_address=0x20)
    bf = reg.a.bf if hasattr(reg, 'a') else list(reg)[0].bf
    writes = []

    with patch_attr(Register, '_reg_read_func', lambda addr: 0xF0), \
            patch_attr(Register, '_reg_write_func',
                       lambda addr, val: writes.append((addr, val))):
        bf.write(0x3)
        assert bf.value == 0x3
        assert writes == [(reg.address, 0x3)]

        writes.clear()
        bf.write(0x3, always_write=False)
        assert len(writes) == 0 or writes == [(reg.address, 0x3)]

        assert bf.read() == 0x0  # 0xF0 read back, low nibble is 0


# ------------------------------------------------------------- RegisterMap

def test_get_dimensions():
    assert get_dimensions(0x100, 0x104, 32) == (4, 8, 4)
    assert get_dimensions(0x100, 0x100, 12) == (0, 2, 2)


def test_register_map_empty_raises():
    rmap = RegisterMap(name='empty_map')
    with pytest.raises(ValueError):
        rmap.read()


def test_register_map_block_dimensions_and_read():
    rmap = RegisterMap(name='m')
    r0 = make_full_reg('r0', 0)
    r1 = make_full_reg('r1', 4)
    rmap._add(r0)
    rmap._add(r1)

    base = r0.address
    assert rmap._block_dimensions == (base, 8)

    calls = []

    def block_read(addr, size):
        calls.append((addr, size))
        return bytes(range(8))

    with patch_attr(RegisterMap, '_block_read_func', block_read):
        data = rmap.read()

    assert calls == [(base, 8)]
    assert data == bytes(range(8))
    assert r0.value == 0x03020100
    assert r1.value == 0x07060504


def test_register_map_write_from_register_values_and_from_data():
    rmap = RegisterMap(name='m')
    r0 = make_full_reg('r0', 0)
    r1 = make_full_reg('r1', 4)
    rmap._add(r0)
    rmap._add(r1)
    base = r0.address
    writes = []

    with patch_attr(RegisterMap, '_block_write_func',
                    lambda addr, data: writes.append((addr, bytes(data)))):
        r0.value = 0x11223344
        r1.value = 0xAABBCCDD
        rmap.write()
        assert writes[-1] == (base, bytes([0x44, 0x33, 0x22, 0x11,
                                           0xDD, 0xCC, 0xBB, 0xAA]))

        rmap.write(bytes([1, 0, 0, 0, 2, 0, 0, 0]))
        assert r0.value == 1
        assert r1.value == 2
        assert writes[-1] == (base, bytes([1, 0, 0, 0, 2, 0, 0, 0]))
