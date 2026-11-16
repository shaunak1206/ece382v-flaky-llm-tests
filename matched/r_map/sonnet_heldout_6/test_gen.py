import pytest

from r_map.Register import Register
from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.RegisterMap import RegisterMap, get_dimensions


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

IO_LOG = []
READ_VALUE = [0]
BLOCK_READ_DATA = [b""]


class IORegister(Register):
    """Register with fake IO functions (class attributes override whatever
    the base class provides)."""

    def _reg_read_func(self, address):
        IO_LOG.append(('r', address))
        return READ_VALUE[0]

    def _reg_write_func(self, address, value):
        IO_LOG.append(('w', address, value))


class IOMap(RegisterMap):
    def _block_read_func(self, address, size):
        IO_LOG.append(('br', address, size))
        return BLOCK_READ_DATA[0]

    def _block_write_func(self, address, data):
        IO_LOG.append(('bw', address, bytes(data)))


@pytest.fixture(autouse=True)
def _clean_io_state():
    del IO_LOG[:]
    READ_VALUE[0] = 0
    BLOCK_READ_DATA[0] = b""
    yield


def make_field(name, width=8, reset_val=0, access='RW'):
    return BitField(name=name, width=width, reset_val=reset_val, access=access)


def make_ref(name, bf, reg_offset=0, slice_width=0, field_offset=0):
    ref = BitFieldRef(name=name, reg_offset=reg_offset,
                      slice_width=slice_width, field_offset=field_offset)
    ref._add(bf)
    return ref


def make_reg(specs, cls=Register, name='reg', width=32, addr=0):
    """specs: iterable of (name, reg_offset, bf_width, reset_val, access)"""
    reg = cls(name=name, width=width, local_address=addr)
    for spec in specs:
        fname, offset, fwidth, reset = spec[:4]
        access = spec[4] if len(spec) > 4 else 'RW'
        bf = make_field(fname, width=fwidth, reset_val=reset, access=access)
        reg._add(make_ref(fname + '_ref', bf, reg_offset=offset))
    return reg


# ---------------------------------------------------------------------------
# BitField
# ---------------------------------------------------------------------------

def test_bitfield_invalid_arguments():
    with pytest.raises(ValueError):
        BitField(name='bf', width=0)
    with pytest.raises(ValueError):
        BitField(name='bf', width=4, reset_val=-1)


def test_bitfield_initial_value_and_mask():
    bf = make_field('bf', width=4, reset_val=3)
    assert bf.mask == 0xF
    assert bf.value == 3
    assert bf.reset_val == 3


def test_bitfield_value_is_masked_and_reset():
    bf = make_field('bf', width=4, reset_val=3)
    bf.value = 0x1F
    assert bf.value == 0xF
    bf.reset()
    assert bf.value == 3


def test_bitfield_rejects_unsupported_value_type():
    bf = make_field('bf', width=4)
    with pytest.raises(NotImplementedError):
        bf.value = 1.5


def test_bitfield_string_value_without_matching_enumeration():
    bf = make_field('bf', width=4)
    with pytest.raises(ValueError):
        bf.value = 'does_not_exist'


def test_bitfield_annotation_and_str_without_enumerations():
    bf = make_field('bf', width=4)
    bf.value = 5
    assert bf.annotation == hex(5)
    text = str(bf)
    assert 'width: 4' in text
    assert '0x5' in text


def test_bitfield_io_without_parent_raises():
    bf = make_field('bf', width=4)
    with pytest.raises(ValueError):
        bf.read()
    with pytest.raises(ValueError):
        bf.write(1)


def test_bitfield_read_and_write_through_register():
    reg = make_reg([('f', 0, 8, 0)], cls=IORegister, addr=0x20)
    bf = next(iter(reg)).bf

    READ_VALUE[0] = 0xF0
    assert bf.read() == 0xF0
    assert ('r', 0x20) in IO_LOG

    del IO_LOG[:]
    READ_VALUE[0] = 0xF0
    bf.write(0x0A)
    assert bf.value == 0x0A
    assert IO_LOG[-1] == ('w', 0x20, 0x0A)


def test_bitfield_write_skipped_when_unchanged_and_not_always_write():
    reg = make_reg([('f', 0, 8, 0)], cls=IORegister, addr=0x20)
    bf = next(iter(reg)).bf
    READ_VALUE[0] = 0x0A
    bf.write(0x0A, always_write=False)
    assert not any(entry[0] == 'w' for entry in IO_LOG)
    bf.write(0x0A, always_write=True)
    assert any(entry[0] == 'w' for entry in IO_LOG)


# ---------------------------------------------------------------------------
# BitFieldRef
# ---------------------------------------------------------------------------

def test_bitfieldref_invalid_arguments():
    with pytest.raises(ValueError):
        BitFieldRef(name='r', reg_offset=0, slice_width=-1)
    with pytest.raises(ValueError):
        BitFieldRef(name='r', reg_offset=0, field_offset=-1)


def test_bitfieldref_defaults_slice_width_to_field_width():
    bf = make_field('bf', width=6)
    ref = make_ref('ref', bf)
    assert ref.slice_width == 6
    assert ref.mask == 0x3F
    assert ref.bf is bf


def test_bitfieldref_add_rejects_non_bitfield():
    ref = BitFieldRef(name='ref', reg_offset=0)
    with pytest.raises(ValueError):
        ref._add(object())


def test_bitfieldref_value_with_offsets():
    bf = make_field('bf', width=8, reset_val=0xAB)
    bf.value = 0xAB
    ref = make_ref('ref', bf, reg_offset=8, slice_width=4, field_offset=4)
    assert ref.value == 0xA00
    assert ref.reset_val == 0xA00

    ref.value = 0x500
    assert bf.value == 0x5B


def test_bitfieldref_second_reference_marks_alias():
    bf = make_field('bf', width=8)
    ref1 = make_ref('ref1', bf)
    ref2 = make_ref('ref2', bf, reg_offset=8)
    assert ref2._alias is True
    assert ref2._ref is ref1
    assert ref1.bf is ref2.bf


def test_bitfieldref_validate_slice_wider_than_bitfield():
    bf = make_field('bf', width=4)
    good = make_ref('good', bf)
    bad = make_ref('bad', make_field('bf2', width=4), slice_width=8)
    assert len(list(bad.validate())) > len(list(good.validate()))


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

def test_register_value_combines_fields_and_setter_distributes():
    reg = make_reg([('a', 0, 4, 0), ('b', 4, 4, 0)], width=8)
    for ref in reg:
        ref.bf.value = 0xA if ref.bf.name == 'a' else 0x5
    assert reg.value == 0x5A

    reg.value = 0x3C
    fields = {ref.bf.name: ref.bf.value for ref in reg}
    assert fields == {'a': 0xC, 'b': 0x3}
    assert reg.value == 0x3C
    assert '0x3c' in str(reg)


def test_register_iterates_sorted_by_reg_offset():
    reg = Register(name='reg', width=16, local_address=0)
    for name, offset in (('hi', 8), ('lo', 0), ('mid', 4)):
        reg._add(make_ref(name + '_ref', make_field(name, width=4),
                          reg_offset=offset))
    assert [r.reg_offset for r in reg] == [0, 4, 8]


def test_register_reset_and_reset_val():
    reg = make_reg([('a', 0, 4, 3), ('b', 4, 4, 9)], width=8)
    reg.value = 0xFF
    assert reg.value == 0xFF
    assert reg.reset_val == 0x93
    reg.reset()
    assert reg.value == 0x93


def test_register_access_is_sorted_unique_join():
    reg = make_reg([('a', 0, 4, 0, 'RW'), ('b', 4, 2, 0, 'RO'),
                    ('c', 6, 2, 0, 'RW')], width=8)
    assert reg.access == 'RO|RW'


def test_register_validate_detects_empty_and_overlap():
    empty = Register(name='empty', width=8, local_address=0)
    assert len(list(empty.validate())) >= 1

    ok = make_reg([('a', 0, 4, 0), ('b', 4, 4, 0)], width=8, name='ok')
    overlapping = make_reg([('a', 0, 8, 0), ('b', 4, 4, 0)], width=8,
                           name='ovl')
    assert len(list(overlapping.validate())) > len(list(ok.validate()))


def test_register_read_and_write_use_io_functions():
    reg = make_reg([('f', 0, 8, 0)], cls=IORegister, width=16, addr=0x10)

    READ_VALUE[0] = 0x1FF
    assert reg.read() == 0x1FF  # raw value returned, not the masked one
    assert reg.value == 0xFF
    assert IO_LOG[-1] == ('r', 0x10)

    reg.write(0x12)
    assert reg.value == 0x12
    assert IO_LOG[-1] == ('w', 0x10, 0x12)

    reg.value = 0x34
    reg.write()
    assert IO_LOG[-1] == ('w', 0x10, 0x34)


# ---------------------------------------------------------------------------
# RegisterMap
# ---------------------------------------------------------------------------

def make_map():
    rmap = IOMap(name='map', local_address=0)
    rmap._add(make_reg([('fa', 0, 32, 0)], cls=Register, name='A',
                       width=32, addr=0))
    rmap._add(make_reg([('fb', 0, 16, 0)], cls=Register, name='B',
                       width=16, addr=4))
    return rmap


def test_get_dimensions():
    assert get_dimensions(0x10, 0x14, 32) == (4, 8, 4)
    assert get_dimensions(0, 2, 12) == (2, 4, 2)
    assert get_dimensions(0, 0, 8) == (0, 1, 1)


def test_register_map_block_dimensions():
    rmap = make_map()
    assert rmap._block_dimensions == (0, 6)


def test_register_map_empty_raises():
    rmap = IOMap(name='empty', local_address=0)
    with pytest.raises(ValueError):
        rmap._block_dimensions
    with pytest.raises(ValueError):
        rmap.write()


def test_register_map_read_updates_registers():
    rmap = make_map()
    data = bytes([0xEF, 0xBE, 0xAD, 0xDE, 0xFE, 0xCA])
    BLOCK_READ_DATA[0] = data
    assert rmap.read() == data
    assert IO_LOG[0] == ('br', 0, 6)
    values = {r.name: r.value for r in rmap}
    assert values == {'A': 0xDEADBEEF, 'B': 0xCAFE}


def test_register_map_write_without_data_packs_register_values():
    rmap = make_map()
    regs = {r.name: r for r in rmap}
    regs['A'].value = 0xDEADBEEF
    regs['B'].value = 0xCAFE
    rmap.write()
    assert IO_LOG[-1] == ('bw', 0, bytes([0xEF, 0xBE, 0xAD, 0xDE, 0xFE, 0xCA]))


def test_register_map_write_with_data_updates_registers_and_partial_write():
    rmap = make_map()
    regs = {r.name: r for r in rmap}

    rmap.write(bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06]))
    assert regs['A'].value == 0x04030201
    assert regs['B'].value == 0x0605
    assert IO_LOG[-1] == ('bw', 0, bytes([1, 2, 3, 4, 5, 6]))

    # Partial block write: only register A is covered by the supplied data
    rmap.write(bytes([0x11, 0x22, 0x33, 0x44]))
    assert regs['A'].value == 0x44332211
    assert regs['B'].value == 0x0605
    assert IO_LOG[-1] == ('bw', 0, bytes([0x11, 0x22, 0x33, 0x44]))
