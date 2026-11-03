import pytest

from r_map import BitField, BitFieldRef, Enumeration, Register, RegisterMap


def _ref(name, reg_offset, bf, slice_width=0, field_offset=0):
    """Build a BitFieldRef and attach it to a bitfield."""
    ref = BitFieldRef(name=name, reg_offset=reg_offset,
                      slice_width=slice_width, field_offset=field_offset)
    ref._add(bf)
    return ref


def _register(name, local_address=0, width=32, slices=()):
    reg = Register(name=name, local_address=local_address, width=width)
    for s in slices:
        reg._add(s)
    return reg


def _full_register(name, local_address, width=32):
    """A register backed by one bitfield spanning its full width."""
    bf = BitField(name=name + '_f', width=width)
    return _register(name, local_address, width, [_ref(name + '_r', 0, bf)])


def test_bitfield_defaults():
    bf = BitField(name='f')
    assert bf.width == 1
    assert bf.reset_val == 0
    assert bf.access == 'XX'
    assert bf.value == 0
    assert bf.mask == 1


def test_bitfield_int_assignment_is_masked():
    bf = BitField(name='f', width=4)
    bf.value = 0x1F
    assert bf.value == 0xF
    bf.value = 0x25
    assert bf.value == 0x5


def test_bitfield_string_assignment_selects_enumeration():
    bf = BitField(name='mode', width=2)
    bf._add(Enumeration(name='OFF', value=0))
    bf._add(Enumeration(name='ON', value=1))
    bf.value = 'ON'
    assert bf.value == 1
    bf.value = 'OFF'
    assert bf.value == 0


def test_bitfield_unknown_enumeration_name_raises():
    bf = BitField(name='mode', width=2)
    bf._add(Enumeration(name='ON', value=1))
    with pytest.raises(ValueError):
        bf.value = 'BLINK'


def test_bitfield_validate_reports_duplicate_enum_values():
    bf = BitField(name='mode', width=2)
    bf._add(Enumeration(name='A', value=1))
    bf._add(Enumeration(name='B', value=1))
    errors = list(bf.validate())
    assert any('same value' in e.error for e in errors)


def test_bitfieldref_slice_width_defaults_to_bitfield_width():
    bf = BitField(name='f', width=6)
    ref = BitFieldRef(name='r', reg_offset=2)
    ref._add(bf)
    assert ref.slice_width == 6
    assert ref.mask == 0x3F
    assert ref.bf is bf


def test_bitfieldref_value_places_slice_at_reg_offset():
    bf = BitField(name='f', width=4)
    bf.value = 0xA
    ref = _ref('r', reg_offset=8, bf=bf)
    assert ref.value == 0xA00


def test_bitfieldref_setter_only_touches_its_slice():
    bf = BitField(name='f', width=8)
    bf.value = 0xFF
    ref = _ref('r', reg_offset=4, bf=bf, slice_width=4)
    ref.value = 0x0A0
    assert bf.value == 0xFA


def test_bitfieldref_field_offset_selects_high_nibble():
    bf = BitField(name='f', width=8)
    bf.value = 0xD6
    ref = _ref('r', reg_offset=0, bf=bf, slice_width=4, field_offset=4)
    assert ref.value == 0xD


def test_bitfieldref_validate_flags_slice_wider_than_field():
    bf = BitField(name='f', width=4)
    ref = _ref('r', reg_offset=0, bf=bf, slice_width=8)
    errors = list(ref.validate())
    assert any('larger than bitfield' in e.error for e in errors)


def test_bitfieldref_second_reference_becomes_alias():
    bf = BitField(name='f', width=4)
    first = _ref('a', reg_offset=0, bf=bf)
    second = _ref('b', reg_offset=4, bf=bf)
    assert second._alias is True
    assert second._ref is first
    assert not first._alias


def test_bitfieldref_copy_preserves_geometry_and_clones_field():
    bf = BitField(name='f', width=8)
    ref = _ref('r', reg_offset=4, bf=bf, slice_width=4, field_offset=2)
    dup = ref._copy()
    assert dup is not ref
    assert (dup.reg_offset, dup.slice_width, dup.field_offset) == (4, 4, 2)
    assert dup.bf is not bf
    assert dup.bf.name == 'f'
    assert dup.bf.width == 8


def test_register_value_ors_slices_at_their_offsets():
    lo = BitField(name='lo', width=8)
    lo.value = 0x34
    hi = BitField(name='hi', width=8)
    hi.value = 0x12
    reg = _register('r', slices=[_ref('lo_r', 0, lo), _ref('hi_r', 8, hi)])
    assert reg.value == 0x1234


def test_register_value_setter_distributes_to_slices():
    lo = BitField(name='lo', width=8)
    hi = BitField(name='hi', width=8)
    reg = _register('r', slices=[_ref('lo_r', 0, lo), _ref('hi_r', 8, hi)])
    reg.value = 0xABCD
    assert lo.value == 0xCD
    assert hi.value == 0xAB
    assert reg.value == 0xABCD


def test_register_iterates_children_in_offset_order():
    hi = BitField(name='hi', width=4)
    lo = BitField(name='lo', width=4)
    reg = _register('r', slices=[_ref('hi_r', 4, hi), _ref('lo_r', 0, lo)])
    assert [c.name for c in reg] == ['lo_r', 'hi_r']


def test_register_validate_reports_missing_slices():
    reg = _register('empty')
    errors = list(reg.validate())
    assert any('No bitfieldrefs' in e.error for e in errors)


def test_register_validate_reports_overlapping_slices():
    a = BitField(name='a', width=8)
    b = BitField(name='b', width=8)
    reg = _register('r', slices=[_ref('a_r', 0, a), _ref('b_r', 4, b)])
    errors = list(reg.validate())
    assert any('overlaps' in e.error for e in errors)


def test_register_validate_reports_non_ref_child():
    reg = _register('r', slices=[_ref('a_r', 0, BitField(name='a', width=4))])
    reg._add(BitField(name='stray', width=1))
    errors = list(reg.validate())
    assert any('is not of type' in e.error for e in errors)


def test_register_read_write_use_parent_map_io():
    rm = RegisterMap(name='rm', local_address=0x1000)
    mem = {}
    rm._reg_read_func = lambda addr: mem.get(addr, 0)
    rm._reg_write_func = lambda addr, val: mem.__setitem__(addr, val)
    reg = _full_register('r0', local_address=0x4)
    rm._add(reg)
    reg.value = 0x12345678
    reg.write()
    assert mem[0x1004] == 0x12345678
    mem[0x1004] = 0xCAFE
    assert reg.read() == 0xCAFE
    assert reg.value == 0xCAFE


def test_bitfield_write_then_read_round_trips_through_register():
    rm = RegisterMap(name='rm', local_address=0x1000)
    mem = {}
    rm._reg_read_func = lambda addr: mem.get(addr, 0)
    rm._reg_write_func = lambda addr, val: mem.__setitem__(addr, val)
    bf = BitField(name='f', width=4)
    reg = _register('r0', local_address=0x4,
                    slices=[_ref('f_r', reg_offset=4, bf=bf)])
    rm._add(reg)
    bf.write(0x5)
    assert mem[0x1004] == 0x50
    mem[0x1004] = 0xA0
    assert bf.read() == 0xA
    assert bf.value == 0xA


def test_regmap_block_read_splits_bytes_across_registers():
    rm = RegisterMap(name='rm', local_address=0x0)
    r0 = _full_register('r0', local_address=0x0)
    r1 = _full_register('r1', local_address=0x4)
    rm._add(r0)
    rm._add(r1)
    payload = (0x11223344).to_bytes(4, 'little') + (0xAABBCCDD).to_bytes(4, 'little')
    rm._block_read_func = lambda addr, size: payload[addr:addr + size]
    data = rm.read()
    assert data == payload
    assert r0.value == 0x11223344
    assert r1.value == 0xAABBCCDD


def test_regmap_block_write_packs_little_endian():
    rm = RegisterMap(name='rm', local_address=0x0)
    r0 = _full_register('r0', local_address=0x0)
    r1 = _full_register('r1', local_address=0x4)
    rm._add(r0)
    rm._add(r1)
    r0.value = 0x11223344
    r1.value = 0xAABBCCDD
    seen = []
    rm._block_write_func = lambda addr, data: seen.append((addr, bytes(data)))
    rm.write()
    expected = (0x11223344).to_bytes(4, 'little') + (0xAABBCCDD).to_bytes(4, 'little')
    assert seen == [(0, expected)]


def test_regmap_write_with_buffer_updates_only_covered_registers():
    rm = RegisterMap(name='rm', local_address=0x0)
    r0 = _full_register('r0', local_address=0x0)
    r1 = _full_register('r1', local_address=0x4)
    rm._add(r0)
    rm._add(r1)
    r1.value = 0x99
    seen = []
    rm._block_write_func = lambda addr, data: seen.append((addr, bytes(data)))
    short = (0x11223344).to_bytes(4, 'little')
    rm.write(short)
    assert r0.value == 0x11223344
    assert r1.value == 0x99
    assert seen == [(0, short)]


def test_regmap_read_and_write_require_registers():
    rm = RegisterMap(name='empty', local_address=0x0)
    with pytest.raises(ValueError):
        rm.read()
    with pytest.raises(ValueError):
        rm.write()
