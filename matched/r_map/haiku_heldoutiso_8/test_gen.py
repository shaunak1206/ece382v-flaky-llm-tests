import pytest

from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.Register import Register
from r_map.RegisterMap import RegisterMap, get_dimensions
from r_map.ValidationError import ValidationError


def make_field(reg, name, reg_offset, slice_width, field_offset=0,
               bf_width=8, bf_value=None):
    """Attach a BitFieldRef backed by a new BitField to a register."""
    bf = BitField(name=name + '_bf', width=bf_width)
    if bf_value is not None:
        bf.value = bf_value
    ref = BitFieldRef(name=name, reg_offset=reg_offset,
                      slice_width=slice_width, field_offset=field_offset)
    ref._add(bf)
    reg._children[ref.name] = ref
    return ref


# ---------------------------------------------------------------- BitField

def test_bitfield_defaults():
    bf = BitField(name='f')
    assert bf.width == 1
    assert bf.value == 0
    assert bf.mask == 0b1


def test_bitfield_rejects_invalid_width_and_reset_val():
    with pytest.raises(ValueError):
        BitField(name='f', width=0)
    with pytest.raises(ValueError):
        BitField(name='f', reset_val=-1)


def test_bitfield_value_is_masked_to_width():
    bf = BitField(name='f', width=4)
    bf.value = 0x1F
    assert bf.value == 0xF
    bf.value = 0x12
    assert bf.value == 0x2


def test_bitfield_initial_value_is_reset_val():
    bf = BitField(name='f', width=8, reset_val=0x5A)
    assert bf.value == 0x5A


def test_bitfield_reset_restores_reset_val():
    bf = BitField(name='f', width=4, reset_val=3)
    bf.value = 7
    bf.reset()
    assert bf.value == 3


def test_bitfield_rejects_unsupported_value_type():
    bf = BitField(name='f', width=4)
    with pytest.raises(NotImplementedError):
        bf.value = 1.5


def test_bitfield_annotation_falls_back_to_hex_without_enumerations():
    bf = BitField(name='f', width=8)
    bf.value = 0x2A
    assert bf.annotation == hex(0x2A)


def test_bitfield_read_and_write_require_parent_register():
    bf = BitField(name='f', width=4)
    with pytest.raises(ValueError):
        bf.read()
    with pytest.raises(ValueError):
        bf.write(1)


# ---------------------------------------------------------------- BitFieldRef

def test_bitfieldref_slice_width_taken_from_bitfield_when_zero():
    ref = BitFieldRef(name='r', reg_offset=0)
    assert ref.slice_width == 0
    ref._add(BitField(name='bf', width=8))
    assert ref.slice_width == 8
    assert ref.mask == 0xFF


def test_bitfieldref_add_rejects_non_bitfield():
    ref = BitFieldRef(name='r', reg_offset=0, slice_width=4)
    with pytest.raises(ValueError):
        ref._add("not a bitfield")


def test_bitfieldref_add_links_bitfield_and_references():
    bf = BitField(name='bf', width=8)
    ref = BitFieldRef(name='r', reg_offset=0, slice_width=4)
    ref._add(bf)
    assert ref.bf is bf
    assert ref in bf._references


def test_bitfieldref_rejects_negative_arguments():
    with pytest.raises(ValueError):
        BitFieldRef(name='r', reg_offset=0, slice_width=-1)
    with pytest.raises(ValueError):
        BitFieldRef(name='r', reg_offset=0, field_offset=-1)


def test_bitfieldref_value_extracts_field_and_shifts_to_reg_offset():
    bf = BitField(name='bf', width=8, reset_val=0xAB)
    ref = BitFieldRef(name='r', reg_offset=4, slice_width=4, field_offset=4)
    ref._add(bf)
    assert ref.value == 0xA0
    assert ref.reset_val == 0xA0


def test_bitfieldref_value_setter_preserves_other_bits():
    bf = BitField(name='bf', width=8)
    bf.value = 0x0F
    ref = BitFieldRef(name='r', reg_offset=0, slice_width=4, field_offset=4)
    ref._add(bf)
    ref.value = 0x05
    assert bf.value == 0x5F


def test_bitfieldref_validate_flags_slice_wider_than_bitfield():
    bf = BitField(name='bf', width=4)
    ref = BitFieldRef(name='r', reg_offset=0, slice_width=8)
    ref._add(bf)
    errors = [e for e in ref.validate() if isinstance(e, ValidationError)]
    assert errors


# ---------------------------------------------------------------- Register

def test_register_value_combines_fields_with_or():
    reg = Register(name='reg', address=0, width=8)
    make_field(reg, 'a', reg_offset=0, slice_width=4, bf_width=4, bf_value=0x3)
    make_field(reg, 'b', reg_offset=4, slice_width=4, bf_width=4, bf_value=0x2)
    assert reg.value == 0x23


def test_register_value_setter_distributes_to_fields():
    reg = Register(name='reg', address=0, width=8)
    make_field(reg, 'a', reg_offset=0, slice_width=4, bf_width=4)
    make_field(reg, 'b', reg_offset=4, slice_width=4, bf_width=4)
    reg.value = 0x5A
    assert reg.value == 0x5A


def test_register_reset_restores_reset_values():
    reg = Register(name='reg', address=0, width=8)
    ref = make_field(reg, 'a', reg_offset=0, slice_width=8, bf_width=8)
    ref.bf.reset_val = 0x3C
    ref.bf.value = 0xFF
    reg.reset()
    assert reg.value == 0x3C


def test_register_iterates_children_sorted_by_reg_offset():
    reg = Register(name='reg', address=0, width=8)
    make_field(reg, 'hi', reg_offset=4, slice_width=4)
    make_field(reg, 'lo', reg_offset=0, slice_width=4)
    offsets = [c.reg_offset for c in reg]
    assert offsets == [0, 4]


def test_register_validate_reports_missing_bitfieldrefs():
    reg = Register(name='reg', address=0, width=8)
    errors = [e for e in reg.validate() if isinstance(e, ValidationError)]
    assert errors


def test_register_validate_reports_non_bitfieldref_child():
    reg = Register(name='reg', address=0, width=8)
    reg._children['bogus'] = BitField(name='bogus')
    errors = [e for e in reg.validate() if isinstance(e, ValidationError)]
    assert errors


def test_register_validate_detects_overlapping_refs():
    reg = Register(name='reg', address=0, width=8)
    make_field(reg, 'a', reg_offset=0, slice_width=4)
    make_field(reg, 'b', reg_offset=2, slice_width=4)
    errors = [e for e in reg.validate() if isinstance(e, ValidationError)]
    assert errors


def test_register_read_returns_raw_value_and_updates_fields():
    reg = Register(name='reg', address=0x10, width=8)
    ref = make_field(reg, 'a', reg_offset=0, slice_width=4, bf_width=4)
    reg._reg_read_func = lambda addr: 0xA5
    assert reg.read() == 0xA5
    # Only the low nibble belongs to a defined field
    assert ref.bf.value == 0x5
    assert reg.value == 0x05


def test_register_write_uses_current_value_when_none_given():
    reg = Register(name='reg', address=0x10, width=8)
    make_field(reg, 'a', reg_offset=0, slice_width=8, bf_width=8, bf_value=0x3C)
    calls = []
    reg._reg_write_func = lambda addr, val: calls.append((addr, val))
    reg.write()
    assert calls == [(0x10, 0x3C)]


def test_register_write_with_value_updates_fields_then_writes():
    reg = Register(name='reg', address=0x10, width=8)
    make_field(reg, 'a', reg_offset=0, slice_width=8, bf_width=8)
    calls = []
    reg._reg_write_func = lambda addr, val: calls.append((addr, val))
    reg.write(0x11)
    assert reg.value == 0x11
    assert calls == [(0x10, 0x11)]


# ---------------------------------------------------------------- RegisterMap

def test_get_dimensions_computes_indices_and_byte_count():
    start_index, end_index, num_bytes = get_dimensions(0, 4, 16)
    assert (start_index, end_index, num_bytes) == (4, 6, 2)


def test_registermap_read_splits_block_into_registers_little_endian():
    rm = RegisterMap(name='map')
    r0 = Register(name='r0', address=0, width=8)
    make_field(r0, 'f0', reg_offset=0, slice_width=8)
    r1 = Register(name='r1', address=1, width=16)
    make_field(r1, 'f1', reg_offset=0, slice_width=16, bf_width=16)
    rm._children['r0'] = r0
    rm._children['r1'] = r1

    rm._block_read_func = lambda addr, size: b'\x01\x02\x03'
    data = rm.read()

    assert bytes(data) == b'\x01\x02\x03'
    assert r0.value == 0x01
    assert r1.value == 0x0302


def test_registermap_write_builds_block_little_endian():
    rm = RegisterMap(name='map')
    r0 = Register(name='r0', address=0, width=8)
    make_field(r0, 'f0', reg_offset=0, slice_width=8, bf_value=0x01)
    r1 = Register(name='r1', address=1, width=16)
    make_field(r1, 'f1', reg_offset=0, slice_width=16, bf_width=16,
               bf_value=0x0302)
    rm._children['r0'] = r0
    rm._children['r1'] = r1

    written = []
    rm._block_write_func = lambda addr, data: written.append((addr, bytes(data)))
    rm.write()

    assert written == [(0, b'\x01\x02\x03')]


def test_registermap_write_without_registers_raises():
    rm = RegisterMap(name='empty')
    with pytest.raises(ValueError):
        rm.write()
