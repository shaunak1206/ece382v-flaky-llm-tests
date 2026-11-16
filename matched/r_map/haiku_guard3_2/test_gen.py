import pytest

from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.Register import Register
from r_map.RegisterMap import RegisterMap, get_dimensions
from r_map.ValidationError import ValidationError

# No random number generators are used by the code under test, so nothing needs
# seeding. The code does not call hash() on keys either. All objects are built
# inside each test or helper, and no module or global state is changed.


def _add_ref(parent, bf, name, reg_offset, slice_width, field_offset=0):
    """Attach a BitFieldRef to a register, wiring it to bf the same way the library does."""
    ref = BitFieldRef(name=name, reg_offset=reg_offset,
                      slice_width=slice_width, field_offset=field_offset)
    ref._add(bf)
    parent._children[ref.name] = ref
    ref._parent = parent
    return ref


def _add_child(block, child):
    block._children[child.name] = child
    child._parent = block


def _two_reg_map():
    """Map with r0 (8-bit at 0x100) and r1 (16-bit at 0x102), each with one full-width field."""
    m = RegisterMap(name='m')
    r0 = Register(name='r0', width=8)
    r0.address = 0x100
    bf0 = BitField(name='b0', width=8)
    _add_ref(r0, bf0, 'x0', 0, 8)
    r1 = Register(name='r1', width=16)
    r1.address = 0x102
    bf1 = BitField(name='b1', width=16)
    _add_ref(r1, bf1, 'x1', 0, 16)
    _add_child(m, r0)
    _add_child(m, r1)
    return m, r0, r1, bf0, bf1


# ---------------------------------------------------------------- BitField

def test_bitfield_constructor_rejects_invalid_args():
    with pytest.raises(ValueError):
        BitField(name='f', width=0)
    with pytest.raises(ValueError):
        BitField(name='f', width=4, reset_val=-1)


def test_bitfield_int_value_is_masked_to_width():
    bf = BitField(name='f', width=4)
    bf.value = 0x1F
    assert bf.value == 0xF
    bf.value = 0x5
    assert bf.value == 0x5


def test_bitfield_rejects_unsupported_value_types():
    bf = BitField(name='f', width=4)
    # No enumerations are attached, so a string cannot match any name.
    with pytest.raises(ValueError):
        bf.value = 'missing'
    with pytest.raises(NotImplementedError):
        bf.value = 1.5


def test_bitfield_reset_restores_reset_val():
    bf = BitField(name='f', width=8, reset_val=0x5A)
    bf.value = 0
    bf.reset()
    assert bf.value == 0x5A


def test_bitfield_read_and_write_without_parent_raise():
    bf = BitField(name='f', width=4)
    with pytest.raises(ValueError):
        bf.read()
    with pytest.raises(ValueError):
        bf.write(1)


def test_bitfield_write_performs_read_modify_write_through_register():
    reg = Register(name='r', width=8)
    reg.address = 0x40
    bf = BitField(name='f', width=4)
    _add_ref(reg, bf, 'rf', 4, 4)

    reads = []
    writes = []
    reg._reg_read_func = lambda addr: reads.append(addr) or 0x00
    reg._reg_write_func = lambda addr, val: writes.append((addr, val))

    bf.write(val=5)

    assert reads == [0x40]
    assert writes == [(0x40, 0x50)]
    assert bf.value == 5


# ---------------------------------------------------------------- BitFieldRef

def test_bitfieldref_rejects_negative_slice_width_or_field_offset():
    with pytest.raises(ValueError):
        BitFieldRef(name='r', reg_offset=0, slice_width=-1)
    with pytest.raises(ValueError):
        BitFieldRef(name='r', reg_offset=0, slice_width=4, field_offset=-1)


def test_bitfieldref_slice_width_defaults_to_bitfield_width():
    ref = BitFieldRef(name='r', reg_offset=0)
    ref._add(BitField(name='f', width=6))
    assert ref.slice_width == 6
    assert ref.mask == 0x3F


def test_bitfieldref_rejects_non_bitfield_child():
    ref = BitFieldRef(name='r', reg_offset=0, slice_width=4)
    with pytest.raises(ValueError):
        ref._add(object())


def test_bitfieldref_value_getter_extracts_shifted_slice():
    bf = BitField(name='f', width=8)
    bf.value = 0xC4  # binary 1100_0100
    ref = BitFieldRef(name='r', reg_offset=4, slice_width=4, field_offset=2)
    ref._add(bf)
    # (0xC4 >> 2) & 0xF == 0x1, placed at reg offset 4 -> 0x10
    assert ref.value == 0x10


def test_bitfieldref_value_setter_updates_only_its_slice():
    bf = BitField(name='f', width=8)
    bf.value = 0
    ref = BitFieldRef(name='r', reg_offset=4, slice_width=4, field_offset=2)
    ref._add(bf)
    ref.value = 0x30  # (0x30 >> 4) & 0xF == 3, stored at field offset 2
    assert bf.value == 0xC


def test_bitfieldref_second_reference_marks_alias():
    reg = Register(name='r', width=8)
    bf = BitField(name='f', width=4)
    ref1 = _add_ref(reg, bf, 'a', 0, 4)
    ref2 = BitFieldRef(name='b', reg_offset=4, slice_width=4)
    ref2._add(bf)
    assert ref2._alias is True
    assert ref2._ref is ref1


def test_bitfieldref_validate_flags_slice_wider_than_field():
    bf = BitField(name='f', width=4)
    ref = BitFieldRef(name='r', reg_offset=0, slice_width=8)
    ref._add(bf)
    errors = [e for e in ref.validate() if isinstance(e, ValidationError)]
    assert len(errors) >= 1


# ---------------------------------------------------------------- Register

def test_register_value_is_or_of_children():
    reg = Register(name='r', width=8)
    bf_a = BitField(name='a', width=4)
    bf_a.value = 0x5
    bf_b = BitField(name='b', width=4)
    bf_b.value = 0xA
    _add_ref(reg, bf_a, 'ra', 0, 4)
    _add_ref(reg, bf_b, 'rb', 4, 4)
    assert reg.value == 0xA5


def test_register_value_setter_distributes_to_fields():
    reg = Register(name='r', width=8)
    bf_a = BitField(name='a', width=4)
    bf_b = BitField(name='b', width=4)
    _add_ref(reg, bf_a, 'ra', 0, 4)
    _add_ref(reg, bf_b, 'rb', 4, 4)
    reg.value = 0x3C
    assert bf_a.value == 0xC
    assert bf_b.value == 0x3


def test_register_iterates_in_reg_offset_order():
    reg = Register(name='r', width=8)
    # Add in reverse offset order; iteration must still be sorted by reg_offset.
    _add_ref(reg, BitField(name='h', width=4), 'hi', 4, 4)
    _add_ref(reg, BitField(name='l', width=4), 'lo', 0, 4)
    assert [c.name for c in reg] == ['lo', 'hi']


def test_register_validate_reports_overlapping_refs():
    reg = Register(name='r', width=8)
    _add_ref(reg, BitField(name='a', width=4), 'r0', 0, 4)
    _add_ref(reg, BitField(name='b', width=4), 'r1', 2, 4)  # overlaps bits 2-3
    errors = [e for e in reg.validate() if isinstance(e, ValidationError)]
    assert len(errors) >= 1


def test_register_reset_restores_field_reset_values():
    reg = Register(name='r', width=8)
    bf = BitField(name='a', width=4, reset_val=0x3)
    _add_ref(reg, bf, 'ra', 0, 4)
    bf.value = 0
    reg.reset()
    assert bf.value == 0x3


def test_register_read_updates_fields_and_returns_raw_value():
    reg = Register(name='r', width=8)
    reg.address = 0x40
    bf = BitField(name='a', width=8)
    _add_ref(reg, bf, 'ra', 0, 8)

    calls = []

    def read_fn(addr):
        calls.append(addr)
        return 0xA5

    reg._reg_read_func = read_fn
    assert reg.read() == 0xA5
    assert calls == [0x40]
    assert bf.value == 0xA5


# ---------------------------------------------------------------- RegisterMap

def test_get_dimensions_returns_offsets_and_byte_count():
    # 32-bit register at 0x104 inside a block starting at 0x100
    assert get_dimensions(0x100, 0x104, 32) == (4, 8, 4)
    # 12-bit register rounds up to 2 bytes
    assert get_dimensions(0x100, 0x100, 12) == (0, 2, 2)


def test_register_map_block_dimensions_span_all_registers():
    m = RegisterMap(name='m')
    r0 = Register(name='r0', width=32)
    r0.address = 0x100
    r1 = Register(name='r1', width=16)
    r1.address = 0x104
    _add_child(m, r0)
    _add_child(m, r1)
    # block runs from 0x100 to 0x104 + 2 bytes = 0x106
    assert m._block_dimensions == (0x100, 6)


def test_register_map_read_unpacks_little_endian_values():
    m, r0, r1, bf0, bf1 = _two_reg_map()

    block_calls = []
    data = bytes([0x11, 0xAA, 0x34, 0x12])

    def block_read(addr, size):
        block_calls.append((addr, size))
        return data

    m._block_read_func = block_read
    assert m.read() == data
    assert block_calls == [(0x100, 4)]
    assert bf0.value == 0x11
    assert bf1.value == 0x1234


def test_register_map_write_without_data_builds_block_from_registers():
    m, r0, r1, bf0, bf1 = _two_reg_map()
    bf0.value = 0x11
    bf1.value = 0x1234

    block_writes = []
    m._block_write_func = lambda addr, d: block_writes.append((addr, d))

    m.write()

    assert len(block_writes) == 1
    addr, written = block_writes[0]
    assert addr == 0x100
    # byte 1 is a gap between the registers and stays zero
    assert written == bytearray(b'\x11\x00\x34\x12')


def test_register_map_write_with_data_updates_registers():
    m, r0, r1, bf0, bf1 = _two_reg_map()

    block_writes = []
    m._block_write_func = lambda addr, d: block_writes.append((addr, d))

    data = bytearray(b'\x7f\x00\x78\x56')
    m.write(data)

    assert bf0.value == 0x7F
    assert bf1.value == 0x5678
    assert block_writes == [(0x100, data)]
