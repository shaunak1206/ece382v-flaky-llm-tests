import pytest

from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.Register import Register
from r_map.RegisterMap import RegisterMap, get_dimensions
from r_map.ValidationError import ValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bf(name='bf', width=8, reset_val=0):
    return BitField(name=name, width=width, reset_val=reset_val)


def make_ref(name, bf, reg_offset=0, slice_width=0, field_offset=0):
    ref = BitFieldRef(name=name, reg_offset=reg_offset,
                      slice_width=slice_width, field_offset=field_offset)
    ref._add(bf)
    return ref


def make_reg(name='reg', width=32, address=0, refs=()):
    reg = Register(name=name, width=width, address=address)
    for ref in refs:
        reg._add(ref)
    return reg


def make_field_reg(name, address, width=32):
    """Register with a single full-width bitfield reference."""
    bf = make_bf(name + '_bf', width)
    ref = make_ref(name + '_ref', bf, reg_offset=0, slice_width=width)
    return make_reg(name, width, address, [ref])


# ---------------------------------------------------------------------------
# BitField
# ---------------------------------------------------------------------------

def test_bitfield_defaults_to_reset_value():
    bf = make_bf(width=4, reset_val=0x5)
    assert bf.value == 0x5
    assert bf.mask == 0xF


def test_bitfield_rejects_zero_width():
    with pytest.raises(ValueError):
        BitField(name='bad', width=0)


def test_bitfield_rejects_negative_reset_val():
    with pytest.raises(ValueError):
        BitField(name='bad', width=4, reset_val=-1)


def test_bitfield_int_assignment_is_masked_to_width():
    bf = make_bf(width=4)
    bf.value = 0x1F
    assert bf.value == 0xF


def test_bitfield_reset_restores_reset_val():
    bf = make_bf(width=8, reset_val=0x3C)
    bf.value = 0x00
    assert bf.value == 0x00
    bf.reset()
    assert bf.value == 0x3C


def test_bitfield_annotation_falls_back_to_hex_without_enumerations():
    bf = make_bf(width=4, reset_val=5)
    assert bf.annotation == hex(5)


# ---------------------------------------------------------------------------
# BitFieldRef
# ---------------------------------------------------------------------------

def test_bitfieldref_reads_field_slice_shifted_to_reg_offset():
    bf = make_bf(width=8)
    bf.value = 0xA5
    ref = make_ref('ref', bf, reg_offset=0, slice_width=4, field_offset=4)
    assert ref.value == 0xA

    ref2 = make_ref('ref2', bf, reg_offset=8, slice_width=4, field_offset=0)
    assert ref2.value == 0x500


def test_bitfieldref_writes_only_its_slice():
    bf = make_bf(width=8)
    bf.value = 0xF0
    ref = make_ref('ref', bf, reg_offset=0, slice_width=4, field_offset=0)
    ref.value = 0x03
    assert bf.value == 0xF3
    assert ref.value == 0x3


def test_bitfieldref_reset_val_uses_bitfield_reset_value():
    bf = make_bf(width=4, reset_val=0x5)
    ref = make_ref('ref', bf, reg_offset=8, slice_width=4)
    assert ref.reset_val == 0x500


def test_bitfieldref_slice_width_defaults_to_bitfield_width():
    bf = make_bf(width=6)
    ref = make_ref('ref', bf, slice_width=0)
    assert ref.slice_width == 6
    assert ref.mask == 0x3F


def test_bitfieldref_rejects_negative_arguments():
    with pytest.raises(ValueError):
        BitFieldRef(name='neg', reg_offset=0, slice_width=-1)
    with pytest.raises(ValueError):
        BitFieldRef(name='neg', reg_offset=0, field_offset=-1)


def test_bitfieldref_validate_reports_oversized_slice():
    bf = make_bf(width=4)
    ref = make_ref('ref', bf, slice_width=8)
    errors = list(ref.validate())
    assert any(isinstance(e, ValidationError) for e in errors)


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

def test_register_value_combines_refs():
    bf1 = make_bf('bf1', width=8)
    bf2 = make_bf('bf2', width=8)
    bf1.value = 0x34
    bf2.value = 0x12
    reg = make_reg(refs=[
        make_ref('r1', bf1, reg_offset=0, slice_width=8),
        make_ref('r2', bf2, reg_offset=8, slice_width=8),
    ])
    assert reg.value == 0x1234


def test_register_value_setter_distributes_to_bitfields():
    bf1 = make_bf('bf1', width=8)
    bf2 = make_bf('bf2', width=8)
    reg = make_reg(refs=[
        make_ref('r1', bf1, reg_offset=0, slice_width=8),
        make_ref('r2', bf2, reg_offset=8, slice_width=8),
    ])
    reg.value = 0xABCD
    assert bf1.value == 0xCD
    assert bf2.value == 0xAB
    assert reg.value == 0xABCD


def test_register_reset_restores_bitfields():
    bf = make_bf(width=8, reset_val=0x7)
    reg = make_reg(refs=[make_ref('r', bf, slice_width=8)])
    bf.value = 0xFF
    reg.reset()
    assert bf.value == 0x7


def test_register_validate_requires_children():
    reg = make_reg()
    errors = list(reg.validate())
    assert len(errors) >= 1
    assert all(isinstance(e, ValidationError) for e in errors)


def test_register_validate_detects_overlapping_refs():
    bf1 = make_bf('bf1', width=8)
    bf2 = make_bf('bf2', width=8)
    reg = make_reg(refs=[
        make_ref('r1', bf1, reg_offset=0, slice_width=8),
        make_ref('r2', bf2, reg_offset=4, slice_width=8),
    ])
    errors = list(reg.validate())
    assert any(isinstance(e, ValidationError) for e in errors)


def test_register_validate_accepts_disjoint_refs():
    bf1 = make_bf('bf1', width=8)
    bf2 = make_bf('bf2', width=8)
    reg = make_reg(refs=[
        make_ref('r1', bf1, reg_offset=0, slice_width=8),
        make_ref('r2', bf2, reg_offset=8, slice_width=8),
    ])
    errors = [e for e in reg.validate() if 'overlap' in str(e).lower()]
    assert errors == []


def test_register_iterates_children_in_offset_order():
    bf1 = make_bf('bf1', width=8)
    bf2 = make_bf('bf2', width=8)
    # Insert in reverse offset order
    reg = make_reg(refs=[
        make_ref('high', bf2, reg_offset=8, slice_width=8),
        make_ref('low', bf1, reg_offset=0, slice_width=8),
    ])
    offsets = [child.reg_offset for child in reg]
    assert offsets == [0, 8]


def test_register_read_and_write_use_io_functions():
    reg = make_field_reg('ctrl', address=0x10, width=16)

    read_calls = []

    def fake_read(address):
        read_calls.append(address)
        return 0x1234

    write_calls = []

    def fake_write(address, val):
        write_calls.append((address, val))

    reg._reg_read_func = fake_read
    reg._reg_write_func = fake_write

    assert reg.read() == 0x1234
    assert reg.value == 0x1234
    assert read_calls == [0x10]

    reg.write(0x00AB)
    assert write_calls == [(0x10, 0x00AB)]


# ---------------------------------------------------------------------------
# RegisterMap
# ---------------------------------------------------------------------------

def test_get_dimensions_computes_indices_and_byte_count():
    assert get_dimensions(0, 4, 32) == (4, 8, 4)
    # 12 bits occupies 2 bytes
    assert get_dimensions(16, 18, 12) == (2, 4, 2)


def test_registermap_read_unpacks_little_endian():
    rm = RegisterMap(name='map')
    rm._add(make_field_reg('r0', address=0, width=32))
    rm._add(make_field_reg('r1', address=4, width=32))

    payload = bytes([0x44, 0x33, 0x22, 0x11, 0xDD, 0xCC, 0xBB, 0xAA])
    calls = []

    def fake_block_read(address, size):
        calls.append((address, size))
        return payload

    rm._block_read_func = fake_block_read
    result = rm.read()

    assert result == payload
    assert calls == [(0, 8)]
    regs = list(rm)
    assert regs[0].value == 0x11223344
    assert regs[1].value == 0xAABBCCDD


def test_registermap_write_packs_little_endian():
    rm = RegisterMap(name='map')
    r0 = make_field_reg('r0', address=0, width=32)
    r1 = make_field_reg('r1', address=4, width=32)
    rm._add(r0)
    rm._add(r1)
    r0.value = 0x11223344
    r1.value = 0xAABBCCDD

    written = []

    def fake_block_write(address, data):
        written.append((address, bytes(data)))

    rm._block_write_func = fake_block_write
    rm.write()

    expected = bytes([0x44, 0x33, 0x22, 0x11, 0xDD, 0xCC, 0xBB, 0xAA])
    assert written == [(0, expected)]


def test_registermap_write_without_registers_raises():
    rm = RegisterMap(name='empty')
    rm._block_write_func = lambda address, data: None
    with pytest.raises(ValueError):
        rm.write()
