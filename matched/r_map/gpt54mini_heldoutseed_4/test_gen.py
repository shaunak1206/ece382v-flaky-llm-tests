import pytest

from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.Register import Register
from r_map.RegisterMap import RegisterMap, get_dimensions
from r_map.ValidationError import ValidationError


def _make_register(name="reg", address=0, width=16):
    return Register(name=name, address=address, width=width)


def _make_bitfield(name="bf", width=8, reset_val=0, access="RW"):
    return BitField(name=name, width=width, reset_val=reset_val, access=access)


def _attach(reg, ref, bf):
    reg._add(ref)
    ref._add(bf)
    return reg, ref, bf


def test_bitfield_init_rejects_invalid_width_and_reset_val():
    with pytest.raises(ValueError):
        BitField(width=0)

    with pytest.raises(ValueError):
        BitField(width=1, reset_val=-1)


def test_bitfield_mask_and_reset_value_are_clipped_to_width():
    bf = BitField(name="bf", width=3, reset_val=0xF)
    assert bf.mask == 0b111
    assert bf.value == 0b111
    assert bf.reset_val == 0b111


def test_bitfield_value_masks_integer_assignments():
    bf = BitField(name="bf", width=4, reset_val=0)
    bf.value = 0xAB
    assert bf.value == 0xB


def test_bitfield_string_value_sets_matching_enumeration_name():
    bf = BitField(name="mode", width=2, reset_val=0)
    enum_a = r_map.Enumeration(name="A", value=0)
    enum_b = r_map.Enumeration(name="B", value=1)
    bf._add(enum_a)
    bf._add(enum_b)

    bf.value = "B"
    assert bf.value == 1
    assert bf.annotation == "B"


def test_bitfield_write_and_read_use_parent_registers(monkeypatch):
    reg = _make_register(address=0)
    ref = BitFieldRef(name="ref", reg_offset=0, slice_width=4, field_offset=0)
    bf = _make_bitfield(name="bf", width=4, reset_val=0)
    _attach(reg, ref, bf)

    reads = []
    writes = []

    def reg_read():
        reads.append("read")
        reg.value = 0x0

    def reg_write():
        writes.append(reg.value)

    reg._reg_read_func = reg_read
    reg._reg_write_func = reg_write

    bf.value = 0xA
    assert bf.read() == 0xA
    assert reads == ["read"]

    bf.write(0x5)
    assert bf.value == 0x5
    assert reads == ["read", "read"]
    assert writes == [0x5]


def test_bitfieldref_defaults_slice_width_to_bitfield_width():
    reg = _make_register(address=0)
    ref = BitFieldRef(name="ref", reg_offset=3, slice_width=0, field_offset=0)
    bf = _make_bitfield(name="bf", width=5, reset_val=0)
    _attach(reg, ref, bf)

    assert ref.slice_width == 5
    assert ref.mask == 0b11111


def test_bitfieldref_init_rejects_negative_values():
    with pytest.raises(ValueError):
        BitFieldRef(name="ref", reg_offset=0, slice_width=-1, field_offset=0)

    with pytest.raises(ValueError):
        BitFieldRef(name="ref", reg_offset=0, slice_width=1, field_offset=-1)


def test_bitfieldref_value_property_maps_bits_between_register_and_bitfield():
    reg = _make_register(address=0)
    ref = BitFieldRef(name="ref", reg_offset=4, slice_width=3, field_offset=1)
    bf = _make_bitfield(name="bf", width=8, reset_val=0)
    _attach(reg, ref, bf)

    bf.value = 0b1010_1100
    assert ref.value == 0b110 << 4

    ref.value = 0b1110_0000
    assert bf.value == 0b1010_1110


def test_bitfieldref_validate_requires_child_bitfield():
    ref = BitFieldRef(name="ref", reg_offset=0, slice_width=1, field_offset=0)
    errors = list(ref.validate())
    assert any("No child bitfield present" in str(e) for e in errors)


def test_register_iteration_sorts_by_reg_offset():
    reg = _make_register(address=0)
    bf = _make_bitfield(name="bf", width=8, reset_val=0)

    ref_late = BitFieldRef(name="late", reg_offset=8, slice_width=8, field_offset=0)
    ref_early = BitFieldRef(name="early", reg_offset=0, slice_width=8, field_offset=0)
    _attach(reg, ref_late, bf)
    _attach(reg, ref_early, bf)

    assert [c.name for c in reg] == ["early", "late"]


def test_register_validate_detects_overlap_and_empty_register():
    reg = _make_register(address=0)
    assert any("No bitfieldrefs present" in str(e) for e in reg.validate())

    bf = _make_bitfield(name="bf", width=8, reset_val=0)
    r1 = BitFieldRef(name="r1", reg_offset=0, slice_width=4, field_offset=0)
    r2 = BitFieldRef(name="r2", reg_offset=2, slice_width=4, field_offset=0)
    _attach(reg, r1, bf)
    _attach(reg, r2, bf)

    errors = list(reg.validate())
    assert any("overlaps" in str(e) for e in errors)


def test_register_read_and_write_round_trip(monkeypatch):
    reg = _make_register(address=0)
    bf = _make_bitfield(name="bf", width=8, reset_val=0)
    ref = BitFieldRef(name="ref", reg_offset=0, slice_width=8, field_offset=0)
    _attach(reg, ref, bf)

    read_values = [0x3C]
    written = []

    def reg_read(addr):
        assert addr == 0
        return read_values.pop(0)

    def reg_write(addr, val):
        written.append((addr, val))

    reg._reg_read_func = reg_read
    reg._reg_write_func = reg_write

    val = reg.read()
    assert val == 0x3C
    assert reg.value == 0x3C

    reg.write(0xA5)
    assert written == [(0, 0xA5)]


def test_register_map_get_dimensions_and_block_dimensions():
    start_index, end_index, num_bytes = get_dimensions(0x100, 0x104, 16)
    assert (start_index, end_index, num_bytes) == (4, 6, 2)

    rm = RegisterMap(name="rm")
    r0 = _make_register(name="r0", address=0x100, width=16)
    r1 = _make_register(name="r1", address=0x104, width=8)
    rm._add(r0)
    rm._add(r1)

    assert rm._block_dimensions == (0x100, 5)


def test_register_map_read_updates_registers_from_block_bytes():
    rm = RegisterMap(name="rm")
    r0 = _make_register(name="r0", address=0x100, width=16)
    r1 = _make_register(name="r1", address=0x102, width=8)
    rm._add(r0)
    rm._add(r1)

    def block_read(addr, size):
        assert addr == 0x100
        assert size == 3
        return bytes([0x34, 0x12, 0xAB])

    rm._block_read_func = block_read
    data = rm.read()

    assert data == bytes([0x34, 0x12, 0xAB])
    assert r0.value == 0x1234
    assert r1.value == 0xAB


def test_register_map_write_assembles_block_bytes_and_calls_io():
    rm = RegisterMap(name="rm")
    r0 = _make_register(name="r0", address=0x100, width=16)
    r1 = _make_register(name="r1", address=0x102, width=8)
    rm._add(r0)
    rm._add(r1)

    r0.value = 0x1234
    r1.value = 0xAB

    calls = []

    def block_write(addr, data):
        calls.append((addr, bytes(data)))

    rm._block_write_func = block_write
    rm.write()

    assert calls == [(0x100, bytes([0x34, 0x12, 0xAB]))]


def test_register_map_write_with_partial_data_updates_covered_registers_only():
    rm = RegisterMap(name="rm")
    r0 = _make_register(name="r0", address=0x100, width=16)
    r1 = _make_register(name="r1", address=0x102, width=8)
    rm._add(r0)
    rm._add(r1)

    calls = []

    def block_write(addr, data):
        calls.append((addr, bytes(data)))

    rm._block_write_func = block_write
    rm.write(data=bytearray([0x78, 0x56]))  # only enough for first register

    assert calls == [(0x100, bytes([0x78, 0x56]))]
    assert r0.value == 0x5678
    assert r1.value == 0  # unchanged because partial block write stops early
