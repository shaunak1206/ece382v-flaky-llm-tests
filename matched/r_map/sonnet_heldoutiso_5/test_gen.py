import types
from unittest.mock import patch

import pytest

import r_map
from r_map import BitField, BitFieldRef, Register, RegisterMap
from r_map.RegisterMap import get_dimensions


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def make_ref(bf, reg_offset=0, slice_width=0, field_offset=0, name=None):
    ref = BitFieldRef(name=name or (bf.name + "_ref"), reg_offset=reg_offset,
                      slice_width=slice_width, field_offset=field_offset)
    ref._add(bf)
    return ref


def make_reg(name="reg", width=32, fields=(), **kwargs):
    """fields: iterable of (name, reg_offset, width, reset_val)"""
    reg = Register(name=name, width=width, **kwargs)
    for fname, offset, fwidth, reset in fields:
        bf = BitField(name=fname, width=fwidth, reset_val=reset)
        reg._add(make_ref(bf, reg_offset=offset))
    return reg


def msgs(errors):
    return " ".join(f"{e!s} {e!r} {getattr(e, 'args', '')}" for e in errors)


@pytest.fixture
def reg_io():
    io = types.SimpleNamespace(read_value=0, reads=[], writes=[])

    def rd(address):
        io.reads.append(address)
        return io.read_value

    def wr(address, val):
        io.writes.append((address, val))

    with patch.object(Register, "_reg_read_func", new=staticmethod(rd),
                      create=True), \
            patch.object(Register, "_reg_write_func", new=staticmethod(wr),
                         create=True):
        yield io


@pytest.fixture
def block_io():
    io = types.SimpleNamespace(read_data=b"", reads=[], writes=[])

    def rd(address, size):
        io.reads.append((address, size))
        return io.read_data

    def wr(address, data):
        io.writes.append((address, bytes(data)))

    with patch.object(RegisterMap, "_block_read_func", new=staticmethod(rd),
                      create=True), \
            patch.object(RegisterMap, "_block_write_func", new=staticmethod(wr),
                         create=True):
        yield io


def make_map():
    rmap = RegisterMap(name="map")
    r0 = make_reg("r0", 32, [("f0", 0, 32, 0)], local_address=0)
    r1 = make_reg("r1", 16, [("f1", 0, 16, 0)], local_address=4)
    rmap._add(r0)
    rmap._add(r1)
    return rmap, r0, r1


# --------------------------------------------------------------------------
# BitField
# --------------------------------------------------------------------------
def test_bitfield_invalid_args():
    with pytest.raises(ValueError):
        BitField(name="bad", width=0)
    with pytest.raises(ValueError):
        BitField(name="bad", width=4, reset_val=-1)


def test_bitfield_initial_value_and_reset():
    bf = BitField(name="bf", width=8, reset_val=0x12)
    assert bf.value == 0x12
    bf.value = 0x34
    assert bf.value == 0x34
    bf.reset()
    assert bf.value == 0x12


def test_bitfield_value_masked_to_width():
    bf = BitField(name="bf", width=4)
    bf.value = 0x1F
    assert bf.value == 0xF


def test_bitfield_value_unsupported_type():
    bf = BitField(name="bf", width=4)
    with pytest.raises(NotImplementedError):
        bf.value = 1.5


def test_bitfield_enumeration_assignment_and_annotation():
    from r_map import Enumeration
    bf = BitField(name="bf", width=4)
    bf._add(Enumeration(name="A", value=1))
    bf._add(Enumeration(name="B", value=2))
    bf.value = "B"
    assert bf.value == 2
    assert bf.annotation == "B"
    bf.value = 7
    assert bf.annotation == hex(7)
    with pytest.raises(ValueError):
        bf.value = "NOPE"


def test_bitfield_validate_duplicate_enum_values():
    from r_map import Enumeration
    good = BitField(name="good", width=4)
    good._add(Enumeration(name="A", value=1))
    good._add(Enumeration(name="B", value=2))
    bad = BitField(name="bad", width=4)
    bad._add(Enumeration(name="A", value=1))
    bad._add(Enumeration(name="B", value=1))
    assert len(list(bad.validate())) > len(list(good.validate()))


def test_bitfield_io_without_parent_raises():
    bf = BitField(name="bf", width=4)
    with pytest.raises(ValueError):
        bf.read()
    with pytest.raises(ValueError):
        bf.write(1)


def test_bitfield_read_uses_parent_register(reg_io):
    reg = make_reg("reg", 8, [("f", 0, 4, 0)])
    bf = list(reg)[0].bf
    reg_io.read_value = 0xFF
    assert bf.read() == 0xF
    assert len(reg_io.reads) == 1


def test_bitfield_write_always_write(reg_io):
    reg = make_reg("reg", 8, [("f", 0, 8, 0)])
    bf = list(reg)[0].bf
    reg_io.read_value = 0x12
    bf.write(0x12, always_write=True)
    assert len(reg_io.writes) == 1
    assert reg_io.writes[0][1] == 0x12


def test_bitfield_write_skipped_when_unchanged(reg_io):
    reg = make_reg("reg", 8, [("f", 0, 8, 0)])
    bf = list(reg)[0].bf
    reg_io.read_value = 0x12
    bf.write(0x12, always_write=False)
    assert reg_io.writes == []
    reg_io.read_value = 0x00
    bf.write(0x12, always_write=False)
    assert len(reg_io.writes) == 1
    assert reg_io.writes[0][1] == 0x12


# --------------------------------------------------------------------------
# BitFieldRef
# --------------------------------------------------------------------------
def test_bitfieldref_default_slice_width_uses_bitfield_width():
    bf = BitField(name="bf", width=6)
    ref = make_ref(bf)
    assert ref.slice_width == 6
    assert ref.mask == 0x3F
    assert ref.bf is bf


def test_bitfieldref_invalid_args():
    with pytest.raises(ValueError):
        BitFieldRef(name="r", reg_offset=0, slice_width=-1)
    with pytest.raises(ValueError):
        BitFieldRef(name="r", reg_offset=0, field_offset=-1)


def test_bitfieldref_add_rejects_non_bitfield():
    ref = BitFieldRef(name="r", reg_offset=0)
    with pytest.raises(ValueError):
        ref._add("not a bitfield")


def test_bitfieldref_value_and_reset_val_with_offsets():
    bf = BitField(name="bf", width=8, reset_val=0xAB)
    ref = make_ref(bf, reg_offset=4, slice_width=4, field_offset=4)
    assert ref.value == 0xA0
    assert ref.reset_val == 0xA0


def test_bitfieldref_value_setter_only_touches_slice():
    bf = BitField(name="bf", width=8, reset_val=0xAB)
    ref = make_ref(bf, reg_offset=4, slice_width=4, field_offset=4)
    ref.value = 0x30
    assert bf.value == 0x3B


def test_bitfieldref_shared_bitfield_marks_alias():
    bf = BitField(name="bf", width=8)
    ref1 = make_ref(bf, name="ref1")
    ref2 = make_ref(bf, name="ref2")
    assert bf._references == {ref1, ref2}
    assert ref2._alias is True
    assert ref2._ref is ref1


# --------------------------------------------------------------------------
# Register
# --------------------------------------------------------------------------
def test_register_value_combines_fields_and_setter_distributes():
    reg = make_reg("reg", 8, [("lo", 0, 4, 0x1), ("hi", 4, 4, 0x2)])
    assert reg.value == 0x21
    reg.value = 0xA5
    lo, hi = [r.bf for r in reg]
    assert lo.value == 0x5
    assert hi.value == 0xA
    assert reg.value == 0xA5


def test_register_reset_val_and_reset():
    reg = make_reg("reg", 8, [("lo", 0, 4, 0x1), ("hi", 4, 4, 0x2)])
    assert reg.reset_val == 0x21
    reg.value = 0xFF
    reg.reset()
    assert reg.value == 0x21


def test_register_iterates_sorted_by_reg_offset():
    reg = Register(name="reg", width=16)
    hi = BitField(name="hi", width=8)
    lo = BitField(name="lo", width=8)
    reg._add(make_ref(hi, reg_offset=8))
    reg._add(make_ref(lo, reg_offset=0))
    assert [r.reg_offset for r in reg] == [0, 8]


def test_register_access_combines_unique_values():
    reg = Register(name="reg", width=16)
    a = BitField(name="a", width=8, access="RW")
    b = BitField(name="b", width=8, access="RO")
    reg._add(make_ref(a, reg_offset=0))
    reg._add(make_ref(b, reg_offset=8))
    assert reg.access == "RO|RW"


def test_register_str_contains_formatted_value():
    reg = make_reg("reg", 8, [("f", 0, 8, 0x5)])
    assert str(reg).endswith("value: 0x05")


def test_register_validate_empty_and_overlap():
    empty = Register(name="empty", width=8)
    assert "No bitfieldrefs" in msgs(empty.validate())

    ok = Register(name="ok", width=16)
    ok._add(make_ref(BitField(name="a", width=8), reg_offset=0))
    ok._add(make_ref(BitField(name="b", width=8), reg_offset=8))

    overlap = Register(name="overlap", width=16)
    overlap._add(make_ref(BitField(name="a", width=8), reg_offset=0))
    overlap._add(make_ref(BitField(name="b", width=8), reg_offset=4))

    assert len(list(overlap.validate())) > len(list(ok.validate()))
    assert "overlaps" in msgs(overlap.validate())


def test_register_read_updates_value_and_returns_raw(reg_io):
    reg = make_reg("reg", 8, [("lo", 0, 4, 0)])
    reg_io.read_value = 0xFF
    result = reg.read()
    assert result == 0xFF          # raw value, not the masked register value
    assert reg.value == 0xF
    assert len(reg_io.reads) == 1


def test_register_write_with_and_without_value(reg_io):
    reg = make_reg("reg", 8, [("lo", 0, 4, 0x3), ("hi", 4, 4, 0x2)])
    reg.write()
    assert reg_io.writes[-1][1] == 0x23
    reg.write(0x51)
    assert reg.value == 0x51
    assert reg_io.writes[-1][1] == 0x51


# --------------------------------------------------------------------------
# RegisterMap
# --------------------------------------------------------------------------
def test_get_dimensions():
    assert get_dimensions(0x10, 0x14, 32) == (4, 8, 4)
    assert get_dimensions(0x10, 0x14, 12) == (4, 6, 2)


def test_registermap_empty_raises():
    rmap = RegisterMap(name="empty")
    with pytest.raises(ValueError):
        rmap.read()
    with pytest.raises(ValueError):
        rmap.write()


def test_registermap_read_splits_data_little_endian(block_io):
    rmap, r0, r1 = make_map()
    block_io.read_data = bytes([1, 2, 3, 4, 5, 6])
    data = rmap.read()
    assert block_io.reads == [(0, 6)]
    assert data == block_io.read_data
    assert r0.value == 0x04030201
    assert r1.value == 0x0605


def test_registermap_write_packs_register_values(block_io):
    rmap, r0, r1 = make_map()
    r0.value = 0x04030201
    r1.value = 0x0605
    rmap.write()
    assert block_io.writes == [(0, bytes([1, 2, 3, 4, 5, 6]))]


def test_registermap_partial_write_updates_only_covered_registers(block_io):
    rmap, r0, r1 = make_map()
    rmap.write(bytearray([9, 8, 7, 6]))
    assert r0.value == 0x06070809
    assert r1.value == 0
    assert block_io.writes == [(0, bytes([9, 8, 7, 6]))]
