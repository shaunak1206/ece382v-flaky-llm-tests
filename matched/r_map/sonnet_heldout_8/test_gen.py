import pytest

from r_map.Register import Register
from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.RegisterMap import RegisterMap, get_dimensions


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
class _Reg(Register):
    """Register whose address and IO functions are set per-instance."""

    @property
    def address(self):
        return self._test_addr

    @property
    def _reg_read_func(self):
        return self._test_read

    @property
    def _reg_write_func(self):
        return self._test_write


class _Map(RegisterMap):
    @property
    def _block_read_func(self):
        return self._test_bread

    @property
    def _block_write_func(self):
        return self._test_bwrite


def make_reg(name="reg", addr=0, fields=(("bf", 32, 0, 0),), width=32,
             access="RW"):
    """fields: iterable of (bf_name, bf_width, reset_val, reg_offset)"""
    reg = _Reg(name=name, width=width)
    reg._test_addr = addr
    reg.writes = []
    reg._test_read = lambda a: 0
    reg._test_write = lambda a, v: reg.writes.append((a, v))
    for bf_name, bf_width, reset, offset in fields:
        bf = BitField(name=name + "_" + bf_name, width=bf_width,
                      reset_val=reset, access=access)
        ref = BitFieldRef(name=name + "_" + bf_name + "_ref",
                          reg_offset=offset)
        ref._add(bf)
        reg._add(ref)
    return reg


def error_text(errors):
    out = []
    for e in errors:
        out.append(str(e) + repr(e) + str(getattr(e, "args", ""))
                   + str(getattr(e, "__dict__", "")))
    return " ".join(out)


# --------------------------------------------------------------------------
# BitField
# --------------------------------------------------------------------------
def test_bitfield_defaults_and_mask():
    bf = BitField(name="bf", width=4, reset_val=0x5)
    assert bf.mask == 0xF
    assert bf.value == 0x5
    assert bf.reset_val == 0x5


def test_bitfield_invalid_arguments():
    with pytest.raises(ValueError):
        BitField(name="bf", width=0)
    with pytest.raises(ValueError):
        BitField(name="bf", width=4, reset_val=-1)


def test_bitfield_value_is_masked():
    bf = BitField(name="bf", width=4)
    bf.value = 0x1F3
    assert bf.value == 0x3


def test_bitfield_value_setter_bad_inputs():
    bf = BitField(name="bf", width=4)
    with pytest.raises(ValueError):
        bf.value = "does_not_exist"
    with pytest.raises(NotImplementedError):
        bf.value = 1.5


def test_bitfield_reset_restores_reset_val():
    bf = BitField(name="bf", width=8, reset_val=0x12)
    bf.value = 0xFF
    assert bf.value == 0xFF
    bf.reset()
    assert bf.value == 0x12


def test_bitfield_annotation_without_enumerations():
    bf = BitField(name="bf", width=8, reset_val=0x2A)
    assert bf.annotation == hex(0x2A)


def test_bitfield_io_without_parent_raises():
    bf = BitField(name="bf", width=8)
    with pytest.raises(ValueError):
        bf.read()
    with pytest.raises(ValueError):
        bf.write(1)


def test_bitfield_write_does_read_modify_write():
    reg = make_reg("r", addr=0x40, fields=[("f", 8, 0, 0)])
    reg._test_read = lambda a: 0x00
    bf = next(iter(reg)).bf
    bf.write(0xAB)
    assert bf.value == 0xAB
    assert reg.writes == [(0x40, 0xAB)]


def test_bitfield_write_skips_when_unchanged_and_not_always():
    reg = make_reg("r", addr=0x40, fields=[("f", 8, 0, 0)])
    reg._test_read = lambda a: 0xAB
    bf = next(iter(reg)).bf
    bf.write(0xAB, always_write=False)
    assert reg.writes == []
    bf.write(0xAB, always_write=True)
    assert reg.writes == [(0x40, 0xAB)]


def test_bitfield_read_returns_value_from_register():
    reg = make_reg("r", addr=0x8, fields=[("f", 8, 0, 0)])
    reg._test_read = lambda a: 0x7E
    bf = next(iter(reg)).bf
    assert bf.read() == 0x7E


# --------------------------------------------------------------------------
# BitFieldRef
# --------------------------------------------------------------------------
def test_bitfieldref_invalid_arguments():
    with pytest.raises(ValueError):
        BitFieldRef(name="r", reg_offset=0, slice_width=-1)
    with pytest.raises(ValueError):
        BitFieldRef(name="r", reg_offset=0, field_offset=-1)


def test_bitfieldref_adds_only_bitfields():
    ref = BitFieldRef(name="r", reg_offset=0)
    with pytest.raises(ValueError):
        ref._add("not a bitfield")


def test_bitfieldref_default_slice_width_uses_bitfield_width():
    bf = BitField(name="bf", width=6)
    ref = BitFieldRef(name="r", reg_offset=0)
    ref._add(bf)
    assert ref.slice_width == 6
    assert ref.mask == 0x3F
    assert ref.bf is bf


def test_bitfieldref_value_and_reset_val_with_offsets():
    bf = BitField(name="bf", width=8, reset_val=0xAB)
    ref = BitFieldRef(name="r", reg_offset=4, slice_width=4, field_offset=2)
    ref._add(bf)
    assert ref.value == 0xA0
    assert ref.reset_val == 0xA0


def test_bitfieldref_value_setter_updates_only_slice():
    bf = BitField(name="bf", width=8, reset_val=0xAB)
    ref = BitFieldRef(name="r", reg_offset=4, slice_width=4, field_offset=2)
    ref._add(bf)
    ref.value = 0x50
    assert bf.value == 0x97


def test_bitfieldref_multiple_references_tracked():
    bf = BitField(name="bf", width=8)
    ref1 = BitFieldRef(name="r1", reg_offset=0)
    ref2 = BitFieldRef(name="r2", reg_offset=0)
    ref1._add(bf)
    ref2._add(bf)
    assert bf._references == {ref1, ref2}


# --------------------------------------------------------------------------
# Register
# --------------------------------------------------------------------------
def test_register_value_combines_bitfields():
    reg = make_reg("r", fields=[("a", 4, 0x3, 0), ("b", 4, 0x1, 8)])
    assert reg.value == 0x103
    assert reg.reset_val == 0x103


def test_register_value_setter_distributes_to_bitfields():
    reg = make_reg("r", fields=[("a", 4, 0, 0), ("b", 4, 0, 8)])
    reg.value = 0xF5A
    assert reg.value == 0xF0A


def test_register_reset_restores_reset_values():
    reg = make_reg("r", fields=[("a", 4, 0x3, 0), ("b", 4, 0x1, 8)])
    reg.value = 0xFFF
    assert reg.value == 0xF0F
    reg.reset()
    assert reg.value == 0x103


def test_register_iterates_sorted_by_reg_offset():
    reg = make_reg("r", fields=[("b", 4, 0, 8), ("a", 4, 0, 0),
                                ("c", 4, 0, 16)])
    offsets = [ref.reg_offset for ref in reg]
    assert offsets == [0, 8, 16]


def test_register_access_joins_unique_sorted():
    reg = _Reg(name="r", width=32)
    for i, (acc, off) in enumerate([("RW", 0), ("RO", 8), ("RW", 16)]):
        bf = BitField(name=f"bf{i}", width=4, access=acc)
        ref = BitFieldRef(name=f"ref{i}", reg_offset=off)
        ref._add(bf)
        reg._add(ref)
    assert reg.access == "RO|RW"


def test_register_read_and_write_use_io_functions():
    reg = make_reg("r", addr=0x20, fields=[("f", 8, 0, 4)])
    seen = []

    def rd(addr):
        seen.append(addr)
        return 0xFFFF

    reg._test_read = rd
    assert reg.read() == 0xFFFF  # raw value returned
    assert seen == [0x20]
    assert reg.value == 0xFF0  # only mapped bits kept

    reg.write()
    assert reg.writes[-1] == (0x20, 0xFF0)
    reg.write(0x120)
    assert reg.writes[-1] == (0x20, 0x120)
    assert reg.value == 0x120


def test_register_validate_detects_missing_and_overlapping_refs():
    good = make_reg("good", fields=[("a", 4, 0, 0), ("b", 4, 0, 8)])
    good_errors = list(good.validate())

    empty = _Reg(name="empty", width=32)
    empty_text = error_text(empty.validate())
    assert "No bitfieldrefs present" in empty_text

    overlap = make_reg("ov", fields=[("a", 8, 0, 0), ("b", 8, 0, 4)])
    overlap_errors = list(overlap.validate())
    assert len(overlap_errors) > len(good_errors)
    assert "overlaps" in error_text(overlap_errors)


# --------------------------------------------------------------------------
# RegisterMap
# --------------------------------------------------------------------------
def make_map():
    m = _Map(name="map")
    r0 = make_reg("r0", addr=0x100)
    r1 = make_reg("r1", addr=0x104)
    m._add(r0)
    m._add(r1)
    return m, r0, r1


def test_get_dimensions():
    assert get_dimensions(0x100, 0x104, 32) == (4, 8, 4)
    assert get_dimensions(0x100, 0x100, 12) == (0, 2, 2)


def test_registermap_empty_raises():
    m = _Map(name="empty")
    with pytest.raises(ValueError):
        m.read()
    with pytest.raises(ValueError):
        m.write()


def test_registermap_read_populates_registers():
    m, r0, r1 = make_map()
    calls = []
    data = bytes([0x44, 0x33, 0x22, 0x11, 0xDD, 0xCC, 0xBB, 0xAA])

    def bread(addr, size):
        calls.append((addr, size))
        return data

    m._test_bread = bread
    assert m.read() == data
    assert calls == [(0x100, 8)]
    assert r0.value == 0x11223344
    assert r1.value == 0xAABBCCDD


def test_registermap_write_without_data_serialises_registers():
    m, r0, r1 = make_map()
    r0.value = 0x11223344
    r1.value = 0xAABBCCDD
    written = []
    m._test_bwrite = lambda addr, data: written.append((addr, bytes(data)))
    m.write()
    assert written == [(0x100, bytes([0x44, 0x33, 0x22, 0x11,
                                      0xDD, 0xCC, 0xBB, 0xAA]))]


def test_registermap_write_with_data_updates_registers():
    m, r0, r1 = make_map()
    written = []
    m._test_bwrite = lambda addr, data: written.append((addr, bytes(data)))
    data = bytes([1, 0, 0, 0, 2, 0, 0, 0])
    m.write(data)
    assert r0.value == 1
    assert r1.value == 2
    assert written == [(0x100, data)]


def test_registermap_partial_block_write_stops_early():
    m, r0, r1 = make_map()
    written = []
    m._test_bwrite = lambda addr, data: written.append((addr, bytes(data)))
    data = bytes([9, 0, 0, 0])
    m.write(data)
    assert r0.value == 9
    assert r1.value == 0
    assert written == [(0x100, data)]
