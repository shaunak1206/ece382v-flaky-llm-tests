import pytest

from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.Register import Register
from r_map.RegisterMap import RegisterMap, get_dimensions
from r_map.ValidationError import ValidationError


def make_bitfield_ref(name, bf, *, reg_offset, slice_width=0, field_offset=0):
    ref = BitFieldRef(
        name=name,
        reg_offset=reg_offset,
        slice_width=slice_width,
        field_offset=field_offset,
    )
    ref._add(bf)
    return ref


def make_register(name, *refs, address=0, width=8):
    reg = Register(name=name, address=address, width=width)
    for ref in refs:
        reg._add(ref)
    return reg


def make_register_map(name, *regs):
    rm = RegisterMap(name=name)
    for reg in regs:
        rm._add(reg)
    return rm


def test_bitfield_rejects_invalid_width_and_reset_val():
    with pytest.raises(ValueError):
        BitField(name="bf0", width=0)
    with pytest.raises(ValueError):
        BitField(name="bf1", width=1, reset_val=-1)


def test_bitfield_masks_value_on_assignment():
    bf = BitField(name="bf", width=3, reset_val=0)
    bf.value = 0b1111
    assert bf.value == 0b111


def test_bitfield_accepts_enumeration_name_and_annotation():
    enum = pytest.importorskip("r_map.Enumeration")
    Enumeration = enum.Enumeration

    bf = BitField(name="mode", width=2, reset_val=0)
    e0 = Enumeration(name="ZERO", value=0)
    e1 = Enumeration(name="ONE", value=1)
    bf._add(e0)
    bf._add(e1)

    bf.value = "ONE"
    assert bf.value == 1
    assert bf.annotation == "ONE"


def test_bitfield_value_from_unknown_enumeration_name_raises():
    bf = BitField(name="mode", width=2, reset_val=0)
    with pytest.raises(ValueError):
        bf.value = "MISSING"


def test_bitfield_write_without_parent_raises_value_error():
    bf = BitField(name="orphan", width=4, reset_val=0)
    with pytest.raises(ValueError, match="doesn't have any parents"):
        bf.write(1)


def test_bitfield_validate_reports_duplicate_enumeration_values():
    enum = pytest.importorskip("r_map.Enumeration")
    Enumeration = enum.Enumeration

    bf = BitField(name="bf", width=2, reset_val=0)
    bf._add(Enumeration(name="A", value=1))
    bf._add(Enumeration(name="B", value=1))

    errors = list(bf.validate())
    assert len(errors) == 1
    assert isinstance(errors[0], ValidationError)
    assert "same value" in str(errors[0])


def test_bitfieldref_rejects_negative_offsets():
    with pytest.raises(ValueError):
        BitFieldRef(name="r0", reg_offset=-1)
    with pytest.raises(ValueError):
        BitFieldRef(name="r1", reg_offset=0, field_offset=-1)


def test_bitfieldref_uses_child_width_when_slice_width_is_zero():
    bf = BitField(name="bf", width=8, reset_val=0xAB)
    ref = make_bitfield_ref("ref", bf, reg_offset=2, slice_width=0, field_offset=1)
    assert ref.slice_width == 8
    assert ref.mask == 0xFF


def test_bitfieldref_value_maps_between_register_and_bitfield_space():
    bf = BitField(name="bf", width=8, reset_val=0)
    ref = make_bitfield_ref("ref", bf, reg_offset=2, slice_width=3, field_offset=1)

    ref.value = 0b11100
    assert bf.value == 0b1110

    bf.value = 0b1010
    assert ref.value == ((0b1010 >> 1) & 0b111) << 2


def test_bitfieldref_reset_val_uses_slice_and_offsets():
    bf = BitField(name="bf", width=8, reset_val=0b10110100)
    ref = make_bitfield_ref("ref", bf, reg_offset=1, slice_width=3, field_offset=2)
    expected = ((bf.reset_val >> 2) & 0b111) << 1
    assert ref.reset_val == expected


def test_bitfieldref_validate_reports_missing_child_bitfield():
    ref = BitFieldRef(name="ref", reg_offset=0, slice_width=1, field_offset=0)
    errors = list(ref.validate())
    assert len(errors) == 1
    assert isinstance(errors[0], ValidationError)
    assert "No child bitfield present" in str(errors[0])


def test_register_value_access_and_reset_are_aggregated_from_children():
    bf1 = BitField(name="bf1", width=4, reset_val=0x5, access="RO")
    bf2 = BitField(name="bf2", width=4, reset_val=0xA, access="RW")
    r1 = make_bitfield_ref("r1", bf1, reg_offset=0, slice_width=4, field_offset=0)
    r2 = make_bitfield_ref("r2", bf2, reg_offset=4, slice_width=4, field_offset=0)
    reg = make_register("reg", r1, r2, address=0, width=8)

    assert reg.reset_val == 0xA5
    assert reg.value == 0xA5
    assert reg.access == "RO|RW"

    reg.value = 0x3C
    assert bf1.value == 0xC
    assert bf2.value == 0x3


def test_register_str_contains_hex_value():
    bf = BitField(name="bf", width=8, reset_val=0x12)
    ref = make_bitfield_ref("ref", bf, reg_offset=0, slice_width=8, field_offset=0)
    reg = make_register("reg", ref, address=0, width=8)

    s = str(reg)
    assert "value:" in s
    assert "0x12" in s


def test_register_validate_detects_no_children_and_non_bitfieldref_child():
    reg0 = Register(name="empty", address=0, width=8)
    errors0 = list(reg0.validate())
    assert any("No bitfieldrefs present" in str(e) for e in errors0)

    reg1 = Register(name="bad", address=0, width=8)
    reg1._add(BitField(name="bf", width=1, reset_val=0))
    errors1 = list(reg1.validate())
    assert any("is not of typeBitFieldRef" in str(e) for e in errors1)


def test_register_validate_reports_overlapping_bitfieldrefs():
    bf = BitField(name="bf", width=8, reset_val=0)
    r1 = make_bitfield_ref("r1", bf, reg_offset=0, slice_width=4, field_offset=0)
    r2 = make_bitfield_ref("r2", bf, reg_offset=2, slice_width=4, field_offset=0)
    reg = make_register("reg", r1, r2, address=0, width=8)

    errors = list(reg.validate())
    assert any("overlaps with" in str(e) for e in errors)


def test_register_read_calls_io_and_updates_value(monkeypatch):
    bf = BitField(name="bf", width=8, reset_val=0)
    ref = make_bitfield_ref("ref", bf, reg_offset=0, slice_width=8, field_offset=0)
    reg = make_register("reg", ref, address=0x10, width=8)

    def fake_read(addr):
        assert addr == 0x10
        return 0xAB

    monkeypatch.setattr(reg, "_reg_read_func", fake_read)
    returned = reg.read()

    assert returned == 0xAB
    assert reg.value == 0xAB
    assert bf.value == 0xAB


def test_register_write_calls_io_with_current_or_explicit_value(monkeypatch):
    bf = BitField(name="bf", width=8, reset_val=0x11)
    ref = make_bitfield_ref("ref", bf, reg_offset=0, slice_width=8, field_offset=0)
    reg = make_register("reg", ref, address=0x20, width=8)

    calls = []

    def fake_write(addr, val):
        calls.append((addr, val))

    monkeypatch.setattr(reg, "_reg_write_func", fake_write)

    reg.write()
    reg.write(0x55)

    assert calls == [(0x20, 0x11), (0x20, 0x55)]
    assert reg.value == 0x55


def test_get_dimensions_and_registermap_read_write_roundtrip(monkeypatch):
    bf1 = BitField(name="bf1", width=8, reset_val=0x00)
    bf2 = BitField(name="bf2", width=16, reset_val=0x0000)
    r1 = make_bitfield_ref("r1", bf1, reg_offset=0, slice_width=8, field_offset=0)
    r2 = make_bitfield_ref("r2", bf2, reg_offset=0, slice_width=16, field_offset=0)
    reg1 = make_register("reg1", r1, address=0x100, width=8)
    reg2 = make_register("reg2", r2, address=0x102, width=16)
    rm = make_register_map("rm", reg1, reg2)

    assert get_dimensions(0x100, 0x102, 16) == (2, 4, 2)

    def fake_block_read(addr, size):
        assert addr == 0x100
        assert size == 4
        return bytes([0x11, 0x00, 0x22, 0x33])

    monkeypatch.setattr(rm, "_block_read_func", fake_block_read)

    data = rm.read()
    assert data == bytes([0x11, 0x00, 0x22, 0x33])
    assert reg1.value == 0x11
    assert reg2.value == 0x3322

    written = []

    def fake_block_write(addr, payload):
        written.append((addr, bytes(payload)))

    monkeypatch.setattr(rm, "_block_write_func", fake_block_write)

    reg1.value = 0xAA
    reg2.value = 0x55CC
    rm.write()

    assert written == [(0x100, bytes([0xAA, 0x00, 0xCC, 0x55]))]
