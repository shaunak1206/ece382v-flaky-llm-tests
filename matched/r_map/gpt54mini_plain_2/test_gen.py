import pytest

from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.Register import Register
from r_map.RegisterMap import RegisterMap


def make_bitfield(name="bf", width=8, reset_val=0, access="RW", **kwargs):
    return BitField(name=name, width=width, reset_val=reset_val, access=access, **kwargs)


def make_ref(name="ref", reg_offset=0, slice_width=0, field_offset=0, bf=None, **kwargs):
    ref = BitFieldRef(
        name=name,
        reg_offset=reg_offset,
        slice_width=slice_width,
        field_offset=field_offset,
        **kwargs,
    )
    if bf is not None:
        ref._add(bf)
    return ref


def make_register(name="reg", address=0, width=32, refs=()):
    reg = Register(name=name, address=address, width=width)
    for ref in refs:
        reg._add(ref)
    return reg


def make_regmap(name="map", regs=()):
    rm = RegisterMap(name=name)
    for reg in regs:
        rm._add(reg)
    return rm


def test_bitfield_rejects_invalid_width_and_reset_val():
    with pytest.raises(ValueError):
        make_bitfield(width=0)
    with pytest.raises(ValueError):
        make_bitfield(reset_val=-1)


def test_bitfield_masks_value_to_width():
    bf = make_bitfield(width=4, reset_val=0)
    bf.value = 0x1F
    assert bf.value == 0xF


def test_bitfield_accepts_enumeration_name_and_instance():
    from r_map.Enumeration import Enumeration

    bf = make_bitfield(width=2)
    e0 = Enumeration(name="ZERO", value=0)
    e1 = Enumeration(name="ONE", value=1)
    bf._add(e0)
    bf._add(e1)

    bf.value = "ONE"
    assert bf.value == 1
    bf.value = e0
    assert bf.value == 0


def test_bitfield_annotation_returns_matching_name_or_hex():
    from r_map.Enumeration import Enumeration

    bf = make_bitfield(width=2)
    bf._add(Enumeration(name="ZERO", value=0))
    bf._add(Enumeration(name="ONE", value=1))
    bf.value = 1
    assert bf.annotation == "ONE"
    bf.value = 3
    assert bf.annotation == "0x3"


def test_bitfield_validate_reports_duplicate_enumeration_values():
    from r_map.Enumeration import Enumeration

    bf = make_bitfield(width=2)
    bf._add(Enumeration(name="A", value=1))
    bf._add(Enumeration(name="B", value=1))
    errors = list(bf.validate())
    assert any("same value" in str(e) for e in errors)


def test_bitfieldref_rejects_negative_offsets():
    with pytest.raises(ValueError):
        make_ref(reg_offset=-1, bf=make_bitfield())
    with pytest.raises(ValueError):
        make_ref(reg_offset=0, slice_width=1, field_offset=-1, bf=make_bitfield())


def test_bitfieldref_default_slice_width_uses_child_width():
    bf = make_bitfield(width=8, reset_val=0xAA)
    ref = make_ref(reg_offset=0, slice_width=0, field_offset=0, bf=bf)
    assert ref.slice_width == 8
    assert ref.mask == 0xFF
    assert ref.value == 0xAA


def test_bitfieldref_value_get_and_set_updates_underlying_bitfield():
    bf = make_bitfield(width=8, reset_val=0x00)
    ref = make_ref(reg_offset=4, slice_width=4, field_offset=2, bf=bf)

    ref.value = 0xF0
    assert bf.value == 0x3C  # 0b00111100
    assert ref.value == 0xF0  # shifted into register position


def test_bitfieldref_reset_val_reflects_slice_and_offsets():
    bf = make_bitfield(width=8, reset_val=0b10110110)
    ref = make_ref(reg_offset=2, slice_width=3, field_offset=1, bf=bf)
    assert ref.reset_val == 0b10100


def test_bitfieldref_validate_without_child_reports_error():
    ref = make_ref(reg_offset=0, slice_width=1, field_offset=0)
    errors = list(ref.validate())
    assert any("No child bitfield present" in str(e) for e in errors)


def test_register_value_and_reset_aggregate_child_refs():
    bf = make_bitfield(width=8, reset_val=0x12)
    ref0 = make_ref(reg_offset=0, slice_width=4, field_offset=0, bf=bf)
    ref1 = make_ref(reg_offset=4, slice_width=4, field_offset=4, bf=bf)
    reg = make_register(address=0x10, refs=(ref0, ref1))

    assert reg.reset_val == 0x12
    reg.value = 0xAB
    assert reg.value == 0xAB
    reg.reset()
    assert reg.value == 0x12


def test_register_iteration_sorts_by_reg_offset():
    bf = make_bitfield(width=16, reset_val=0)
    high = make_ref(name="high", reg_offset=8, slice_width=8, field_offset=8, bf=bf)
    low = make_ref(name="low", reg_offset=0, slice_width=8, field_offset=0, bf=bf)
    reg = make_register(refs=(high, low))

    assert [r.name for r in reg] == ["low", "high"]


def test_register_validate_no_children_and_non_bitfieldref_child():
    reg = Register(name="reg", address=0x0, width=32)
    errors = list(reg.validate())
    assert any("No bitfieldrefs present" in str(e) for e in errors)

    reg2 = Register(name="reg2", address=0x0, width=32)
    reg2._add(make_bitfield(name="child_bf"))
    errors2 = list(reg2.validate())
    assert any("not of type BitFieldRef" in str(e) for e in errors2)


def test_register_validate_detects_overlapping_refs():
    bf = make_bitfield(width=16)
    r0 = make_ref(name="r0", reg_offset=0, slice_width=8, field_offset=0, bf=bf)
    r1 = make_ref(name="r1", reg_offset=4, slice_width=8, field_offset=8, bf=bf)
    reg = make_register(refs=(r0, r1))
    errors = list(reg.validate())
    assert any("overlaps" in str(e) for e in errors)


def test_register_read_and_write_delegate_to_callbacks():
    bf = make_bitfield(width=8, reset_val=0)
    ref = make_ref(reg_offset=0, slice_width=8, field_offset=0, bf=bf)
    reg = make_register(address=0x20, refs=(ref,))

    reads = []
    writes = []

    def reg_read(addr):
        reads.append(addr)
        return 0x5A

    def reg_write(addr, val):
        writes.append((addr, val))

    reg._reg_read_func = reg_read
    reg._reg_write_func = reg_write

    assert reg.read() == 0x5A
    assert reg.value == 0x5A
    reg.write()
    assert writes == [(0x20, 0x5A)]
    reg.write(0xA5)
    assert writes[-1] == (0x20, 0xA5)


def test_registermap_dimensions_and_empty_map_error():
    reg = make_register(address=0x100, refs=(make_ref(reg_offset=0, slice_width=8, field_offset=0, bf=make_bitfield(width=8)),))
    rm = make_regmap(regs=(reg,))
    assert rm._block_dimensions == (0x100, 1)

    empty = RegisterMap(name="empty")
    with pytest.raises(ValueError):
        _ = empty._block_dimensions


def test_registermap_read_and_write_roundtrip_bytes():
    bf0 = make_bitfield(width=8, reset_val=0x12)
    bf1 = make_bitfield(width=8, reset_val=0x34)
    r0 = make_register(address=0x10, refs=(make_ref(reg_offset=0, slice_width=8, field_offset=0, bf=bf0),))
    r1 = make_register(address=0x11, refs=(make_ref(reg_offset=0, slice_width=8, field_offset=0, bf=bf1),))
    rm = make_regmap(regs=(r0, r1))

    captured = {}

    def block_read(addr, size):
        captured["read"] = (addr, size)
        return bytes([0xAA, 0xBB])

    def block_write(addr, data):
        captured["write"] = (addr, bytes(data))

    rm._block_read_func = block_read
    rm._block_write_func = block_write

    data = rm.read()
    assert captured["read"] == (0x10, 2)
    assert data == b"\xAA\xBB"
    assert r0.value == 0xAA
    assert r1.value == 0xBB

    r0.value = 0x11
    r1.value = 0x22
    rm.write()
    assert captured["write"] == (0x10, b"\x11\x22")
