import pytest

from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.Register import Register
from r_map.RegisterMap import RegisterMap


def _make_bitfield(name="bf", width=8, reset_val=0, access="RW", children=None):
    bf = BitField(width=width, reset_val=reset_val, access=access, name=name)
    if children:
        bf._children = {c.name: c for c in children}
        for c in children:
            c._parent = bf
    return bf


def _make_bf_ref(name, bf, reg_offset=0, slice_width=0, field_offset=0):
    ref = BitFieldRef(
        reg_offset=reg_offset,
        slice_width=slice_width,
        field_offset=field_offset,
        name=name,
    )
    ref._children = {bf.name: bf}
    ref._bf = bf
    ref._parent = None
    if bf.parent is None:
        bf._parent = ref
    if not hasattr(bf, "_references"):
        bf._references = set()
    bf._references.add(ref)
    return ref


def _make_register(name="reg", width=8, refs=None):
    reg = Register(width=width, name=name)
    reg._children = {r.name: r for r in (refs or [])}
    for r in (refs or []):
        r._parent = reg
    return reg


def _make_regmap(name="map", regs=None):
    rm = RegisterMap(name=name)
    rm._children = {r.name: r for r in (regs or [])}
    for r in (regs or []):
        r._parent = rm
    return rm


def test_bitfield_rejects_invalid_width_and_reset_val():
    with pytest.raises(ValueError):
        BitField(width=0, reset_val=0, name="bf0")
    with pytest.raises(ValueError):
        BitField(width=1, reset_val=-1, name="bf1")


def test_bitfield_masks_reset_val_and_initial_value():
    bf = BitField(width=3, reset_val=0xF, name="bf")
    assert bf.mask == 0b111
    assert bf.reset_val == 0b111
    assert bf.value == 0b111


def test_bitfield_integer_assignment_masks_value():
    bf = BitField(width=4, reset_val=0, name="bf")
    bf.value = 0x3F
    assert bf.value == 0xF


def test_bitfield_reset_restores_reset_val():
    bf = BitField(width=4, reset_val=0xA, name="bf")
    bf.value = 0x0
    bf.reset()
    assert bf.value == 0xA


def test_bitfield_annotation_defaults_to_hex_when_no_enum_matches():
    bf = BitField(width=4, reset_val=0, name="bf")
    bf.value = 3
    assert bf.annotation == "0x3"


def test_bitfield_validate_detects_duplicate_enumeration_values():
    bf = BitField(width=4, reset_val=0, name="bf")
    assert list(bf.validate()) == []


def test_bitfieldref_rejects_negative_dimensions():
    with pytest.raises(ValueError):
        BitFieldRef(reg_offset=0, slice_width=-1, field_offset=0, name="ref")
    with pytest.raises(ValueError):
        BitFieldRef(reg_offset=0, slice_width=1, field_offset=-1, name="ref")


def test_bitfieldref_defaults_slice_width_to_bitfield_width():
    bf = _make_bitfield(width=8, reset_val=0x12, name="bf")
    ref = _make_bf_ref("ref", bf, reg_offset=0, slice_width=0, field_offset=0)
    assert ref.slice_width == 8
    assert ref.mask == 0xFF


def test_bitfieldref_value_reads_selected_slice():
    bf = _make_bitfield(width=8, reset_val=0xA5, name="bf")
    ref = _make_bf_ref("ref", bf, reg_offset=2, slice_width=3, field_offset=1)
    assert ref.value == ((0xA5 >> 1) & 0x7) << 2


def test_bitfieldref_value_writes_only_selected_slice():
    bf = _make_bitfield(width=8, reset_val=0xAA, name="bf")
    ref = _make_bf_ref("ref", bf, reg_offset=2, slice_width=3, field_offset=1)
    ref.value = 0b11100
    expected = (0xAA & ~(0x7 << 1)) | (0x7 << 1)
    assert bf.value == expected


def test_bitfieldref_validate_detects_missing_bitfield():
    ref = BitFieldRef(reg_offset=0, slice_width=1, field_offset=0, name="ref")
    ref._bf = None
    errs = list(ref.validate())
    assert any("No child bitfield present" in str(e) for e in errs)


def test_bitfieldref_validate_detects_too_wide_slice():
    bf = _make_bitfield(width=4, reset_val=0, name="bf")
    ref = _make_bf_ref("ref", bf, reg_offset=0, slice_width=5, field_offset=0)
    errs = list(ref.validate())
    assert any("Slice width is larger than bitfield's width" in str(e) for e in errs)


def test_register_value_roundtrip_through_refs_and_sorting_by_reg_offset():
    bf = _make_bitfield(width=8, reset_val=0, name="bf")
    ref_hi = _make_bf_ref("hi", bf, reg_offset=4, slice_width=4, field_offset=4)
    ref_lo = _make_bf_ref("lo", bf, reg_offset=0, slice_width=4, field_offset=0)
    reg = _make_register(refs=[ref_hi, ref_lo])

    assert [c.name for c in reg] == ["lo", "hi"]
    reg.value = 0xAB
    assert bf.value == 0xAB
    assert reg.value == 0xAB


def test_register_reset_resets_all_child_bitfields():
    bf = _make_bitfield(width=8, reset_val=0x12, name="bf")
    ref = _make_bf_ref("ref", bf, reg_offset=0, slice_width=8, field_offset=0)
    reg = _make_register(refs=[ref])
    reg.value = 0xFF
    reg.reset()
    assert bf.value == 0x12


def test_register_validate_rejects_no_bitfieldrefs_and_wrong_child_type():
    reg = _make_register(refs=[])
    errs = list(reg.validate())
    assert any("No bitfieldrefs present" in str(e) for e in errs)

    reg2 = _make_register(refs=[])
    reg2._children = {"x": object()}
    errs2 = list(reg2.validate())
    assert any("is not of typeBitFieldRef" in str(e) for e in errs2)


def test_register_validate_detects_overlapping_refs():
    bf = _make_bitfield(width=8, reset_val=0, name="bf")
    r1 = _make_bf_ref("r1", bf, reg_offset=0, slice_width=4, field_offset=0)
    r2 = _make_bf_ref("r2", bf, reg_offset=2, slice_width=4, field_offset=4)
    reg = _make_register(refs=[r1, r2])
    errs = list(reg.validate())
    assert any("overlaps" in str(e) for e in errs)


def test_register_read_and_write_delegate_to_io_functions():
    bf = _make_bitfield(width=8, reset_val=0, name="bf")
    ref = _make_bf_ref("ref", bf, reg_offset=0, slice_width=8, field_offset=0)
    reg = _make_register(refs=[ref])

    calls = {"read": [], "write": []}

    def reg_read(addr):
        calls["read"].append(addr)
        return 0x5A

    def reg_write(addr, val):
        calls["write"].append((addr, val))

    reg._reg_read_func = reg_read
    reg._reg_write_func = reg_write
    reg._address = 0x10

    assert reg.read() == 0x5A
    assert bf.value == 0x5A

    reg.write()
    assert calls["write"] == [(0x10, 0x5A)]


def test_registermap_block_dimensions_read_and_write_roundtrip():
    r1_bf = _make_bitfield(width=8, reset_val=0x12, name="bf1")
    r1_ref = _make_bf_ref("r1ref", r1_bf, reg_offset=0, slice_width=8, field_offset=0)
    r1 = _make_register(name="r1", refs=[r1_ref])
    r1._address = 0x1000

    r2_bf = _make_bitfield(width=16, reset_val=0x3456, name="bf2")
    r2_ref = _make_bf_ref("r2ref", r2_bf, reg_offset=0, slice_width=16, field_offset=0)
    r2 = _make_register(name="r2", width=16, refs=[r2_ref])
    r2._address = 0x1001

    rm = _make_regmap(regs=[r1, r2])

    assert rm._block_dimensions == (0x1000, 3)

    written = {}

    def block_read(addr, size):
        assert addr == 0x1000
        assert size == 3
        return bytes([0x11, 0x22, 0x33])

    def block_write(addr, data):
        written["call"] = (addr, bytes(data))

    rm._block_read_func = block_read
    rm._block_write_func = block_write

    data = rm.read()
    assert data == bytes([0x11, 0x22, 0x33])
    assert r1.value == 0x11
    assert r2.value == 0x3322

    rm.write()
    assert written["call"][0] == 0x1000
    assert len(written["call"][1]) == 3


def test_registermap_write_empty_raises_valueerror():
    rm = _make_regmap(regs=[])
    rm._block_dimensions = (0x0, 0)
    rm._block_write_func = lambda addr, data: None
    with pytest.raises(ValueError):
        rm.write()


def test_registermap_partial_write_updates_only_available_registers():
    r1_bf = _make_bitfield(width=8, reset_val=0, name="bf1")
    r1_ref = _make_bf_ref("r1ref", r1_bf, reg_offset=0, slice_width=8, field_offset=0)
    r1 = _make_register(name="r1", refs=[r1_ref])
    r1._address = 0x0

    r2_bf = _make_bitfield(width=8, reset_val=0, name="bf2")
    r2_ref = _make_bf_ref("r2ref", r2_bf, reg_offset=0, slice_width=8, field_offset=0)
    r2 = _make_register(name="r2", refs=[r2_ref])
    r2._address = 0x1

    rm = _make_regmap(regs=[r1, r2])
    rm._block_dimensions = (0x0, 2)
    rm._block_write_func = lambda addr, data: None

    rm.write(bytearray([0xAA]))
    assert r1.value == 0xAA
    assert r2.value == 0x00
