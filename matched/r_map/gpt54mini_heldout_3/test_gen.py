import pytest

import r_map
from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.Register import Register
from r_map.RegisterMap import RegisterMap, get_dimensions
from r_map.ValidationError import ValidationError


class DummyEnum:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        return self.name


def make_bitfield(*, name="bf", width=4, reset_val=0, access="RW", enums=None):
    bf = BitField(name=name, width=width, reset_val=reset_val, access=access)
    bf._children = {}
    bf._parent = None
    if enums:
        for i, e in enumerate(enums):
            bf._children[f"e{i}"] = e
    return bf


def make_bfref(
    bf,
    *,
    name="ref",
    reg_offset=0,
    slice_width=0,
    field_offset=0,
):
    ref = BitFieldRef(
        name=name,
        reg_offset=reg_offset,
        slice_width=slice_width,
        field_offset=field_offset,
    )
    ref._children = {}
    ref._parent = None
    ref._bf = None
    ref._add(bf)
    return ref


def make_register(*, name="reg", address=0, width=8, refs=None):
    reg = Register(name=name, address=address, width=width)
    reg._children = {}
    reg._parent = None
    if refs:
        for i, ref in enumerate(refs):
            reg._children[f"r{i}"] = ref
            ref._parent = reg
    return reg


def make_register_map(*, name="rm", regs=None, block_read=None, block_write=None):
    rm = RegisterMap(name=name)
    rm._children = {}
    rm._parent = None
    if regs:
        for i, reg in enumerate(regs):
            rm._children[f"reg{i}"] = reg
            reg._parent = rm
    rm._block_read_func = block_read or (lambda address, size: bytes(size))
    rm._block_write_func = block_write or (lambda address, data: None)
    return rm


def test_bitfield_init_masks_reset_value_to_width():
    bf = BitField(name="bf", width=3, reset_val=15, access="RW")
    assert bf.mask == 0b111
    assert bf.reset_val == 7
    assert bf.value == 7


def test_bitfield_rejects_invalid_width_and_negative_reset():
    with pytest.raises(ValueError):
        BitField(name="bf", width=0, reset_val=0)
    with pytest.raises(ValueError):
        BitField(name="bf", width=1, reset_val=-1)


def test_bitfield_value_assignment_from_int_string_and_enumeration():
    bf = make_bitfield(enums=[DummyEnum("ZERO", 0), DummyEnum("ONE", 1), DummyEnum("TWO", 2)])

    bf.value = 5
    assert bf.value == 5 & bf.mask

    bf.value = "ONE"
    assert bf.value == 1

    bf.value = r_map.Enumeration("dummy", 2)
    assert bf.value == 2


def test_bitfield_value_assignment_invalid_string_and_type():
    bf = make_bitfield(enums=[DummyEnum("ZERO", 0)])
    with pytest.raises(ValueError):
        bf.value = "MISSING"
    with pytest.raises(NotImplementedError):
        bf.value = 3.14


def test_bitfield_annotation_prefers_matching_enum_name():
    bf = make_bitfield(enums=[DummyEnum("ZERO", 0), DummyEnum("ONE", 1)])
    bf.value = 1
    assert bf.annotation == "ONE"
    bf.value = 3
    assert bf.annotation == hex(3)


def test_bitfield_validate_reports_duplicate_enumeration_values():
    bf = make_bitfield(enums=[DummyEnum("A", 1), DummyEnum("B", 1), DummyEnum("C", 2)])
    errors = list(bf.validate())
    assert any("same value" in str(e) for e in errors)
    assert all(isinstance(e, ValidationError) for e in errors)


def test_bitfieldref_add_sets_parent_reference_and_defaults_slice_width():
    bf = make_bitfield(width=8)
    ref = BitFieldRef(name="ref", reg_offset=0, slice_width=0, field_offset=0)
    ref._children = {}
    ref._parent = None
    ref._bf = None
    ref._add(bf)

    assert ref.bf is bf
    assert ref.slice_width == 8
    assert ref.mask == 0xFF
    assert bf._references and ref in bf._references
    assert bf.parent is ref


def test_bitfieldref_value_and_reset_value_mapping():
    bf = make_bitfield(width=8, reset_val=0xAB)
    ref = make_bfref(bf, reg_offset=2, slice_width=3, field_offset=4)

    assert ref.reset_val == ((0xAB >> 4) & 0x7) << 2

    bf.value = 0xF0
    assert ref.value == ((0xF0 >> 4) & 0x7) << 2

    ref.value = 0b11100
    expected = (0xF0 & ~(0x7 << 4)) | (0x7 << 4)
    assert bf.value == expected


def test_bitfieldref_validate_without_bitfield_and_with_oversized_slice():
    ref = BitFieldRef(name="ref", reg_offset=0, slice_width=4, field_offset=0)
    ref._children = {}
    ref._parent = None
    ref._bf = None
    errors = list(ref.validate())
    assert any("No child bitfield present" in str(e) for e in errors)

    bf = make_bitfield(width=3)
    ref2 = make_bfref(bf, slice_width=4)
    errors2 = list(ref2.validate())
    assert any("Slice width is larger" in str(e) for e in errors2)


def test_bitfieldref_copy_of_alias_reuses_existing_bitfield():
    bf = make_bitfield(width=8)
    ref1 = make_bfref(bf, name="ref1", reg_offset=0)
    ref2 = BitFieldRef(name="ref2", reg_offset=4, slice_width=0, field_offset=0)
    ref2._children = {}
    ref2._parent = None
    ref2._bf = None
    ref2._add(bf)

    assert getattr(ref2, "_alias", False) is True
    copied = ref2._copy(new_instance=False, new_alias=False)
    assert copied.bf is bf
    assert copied._alias is True


def test_register_iteration_sorts_bitfieldrefs_by_reg_offset():
    bf = make_bitfield(width=8)
    ref_hi = make_bfref(bf, name="hi", reg_offset=4, slice_width=4, field_offset=4)
    ref_lo = make_bfref(bf, name="lo", reg_offset=0, slice_width=4, field_offset=0)
    reg = make_register(refs=[ref_hi, ref_lo])

    offsets = [r.reg_offset for r in reg]
    assert offsets == [0, 4]


def test_register_value_and_access_are_aggregated_from_children():
    bf = make_bitfield(width=8)
    lo = make_bfref(bf, name="lo", reg_offset=0, slice_width=4, field_offset=0)
    hi = make_bfref(bf, name="hi", reg_offset=4, slice_width=4, field_offset=4)
    lo.bf.access = "RO"
    hi.bf.access = "RW"
    reg = make_register(refs=[lo, hi])

    bf.value = 0xAB
    assert reg.value == 0xAB
    assert reg.access == "RO|RW"

    reg.value = 0x12
    assert bf.value == 0x12


def test_register_read_write_calls_underlying_io():
    bf = make_bitfield(width=8)
    ref = make_bfref(bf, reg_offset=0, slice_width=8, field_offset=0)

    reads = []
    writes = []

    def reg_read(address):
        reads.append(address)
        return 0x5A

    def reg_write(address, value):
        writes.append((address, value))

    reg = make_register(address=0x20, refs=[ref])
    reg._reg_read_func = reg_read
    reg._reg_write_func = reg_write

    rv = reg.read()
    assert rv == 0x5A
    assert reads == [0x20]
    assert reg.value == 0x5A

    reg.write()
    assert writes == [(0x20, 0x5A)]


def test_register_validate_reports_missing_children_and_wrong_child_type():
    reg = make_register()
    errors = list(reg.validate())
    assert any("No bitfieldrefs present" in str(e) for e in errors)

    reg2 = make_register()
    reg2._children = {"bad": object()}
    errors2 = list(reg2.validate())
    assert any("not of typeBitFieldRef" in str(e) or "not of type" in str(e) for e in errors2)


def test_register_validate_reports_overlapping_bitfieldrefs():
    bf = make_bitfield(width=16)
    r1 = make_bfref(bf, name="r1", reg_offset=0, slice_width=8, field_offset=0)
    r2 = make_bfref(bf, name="r2", reg_offset=4, slice_width=8, field_offset=4)
    reg = make_register(refs=[r1, r2])
    errors = list(reg.validate())
    assert any("overlaps" in str(e) for e in errors)


def test_registermap_block_dimensions_read_and_write_roundtrip():
    reg1 = make_register(name="r1", address=0x1000, width=8)
    reg2 = make_register(name="r2", address=0x1002, width=16)
    rm = make_register_map(regs=[reg1, reg2])

    # Force deterministic layout
    start, size = rm._block_dimensions
    assert start == 0x1000
    assert size == 4

    data = bytearray([0x11, 0x22, 0x33, 0x44])
    reads = []
    writes = []

    def block_read(address, size):
        reads.append((address, size))
        return bytes(data)

    def block_write(address, out):
        writes.append((address, bytes(out)))

    rm._block_read_func = block_read
    rm._block_write_func = block_write

    out = rm.read()
    assert out == bytes(data)
    assert reads == [(0x1000, 4)]
    assert reg1.value == 0x11
    assert reg2.value == 0x4433

    reg1.value = 0xAA
    reg2.value = 0xBEEF
    rm.write()
    assert writes == [(0x1000, bytes([0xAA, 0x00, 0xEF, 0xBE]))]


def test_registermap_write_partial_data_updates_prefix_only():
    reg1 = make_register(name="r1", address=0x1000, width=8)
    reg2 = make_register(name="r2", address=0x1002, width=16)
    rm = make_register_map(regs=[reg1, reg2])

    captured = []

    def block_write(address, out):
        captured.append((address, bytes(out)))

    rm._block_write_func = block_write
    rm.write(data=bytearray([0x01, 0x02]))
    assert captured == [(0x1000, bytes([0x01, 0x02, 0x00, 0x00]))]
    assert reg1.value == 0x01
    assert reg2.value == 0x0000


def test_get_dimensions_returns_expected_indices():
    start, end, n = get_dimensions(0x10, 0x14, 16)
    assert (start, end, n) == (4, 6, 2)
