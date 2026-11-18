import pytest

from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.Register import Register
from r_map.RegisterMap import RegisterMap


class DummyEnum:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        return f"{self.name}={self.value}"


def make_bf_ref(bitfield, *, reg_offset=0, slice_width=0, field_offset=0, name="ref"):
    ref = BitFieldRef(reg_offset=reg_offset, slice_width=slice_width, field_offset=field_offset, name=name)
    ref._add(bitfield)
    return ref


def make_register_with_refs(*refs, name="reg", address=0, width=16):
    reg = Register(name=name, address=address, width=width)
    for ref in refs:
        reg._add(ref)
    return reg


def test_bitfield_rejects_invalid_width_and_reset():
    with pytest.raises(ValueError):
        BitField(width=0)
    with pytest.raises(ValueError):
        BitField(width=1, reset_val=-1)


def test_bitfield_masks_integer_values_and_resets():
    bf = BitField(width=3, reset_val=0b1111, name="bf")
    assert bf.reset_val == 0b111  # masked on init
    bf.value = 0b10101
    assert bf.value == 0b101
    bf.reset()
    assert bf.value == 0b111


def test_bitfield_string_value_and_annotation_uses_enumeration_names():
    bf = BitField(width=2, name="bf")
    bf._children = {
        "zero": DummyEnum("zero", 0),
        "one": DummyEnum("one", 1),
        "two": DummyEnum("two", 2),
    }
    bf.value = "two"
    assert bf.value == 2
    assert bf.annotation == "two"


def test_bitfield_string_value_unknown_name_raises():
    bf = BitField(width=2, name="bf")
    bf._children = {"zero": DummyEnum("zero", 0)}
    with pytest.raises(ValueError):
        bf.value = "missing"


def test_bitfield_validate_detects_duplicate_enumeration_values():
    bf = BitField(width=2, name="bf")
    bf._children = {
        "a": DummyEnum("a", 1),
        "b": DummyEnum("b", 1),
    }
    msgs = [v.message for v in bf.validate()]
    assert any("same value" in msg for msg in msgs)


def test_bitfieldref_default_slice_width_and_value_projection():
    bf = BitField(width=8, reset_val=0xAB, name="bf")
    ref = make_bf_ref(bf, reg_offset=0, slice_width=0, field_offset=0, name="ref")
    assert ref.slice_width == 8
    assert ref.reset_val == 0xAB
    assert ref.value == 0xAB


def test_bitfieldref_value_setter_updates_only_targeted_bits():
    bf = BitField(width=8, reset_val=0xF0, name="bf")
    ref = make_bf_ref(bf, reg_offset=4, slice_width=4, field_offset=0, name="ref")
    ref.value = 0xA0  # register bits 4..7 should become 0xA
    assert bf.value == 0xFA
    assert ref.value == 0xA0


def test_bitfieldref_validate_missing_child_and_large_slice_width():
    ref = BitFieldRef(reg_offset=0, slice_width=4, field_offset=0, name="ref")
    msgs = [v.message for v in ref.validate()]
    assert any("No child bitfield present" in msg for msg in msgs)

    bf = BitField(width=2, name="bf")
    ref2 = make_bf_ref(bf, reg_offset=0, slice_width=4, field_offset=0, name="ref2")
    msgs2 = [v.message for v in ref2.validate()]
    assert any("Slice width is larger" in msg for msg in msgs2)


def test_register_iterates_bitfieldrefs_sorted_by_reg_offset():
    bf = BitField(width=8, reset_val=0, name="bf")
    ref_hi = make_bf_ref(bf, reg_offset=8, slice_width=4, field_offset=4, name="hi")
    ref_lo = make_bf_ref(bf, reg_offset=0, slice_width=4, field_offset=0, name="lo")
    reg = make_register_with_refs(ref_hi, ref_lo, width=16)
    assert [c.name for c in reg] == ["lo", "hi"]


def test_register_value_aggregates_and_distributes_across_refs():
    bf = BitField(width=8, reset_val=0, name="bf")
    low = make_bf_ref(bf, reg_offset=0, slice_width=4, field_offset=0, name="low")
    high = make_bf_ref(bf, reg_offset=4, slice_width=4, field_offset=4, name="high")
    reg = make_register_with_refs(low, high, width=8)
    reg.value = 0xAB
    assert bf.value == 0xAB
    assert reg.value == 0xAB


def test_register_validate_empty_non_ref_and_overlap():
    empty = Register(name="empty", width=8)
    msgs = [v.message for v in empty.validate()]
    assert any("No bitfieldrefs present" in msg for msg in msgs)

    reg = Register(name="bad", width=8)
    reg._children = {"x": object()}
    msgs2 = [v.message for v in reg.validate()]
    assert any("is not of typeBitFieldRef" in msg for msg in msgs2)

    bf = BitField(width=8, name="bf")
    a = make_bf_ref(bf, reg_offset=0, slice_width=6, field_offset=0, name="a")
    b = make_bf_ref(bf, reg_offset=4, slice_width=4, field_offset=4, name="b")
    reg2 = make_register_with_refs(a, b, width=8)
    msgs3 = [v.message for v in reg2.validate()]
    assert any("overlaps" in msg for msg in msgs3)


def test_register_read_updates_value_but_returns_raw_io_value():
    bf = BitField(width=8, reset_val=0, name="bf")
    ref = make_bf_ref(bf, reg_offset=0, slice_width=8, field_offset=0, name="ref")
    reg = make_register_with_refs(ref, width=8, address=0x10)

    calls = {}

    def read_func(addr):
        calls["addr"] = addr
        return 0x5A

    reg._reg_read_func = read_func
    raw = reg.read()
    assert raw == 0x5A
    assert calls["addr"] == 0x10
    assert reg.value == 0x5A


def test_register_write_uses_current_value_when_no_argument_supplied():
    bf = BitField(width=8, reset_val=0, name="bf")
    ref = make_bf_ref(bf, reg_offset=0, slice_width=8, field_offset=0, name="ref")
    reg = make_register_with_refs(ref, width=8, address=0x20)
    reg.value = 0x3C

    calls = {}

    def write_func(addr, val):
        calls["addr"] = addr
        calls["val"] = val

    reg._reg_write_func = write_func
    reg.write()
    assert calls == {"addr": 0x20, "val": 0x3C}


def test_registermap_block_dimensions_read_and_partial_write():
    rm = RegisterMap(name="rm")
    r0 = Register(name="r0", address=0x100, width=16)
    r1 = Register(name="r1", address=0x102, width=8)
    rm._add(r0)
    rm._add(r1)

    assert rm._block_dimensions == (0x100, 3)

    read_calls = {}
    write_calls = {}

    def block_read(addr, size):
        read_calls["args"] = (addr, size)
        return bytes([0x34, 0x12, 0x56])

    def block_write(addr, data):
        write_calls["args"] = (addr, bytes(data))

    rm._block_read_func = block_read
    rm._block_write_func = block_write

    data = rm.read()
    assert data == bytes([0x34, 0x12, 0x56])
    assert read_calls["args"] == (0x100, 3)
    assert r0.value == 0x1234
    assert r1.value == 0x56

    rm.write()
    assert write_calls["args"] == (0x100, bytes([0x34, 0x12, 0x56]))

    rm.write(bytearray([0xAA, 0xBB]))
    assert r0.value == 0xBBAA
    assert r1.value == 0x56


def test_registermap_write_without_registers_raises():
    rm = RegisterMap(name="empty_map")
    rm._block_write_func = lambda addr, data: None
    with pytest.raises(ValueError):
        rm.write()
