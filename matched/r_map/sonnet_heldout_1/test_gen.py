import pytest

import r_map
from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.Register import Register
from r_map.RegisterMap import RegisterMap, get_dimensions


def make_reg(name="reg", width=32, fields=(), local_address=0):
    """fields: iterable of (name, bf_width, reset_val, reg_offset[, access])"""
    reg = Register(name=name, width=width, local_address=local_address)
    bfs = []
    for spec in fields:
        fname, fwidth, reset, offset = spec[:4]
        access = spec[4] if len(spec) > 4 else "RW"
        bf = BitField(name=fname, width=fwidth, reset_val=reset, access=access)
        ref = BitFieldRef(name=fname + "_ref", reg_offset=offset)
        ref._add(bf)
        reg._add(ref)
        bfs.append(bf)
    return reg, bfs


def set_attr(obj, name, value):
    try:
        setattr(obj, name, value)
    except AttributeError:
        pytest.skip(f"cannot set {name} on {type(obj).__name__}")


def error_text(errors):
    out = []
    for e in errors:
        text = str(e)
        try:
            text += " " + " ".join(str(v) for v in vars(e).values())
        except TypeError:
            pass
        out.append(text)
    return out


# ---------------------------------------------------------------- BitField

def test_bitfield_defaults_and_mask():
    bf = BitField(name="bf", width=4, reset_val=3)
    assert bf.width == 4
    assert bf.reset_val == 3
    assert bf.mask == 0xF
    assert bf.value == 3


def test_bitfield_invalid_arguments():
    with pytest.raises(ValueError):
        BitField(name="a", width=0)
    with pytest.raises(ValueError):
        BitField(name="b", width=4, reset_val=-1)


def test_bitfield_value_is_masked_and_reset():
    bf = BitField(name="bf", width=4, reset_val=2)
    bf.value = 0x1F5
    assert bf.value == 0x5
    bf.reset()
    assert bf.value == 2


def test_bitfield_value_from_enumeration_name_and_object():
    bf = BitField(name="bf", width=4)
    e1 = r_map.Enumeration(name="ONE", value=1)
    e2 = r_map.Enumeration(name="TWO", value=2)
    bf._add(e1)
    bf._add(e2)
    bf.value = "TWO"
    assert bf.value == 2
    bf.value = e1
    assert bf.value == 1


def test_bitfield_value_bad_inputs():
    bf = BitField(name="bf", width=4)
    with pytest.raises(ValueError):
        bf.value = "does_not_exist"
    with pytest.raises(NotImplementedError):
        bf.value = 1.5


def test_bitfield_annotation():
    bf = BitField(name="bf", width=4)
    bf._add(r_map.Enumeration(name="ONE", value=1))
    bf.value = 1
    assert bf.annotation == "ONE"
    bf.value = 0xA
    assert bf.annotation == hex(0xA)


def test_bitfield_validate_duplicate_enumeration_values():
    bf = BitField(name="bf", width=4)
    bf._add(r_map.Enumeration(name="A", value=1))
    bf._add(r_map.Enumeration(name="B", value=1))
    errors = list(bf.validate())
    assert any("same value" in t for t in error_text(errors))


def test_bitfield_io_without_parent_raises():
    bf = BitField(name="bf", width=4)
    with pytest.raises(ValueError):
        bf.read()
    with pytest.raises(ValueError):
        bf.write(1)


# ------------------------------------------------------------- BitFieldRef

def test_bitfieldref_value_with_offsets():
    bf = BitField(name="bf", width=8, reset_val=0)
    ref = BitFieldRef(name="ref", reg_offset=4, field_offset=4, slice_width=4)
    ref._add(bf)
    bf.value = 0xAB
    assert ref.value == 0xA0
    assert ref.bf is bf


def test_bitfieldref_default_slice_width_uses_bitfield_width():
    bf = BitField(name="bf", width=6)
    ref = BitFieldRef(name="ref", reg_offset=0)
    ref._add(bf)
    assert ref.slice_width == 6
    assert ref.mask == 0x3F


def test_bitfieldref_value_setter_preserves_other_bits():
    bf = BitField(name="bf", width=8, reset_val=0xFF)
    ref = BitFieldRef(name="ref", reg_offset=8, field_offset=4, slice_width=4)
    ref._add(bf)
    ref.value = 0x0 << 8
    assert bf.value == 0x0F
    ref.value = 0x5 << 8
    assert bf.value == 0x5F


def test_bitfieldref_reset_val_and_references():
    bf = BitField(name="bf", width=4, reset_val=0x9)
    ref = BitFieldRef(name="ref", reg_offset=8)
    ref._add(bf)
    assert ref.reset_val == 0x9 << 8
    assert ref in bf._references


def test_bitfieldref_invalid_arguments():
    with pytest.raises(ValueError):
        BitFieldRef(name="r1", reg_offset=0, slice_width=-1)
    with pytest.raises(ValueError):
        BitFieldRef(name="r2", reg_offset=0, field_offset=-1)
    ref = BitFieldRef(name="r3", reg_offset=0)
    with pytest.raises(ValueError):
        ref._add(object())


# ---------------------------------------------------------------- Register

def test_register_value_combines_fields():
    reg, _ = make_reg(width=16, fields=[("lo", 4, 0x3, 0), ("hi", 4, 0xA, 8)])
    assert reg.value == 0xA03
    assert reg.reset_val == 0xA03


def test_register_value_setter_distributes_to_fields():
    reg, (lo, hi) = make_reg(width=16, fields=[("lo", 4, 0, 0), ("hi", 4, 0, 8)])
    reg.value = 0x0B07
    assert lo.value == 0x7
    assert hi.value == 0xB
    assert reg.value == 0x0B07


def test_register_reset_restores_reset_values():
    reg, (lo, hi) = make_reg(width=8, fields=[("lo", 4, 0x1, 0), ("hi", 4, 0x2, 4)])
    reg.value = 0xFF
    assert reg.value == 0xFF
    reg.reset()
    assert reg.value == 0x21


def test_register_iterates_sorted_by_reg_offset_and_access():
    reg, _ = make_reg(width=8, fields=[("hi", 4, 0, 4, "RW"), ("lo", 4, 0, 0, "RO")])
    assert [c.reg_offset for c in reg] == [0, 4]
    assert reg.access == "RO|RW"


def test_register_validate_reports_empty_and_overlap():
    empty = Register(name="empty", width=8, local_address=0)
    assert any("No bitfieldrefs" in t for t in error_text(empty.validate()))

    overlapping, _ = make_reg(
        name="ov", width=8, fields=[("a", 4, 0, 0), ("b", 4, 0, 2)])
    assert any("overlaps" in t for t in error_text(overlapping.validate()))

    good, _ = make_reg(name="good", width=8, fields=[("a", 4, 0, 0), ("b", 4, 0, 4)])
    texts = error_text(good.validate())
    assert not any("overlaps" in t or "No bitfieldrefs" in t for t in texts)


def test_register_read_and_write_use_io_functions():
    reg, (lo, hi) = make_reg(width=8, fields=[("lo", 4, 0, 0), ("hi", 4, 0, 4)])
    writes = []
    set_attr(reg, "_reg_read_func", lambda addr: 0xA5)
    set_attr(reg, "_reg_write_func", lambda addr, val: writes.append((addr, val)))

    assert reg.read() == 0xA5
    assert reg.value == 0xA5

    reg.write(0x3C)
    assert reg.value == 0x3C
    reg.write()
    assert writes == [(reg.address, 0x3C), (reg.address, 0x3C)]


def test_bitfield_write_does_read_modify_write_on_parent():
    reg, (lo, hi) = make_reg(width=8, fields=[("lo", 4, 0, 0), ("hi", 4, 0, 4)])
    writes = []
    set_attr(reg, "_reg_read_func", lambda addr: 0xA5)
    set_attr(reg, "_reg_write_func", lambda addr, val: writes.append((addr, val)))

    lo.write(0x3)
    assert writes == [(reg.address, 0xA3)]
    assert lo.read() == 0x5  # re-read from hardware value 0xA5


# ------------------------------------------------------------- RegisterMap

def test_get_dimensions():
    assert get_dimensions(0x10, 0x14, 32) == (4, 8, 4)
    assert get_dimensions(0x10, 0x10, 12) == (0, 2, 2)


def make_map():
    rm = RegisterMap(name="rm")
    r0, _ = make_reg("r0", 32, [("f0", 32, 0, 0)], local_address=0)
    r1, _ = make_reg("r1", 32, [("f1", 32, 0, 0)], local_address=4)
    rm._add(r0)
    rm._add(r1)
    return rm, r0, r1


def test_register_map_empty_raises():
    rm = RegisterMap(name="empty_rm")
    with pytest.raises(ValueError):
        rm._block_dimensions
    with pytest.raises(ValueError):
        rm.read()


def test_register_map_block_dimensions():
    rm, r0, r1 = make_map()
    assert rm._block_dimensions == (r0.address, 8)


def test_register_map_read_populates_registers():
    rm, r0, r1 = make_map()
    calls = []

    def block_read(addr, size):
        calls.append((addr, size))
        return bytes(range(8))

    set_attr(rm, "_block_read_func", block_read)
    data = rm.read()
    assert data == bytes(range(8))
    assert calls == [(r0.address, 8)]
    assert r0.value == 0x03020100
    assert r1.value == 0x07060504


def test_register_map_write_builds_little_endian_block():
    rm, r0, r1 = make_map()
    r0.value = 0x03020100
    r1.value = 0x07060504
    written = []
    set_attr(rm, "_block_write_func", lambda addr, data: written.append((addr, bytes(data))))
    rm.write()
    assert written == [(r0.address, bytes(range(8)))]


def test_register_map_write_with_data_updates_registers():
    rm, r0, r1 = make_map()
    written = []
    set_attr(rm, "_block_write_func", lambda addr, data: written.append((addr, bytes(data))))
    payload = bytes([0xFF] * 4 + [0x01, 0x00, 0x00, 0x00])
    rm.write(payload)
    assert r0.value == 0xFFFFFFFF
    assert r1.value == 1
    assert written == [(r0.address, payload)]
