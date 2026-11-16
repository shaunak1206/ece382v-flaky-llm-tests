import importlib

import pytest

from r_map import BitField, BitFieldRef, Register, RegisterMap

get_dimensions = importlib.import_module("r_map.RegisterMap").get_dimensions


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _make_addressed(cls, address, **kwargs):
    """Build an addressed node, tolerating either address keyword spelling."""
    first_ok = None
    last_exc = None
    for key in ("local_address", "address"):
        try:
            obj = cls(**{key: address}, **kwargs)
        except (TypeError, ValueError, AttributeError) as exc:
            last_exc = exc
            continue
        if first_ok is None:
            first_ok = obj
        if obj.address == address:
            return obj
    if first_ok is not None:
        return first_ok
    raise last_exc


def _set_io(monkeypatch, obj, name, func):
    """Attach an IO callable to obj (instance attribute, else class property)."""
    try:
        setattr(obj, name, func)
        if getattr(obj, name) is func:
            return
    except AttributeError:
        pass
    monkeypatch.setattr(type(obj), name, property(lambda self: func),
                        raising=False)


def make_field(name, width, reset_val=0, access="RW"):
    return BitField(name=name, width=width, reset_val=reset_val, access=access)


def make_ref(name, bf, reg_offset, slice_width=0, field_offset=0):
    ref = BitFieldRef(name=name, reg_offset=reg_offset,
                      slice_width=slice_width, field_offset=field_offset)
    ref._add(bf)
    return ref


def make_reg(name, address, fields, width=32):
    """fields: list of (name, bf_width, reg_offset, reset_val, access)."""
    reg = _make_addressed(Register, address, name=name, width=width)
    for fname, fwidth, offset, reset_val, access in fields:
        bf = make_field(fname, fwidth, reset_val, access)
        reg._add(make_ref("ref_" + fname, bf, offset))
    return reg


def make_full_reg(name, address, width):
    return make_reg(name, address, [(name + "_bf", width, 0, 0, "RW")],
                    width=width)


def make_map(regs):
    rmap = _make_addressed(RegisterMap, 0, name="rmap")
    for r in regs:
        rmap._add(r)
    return rmap


def byte_reg_two_fields(address=0x10):
    return make_reg("r", address,
                    [("lo", 4, 0, 0, "RW"), ("hi", 4, 4, 0, "RW")], width=8)


# --------------------------------------------------------------------------
# BitField
# --------------------------------------------------------------------------

def test_bitfield_invalid_constructor_args():
    with pytest.raises(ValueError):
        BitField(name="a", width=0)
    with pytest.raises(ValueError):
        BitField(name="a", width=4, reset_val=-1)


def test_bitfield_value_masking_and_reset():
    bf = BitField(name="a", width=4, reset_val=0x3)
    assert bf.value == 0x3
    bf.value = 0x1F
    assert bf.value == 0xF
    bf.reset()
    assert bf.value == 0x3


def test_bitfield_bad_value_types():
    bf = BitField(name="a", width=4)
    with pytest.raises(ValueError):
        bf.value = "nonexistent"
    with pytest.raises(NotImplementedError):
        bf.value = 1.5


def test_bitfield_annotation_without_enumerations_is_hex():
    bf = BitField(name="a", width=8)
    bf.value = 0x2A
    assert bf.annotation == hex(0x2A)


def test_bitfield_io_without_parent_raises():
    bf = BitField(name="a", width=4)
    with pytest.raises(ValueError):
        bf._get_parent_regs()
    with pytest.raises(ValueError):
        bf.read()
    with pytest.raises(ValueError):
        bf.write(1)


def test_bitfield_write_does_read_modify_write(monkeypatch):
    reg = byte_reg_two_fields(0x10)
    mem = {0x10: 0xA0}
    writes = []

    def rd(addr):
        return mem[addr]

    def wr(addr, val):
        writes.append((addr, val))
        mem[addr] = val

    _set_io(monkeypatch, reg, "_reg_read_func", rd)
    _set_io(monkeypatch, reg, "_reg_write_func", wr)
    lo = [r for r in reg if r.reg_offset == 0][0].bf
    lo.write(0x5)
    assert writes == [(0x10, 0xA5)]
    assert mem[0x10] == 0xA5

    # read via the bitfield refreshes it from memory
    mem[0x10] = 0x07
    assert lo.read() == 0x7


# --------------------------------------------------------------------------
# BitFieldRef
# --------------------------------------------------------------------------

def test_bitfieldref_invalid_constructor_args():
    with pytest.raises(ValueError):
        BitFieldRef(name="r", reg_offset=0, slice_width=-1)
    with pytest.raises(ValueError):
        BitFieldRef(name="r", reg_offset=0, field_offset=-1)


def test_bitfieldref_add_rejects_non_bitfield_and_defaults_width():
    ref = BitFieldRef(name="r", reg_offset=0)
    with pytest.raises(ValueError):
        ref._add("not a bitfield")
    bf = make_field("f", 6)
    ref._add(bf)
    assert ref.bf is bf
    assert ref.slice_width == 6
    assert ref.mask == 0x3F


def test_bitfieldref_value_and_reset_val_slicing():
    bf = make_field("f", 8, reset_val=0xCD)
    bf.value = 0xAB
    ref = make_ref("r", bf, reg_offset=8, slice_width=4, field_offset=4)
    assert ref.value == 0xA00
    assert ref.reset_val == 0xC00


def test_bitfieldref_value_setter_updates_only_slice():
    bf = make_field("f", 8)
    bf.value = 0xAB
    ref = make_ref("r", bf, reg_offset=8, slice_width=4, field_offset=4)
    ref.value = 0x300
    assert bf.value == 0x3B


def test_bitfieldref_second_reference_is_alias():
    bf = make_field("f", 8)
    ref1 = make_ref("r1", bf, 0)
    ref2 = make_ref("r2", bf, 8)
    assert len(bf._references) == 2
    assert ref2._alias is True
    assert ref2._ref is ref1


def test_bitfieldref_validate_flags_oversized_slice():
    good = make_ref("g", make_field("fg", 8), 0, slice_width=4)
    bad = make_ref("b", make_field("fb", 8), 0, slice_width=16)
    good_errors = list(good.validate())
    bad_errors = list(bad.validate())
    assert len(bad_errors) == len(good_errors) + 1


# --------------------------------------------------------------------------
# Register
# --------------------------------------------------------------------------

def test_register_value_combines_and_distributes_fields():
    reg = byte_reg_two_fields()
    reg.value = 0xA5
    values = {r.bf.name: r.bf.value for r in reg}
    assert values == {"lo": 0x5, "hi": 0xA}
    assert reg.value == 0xA5


def test_register_iterates_sorted_by_reg_offset():
    reg = _make_addressed(Register, 0, name="r", width=8)
    hi = make_ref("ref_hi", make_field("hi", 4), 4)
    lo = make_ref("ref_lo", make_field("lo", 4), 0)
    reg._add(hi)
    reg._add(lo)
    assert [c.reg_offset for c in reg] == [0, 4]


def test_register_reset_and_reset_val():
    reg = make_reg("r", 0, [("lo", 4, 0, 0x3, "RW"), ("hi", 4, 4, 0x2, "RW")],
                   width=8)
    assert reg.reset_val == 0x23
    reg.value = 0xFF
    assert reg.value == 0xFF
    reg.reset()
    assert reg.value == 0x23


def test_register_access_is_sorted_union():
    reg = make_reg("r", 0, [("a", 2, 0, 0, "RW"),
                            ("b", 2, 2, 0, "RO"),
                            ("c", 2, 4, 0, "RW")], width=8)
    assert reg.access == "RO|RW"


def test_register_read_and_write_use_io_functions(monkeypatch):
    reg = make_full_reg("r", 0x20, 8)
    writes = []
    _set_io(monkeypatch, reg, "_reg_read_func", lambda addr: 0x1FF)
    _set_io(monkeypatch, reg, "_reg_write_func",
            lambda addr, val: writes.append((addr, val)))

    # read returns the raw value, while the register holds the masked one
    assert reg.read() == 0x1FF
    assert reg.value == 0xFF

    reg.write(0x5A)
    assert reg.value == 0x5A
    reg.write()
    assert writes == [(0x20, 0x5A), (0x20, 0x5A)]


def test_register_validate_detects_missing_and_overlapping_refs():
    good = make_reg("g", 0, [("a", 4, 0, 0, "RW"), ("b", 4, 4, 0, "RW")],
                    width=8)
    overlap = make_reg("o", 0, [("a", 4, 0, 0, "RW"), ("b", 4, 2, 0, "RW")],
                       width=8)
    empty = _make_addressed(Register, 0, name="e", width=8)
    base = len(list(good.validate()))
    assert len(list(overlap.validate())) == base + 1
    assert len(list(empty.validate())) == base + 1


# --------------------------------------------------------------------------
# RegisterMap
# --------------------------------------------------------------------------

def test_get_dimensions():
    assert get_dimensions(0x10, 0x14, 32) == (4, 8, 4)
    assert get_dimensions(0, 2, 12) == (2, 4, 2)


def test_registermap_block_dimensions():
    rmap = make_map([make_full_reg("r0", 0, 32), make_full_reg("r1", 8, 16)])
    assert rmap._block_dimensions == (0, 10)


def test_registermap_empty_raises_value_error():
    rmap = _make_addressed(RegisterMap, 0, name="empty")
    with pytest.raises(ValueError):
        rmap._block_dimensions
    with pytest.raises(ValueError):
        rmap.write()


def test_registermap_read_splits_block_little_endian(monkeypatch):
    rmap = make_map([make_full_reg("r0", 0, 32), make_full_reg("r1", 8, 16)])
    data = bytes(range(10))
    calls = []

    def block_read(addr, size):
        calls.append((addr, size))
        return data

    _set_io(monkeypatch, rmap, "_block_read_func", block_read)
    assert rmap.read() == data
    assert calls == [(0, 10)]
    values = {r.name: r.value for r in rmap}
    assert values == {"r0": 0x03020100, "r1": 0x0908}


def test_registermap_write_builds_block_from_register_values(monkeypatch):
    rmap = make_map([make_full_reg("r0", 0, 32), make_full_reg("r1", 8, 16)])
    regs = {r.name: r for r in rmap}
    regs["r0"].value = 0xDEADBEEF
    regs["r1"].value = 0x1234
    captured = []
    _set_io(monkeypatch, rmap, "_block_write_func",
            lambda addr, data: captured.append((addr, bytes(data))))

    rmap.write()
    expected = bytes([0xEF, 0xBE, 0xAD, 0xDE, 0, 0, 0, 0, 0x34, 0x12])
    assert captured == [(0, expected)]


def test_registermap_write_with_data_updates_registers(monkeypatch):
    rmap = make_map([make_full_reg("r0", 0, 32), make_full_reg("r1", 8, 16)])
    captured = []
    _set_io(monkeypatch, rmap, "_block_write_func",
            lambda addr, data: captured.append((addr, bytes(data))))
    data = bytes(range(10))

    rmap.write(data)
    values = {r.name: r.value for r in rmap}
    assert values == {"r0": 0x03020100, "r1": 0x0908}
    assert captured == [(0, data)]
