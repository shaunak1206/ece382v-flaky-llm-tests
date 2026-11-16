import pytest

import r_map
from r_map import Register, BitField, BitFieldRef, RegisterMap


def _patch_io(monkeypatch, cls, name, func):
    """Install an IO function on a class, overriding any base implementation."""
    monkeypatch.setattr(cls, name, property(lambda self: func), raising=False)


def make_bf(name="bf", width=8, reset_val=0, access="RW"):
    return BitField(name=name, width=width, reset_val=reset_val, access=access)


def make_ref(bf, name=None, reg_offset=0, slice_width=0, field_offset=0):
    ref = BitFieldRef(name=name or (bf.name + "_ref"), reg_offset=reg_offset,
                      slice_width=slice_width, field_offset=field_offset)
    ref._add(bf)
    return ref


def make_reg(name="reg", fields=(), width=32, local_address=0):
    """fields: iterable of (name, bf_width, reg_offset, reset_val, access)"""
    reg = Register(name=name, width=width, local_address=local_address)
    for fname, fwidth, offset, reset, access in fields:
        bf = make_bf(fname, fwidth, reset, access)
        reg._add(make_ref(bf, reg_offset=offset))
    return reg


# ---------------------------------------------------------------- BitField

def test_bitfield_defaults_and_reset_value():
    bf = BitField(name="a", width=4, reset_val=0x5)
    assert bf.value == 0x5
    assert bf.mask == 0xF
    assert bf.width == 4


@pytest.mark.parametrize("kwargs", [{"width": 0}, {"width": -3},
                                    {"reset_val": -1}])
def test_bitfield_invalid_args_raise(kwargs):
    with pytest.raises(ValueError):
        BitField(name="bad", **kwargs)


def test_bitfield_value_setter_masks_int():
    bf = make_bf(width=4)
    bf.value = 0xFF
    assert bf.value == 0xF
    bf.value = 0x12
    assert bf.value == 0x2


def test_bitfield_reset_restores_reset_val():
    bf = make_bf(width=8, reset_val=0x12)
    bf.value = 0xFF
    assert bf.value == 0xFF
    bf.reset()
    assert bf.value == 0x12


def test_bitfield_value_setter_errors():
    bf = make_bf(width=4)
    with pytest.raises(ValueError):
        bf.value = "no_such_enum"
    with pytest.raises(NotImplementedError):
        bf.value = 1.5


def test_bitfield_enumeration_assignment_and_annotation():
    bf = make_bf(width=2)
    enum = r_map.Enumeration(name="ON", value=1)
    bf._add(enum)
    bf.value = "ON"
    assert bf.value == 1
    assert bf.annotation == "ON"
    bf.value = 2
    assert bf.annotation == hex(2)


def test_bitfield_io_without_parent_raises():
    bf = make_bf()
    with pytest.raises(ValueError):
        bf.read()
    with pytest.raises(ValueError):
        bf.write(1)


# -------------------------------------------------------------- BitFieldRef

def test_bitfieldref_value_with_offsets():
    bf = make_bf(width=8, reset_val=0xAB)
    ref = make_ref(bf, reg_offset=4, slice_width=4, field_offset=4)
    assert ref.mask == 0xF
    assert ref.value == 0xA0
    assert ref.reset_val == 0xA0
    assert ref.bf is bf


def test_bitfieldref_default_slice_width_uses_bitfield_width():
    bf = make_bf(width=6)
    ref = make_ref(bf)
    assert ref.slice_width == 6
    assert ref.mask == 0x3F


def test_bitfieldref_value_setter_updates_only_slice():
    bf = make_bf(width=8, reset_val=0xFF)
    ref = make_ref(bf, reg_offset=4, slice_width=4, field_offset=0)
    ref.value = 0x30
    assert bf.value == 0xF3


def test_bitfieldref_invalid_args_raise():
    with pytest.raises(ValueError):
        BitFieldRef(name="r", reg_offset=0, slice_width=-1)
    with pytest.raises(ValueError):
        BitFieldRef(name="r", reg_offset=0, field_offset=-1)


def test_bitfieldref_add_rejects_non_bitfield():
    ref = BitFieldRef(name="r", reg_offset=0, slice_width=2)
    with pytest.raises(ValueError):
        ref._add(object())


def test_bitfieldref_validate_flags_oversized_slice():
    good = make_ref(make_bf("g", width=8), slice_width=4)
    bad = make_ref(make_bf("b", width=4), slice_width=8)
    assert len(list(bad.validate())) > len(list(good.validate()))


# ----------------------------------------------------------------- Register

def test_register_value_combines_fields():
    reg = make_reg(fields=[("lo", 4, 0, 0x3, "RW"), ("hi", 4, 8, 0xA, "RW")])
    assert reg.value == 0xA03
    assert reg.reset_val == 0xA03


def test_register_value_setter_distributes_to_fields():
    reg = make_reg(fields=[("lo", 4, 0, 0, "RW"), ("hi", 4, 8, 0, "RW")])
    reg.value = 0x5C7
    refs = {r.name: r for r in reg}
    assert refs["lo_ref"].bf.value == 0x7
    assert refs["hi_ref"].bf.value == 0x5
    assert reg.value == 0x507


def test_register_reset_restores_reset_values():
    reg = make_reg(fields=[("lo", 4, 0, 0x3, "RW"), ("hi", 4, 8, 0xA, "RW")])
    reg.value = 0xFFF
    reg.reset()
    assert reg.value == 0xA03


def test_register_access_is_sorted_union():
    reg = make_reg(fields=[("a", 4, 0, 0, "RW"), ("b", 4, 8, 0, "RO"),
                           ("c", 4, 16, 0, "RW")])
    assert reg.access == "RO|RW"


def test_register_iterates_in_reg_offset_order():
    reg = Register(name="r", width=32, local_address=0)
    reg._add(make_ref(make_bf("high", 4), reg_offset=16))
    reg._add(make_ref(make_bf("low", 4), reg_offset=0))
    reg._add(make_ref(make_bf("mid", 4), reg_offset=8))
    assert [c.name for c in reg] == ["low_ref", "mid_ref", "high_ref"]


def test_register_validate_detects_empty_and_overlap():
    ok = make_reg(fields=[("a", 4, 0, 0, "RW"), ("b", 4, 4, 0, "RW")])
    empty = Register(name="empty", width=32, local_address=0)
    overlap = make_reg(fields=[("a", 4, 0, 0, "RW"), ("b", 4, 2, 0, "RW")])
    n_ok = len(list(ok.validate()))
    assert len(list(empty.validate())) > n_ok
    assert len(list(overlap.validate())) > n_ok


def test_register_read_sets_value_and_returns_raw(monkeypatch):
    reg = make_reg(fields=[("lo", 4, 0, 0, "RW")])
    calls = []

    def fake_read(addr):
        calls.append(addr)
        return 0xA5

    _patch_io(monkeypatch, Register, "_reg_read_func", fake_read)
    assert reg.read() == 0xA5  # raw value, not masked by fields
    assert calls == [reg.address]
    assert reg.value == 0x5


def test_register_write_with_and_without_value(monkeypatch):
    reg = make_reg(fields=[("lo", 8, 0, 0x11, "RW")])
    writes = []
    _patch_io(monkeypatch, Register, "_reg_write_func",
              lambda addr, val: writes.append((addr, val)))
    reg.write()
    reg.write(0x22)
    assert writes == [(reg.address, 0x11), (reg.address, 0x22)]
    assert reg.value == 0x22


def test_bitfield_write_does_read_modify_write(monkeypatch):
    reg = make_reg(fields=[("a", 4, 0, 0, "RW")])
    bf = list(reg)[0].bf
    writes = []
    _patch_io(monkeypatch, Register, "_reg_read_func", lambda addr: 0xF)
    _patch_io(monkeypatch, Register, "_reg_write_func",
              lambda addr, val: writes.append((addr, val)))
    bf.write(0x3)
    assert bf.value == 0x3
    assert writes == [(reg.address, 0x3)]


def test_bitfield_read_reads_parent_registers(monkeypatch):
    reg = make_reg(fields=[("a", 4, 0, 0, "RW")])
    bf = list(reg)[0].bf
    _patch_io(monkeypatch, Register, "_reg_read_func", lambda addr: 0x5)
    assert bf.read() == 0x5


# -------------------------------------------------------------- RegisterMap

def make_map():
    rmap = RegisterMap(name="rm", local_address=0)
    rmap._add(make_reg("r0", [("f0", 32, 0, 0, "RW")], width=32,
                       local_address=0))
    rmap._add(make_reg("r1", [("f1", 32, 0, 0, "RW")], width=32,
                       local_address=4))
    return rmap


def test_registermap_read_splits_block_into_registers(monkeypatch):
    rmap = make_map()
    calls = []
    block = bytes([1, 0, 0, 0, 2, 1, 0, 0])

    def fake_block_read(addr, size):
        calls.append((addr, size))
        return block

    _patch_io(monkeypatch, RegisterMap, "_block_read_func", fake_block_read)
    start = list(rmap)[0].address
    assert rmap.read() == block
    assert calls == [(start, 8)]
    assert [r.value for r in rmap] == [0x1, 0x102]


def test_registermap_write_builds_little_endian_block(monkeypatch):
    rmap = make_map()
    regs = list(rmap)
    regs[0].value = 0x04030201
    regs[1].value = 0xAA
    writes = []
    _patch_io(monkeypatch, RegisterMap, "_block_write_func",
              lambda addr, data: writes.append((addr, bytes(data))))
    rmap.write()
    assert writes == [(regs[0].address,
                       bytes([1, 2, 3, 4, 0xAA, 0, 0, 0]))]


def test_registermap_write_with_data_updates_registers(monkeypatch):
    rmap = make_map()
    writes = []
    _patch_io(monkeypatch, RegisterMap, "_block_write_func",
              lambda addr, data: writes.append((addr, bytes(data))))
    data = bytes([1, 0, 0, 0, 2, 0, 0, 0])
    rmap.write(data)
    assert [r.value for r in rmap] == [1, 2]
    assert writes[0][1] == data


def test_registermap_without_registers_raises(monkeypatch):
    empty = RegisterMap(name="empty", local_address=0)
    _patch_io(monkeypatch, RegisterMap, "_block_read_func",
              lambda a, s: b"")
    _patch_io(monkeypatch, RegisterMap, "_block_write_func",
              lambda a, d: None)
    with pytest.raises(ValueError):
        empty.read()
    with pytest.raises(ValueError):
        empty.write()
