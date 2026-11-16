import importlib

import pytest

from r_map import BitField, BitFieldRef, Register, RegisterMap, Enumeration

rm_module = importlib.import_module("r_map.RegisterMap")


class _IOMixin:
    """Provides deterministic address / IO hooks regardless of how the
    AddressedNode base class resolves them."""
    _test_addr = 0
    _test_reg_read = None
    _test_reg_write = None
    _test_block_read = None
    _test_block_write = None

    @property
    def address(self):
        return self._test_addr

    @address.setter
    def address(self, v):
        self._test_addr = v

    @property
    def _reg_read_func(self):
        return self._test_reg_read

    @_reg_read_func.setter
    def _reg_read_func(self, f):
        if callable(f):
            self._test_reg_read = f

    @property
    def _reg_write_func(self):
        return self._test_reg_write

    @_reg_write_func.setter
    def _reg_write_func(self, f):
        if callable(f):
            self._test_reg_write = f

    @property
    def _block_read_func(self):
        return self._test_block_read

    @_block_read_func.setter
    def _block_read_func(self, f):
        if callable(f):
            self._test_block_read = f

    @property
    def _block_write_func(self):
        return self._test_block_write

    @_block_write_func.setter
    def _block_write_func(self, f):
        if callable(f):
            self._test_block_write = f


class FakeReg(_IOMixin, Register):
    pass


class FakeMap(_IOMixin, RegisterMap):
    pass


def _build(cls, **kw):
    last = None
    for extra in ({}, {'local_address': 0}, {'address': 0}):
        try:
            return cls(**kw, **extra)
        except Exception as e:  # noqa
            last = e
    raise last


def make_bf(name, width=4, reset_val=0, access='RW'):
    return BitField(name=name, width=width, reset_val=reset_val, access=access)


def make_reg(name, fields, width=32, address=0):
    """fields: list of (bf, reg_offset)"""
    reg = _build(FakeReg, name=name, width=width)
    reg.address = address
    for bf, off in fields:
        ref = BitFieldRef(name=f"{name}_{bf.name}_ref", reg_offset=off)
        ref._add(bf)
        reg._add(ref)
    return reg


def make_full_reg(name, address, width=32):
    return make_reg(name, [(make_bf(name + "_f", width=width), 0)],
                    width=width, address=address)


# ---------------------------------------------------------------- BitField

def test_bitfield_defaults_and_mask():
    bf = make_bf("a", width=4, reset_val=0x5)
    assert bf.mask == 0xF
    assert bf.value == 0x5
    assert bf.reset_val == 0x5


def test_bitfield_invalid_args():
    with pytest.raises(ValueError):
        BitField(name="x", width=0)
    with pytest.raises(ValueError):
        BitField(name="x", width=4, reset_val=-1)


def test_bitfield_value_is_masked_and_reset():
    bf = make_bf("a", width=4, reset_val=0x3)
    bf.value = 0x1F
    assert bf.value == 0xF
    bf.reset()
    assert bf.value == 0x3


def test_bitfield_set_by_enumeration_name_and_object():
    bf = make_bf("a", width=4)
    e1 = Enumeration(name="ONE", value=1)
    e2 = Enumeration(name="TWO", value=2)
    bf._add(e1)
    bf._add(e2)
    bf.value = "TWO"
    assert bf.value == 2
    bf.value = e1
    assert bf.value == 1


def test_bitfield_unknown_enum_name_raises():
    bf = make_bf("a", width=4)
    with pytest.raises(ValueError):
        bf.value = "NOPE"


def test_bitfield_unsupported_type_raises():
    bf = make_bf("a", width=4)
    with pytest.raises(NotImplementedError):
        bf.value = 1.5


def test_bitfield_annotation():
    bf = make_bf("a", width=4)
    bf._add(Enumeration(name="THREE", value=3))
    bf.value = 3
    assert bf.annotation == "THREE"
    bf.value = 7
    assert bf.annotation == hex(7)


def test_bitfield_validate_duplicate_enum_values():
    good = make_bf("g", width=4)
    good._add(Enumeration(name="A", value=1))
    good._add(Enumeration(name="B", value=2))
    bad = make_bf("b", width=4)
    bad._add(Enumeration(name="A", value=1))
    bad._add(Enumeration(name="B", value=1))
    assert len(list(bad.validate())) > len(list(good.validate()))


def test_bitfield_io_without_parents_raises():
    bf = make_bf("a")
    with pytest.raises(ValueError):
        bf.read()
    with pytest.raises(ValueError):
        bf.write(1)


# ------------------------------------------------------------- BitFieldRef

def test_bitfieldref_invalid_args():
    with pytest.raises(ValueError):
        BitFieldRef(name="r", reg_offset=0, slice_width=-1)
    with pytest.raises(ValueError):
        BitFieldRef(name="r", reg_offset=0, field_offset=-1)


def test_bitfieldref_add_wrong_type_raises():
    ref = BitFieldRef(name="r", reg_offset=0)
    with pytest.raises(ValueError):
        ref._add(object())


def test_bitfieldref_add_sets_width_and_references():
    bf = make_bf("a", width=6)
    ref = BitFieldRef(name="r", reg_offset=0)
    ref._add(bf)
    assert ref.bf is bf
    assert ref.slice_width == 6
    assert ref.mask == 0x3F
    assert ref in bf._references


def test_bitfieldref_value_and_reset_val_with_offsets():
    bf = make_bf("a", width=8, reset_val=0xA5)
    ref = BitFieldRef(name="r", reg_offset=4, slice_width=4)
    ref._add(bf)
    assert ref.value == 0x50
    assert ref.reset_val == 0x50
    ref2 = BitFieldRef(name="r2", reg_offset=0, slice_width=4, field_offset=4)
    bf2 = make_bf("b", width=8, reset_val=0xA5)
    ref2._add(bf2)
    assert ref2.value == 0xA


def test_bitfieldref_value_setter_preserves_other_bits():
    bf = make_bf("a", width=8, reset_val=0xA5)
    ref = BitFieldRef(name="r", reg_offset=4, slice_width=4)
    ref._add(bf)
    ref.value = 0xF0
    assert bf.value == 0xAF
    ref.value = 0x00
    assert bf.value == 0xA0


def test_bitfieldref_validate_slice_too_wide():
    ok_bf = make_bf("ok", width=8)
    ok = BitFieldRef(name="ok_r", reg_offset=0, slice_width=8)
    ok._add(ok_bf)
    bad_bf = make_bf("bad", width=8)
    bad = BitFieldRef(name="bad_r", reg_offset=0, slice_width=16)
    bad._add(bad_bf)
    assert len(list(bad.validate())) > len(list(ok.validate()))


# ---------------------------------------------------------------- Register

def test_register_value_combines_fields_and_set_splits():
    a, b = make_bf("a"), make_bf("b")
    reg = make_reg("r", [(a, 0), (b, 8)])
    reg.value = 0x0A0B
    assert a.value == 0xB
    assert b.value == 0xA
    assert reg.value == 0x0A0B


def test_register_iterates_sorted_by_reg_offset():
    a, b = make_bf("a"), make_bf("b")
    reg = make_reg("r", [(b, 8), (a, 0)])
    assert [ref.reg_offset for ref in reg] == [0, 8]


def test_register_reset_and_reset_val():
    a, b = make_bf("a", reset_val=3), make_bf("b", reset_val=5)
    reg = make_reg("r", [(a, 0), (b, 8)])
    assert reg.reset_val == 0x503
    reg.value = 0x0F0F
    reg.reset()
    assert reg.value == 0x503


def test_register_access_joined_sorted_unique():
    a = make_bf("a", access="RW")
    b = make_bf("b", access="RO")
    c = make_bf("c", access="RW")
    reg = make_reg("r", [(a, 0), (b, 4), (c, 8)])
    assert reg.access == "RO|RW"


def test_register_str_contains_hex_value():
    reg = make_reg("r", [(make_bf("a", width=8), 0)], width=32)
    reg.value = 0xAB
    s = str(reg)
    assert "value:" in s
    assert "0x000000ab" in s


def test_register_read_and_write_use_io_functions():
    a, b = make_bf("a"), make_bf("b")
    reg = make_reg("r", [(a, 0), (b, 8)], address=0x40)
    writes = []
    reg._test_reg_read = lambda addr: 0x0A0B if addr == 0x40 else 0
    reg._test_reg_write = lambda addr, val: writes.append((addr, val))

    assert reg.read() == 0x0A0B
    assert reg.value == 0x0A0B

    reg.write()
    reg.write(0x0304)
    assert writes == [(0x40, 0x0A0B), (0x40, 0x0304)]
    assert reg.value == 0x0304


def test_register_validate_empty_and_overlap():
    empty = _build(FakeReg, name="empty", width=32)
    assert len(list(empty.validate())) >= 1

    ok = make_reg("ok", [(make_bf("a", width=4), 0), (make_bf("b", width=4), 8)])
    overlap = make_reg("ov", [(make_bf("c", width=8), 0), (make_bf("d", width=8), 4)])
    assert len(list(overlap.validate())) > len(list(ok.validate()))


# ------------------------------------------------------------- RegisterMap

def _make_map():
    m = _build(FakeMap, name="map")
    m.address = 0
    regs = [make_full_reg("r0", 0, 32),
            make_full_reg("r1", 4, 32),
            make_full_reg("r2", 8, 16)]
    for r in regs:
        m._add(r)
    return m, regs


def test_get_dimensions():
    assert rm_module.get_dimensions(0x10, 0x14, 16) == (4, 6, 2)
    assert rm_module.get_dimensions(0, 0, 32) == (0, 4, 4)
    assert rm_module.get_dimensions(0, 3, 9) == (3, 5, 2)


def test_registermap_block_dimensions():
    m, _ = _make_map()
    assert m._block_dimensions == (0, 10)


def test_registermap_empty_raises():
    m = _build(FakeMap, name="empty")
    with pytest.raises(ValueError):
        m._block_dimensions
    with pytest.raises(ValueError):
        m.write()


def test_registermap_read_updates_registers():
    m, regs = _make_map()
    calls = []
    data = bytes([0x01, 0x02, 0x03, 0x04, 0x10, 0x00, 0x00, 0x00, 0xCD, 0xAB])

    def block_read(addr, size):
        calls.append((addr, size))
        return data

    m._test_block_read = block_read
    assert m.read() == data
    assert calls == [(0, 10)]
    assert regs[0].value == 0x04030201
    assert regs[1].value == 0x10
    assert regs[2].value == 0xABCD


def test_registermap_write_builds_block_from_register_values():
    m, regs = _make_map()
    regs[0].value = 0x04030201
    regs[1].value = 0x10
    regs[2].value = 0xABCD
    written = []
    m._test_block_write = lambda addr, d: written.append((addr, bytes(d)))
    m.write()
    assert written == [(0, bytes([1, 2, 3, 4, 0x10, 0, 0, 0, 0xCD, 0xAB]))]


def test_registermap_write_with_data_and_partial_data():
    m, regs = _make_map()
    written = []
    m._test_block_write = lambda addr, d: written.append((addr, bytes(d)))

    full = bytes([1, 0, 0, 0, 2, 0, 0, 0, 3, 0])
    m.write(full)
    assert [r.value for r in regs] == [1, 2, 3]
    assert written[-1] == (0, full)

    partial = bytes([9, 0, 0, 0])
    m.write(partial)
    assert regs[0].value == 9
    assert regs[1].value == 2
    assert regs[2].value == 3
    assert written[-1] == (0, partial)
