import pytest

from r_map import BitField, BitFieldRef, Register, RegisterMap
from r_map.RegisterMap import get_dimensions


def _make(cls, addr, **kwargs):
    """Build an addressed node, tolerating either address keyword name."""
    for key in ("local_address", "address"):
        try:
            obj = cls(**{key: addr}, **kwargs)
        except TypeError:
            continue
        if obj.address == addr:
            return obj
    raise AssertionError("could not construct %s at address %r" % (cls, addr))


def _make_reg(name, addr, width=32, fields=()):
    """fields: tuples of (name, bf_width, reg_offset, reset_val, access)."""
    reg = _make(Register, addr, name=name, width=width)
    bfs = []
    for fname, bf_width, reg_offset, reset_val, access in fields:
        bf = BitField(name=fname, width=bf_width, reset_val=reset_val,
                      access=access)
        ref = BitFieldRef(name=fname + "_ref", reg_offset=reg_offset)
        ref._add(bf)
        reg._add(ref)
        bfs.append(bf)
    return reg, bfs


def _full_reg(name, addr, width=32):
    return _make_reg(name, addr, width, [(name + "_f", width, 0, 0, "RW")])


def _patch_io(monkeypatch, cls, attr, func):
    monkeypatch.setattr(cls, attr, property(lambda self: func), raising=False)


# ---------------------------------------------------------------- get_dimensions

def test_get_dimensions_word():
    assert get_dimensions(0x10, 0x14, 32) == (4, 8, 4)


def test_get_dimensions_rounds_up_partial_byte():
    assert get_dimensions(0, 0, 12) == (0, 2, 2)


# ---------------------------------------------------------------- BitField

def test_bitfield_initial_value_and_mask():
    bf = BitField(name="f", width=4, reset_val=0x5)
    assert bf.value == 5
    assert bf.mask == 0xF
    bf.value = 0x1F
    assert bf.value == 0xF


def test_bitfield_invalid_arguments():
    with pytest.raises(ValueError):
        BitField(name="f", width=0)
    with pytest.raises(ValueError):
        BitField(name="g", width=4, reset_val=-1)


def test_bitfield_reset_restores_reset_val():
    bf = BitField(name="f", width=8, reset_val=0x12)
    bf.value = 0xFF
    assert bf.value == 0xFF
    bf.reset()
    assert bf.value == 0x12


def test_bitfield_value_setter_rejects_bad_types_and_names():
    bf = BitField(name="f", width=8)
    with pytest.raises(ValueError):
        bf.value = "no_such_enum"
    with pytest.raises(NotImplementedError):
        bf.value = 1.5


def test_bitfield_annotation_without_enumerations_is_hex():
    bf = BitField(name="f", width=8, reset_val=0x2A)
    assert bf.annotation == hex(0x2A)


def test_bitfield_io_without_parent_raises():
    bf = BitField(name="f", width=8)
    with pytest.raises(ValueError):
        bf.read()
    with pytest.raises(ValueError):
        bf.write(1)


# ---------------------------------------------------------------- BitFieldRef

def test_bitfieldref_full_width_by_default():
    bf = BitField(name="f", width=4)
    ref = BitFieldRef(name="f_ref", reg_offset=4)
    ref._add(bf)
    assert ref.slice_width == 4
    assert ref.mask == 0xF
    assert ref.bf is bf
    bf.value = 0xA
    assert ref.value == 0xA0


def test_bitfieldref_value_setter_updates_bitfield():
    bf = BitField(name="f", width=4)
    ref = BitFieldRef(name="f_ref", reg_offset=4)
    ref._add(bf)
    ref.value = 0x30
    assert bf.value == 3
    ref.value = 0xFF5  # bits outside the slice are ignored
    assert bf.value == 0xF


def test_bitfieldref_partial_slice_preserves_other_bits():
    bf = BitField(name="f", width=8, reset_val=0xAB)
    ref = BitFieldRef(name="f_ref", reg_offset=0, slice_width=4,
                      field_offset=4)
    ref._add(bf)
    assert ref.value == 0xA
    assert ref.reset_val == 0xA
    ref.value = 0x5
    assert bf.value == 0x5B


def test_bitfieldref_rejects_bad_arguments_and_children():
    with pytest.raises(ValueError):
        BitFieldRef(name="a", reg_offset=0, slice_width=-1)
    with pytest.raises(ValueError):
        BitFieldRef(name="b", reg_offset=0, field_offset=-1)
    ref = BitFieldRef(name="c", reg_offset=0)
    with pytest.raises(ValueError):
        ref._add(object())


def test_bitfieldref_validate_flags_oversized_slice():
    good_bf = BitField(name="g", width=4)
    good = BitFieldRef(name="g_ref", reg_offset=0, slice_width=4)
    good._add(good_bf)

    bad_bf = BitField(name="b", width=4)
    bad = BitFieldRef(name="b_ref", reg_offset=0, slice_width=8)
    bad._add(bad_bf)

    assert len(list(bad.validate())) == len(list(good.validate())) + 1


# ---------------------------------------------------------------- Register

def test_register_value_combines_fields_and_setter_distributes():
    reg, bfs = _make_reg("r", 0, 16, [("lo", 4, 0, 0, "RW"),
                                      ("hi", 4, 8, 0, "RW")])
    reg.value = 0x0A05
    assert bfs[0].value == 0x5
    assert bfs[1].value == 0xA
    assert reg.value == 0x0A05


def test_register_iterates_by_reg_offset():
    reg, _ = _make_reg("r", 0, 16, [("hi", 4, 8, 0, "RW"),
                                    ("lo", 4, 0, 0, "RW")])
    assert [c.reg_offset for c in reg] == [0, 8]


def test_register_access_reset_val_and_reset():
    reg, bfs = _make_reg("r", 0, 16, [("lo", 4, 0, 0x3, "RW"),
                                      ("hi", 4, 8, 0x9, "RO"),
                                      ("x", 2, 12, 0, "RW")])
    assert reg.access == "RO|RW"
    assert reg.reset_val == 0x0903
    reg.value = 0xFFFF
    reg.reset()
    assert reg.value == 0x0903


def test_register_str_shows_padded_value():
    reg, _ = _make_reg("r", 0, 16, [("lo", 8, 0, 0xA5, "RW")])
    assert str(reg).endswith(" value: 0x00a5")


def test_register_validate_detects_empty_and_overlap():
    empty = _make(Register, 0, name="empty", width=16)
    ok, _ = _make_reg("ok", 0, 16, [("a", 8, 0, 0, "RW"),
                                    ("b", 8, 8, 0, "RW")])
    overlap, _ = _make_reg("ov", 0, 16, [("a", 8, 0, 0, "RW"),
                                         ("b", 8, 4, 0, "RW")])
    ok_count = len(list(ok.validate()))
    assert len(list(empty.validate())) >= 1
    assert len(list(overlap.validate())) == ok_count + 1


def test_register_read_and_write_use_io_functions(monkeypatch):
    reg, bfs = _make_reg("r", 0, 16, [("lo", 4, 0, 0, "RW"),
                                      ("hi", 4, 8, 0, "RW")])
    writes = []
    reads = []

    def rd(addr):
        reads.append(addr)
        return 0xFA05

    def wr(addr, val):
        writes.append((addr, val))

    _patch_io(monkeypatch, Register, "_reg_read_func", rd)
    _patch_io(monkeypatch, Register, "_reg_write_func", wr)

    assert reg.read() == 0xFA05  # raw value, not the masked register value
    assert reads == [reg.address]
    assert reg.value == 0x0A05

    reg.write(0x0102)
    assert writes[-1] == (reg.address, 0x0102)
    assert reg.value == 0x0102
    reg.write()
    assert writes[-1] == (reg.address, 0x0102)


def test_bitfield_read_write_through_parent_register(monkeypatch):
    reg, bfs = _make_reg("r", 0, 16, [("f", 8, 0, 0, "RW")])
    bf = bfs[0]
    state = {"mem": 0x00AB}
    writes = []

    def rd(addr):
        return state["mem"]

    def wr(addr, val):
        writes.append((addr, val))
        state["mem"] = val

    _patch_io(monkeypatch, Register, "_reg_read_func", rd)
    _patch_io(monkeypatch, Register, "_reg_write_func", wr)

    assert bf.read() == 0xAB

    bf.write(0x12)
    assert writes == [(reg.address, 0x12)]

    # Same value as read back and always_write disabled: no write occurs.
    bf.write(0x12, always_write=False)
    assert len(writes) == 1
    bf.write(0x12)
    assert len(writes) == 2


# ---------------------------------------------------------------- RegisterMap

def _make_map():
    rmap = _make(RegisterMap, 0, name="map")
    r0, _ = _full_reg("r0", 0, 32)
    r1, _ = _full_reg("r1", 4, 16)
    rmap._add(r0)
    rmap._add(r1)
    return rmap, r0, r1


def test_registermap_block_dimensions():
    rmap, _, _ = _make_map()
    assert rmap._block_dimensions == (0, 6)


def test_registermap_empty_raises():
    rmap = _make(RegisterMap, 0, name="empty_map")
    with pytest.raises(ValueError):
        rmap._block_dimensions
    with pytest.raises(ValueError):
        rmap.read()


def test_registermap_read_splits_block_little_endian(monkeypatch):
    rmap, r0, r1 = _make_map()
    calls = []
    data = bytes([0x78, 0x56, 0x34, 0x12, 0xCD, 0xAB])

    def block_read(addr, size):
        calls.append((addr, size))
        return data

    _patch_io(monkeypatch, RegisterMap, "_block_read_func", block_read)

    assert rmap.read() == data
    assert calls == [(0, 6)]
    assert r0.value == 0x12345678
    assert r1.value == 0xABCD


def test_registermap_write_builds_block_from_register_values(monkeypatch):
    rmap, r0, r1 = _make_map()
    r0.value = 0xDEADBEEF
    r1.value = 0x1234
    written = []

    def block_write(addr, data):
        written.append((addr, bytes(data)))

    _patch_io(monkeypatch, RegisterMap, "_block_write_func", block_write)

    rmap.write()
    assert written == [(0, bytes([0xEF, 0xBE, 0xAD, 0xDE, 0x34, 0x12]))]


def test_registermap_write_with_data_updates_registers(monkeypatch):
    rmap, r0, r1 = _make_map()
    written = []

    def block_write(addr, data):
        written.append((addr, bytes(data)))

    _patch_io(monkeypatch, RegisterMap, "_block_write_func", block_write)

    payload = bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06])
    rmap.write(payload)
    assert r0.value == 0x04030201
    assert r1.value == 0x0605
    assert written == [(0, payload)]
