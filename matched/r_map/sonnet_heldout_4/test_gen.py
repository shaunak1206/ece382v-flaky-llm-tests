import pytest

from r_map import Register, RegisterMap, BitField, BitFieldRef, Enumeration
from r_map.RegisterMap import get_dimensions


def _patch_func(monkeypatch, obj, name, func):
    """Install an IO callable on the class of obj, whether the attribute is a
    plain attribute or a property in the real implementation."""
    monkeypatch.setattr(type(obj), name, property(lambda self: func),
                        raising=False)


def make_ref(name, width, reg_offset=0, value=None, reset_val=0,
             slice_width=0, field_offset=0):
    bf = BitField(name=name + '_bf', width=width, reset_val=reset_val)
    ref = BitFieldRef(name=name, reg_offset=reg_offset,
                      slice_width=slice_width, field_offset=field_offset)
    ref._add(bf)
    if value is not None:
        bf.value = value
    return ref, bf


def make_reg(name='reg', address=0, width=8, bf_width=None, value=0,
             reset_val=0):
    reg = Register(name=name, address=address, width=width)
    ref, bf = make_ref(name + '_f', bf_width or width, value=value,
                       reset_val=reset_val)
    reg._add(ref)
    return reg, ref, bf


# ---------------------------------------------------------------- BitField

def test_bitfield_mask_and_initial_value():
    bf = BitField(name='bf', width=4, reset_val=0x3)
    assert bf.mask == 0xF
    assert bf.value == 0x3
    assert bf.reset_val == 0x3


def test_bitfield_invalid_width_and_reset_val():
    with pytest.raises(ValueError):
        BitField(name='bf', width=0)
    with pytest.raises(ValueError):
        BitField(name='bf', width=4, reset_val=-1)


def test_bitfield_value_is_masked_and_reset():
    bf = BitField(name='bf', width=4, reset_val=0x5)
    bf.value = 0x1F3
    assert bf.value == 0x3
    bf.reset()
    assert bf.value == 0x5


def test_bitfield_value_rejects_unsupported_type():
    bf = BitField(name='bf', width=4)
    with pytest.raises(NotImplementedError):
        bf.value = 1.5


def test_bitfield_enumeration_assignment_and_annotation():
    bf = BitField(name='bf', width=2)
    bf._add(Enumeration(name='ONE', value=1))
    bf._add(Enumeration(name='TWO', value=2))
    bf.value = 'TWO'
    assert bf.value == 2
    assert bf.annotation == 'TWO'
    bf.value = 3
    assert bf.annotation == hex(3)


def test_bitfield_unknown_enumeration_name_raises():
    bf = BitField(name='bf', width=2)
    with pytest.raises(ValueError):
        bf.value = 'NOPE'


def test_bitfield_io_without_parents_raises():
    bf = BitField(name='bf', width=2)
    with pytest.raises(ValueError):
        bf.read()
    with pytest.raises(ValueError):
        bf.write(1)


# ------------------------------------------------------------- BitFieldRef

def test_bitfieldref_invalid_arguments():
    with pytest.raises(ValueError):
        BitFieldRef(name='r', reg_offset=0, slice_width=-1)
    with pytest.raises(ValueError):
        BitFieldRef(name='r', reg_offset=0, field_offset=-1)


def test_bitfieldref_add_rejects_non_bitfield():
    ref = BitFieldRef(name='r', reg_offset=0)
    with pytest.raises(ValueError):
        ref._add(object())


def test_bitfieldref_default_slice_width_from_bitfield():
    ref, bf = make_ref('r', width=6)
    assert ref.slice_width == 6
    assert ref.mask == 0x3F
    assert ref.bf is bf


def test_bitfieldref_value_with_offsets():
    ref, bf = make_ref('r', width=8, reg_offset=4, field_offset=4,
                       slice_width=4, value=0xAB, reset_val=0xCD)
    assert ref.value == 0xA0
    assert ref.reset_val == 0xC0
    ref.value = 0x50
    assert bf.value == 0x5B


# ---------------------------------------------------------------- Register

def test_register_value_combines_bitfieldrefs():
    reg = Register(name='reg', address=0, width=8)
    low, _ = make_ref('low', 4, reg_offset=0, value=0x3)
    high, _ = make_ref('high', 4, reg_offset=4, value=0xA)
    reg._add(low)
    reg._add(high)
    assert reg.value == 0xA3


def test_register_value_setter_distributes_to_fields():
    reg = Register(name='reg', address=0, width=8)
    low, low_bf = make_ref('low', 4, reg_offset=0)
    high, high_bf = make_ref('high', 4, reg_offset=4)
    reg._add(low)
    reg._add(high)
    reg.value = 0x6C
    assert low_bf.value == 0xC
    assert high_bf.value == 0x6


def test_register_iterates_sorted_by_reg_offset():
    reg = Register(name='reg', address=0, width=8)
    high, _ = make_ref('high', 4, reg_offset=4)
    low, _ = make_ref('low', 4, reg_offset=0)
    reg._add(high)
    reg._add(low)
    assert [r.name for r in reg] == ['low', 'high']


def test_register_reset_and_reset_val():
    reg = Register(name='reg', address=0, width=8)
    low, low_bf = make_ref('low', 4, reg_offset=0, reset_val=0x1, value=0xF)
    high, high_bf = make_ref('high', 4, reg_offset=4, reset_val=0x2, value=0xF)
    reg._add(low)
    reg._add(high)
    assert reg.reset_val == 0x21
    assert reg.value == 0xFF
    reg.reset()
    assert reg.value == 0x21


def test_register_validate_flags_empty_and_overlap():
    empty = Register(name='empty', address=0, width=8)
    errors = list(empty.validate())
    assert len(errors) >= 1

    good = Register(name='good', address=0, width=8)
    a, _ = make_ref('a', 4, reg_offset=0)
    b, _ = make_ref('b', 4, reg_offset=4)
    good._add(a)
    good._add(b)
    good_count = len(list(good.validate()))

    bad = Register(name='bad', address=0, width=8)
    c, _ = make_ref('c', 4, reg_offset=0)
    d, _ = make_ref('d', 4, reg_offset=2)
    bad._add(c)
    bad._add(d)
    assert len(list(bad.validate())) > good_count


def test_register_read_and_write_use_io_functions(monkeypatch):
    reg, ref, bf = make_reg(address=0x40, width=8)
    reads, writes = [], []

    def read_func(addr):
        reads.append(addr)
        return 0x5A

    def write_func(addr, val):
        writes.append((addr, val))

    _patch_func(monkeypatch, reg, '_reg_read_func', read_func)
    _patch_func(monkeypatch, reg, '_reg_write_func', write_func)

    assert reg.read() == 0x5A
    assert reads == [0x40]
    assert reg.value == 0x5A

    reg.write(0x3C)
    assert reg.value == 0x3C
    reg.write()
    assert writes == [(0x40, 0x3C), (0x40, 0x3C)]


# ------------------------------------------------------------- RegisterMap

def test_get_dimensions():
    assert get_dimensions(0x10, 0x12, 16) == (2, 4, 2)
    assert get_dimensions(0, 0, 8) == (0, 1, 1)
    assert get_dimensions(0, 4, 9) == (4, 6, 2)


def _make_map():
    rmap = RegisterMap(name='map', address=0)
    r0, _, _ = make_reg('r0', address=0, width=8)
    r1, _, _ = make_reg('r1', address=1, width=8)
    rmap._add(r0)
    rmap._add(r1)
    return rmap, r0, r1


def test_registermap_block_dimensions_and_empty_error():
    rmap, _, _ = _make_map()
    assert rmap._block_dimensions == (0, 2)

    empty = RegisterMap(name='empty', address=0)
    with pytest.raises(ValueError):
        empty._block_dimensions


def test_registermap_read_updates_registers(monkeypatch):
    rmap, r0, r1 = _make_map()
    calls = []

    def block_read(addr, size):
        calls.append((addr, size))
        return bytes([0x12, 0x34])

    _patch_func(monkeypatch, rmap, '_block_read_func', block_read)
    data = rmap.read()
    assert calls == [(0, 2)]
    assert bytes(data) == b'\x12\x34'
    assert r0.value == 0x12
    assert r1.value == 0x34


def test_registermap_write_without_data_uses_register_values(monkeypatch):
    rmap, r0, r1 = _make_map()
    r0.value = 0xAA
    r1.value = 0x55
    written = []
    _patch_func(monkeypatch, rmap, '_block_write_func',
                lambda addr, data: written.append((addr, bytes(data))))
    rmap.write()
    assert written == [(0, b'\xaa\x55')]


def test_registermap_write_with_data_updates_registers(monkeypatch):
    rmap, r0, r1 = _make_map()
    written = []
    _patch_func(monkeypatch, rmap, '_block_write_func',
                lambda addr, data: written.append((addr, bytes(data))))
    rmap.write(bytearray(b'\x01\x02'))
    assert r0.value == 0x01
    assert r1.value == 0x02
    assert written == [(0, b'\x01\x02')]
