import pytest

from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.Register import Register
from r_map.RegisterMap import RegisterMap, get_dimensions
from r_map.ValidationError import ValidationError


def make_bitfield(name="bf", width=8, reset_val=0, access="RW"):
    bf = BitField(width=width, reset_val=reset_val, access=access)
    bf.name = name
    return bf


def make_bitfieldref(name="ref", bf=None, reg_offset=0, slice_width=0, field_offset=0):
    ref = BitFieldRef(reg_offset=reg_offset, slice_width=slice_width, field_offset=field_offset)
    ref.name = name
    if bf is not None:
        ref._bf = bf
        ref._parent = None
        bf._references = {ref}
        ref._children = {bf.name: bf}
    return ref


def make_register(name="reg", address=0, width=32, children=()):
    reg = Register(width=width)
    reg.name = name
    reg.address = address
    reg._children = {c.name: c for c in children}
    for c in children:
        c._parent = reg
    return reg


def make_regmap(name="rm", regs=()):
    rm = RegisterMap()
    rm.name = name
    rm._children = {r.name: r for r in regs}
    for r in regs:
        r._parent = rm
    return rm


def test_bitfield_initialization_masks_reset_value():
    bf = BitField(width=3, reset_val=0b10110, access="RW")
    assert bf.mask == 0b111
    assert bf.reset_val == 0b110
    assert bf.value == 0b110


def test_bitfield_rejects_invalid_width_and_reset_val():
    with pytest.raises(ValueError):
        BitField(width=0)
    with pytest.raises(ValueError):
        BitField(width=1, reset_val=-1)


def test_bitfield_value_masks_integers():
    bf = BitField(width=4, reset_val=0)
    bf.value = 0x3F
    assert bf.value == 0xF


def test_bitfield_validate_detects_duplicate_enumeration_values():
    bf = BitField(width=4, reset_val=0)
    # Create two minimal enumeration-like children with the same value.
    e1 = type("EnumLike", (), {"name": "A", "value": 1, "__str__": lambda self: self.name})()
    e2 = type("EnumLike", (), {"name": "B", "value": 1, "__str__": lambda self: self.name})()
    bf._children = {"A": e1, "B": e2}
    errs = list(bf.validate())
    assert any(isinstance(e, ValidationError) for e in errs)
    assert any("same value" in str(e) for e in errs)


def test_bitfieldref_value_maps_to_underlying_bitfield():
    bf = BitField(width=8, reset_val=0)
    ref = BitFieldRef(reg_offset=2, slice_width=3, field_offset=1)
    ref.name = "ref"
    ref._bf = bf
    ref._parent = None
    bf._references = {ref}

    ref.value = 0b111 << 2
    assert bf.value == 0b1110
    assert ref.value == 0b111 << 2


def test_bitfieldref_reset_val_and_validate_without_child():
    ref = BitFieldRef(reg_offset=0, slice_width=2, field_offset=0)
    ref.name = "ref"
    ref._bf = None
    errs = list(ref.validate())
    assert any("No child bitfield present" in str(e) for e in errs)


def test_bitfieldref_validate_rejects_slice_width_larger_than_bitfield():
    bf = BitField(width=4, reset_val=0)
    ref = BitFieldRef(reg_offset=0, slice_width=5, field_offset=0)
    ref.name = "ref"
    ref._bf = bf
    bf._references = {ref}
    errs = list(ref.validate())
    assert any("larger than bitfield" in str(e) for e in errs)


def test_register_value_and_reset_propagate_to_child_bitfieldrefs():
    bf1 = BitField(width=4, reset_val=1)
    bf1.name = "bf1"
    ref1 = BitFieldRef(reg_offset=0, slice_width=4, field_offset=0)
    ref1.name = "r1"
    ref1._bf = bf1
    bf1._references = {ref1}
    reg = make_register(children=(ref1,))

    assert reg.value == 1
    reg.value = 0xA
    assert bf1.value == 0xA
    reg.reset()
    assert bf1.value == 1


def test_register_iterates_children_sorted_by_reg_offset():
    bf = BitField(width=8, reset_val=0)
    bf.name = "bf"
    r1 = make_bitfieldref("r1", bf=bf, reg_offset=8, slice_width=4)
    r2 = make_bitfieldref("r2", bf=bf, reg_offset=0, slice_width=4)
    reg = make_register(children=(r1, r2))
    names = [c.name for c in reg]
    assert names == ["r2", "r1"]


def test_register_validate_reports_overlapping_bitfieldrefs():
    bf = BitField(width=16, reset_val=0)
    bf.name = "bf"
    r1 = make_bitfieldref("r1", bf=bf, reg_offset=0, slice_width=8)
    r2 = make_bitfieldref("r2", bf=bf, reg_offset=4, slice_width=8)
    reg = make_register(children=(r1, r2))
    errs = list(reg.validate())
    assert any("overlaps" in str(e) for e in errs)


def test_register_read_and_write_use_io_functions():
    bf = BitField(width=8, reset_val=0)
    bf.name = "bf"
    ref = make_bitfieldref("ref", bf=bf, reg_offset=0, slice_width=8)
    reg = make_register(address=0x100, children=(ref,))

    reads = []
    writes = []

    def _read(addr):
        reads.append(addr)
        return 0xAB

    def _write(addr, val):
        writes.append((addr, val))

    reg._reg_read_func = _read
    reg._reg_write_func = _write

    val = reg.read()
    assert val == 0xAB
    assert reg.value == 0xAB
    reg.write()
    assert writes == [(0x100, 0xAB)]
    assert reads == [0x100]


def test_bitfield_write_triggers_parent_register_io():
    bf = BitField(width=8, reset_val=0x12)
    bf.name = "bf"
    ref = make_bitfieldref("ref", bf=bf, reg_offset=0, slice_width=8)
    reg = make_register(address=0x20, children=(ref,))
    reg_reads = []
    reg_writes = []

    def _read():
        reg_reads.append(True)
        reg.value = 0x34

    def _write():
        reg_writes.append(reg.value)

    reg.read = _read
    reg.write = _write

    bf.write(0x56)
    assert reg_reads == [True]
    assert reg_writes == [0x56]


def test_get_dimensions_basic():
    start, end, num_bytes = get_dimensions(0x10, 0x14, 16)
    assert (start, end, num_bytes) == (4, 6, 2)


def test_registermap_block_dimensions_and_read_write():
    r1_bf = BitField(width=8, reset_val=0)
    r1_bf.name = "bf1"
    r1 = make_register("r1", address=0x00, children=(make_bitfieldref("r1f", bf=r1_bf, reg_offset=0, slice_width=8),))

    r2_bf = BitField(width=16, reset_val=0)
    r2_bf.name = "bf2"
    r2 = make_register("r2", address=0x02, width=16,
                       children=(make_bitfieldref("r2f", bf=r2_bf, reg_offset=0, slice_width=16),))

    rm = make_regmap(regs=(r1, r2))

    # block dimensions are derived from sorted register order
    assert rm._block_dimensions == (0x00, 4)

    captured = []

    def block_read(addr, size):
        assert addr == 0x00
        assert size == 4
        return bytes([0x11, 0x00, 0x22, 0x33])

    def block_write(addr, data):
        captured.append((addr, bytes(data)))

    rm._block_read_func = block_read
    rm._block_write_func = block_write

    data = rm.read()
    assert data == bytes([0x11, 0x00, 0x22, 0x33])
    assert r1.value == 0x11
    assert r2.value == 0x3322

    r1.value = 0xAA
    r2.value = 0xBEEF
    rm.write()
    assert captured == [(0x00, bytes([0xAA, 0x00, 0xEF, 0xBE]))]


def test_registermap_write_rejects_empty_map():
    rm = make_regmap()
    rm._block_dimensions = (0, 0)
    rm._block_write_func = lambda addr, data: None
    with pytest.raises(ValueError):
        rm.write()


def test_registermap_partial_write_updates_only_available_registers():
    r1_bf = BitField(width=8, reset_val=0)
    r1_bf.name = "bf1"
    r1 = make_register("r1", address=0x00, children=(make_bitfieldref("r1f", bf=r1_bf, reg_offset=0, slice_width=8),))

    r2_bf = BitField(width=16, reset_val=0)
    r2_bf.name = "bf2"
    r2 = make_register("r2", address=0x02, width=16,
                       children=(make_bitfieldref("r2f", bf=r2_bf, reg_offset=0, slice_width=16),))

    rm = make_regmap(regs=(r1, r2))
    rm._block_dimensions = (0x00, 4)
    rm._block_write_func = lambda addr, data: None

    rm.write(data=bytes([0x01, 0x02]))
    assert r1.value == 0x01
    assert r2.value == 0
