import pytest
from math import ceil
import r_map
from r_map.Register import Register
from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.RegisterMap import RegisterMap
from r_map.ValidationError import ValidationError

# Helpers for tests
class DummyRegIO:
    def __init__(self):
        self.read_calls = []
        self.write_calls = []

    def reg_read_func(self, addr):
        self.read_calls.append(addr)
        return 0xA5A5A5A5

    def reg_write_func(self, addr, val):
        self.write_calls.append((addr, val))

    def block_read_func(self, addr, size):
        self.read_calls.append((addr, size))
        # return little endian bytes with incremental pattern
        return bytes(range(size))

    def block_write_func(self, addr, data):
        self.write_calls.append((addr, data))


def create_bitfield(name='bf', width=4, reset_val=7, access='RW'):
    bf = BitField(name=name, width=width, reset_val=reset_val, access=access)
    return bf


def create_bitfieldref(reg_offset=0, slice_width=0, field_offset=0, bf=None):
    bfr = BitFieldRef(reg_offset=reg_offset, slice_width=slice_width, field_offset=field_offset, name='bfr')
    if bf:
        bfr._add(bf)
    return bfr


def create_register(name='reg', addr=0, width=32, bitfields=None):
    reg = Register(name=name, address=addr, width=width)
    reg._reg_read_func = lambda x: 0x12345678
    reg._reg_write_func = lambda x, y: None
    if bitfields:
        for bf in bitfields:
            reg._children[bf.name] = bf
            bf._parent = reg
    return reg


def create_register_with_bitfieldrefs(name='reg', addr=0, width=32, bfrs=None):
    reg = Register(name=name, address=addr, width=width)
    reg._reg_read_func = lambda x: 0x12345678
    reg._reg_write_func = lambda x, y: None
    if bfrs:
        for bfr in bfrs:
            reg._children[bfr.name] = bfr
            bfr._parent = reg
    return reg


######################
# Tests for Register  #
######################

def test_register_init_and_str_value_property():
    reg = Register(name='r1', address=0x1000, width=16)
    # Add BitFieldRefs with values to test value property
    bf = create_bitfield('bf1', width=8, reset_val=0xAA)
    bf.value = 0x55
    bfr1 = create_bitfieldref(reg_offset=0, slice_width=8, bf=bf)
    reg._children[bfr1.name] = bfr1
    bfr1._bf = bf
    # The value should equal bf.value shifted by reg_offset => 0x55 << 0 == 0x55
    assert reg.value == (bf.value & 0xFF)

    # Test __str__ returns expected substring (value hex)
    s = str(reg)
    assert f"value: {reg.value:#0{ceil(reg.width/4)+2}x}" in s

def test_register_value_setter_propagates_to_bitfieldrefs():
    bf1 = create_bitfield('bf1', width=8)
    bf2 = create_bitfield('bf2', width=8)
    bf1.value = 0x12
    bf2.value = 0x34
    bfr1 = create_bitfieldref(reg_offset=0, slice_width=8, bf=bf1)
    bfr2 = create_bitfieldref(reg_offset=8, slice_width=8, bf=bf2)
    reg = create_register_with_bitfieldrefs(bfrs=[bfr1, bfr2])

    reg.value = 0xABCD
    # The bf.value should reflect bits within their slice:
    # bfr1 reg_offset=0, slice_width=8 -> bits 0-7 of 0xABCD (0xCD & 0xFF) == 0xCD
    # bfr2 reg_offset=8, slice_width=8 -> bits 8-15 (0xAB & 0xFF) == 0xAB
    assert bf1.value == (reg.value & bfr1.mask)
    assert bf2.value == ((reg.value >> bfr2.reg_offset) & bfr2.mask)

def test_register_access_property_returns_unique_access_sorted():
    bf1 = create_bitfield('bf1', access='RW')
    bf2 = create_bitfield('bf2', access='RO')
    bfr1 = create_bitfieldref(reg_offset=0, slice_width=4, bf=bf1)
    bfr2 = create_bitfieldref(reg_offset=4, slice_width=4, bf=bf2)
    reg = create_register_with_bitfieldrefs(bfrs=[bfr1, bfr2])
    expected_access = '|'.join(sorted({'RW', 'RO'}))
    assert reg.access == expected_access

def test_register_reset_calls_bitfields_reset(monkeypatch):
    bf = create_bitfield('bf1', width=8)
    called = {}

    def bf_reset():
        called['reset'] = True
    bfr = create_bitfieldref(reg_offset=0, slice_width=8, bf=bf)
    bfr._bf.reset = bf_reset
    reg = create_register_with_bitfieldrefs(bfrs=[bfr])
    reg.reset()
    assert 'reset' in called

def test_register_validate_no_bitfieldrefs_yields_error():
    reg = Register(name='empty_reg')
    errors = list(reg.validate())
    assert any(isinstance(e, ValidationError) for e in errors)
    assert any("No bitfieldrefs present" in str(e) for e in errors)

def test_register_validate_non_bitfieldref_child_yields_error():
    class DummyChild:
        pass

    reg = Register(name='reg')
    reg._children['dummy'] = DummyChild()
    errors = list(reg.validate())
    assert any(isinstance(e, ValidationError) for e in errors)
    assert any("not of type BitFieldRef" in str(e) for e in errors)

def test_register_validate_overlapping_bitfieldrefs_yields_error():
    bf = create_bitfield('bf', width=8)
    bfr1 = create_bitfieldref(reg_offset=0, slice_width=4, bf=bf)
    bfr2 = create_bitfieldref(reg_offset=3, slice_width=4, bf=bf)
    reg = create_register_with_bitfieldrefs(bfrs=[bfr1, bfr2])
    errors = list(reg.validate())
    assert any(isinstance(e, ValidationError) for e in errors)
    assert any("overlaps" in str(e) for e in errors)

def test_register_read_and_write_calls_io_functions():
    dummy_io = DummyRegIO()
    reg = Register(name='reg', address=0x10, width=32)
    reg._reg_read_func = dummy_io.reg_read_func
    reg._reg_write_func = dummy_io.reg_write_func

    val = reg.read()
    assert val == 0xA5A5A5A5
    # After read, reg.value should be set to val
    # The read val returned is direct from IO func not necessarily reg.value property, but here reg.value = val

    reg.write(0x1234)
    assert dummy_io.write_calls[-1][1] == 0x1234

def test_register_write_without_val_writes_current_value():
    dummy_io = DummyRegIO()
    reg = Register(name='reg', address=0x10, width=32)
    reg._reg_read_func = dummy_io.reg_read_func
    reg._reg_write_func = dummy_io.reg_write_func

    # Setup a bitfieldref to modify value property
    bf = create_bitfield('bf', width=8)
    bfr = create_bitfieldref(reg_offset=0, slice_width=8, bf=bf)
    reg._children[bfr.name] = bfr

    reg.value = 0x42
    reg.write()  # no val passed, should write reg.value
    assert dummy_io.write_calls[-1][1] == reg.value

def test_register_iter_yields_sorted_by_reg_offset():
    bf = create_bitfield('bf', width=8)
    bfr1 = create_bitfieldref(reg_offset=10, slice_width=4, bf=bf)
    bfr2 = create_bitfieldref(reg_offset=5, slice_width=4, bf=bf)
    reg = create_register_with_bitfieldrefs(bfrs=[bfr1, bfr2])
    # iter should yield in sorted order by reg_offset -> bfr2, bfr1
    iterated = list(reg)
    assert iterated[0].reg_offset == 5
    assert iterated[1].reg_offset == 10

#######################
# Tests for BitField   #
#######################

def test_bitfield_init_invalid_width_and_reset_val():
    with pytest.raises(ValueError):
        BitField(name='bf', width=0)
    with pytest.raises(ValueError):
        BitField(name='bf', reset_val=-1)

def test_bitfield_value_setter_int_and_enumeration():
    bf = create_bitfield('bf', width=4, reset_val=0)
    bf.value = 0xA
    assert bf.value == 0xA & bf.mask

    class DummyEnum:
        name = 'FOO'
        value = 5
    dummy_enum = DummyEnum()

    # Add dummy enumeration to bitfield children to test string setter
    dummy_enum.name = 'BAR'
    bf._children[dummy_enum.name] = dummy_enum
    bf.value = 'BAR'
    assert bf.value == 5

    with pytest.raises(ValueError):
        bf.value = 'NON_EXISTING'

    with pytest.raises(NotImplementedError):
        bf.value = 3.14

def test_bitfield_reset_sets_value_to_reset_val():
    bf = create_bitfield('bf', width=4, reset_val=0xF)
    bf.value = 0
    bf.reset()
    assert bf.value == bf.reset_val == 0xF

def test_bitfield_annotation_returns_name_or_hex():
    bf = create_bitfield('bf', width=4)

    class DummyEnum:
        def __init__(self, name, value):
            self.name = name
            self.value = value

    enum1 = DummyEnum('VAL_A', 3)
    enum2 = DummyEnum('VAL_B', 7)
    bf._children[enum1.name] = enum1
    bf._children[enum2.name] = enum2
    bf.value = 7
    assert bf.annotation == 'VAL_B'
    bf.value = 1
    # No matching enumerations so falls back to hex string annotation
    assert bf.annotation.startswith('0x')

def test_bitfield_validate_duplicate_enumerations_yields_error():
    bf = create_bitfield('bf', width=4)
    class DummyEnum:
        def __init__(self, name, value):
            self.name = name
            self.value = value
        def __str__(self):
            return self.name

    enum1 = DummyEnum('A', 1)
    enum2 = DummyEnum('B', 1)
    bf._children[enum1.name] = enum1
    bf._children[enum2.name] = enum2
    errors = list(bf.validate())
    assert any(isinstance(e, ValidationError) for e in errors)
    assert any("same value" in str(e) for e in errors)

def test_bitfield__get_parent_regs_raises_without_references():
    bf = create_bitfield('bf', width=4)
    if hasattr(bf, '_references'):
        del bf._references
    with pytest.raises(ValueError):
        bf._get_parent_regs()

def test_bitfield_write_and_read_calls_parent_registers(monkeypatch):
    bf = create_bitfield('bf', width=4)
    dummy_reg = Register(name='reg', address=0, width=32)
    dummy_reg.read = lambda : 0
    dummy_reg.write = lambda : None
    bf._references = set()
    bfr = create_bitfieldref(reg_offset=0, slice_width=4, bf=bf)
    bf._references.add(bfr)
    bfr.parent = dummy_reg

    # Patch read and write to track calls
    calls = {'read': 0, 'write': 0}
    dummy_reg.read = lambda: calls.__setitem__('read', calls['read'] + 1)
    dummy_reg.write = lambda: calls.__setitem__('write', calls['write'] + 1)

    bf.write(3)
    assert calls['read'] == 1
    assert calls['write'] == 1

    val = bf.read()
    assert calls['read'] == 2
    assert val == bf.value

##########################
# Tests for BitFieldRef   #
##########################

def test_bitfieldref_init_and_properties():
    bf = create_bitfield('bf', width=8)
    bfr = BitFieldRef(reg_offset=2, slice_width=4, field_offset=1, name='bfr')
    bfr._add(bf)
    assert bfr.reg_offset == 2
    assert bfr.slice_width == 4
    assert bfr.field_offset == 1
    # Mask should be 0b1111 (4 bits)
    assert bfr.mask == 0b1111
    # bf should be set properly
    assert bfr.bf is bf

def test_bitfieldref_init_invalid_values():
    with pytest.raises(ValueError):
        BitFieldRef(reg_offset=0, slice_width=-1)
    with pytest.raises(ValueError):
        BitFieldRef(reg_offset=0, field_offset=-1)

def test_bitfieldref_value_getter_and_setter_masking():
    bf = create_bitfield('bf', width=8)
    bf.value = 0b11110000  # 0xF0
    bfr = create_bitfieldref(reg_offset=4, slice_width=4, field_offset=4, name='bfr')
    bfr._add(bf)

    val = bfr.value
    # Should extract 4 bits starting at field_offset=4 and shift back by reg_offset
    expected_val = ((bf.value >> bfr.field_offset) & bfr.mask) << bfr.reg_offset
    assert val == expected_val

    # Now set the value for the bfr and check bf.value updated correctly
    bfr.value = expected_val
    # The bitfield value should have the 4 bits in position field_offset overwritten
    mask_clear = ~(bfr.mask << bfr.field_offset) & 0xFF
    expected_bf_val = (bf.value & mask_clear) | (((expected_val >> bfr.reg_offset) & bfr.mask) << bfr.field_offset)
    assert bf.value == expected_bf_val

def test_bitfieldref_add_non_bitfield_raises():
    bfr = BitFieldRef(reg_offset=0, slice_width=0)
    with pytest.raises(ValueError):
        bfr._add(object())

def test_bitfieldref_reset_val_property_calculates_correctly():
    bf = create_bitfield('bf', width=8, reset_val=0b11001100)
    bfr = create_bitfieldref(reg_offset=2, slice_width=4, field_offset=2, name='bfr')
    bfr._add(bf)
    # reset_val should be bits reset_val shifted by field_offset masked and shifted by reg_offset
    expected = ((bf.reset_val >> bfr.field_offset) & bfr.mask) << bfr.reg_offset
    assert bfr.reset_val == expected

def test_bitfieldref_copy_with_alias_and_new_instance(monkeypatch):
    bf = create_bitfield('bf', width=8)
    bfr = BitFieldRef(reg_offset=0, slice_width=8, field_offset=0, name='bfr')
    bfr._add(bf)
    bfr._alias = True
    bfr._ref = None
    # Copy with new_alias True forces shallow copy path
    copy_obj = bfr._copy(new_alias=True, new_instance=False)
    assert copy_obj is not bfr
    # Copy with new_alias False makes deep copy, adds bf children anew
    copy_deep = bfr._copy(new_alias=False, new_instance=True)
    assert copy_deep is not bfr
    assert copy_deep.bf is bf

def test_bitfieldref_validate_no_bitfield_and_slice_width_error():
    bfr = BitFieldRef(reg_offset=0, slice_width=4, field_offset=0)
    errors = list(bfr.validate())
    assert any(isinstance(e, ValidationError) for e in errors)
    assert any("No child bitfield present" in str(e) for e in errors)

    bf = create_bitfield('bf', width=4)
    bfr = BitFieldRef(reg_offset=0, slice_width=5, field_offset=0)
    bfr._add(bf)
    # slice_width > bf.width triggers validation error
    errors = list(bfr.validate())
    assert any(isinstance(e, ValidationError) for e in errors)
    assert any("Slice width is larger" in str(e) for e in errors)

#######################
# Tests for RegisterMap#
#######################

def test_registermap_block_dimensions_setter_and_getter():
    rm = RegisterMap(name='map')
    reg1 = Register(name='r1', address=0x1000, width=16)
    reg2 = Register(name='r2', address=0x1004, width=32)
    rm._children = {reg1.name: reg1, reg2.name: reg2}
    # Calculated property works
    start, size = rm._block_dimensions
    assert start == reg1.address
    # size should cover from start to last reg + width(in bytes)
    expected_size = (reg2.address + ceil(reg2.width / 8)) - start
    assert size == expected_size

    # Explicit setter works
    rm._block_dimensions = (0x2000, 0x10)
    assert rm._block_dimensions == (0x2000, 0x10)

def test_registermap_block_dimensions_raises_on_empty():
    rm = RegisterMap(name='empty_map')
    with pytest.raises(ValueError):
        _ = rm._block_dimensions

def test_registermap_read_and_write_behaviour(monkeypatch):
    dummy_io = DummyRegIO()
    rm = RegisterMap(name='map')
    reg1 = Register(name='r1', address=0x1000, width=16)
    reg2 = Register(name='r2', address=0x1002, width=16)
    reg1.value = 0xAA55
    reg2.value = 0x55AA
    rm._children = {reg1.name: reg1, reg2.name: reg2}

    rm._block_read_func = dummy_io.block_read_func
    rm._block_write_func = dummy_io.block_write_func

    # Trigger _block_dimensions calculation
    _ = rm._block_dimensions
    data = rm.read()
    # Data returned should be bytes of total block size
    assert isinstance(data, bytes)
    assert len(data) == rm._block_dimensions[1]

    # Write without data argument initializes zero bytes and writes per registers
    dummy_io.write_calls.clear()
    rm.write()
    assert dummy_io.write_calls
    assert dummy_io.write_calls[-1][0] == rm._block_dimensions[0]
    assert isinstance(dummy_io.write_calls[-1][1], (bytearray, bytes))

def test_registermap_write_raises_if_no_registers():
    rm = RegisterMap(name='empty_map')
    rm._block_write_func = lambda a, b: None
    with pytest.raises(ValueError):
        rm.write(None)
