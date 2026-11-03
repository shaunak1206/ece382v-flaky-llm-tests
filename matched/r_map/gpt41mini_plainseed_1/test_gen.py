import pytest
import r_map
from r_map.Register import Register
from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.RegisterMap import RegisterMap
from r_map.ValidationError import ValidationError


def make_simple_bitfield(name='bf', width=4, reset_val=2, access='RW'):
    bf = BitField(name=name, width=width, reset_val=reset_val, access=access)
    return bf

def make_simple_bfref(bf=None, reg_offset=0, slice_width=0, field_offset=0):
    bfr = BitFieldRef(reg_offset=reg_offset, slice_width=slice_width, field_offset=field_offset, name='bfr')
    if bf:
        bfr._add(bf)
    return bfr

def make_simple_register(name='reg', width=8):
    reg = Register(name=name, width=width)
    # patch register IO funcs to dummy functions to allow reads/writes
    reg._reg_read_func = lambda address: 0x0
    reg._reg_write_func = lambda address, val: None
    return reg

def make_simple_register_with_bfrefs():
    reg = make_simple_register()
    bf1 = make_simple_bitfield(name='bf1', width=4, reset_val=1)
    bf2 = make_simple_bitfield(name='bf2', width=4, reset_val=2)
    bfr1 = make_simple_bfref(bf=bf1, reg_offset=0, slice_width=4)
    bfr2 = make_simple_bfref(bf=bf2, reg_offset=4, slice_width=4)
    reg._children = {}
    reg._added = set()
    # Register.__iter__ relies on Node.__iter__, which yields children in _children.
    reg._children[bfr1.name] = bfr1
    reg._children[bfr2.name] = bfr2
    # Also patch register with required bitfield refs
    reg._reg_read_func = lambda address: 0x12
    reg._reg_write_func = lambda address, val: None
    return reg, bf1, bf2, bfr1, bfr2


# --- BitField Tests ---

def test_bitfield_init_and_str():
    bf = BitField(name='testbf', width=8, reset_val=0x55, access='RW')
    assert bf.width == 8
    assert bf.reset_val == 0x55
    s = str(bf)
    assert 'width: 8' in s and 'reset_val' in s and 'value' in s

def test_bitfield_value_set_get():
    bf = BitField(width=4, reset_val=0)
    bf.value = 0xF
    assert bf.value == 0xF & ((1 << 4) - 1)
    bf.value = 0x20
    assert bf.value == 0x0  # 0x20 masked to 4 bits is zero

def test_bitfield_value_set_str_and_enum(monkeypatch):
    # add enumeration child mock with name and value
    e1 = type('Enumeration', (), {'name': 'OK', 'value': 1})
    bf = BitField(width=4)
    bf._children = { 'OK': e1 }
    bf.__iter__ = lambda: iter(bf._children.values())

    bf.value = 'OK'
    assert bf.value == 1
    with pytest.raises(ValueError):
        bf.value = 'BAD_NAME'

def test_bitfield_value_set_wrong_type():
    bf = BitField(width=4)
    with pytest.raises(NotImplementedError):
        bf.value = 3.14

def test_bitfield_annotation_returns_enum_or_hex():
    e1 = type('Enumeration', (), {'name': 'OK', 'value': 5})
    bf = BitField(width=8)
    bf._children = { 'OK': e1 }
    bf.__iter__ = lambda: iter(bf._children.values())
    bf.value = 5
    assert bf.annotation == 'OK'
    bf.value = 6
    assert bf.annotation == hex(6)

def test_bitfield_validate_duplicate_values():
    e1 = type('Enumeration', (), {'name': 'E1', 'value': 2, '__str__': lambda s: 'E1'})
    e2 = type('Enumeration', (), {'name': 'E2', 'value': 2, '__str__': lambda s: 'E2'})
    bf = BitField(width=4)
    bf._children = { 'E1': e1, 'E2': e2 }
    bf.__iter__ = lambda: iter(bf._children.values())
    errors = list(bf.validate())
    assert any(isinstance(e, ValidationError) for e in errors)
    assert "E1" in str(errors[0]) or "E2" in str(errors[0])

def test_bitfield_write_and_read(monkeypatch):
    reg = make_simple_register()
    bf = BitField(width=4)
    bfr = BitFieldRef(reg_offset=0, slice_width=4, field_offset=0)
    bfr._add(bf)

    # patch BitField._get_parent_regs to produce our mock register
    bf._references = {bfr}
    bf._get_parent_regs = lambda: [reg]

    called = {'read': 0, 'write': 0}
    reg._reg_read_func = lambda address: 10
    reg._reg_write_func = lambda address, val: called.update({'write': called['write'] + 1})
    reg.read = lambda: (called.update({'read': called['read'] + 1}) or 10)
    reg.write = lambda: called.update({'write': called['write'] + 1})

    bf.value = 3
    bf.write(val=5)
    assert called['read'] > 0 and called['write'] > 0

    # test read triggers parent's read
    val = bf.read()
    assert val == bf.value
    assert called['read'] > 1

# --- BitFieldRef Tests ---

def test_bitfieldref_init_invalid_values():
    with pytest.raises(ValueError):
        BitFieldRef(reg_offset=0, slice_width=-1)
    with pytest.raises(ValueError):
        BitFieldRef(reg_offset=0, slice_width=1, field_offset=-1)

def test_bitfieldref_add_and_properties():
    bf = BitField(width=8, reset_val=0xAA)
    bfr = BitFieldRef(reg_offset=2, slice_width=4, field_offset=1)
    bfr._add(bf)
    assert bfr.bf == bf
    # reset_val property encoding
    expected = ((bf.reset_val >> bfr.field_offset) & bfr.mask) << bfr.reg_offset
    assert bfr.reset_val == expected
    # value property encoding and setter
    bf.value = 0b10101010
    val = ((bf.value >> bfr.field_offset) & bfr.mask) << bfr.reg_offset
    assert bfr.value == val
    # setting bfr.value alters bf.value correctly
    orig_value = bf.value
    new_val = val ^ (1 << bfr.reg_offset)
    bfr.value = new_val
    # BitField value should have updated in bits corresponding to mask shifted by field_offset
    masked_val = (bf.value >> bfr.field_offset) & bfr.mask
    expected_val = (new_val >> bfr.reg_offset) & bfr.mask
    assert masked_val == expected_val
    # old bits outside slice should remain intact
    outside_mask = ~ (bfr.mask << bfr.field_offset)
    assert (bf.value & outside_mask) == (orig_value & outside_mask)

def test_bitfieldref_copy_alias_behavior():
    bf = BitField(width=4)
    bfr = BitFieldRef(reg_offset=0, slice_width=4)
    bfr._add(bf)
    # simulate multiple references to same bitfield
    bf._references.add(bfr)
    bfr._alias = True
    bfr._ref = None
    copy_obj = bfr._copy(new_alias=True, new_instance=True)
    # The copied object should have _bf set
    assert copy_obj.bf is not None

def test_bitfieldref_validate_errors():
    bfr = BitFieldRef(reg_offset=0, slice_width=2)
    errors = list(bfr.validate())
    # No bf assigned, expect error
    assert any(isinstance(e, ValidationError) for e in errors)
    # Attach a bf with smaller width than slice_width to trigger error
    bf = BitField(width=1)
    bfr._add(bf)
    errors2 = list(bfr.validate())
    assert any(isinstance(e, ValidationError) for e in errors2)

# --- Register Tests ---

def test_register_basic_value_set_get():
    reg = make_simple_register()
    bf = make_simple_bitfield(width=8, reset_val=0)
    bfr = BitFieldRef(reg_offset=0, slice_width=8)
    bfr._add(bf)
    reg._children = {bfr.name: bfr}
    # patch reg's _reg_read_func and _reg_write_func
    reg._reg_read_func = lambda addr: 0xAB
    reg._reg_write_func = lambda addr, val: None

    reg.value = 0xC3
    # Register value should match bitfield value setting
    assert reg.value == bfr.value

def test_register_reset_resets_all_bitfieldrefs():
    reg, bf1, bf2, bfr1, bfr2 = make_simple_register_with_bfrefs()
    bf1.value = 0xF
    bf2.value = 0xA
    reg.reset()
    assert bf1.value == bf1.reset_val
    assert bf2.value == bf2.reset_val

def test_register_str_includes_value():
    reg, *_ = make_simple_register_with_bfrefs()
    reg.value = 0x55
    s = str(reg)
    assert 'value:' in s
    assert f"{reg.value:#0{int(reg.width/4)+2}x}" in s

def test_register_access_property():
    reg, bf1, bf2, bfr1, bfr2 = make_simple_register_with_bfrefs()
    # bitfields access string set
    bf1.access = 'R'
    bf2.access = 'W'
    acc = reg.access
    assert 'R' in acc and 'W' in acc

def test_register_validate_errors():
    reg = make_simple_register()
    # no bitfieldrefs at all
    errors = list(reg.validate())
    assert any(isinstance(e, ValidationError) for e in errors)
    # add a child that is not BitFieldRef
    reg._children['bad'] = make_simple_bitfield()
    errors = list(reg.validate())
    assert any(isinstance(e, ValidationError) for e in errors)
    # add overlapping bitfieldrefs
    bf1 = make_simple_bitfield(width=4)
    bf2 = make_simple_bitfield(width=4)
    bfr1 = make_simple_bfref(bf=bf1, reg_offset=0, slice_width=4)
    bfr2 = make_simple_bfref(bf=bf2, reg_offset=2, slice_width=4)  # overlaps with bfr1
    reg._children.clear()
    reg._children[bfr1.name] = bfr1
    reg._children[bfr2.name] = bfr2
    errors = list(reg.validate())
    assert any('overlaps' in str(e) for e in errors)

def test_register_read_write_calls(monkeypatch):
    reg, bf1, bf2, bfr1, bfr2 = make_simple_register_with_bfrefs()
    called = {'read': False, 'write': False}
    def read_func(addr):
        called['read'] = True
        return 0xAB
    def write_func(addr, val):
        called['write'] = True
        assert val == reg.value
    reg._reg_read_func = read_func
    reg._reg_write_func = write_func
    val = reg.read()
    assert val == 0xAB
    reg.write()
    assert called['read'] is True
    assert called['write'] is True

# --- RegisterMap Tests ---

def test_register_map_block_dimensions_and_errors():
    rm = RegisterMap(name='map')
    # empty register map: error on _block_dimensions access
    with pytest.raises(ValueError):
        _ = rm._block_dimensions
    # Add one register at address 0x10 with width 16 bits (2 bytes)
    reg = make_simple_register()
    reg.address = 0x10
    reg.width = 16
    rm._children = {reg.name: reg}
    rm._added = set()
    start, width = rm._block_dimensions
    assert start == 0x10
    assert width >= 2

def test_register_map_read_write_flow(monkeypatch):
    rm = RegisterMap(name='map')
    # add two registers
    reg1 = make_simple_register(name='r1', width=8)
    reg1.address = 0x10
    reg1.value = 0x12
    reg1._reg_read_func = lambda a: 0x12
    reg1._reg_write_func = lambda a, v: None
    reg2 = make_simple_register(name='r2', width=16)
    reg2.address = 0x11
    reg2.value = 0x3456
    reg2._reg_read_func = lambda a: 0x3456
    reg2._reg_write_func = lambda a, v: None
    rm._children = {reg1.name: reg1, reg2.name: reg2}
    rm._added = set()
    # Prepare data to be read: bytes for r1 and r2, little endian
    data_bytes = b'\x12\x56\x34'
    rm._block_read_func = lambda address, size: data_bytes
    written_data = {}
    def fake_write_func(address, data):
        written_data['address'] = address
        written_data['data'] = data
    rm._block_write_func = fake_write_func

    # test read updates registers
    read_data = rm.read()
    assert read_data == data_bytes
    assert reg1.value == 0x12
    assert reg2.value == 0x3456

    # test write composes block and calls write func
    rm.write()
    assert 'address' in written_data
    assert 'data' in written_data

def test_register_map_write_partial_data():
    rm = RegisterMap(name='map')
    reg1 = make_simple_register(name='r1', width=8)
    reg1.address = 0x00
    reg1.value = 0x11
    reg2 = make_simple_register(name='r2', width=16)
    reg2.address = 0x01
    reg2.value = 0x2233
    rm._children = {reg1.name: reg1, reg2.name: reg2}
    rm._added = set()
    rm._block_read_func = lambda a, s: b'\x11\x33\x22'

    data = bytearray(b'\x11\x33')  # less bytes than block (3 bytes expected)
    called = {}
    def fake_write_func(address, data):
        called['called'] = True
        # Should not raise error
    rm._block_write_func = fake_write_func

    rm.write(data)
    assert called.get('called', False)

def test_register_map_write_raises_on_empty(monkeypatch):
    rm = RegisterMap(name='mapempty')
    rm._children.clear()
    rm._added = set()
    with pytest.raises(ValueError):
        rm.write()


# EOF
