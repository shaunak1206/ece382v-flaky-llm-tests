import pytest
from r_map import Register, BitField, BitFieldRef, RegisterMap
from r_map.ValidationError import ValidationError

# Helper dummy IO functions for Registers and RegisterMaps
def dummy_reg_read(address):
    return 0xA5A5A5A5

def dummy_reg_write(address, value):
    dummy_reg_write.called = (address, value)
dummy_reg_write.called = None

def dummy_block_read(address, size):
    return bytearray(range(size))

def dummy_block_write(address, data):
    dummy_block_write.called = (address, data)
dummy_block_write.called = None

# ---------------- Register tests ----------------

def test_register_str_and_value_setter_getter():
    bf1 = BitField(name='bf1', width=8, reset_val=0x0F)
    bfref1 = BitFieldRef(name='bfref1', reg_offset=0)
    bfref1._add(bf1)
    reg = Register(name='reg1', width=16)
    reg._reg_read_func = dummy_reg_read
    reg._reg_write_func = dummy_reg_write
    reg._children = {bfref1.name: bfref1}
    bfref1._parent = reg
    # Register iter yields children
    assert list(reg) == [bfref1]

    # Set value propagates to bitfield refs correctly
    reg.value = 0xAAAA
    v_bfref1 = bfref1.value
    assert isinstance(v_bfref1, int)
    # Value string includes 'value'
    s = str(reg)
    assert 'value' in s

def test_register_reset_resets_all_bitfields():
    bf1 = BitField(name='bf1', width=8, reset_val=0x12)
    bfref1 = BitFieldRef(name='bfref1', reg_offset=0)
    bfref1._add(bf1)
    reg = Register(name='reg1', width=16, _reg_write_func=dummy_reg_write, _reg_read_func=dummy_reg_read)
    reg._children = {bfref1.name: bfref1}
    bfref1._parent = reg

    bfref1.value = 0x34
    assert bfref1.value != bfref1.reset_val
    reg.reset()
    assert bfref1.value == bfref1.reset_val

def test_register_access_property_joins_access_attrs():
    bf1 = BitField(name='bf1', width=8, access='RW')
    bfref1 = BitFieldRef(name='bfref1', reg_offset=0)
    bfref1._add(bf1)
    bf2 = BitField(name='bf2', width=4, access='RO')
    bfref2 = BitFieldRef(name='bfref2', reg_offset=8)
    bfref2._add(bf2)
    reg = Register(name='reg1')
    reg._children = {bfref1.name: bfref1, bfref2.name: bfref2}
    bfref1._parent = bfref2._parent = reg

    access = reg.access
    # Access is sorted, unique and joined by '|'
    expected = '|'.join(sorted({'RW','RO'}))
    assert access == expected

def test_register_value_and_reset_val_properties():
    bf1 = BitField(name='bf1', width=8, reset_val=0x0F)
    bfref1 = BitFieldRef(name='bfref1', reg_offset=0)
    bfref1._add(bf1)
    bf2 = BitField(name='bf2', width=8, reset_val=0xF0)
    bfref2 = BitFieldRef(name='bfref2', reg_offset=8)
    bfref2._add(bf2)
    reg = Register(name='reg1')
    reg._children = {bfref1.name: bfref1, bfref2.name: bfref2}
    bfref1._parent = bfref2._parent = reg

    # Value = combination of bitfieldref values
    bf1.value = 0x01
    bf2.value = 0x02
    actual_value = reg.value
    expected = bfref1.value | bfref2.value
    assert actual_value == expected

    # reset_val combines bitfields reset_val shifted accordingly
    expected_reset_val = bfref1.reset_val | bfref2.reset_val
    assert reg.reset_val == expected_reset_val

def test_register_validate_reports_errors_on_no_bitfieldrefs_and_bad_children():
    reg = Register(name='reg1')
    reg._children = {}
    errors = list(reg.validate())
    # Should return ValidationError about no bitfieldrefs present
    assert any(isinstance(e, ValidationError) and "No bitfieldrefs" in str(e) for e in errors)

    # Add child not BitFieldRef should error
    class FakeChild:
        pass
    fake = FakeChild()
    reg._children = {'fake': fake}
    errors = list(reg.validate())
    # Should report type error for child not BitFieldRef
    assert any(isinstance(e, ValidationError) and "not of type BitFieldRef" in str(e) for e in errors)

def test_register_validate_detects_overlapping_bitfieldrefs():
    bf1 = BitField(name='bf1', width=8)
    bfref1 = BitFieldRef(name='bfref1', reg_offset=0, slice_width=8)
    bfref1._add(bf1)
    bf2 = BitField(name='bf2', width=8)
    bfref2 = BitFieldRef(name='bfref2', reg_offset=4, slice_width=8)
    bfref2._add(bf2)
    reg = Register(name='reg1')
    reg._children = {bfref1.name: bfref1, bfref2.name: bfref2}
    bfref1._parent = bfref2._parent = reg

    errors = list(reg.validate())
    assert any(isinstance(e, ValidationError) and "overlaps" in str(e) for e in errors)

def test_register_read_and_write_calls_io_functions(monkeypatch):
    bf1 = BitField(name='bf1', width=8)
    bfref1 = BitFieldRef(name='bfref1', reg_offset=0)
    bfref1._add(bf1)
    reg = Register(name='reg1')
    reg._children = {bfref1.name: bfref1}
    bfref1._parent = reg

    called = {}
    def fake_read(addr):
        called['read'] = addr
        return 0x1234
    def fake_write(addr, val):
        called['write'] = (addr, val)
    reg._reg_read_func = fake_read
    reg._reg_write_func = fake_write
    reg.address = 0x10

    val_returned = reg.read()
    # Should read from address and set values
    assert val_returned == 0x1234
    assert 'read' in called and called['read'] == 0x10

    reg.write(0xABCD)
    assert 'write' in called and called['write'] == (0x10, 0xABCD)

    # Writing with no arg writes current value
    called.clear()
    reg.write()
    assert 'write' in called and called['write'][0] == 0x10

def test_register_iter_sorts_bitfieldrefs_by_reg_offset_and_yields_all():
    bf1 = BitField(name='bf1', width=8)
    bfref1 = BitFieldRef(name='bfref1', reg_offset=1)
    bfref1._add(bf1)
    bf2 = BitField(name='bf2', width=8)
    bfref2 = BitFieldRef(name='bfref2', reg_offset=0)
    bfref2._add(bf2)
    reg = Register(name='reg1')
    reg._children = {bfref1.name: bfref1, bfref2.name: bfref2}
    bfref1._parent = bfref2._parent = reg

    # Iter returns items sorted by reg_offset
    iter_list = list(reg)
    assert iter_list == sorted(iter_list, key=lambda c: c.reg_offset)

# ---------------- BitField tests ----------------

def test_bitfield_init_and_mask_and_value_behavior():
    bf = BitField(name='bf', width=4, reset_val=0xF)
    # Mask should be (1<<width)-1=0xF for width=4
    assert bf.mask == 0xF
    # Initial value is reset_val masked
    assert bf.value == (0xF & bf.mask)

def test_bitfield_value_setter_with_int_enumeration_and_str(monkeypatch):
    class MockEnum:
        def __init__(self, name, value):
            self.name = name
            self.value = value
        def __str__(self):
            return f"MockEnum: {self.name}"

    bf = BitField(name='bf', width=8)
    # Add mock enumeration children
    enum1 = MockEnum('ENUM1', 1)
    enum2 = MockEnum('ENUM2', 2)
    bf._children = {'ENUM1': enum1, 'ENUM2': enum2}

    # Setting int value masks correctly
    bf.value = 0x123
    assert bf.value == 0x23  # masked to width=8 bits

    # Setting value using Enumeration instance sets underlying int value
    bf.value = enum1
    assert bf.value == enum1.value

    # Setting value using string matches enumeration name
    bf.value = 'ENUM2'
    assert bf.value == enum2.value

    # Setting string with no matching enum raises ValueError
    with pytest.raises(ValueError):
        bf.value = 'NO_SUCH_ENUM'

    # Setting unsupported type raises NotImplementedError
    with pytest.raises(NotImplementedError):
        bf.value = 3.14

def test_bitfield_annotation_returns_name_or_hex():
    class MockEnum:
        def __init__(self, name, value):
            self.name = name
            self.value = value

    bf = BitField(name='bf', width=8)
    enum1 = MockEnum('ENUM1', 1)
    enum2 = MockEnum('ENUM2', 2)
    bf._children = {'ENUM1': enum1, 'ENUM2': enum2}

    # Value matching ENUM1 returns 'ENUM1'
    bf.value = 1
    assert bf.annotation == 'ENUM1'

    # Value not matching any enum returns hex string
    bf.value = 99
    assert bf.annotation == hex(99)

def test_bitfield_validate_reports_duplicate_enum_values():
    class MockEnum:
        def __init__(self, name, value):
            self.name = name
            self.value = value
        def __str__(self):
            return self.name

    bf = BitField(name='bf', width=8)
    e1 = MockEnum('e1', 1)
    e2 = MockEnum('e2', 1)  # duplicate value

    bf._children = {'e1': e1, 'e2': e2}
    errs = list(bf.validate())
    assert any(isinstance(e, ValidationError) and 'same value' in str(e) for e in errs)

def test_bitfield__get_parent_regs_errors_when_no_refs():
    bf = BitField(name='bf')
    # No _references attribute
    with pytest.raises(ValueError):
        bf._get_parent_regs()

def test_bitfield_write_and_read_io(monkeypatch):
    bf = BitField(name='bf')
    # Prepare fake parent register with read/write tracking
    class FakeReg:
        def __init__(self):
            self.read_called = False
            self.write_called = False
            self.value = 0

        def read(self):
            self.read_called = True
            self.value = 0x55

        def write(self):
            self.write_called = True

    reg = FakeReg()
    # Attach a BitFieldRef to the BitField that points to reg as parent
    bf._references = set()
    bf_ref = BitFieldRef(name='bfref', reg_offset=0)
    bf_ref._add(bf)
    bf._references.add(bf_ref)
    bf_ref._parent = reg

    # monkeypatch _get_parent_regs to return [reg]
    bf._get_parent_regs = lambda: [reg]

    # Write val, should read before write, write called on reg
    bf.write(0xAA, always_write=True)
    assert reg.read_called is True
    assert reg.write_called is True
    # Write with val None uses current value
    reg.read_called = False
    reg.write_called = False
    bf.write(val=None)
    assert reg.read_called is True
    assert reg.write_called is True

    # Read returns current value and calls parent read
    reg.read_called = False
    val = bf.read()
    assert reg.read_called is True
    assert val == bf.value

# ------------- BitFieldRef tests -------------

def test_bitfieldref_init_and_mask_and_reset_val_value(monkeypatch):
    bf = BitField(name='b', width=8, reset_val=0xF0)
    bfref = BitFieldRef(name='ref', reg_offset=4, slice_width=4, field_offset=4)
    bfref._add(bf)
    # Mask should be (1<<slice_width)-1 = 0b1111 = 15
    assert bfref.mask == 0xF

    # Reset val property reflects bitfield reset_val shifted accordingly
    expected_reset = ((bf.reset_val >> bfref.field_offset) & bfref.mask) << bfref.reg_offset
    assert bfref.reset_val == expected_reset

    # Value property getter returns bitfield's relevant shifted value
    bf.value = 0xFF
    expected_val = ((bf.value >> bfref.field_offset) & bfref.mask) << bfref.reg_offset
    assert bfref.value == expected_val

def test_bitfieldref_value_setter_updates_bitfield_value():
    bf = BitField(name='b', width=8)
    bfref = BitFieldRef(name='ref', reg_offset=4, slice_width=4, field_offset=0)
    bfref._add(bf)
    bf.value = 0x00
    # Set bitfieldref value which affects bf.value masked and shifted
    bfref.value = (0xA << bfref.reg_offset)  # bfref expects value shifted by reg_offset
    # The bits in bf.value corresponding to bitfieldref should be updated
    masked_old = bf.value & ~(bfref.mask << bfref.field_offset)
    expected_val = masked_old | (( ( (0xA << bfref.reg_offset) >> bfref.reg_offset) & bfref.mask) << bfref.field_offset)
    assert bf.value == expected_val

def test_bitfieldref_add_invalid_type_raises():
    bfref = BitFieldRef(name='ref', reg_offset=0)
    with pytest.raises(ValueError):
        bfref._add("not a BitField")

def test_bitfieldref_validate_no_bitfield_and_slice_wrong_width():
    bfref = BitFieldRef(name='ref', reg_offset=0, slice_width=8)
    # No _bf set yet
    errors1 = list(bfref.validate())
    assert any(isinstance(e, ValidationError) and 'No child bitfield' in str(e) for e in errors1)

    # Assign bitfield with smaller width than slice_width triggers validation error
    bf = BitField(name='b', width=4)
    bfref._add(bf)
    errors2 = list(bfref.validate())
    assert any(isinstance(e, ValidationError) and 'Slice width is larger than bitfield' in str(e) for e in errors2)

def test_bitfieldref_copy_behavior():
    bf = BitField(name='b', width=8)
    bfref = BitFieldRef(name='ref', reg_offset=0)
    bfref._add(bf)

    # _alias False triggers deep copy = True
    new_obj = bfref._copy(new_instance=True, new_alias=False)
    assert new_obj is not bfref
    # _alias True triggers deep copy = False and uses _ref if set
    bfref._alias = True
    bfref._ref = bfref  # self-reference for testing
    new_obj2 = bfref._copy(new_instance=True, new_alias=True)
    assert new_obj2 is not bfref
    # Should have _bf set (either new_obj2._bf or via _add)
    assert hasattr(new_obj2, 'bf')

# ------------- RegisterMap tests ------------------

def test_registermap_block_dimensions_and_setter():
    regmap = RegisterMap(name='map1')
    # No children, _block_dimensions should raise
    with pytest.raises(ValueError):
        _ = regmap._block_dimensions

    # Add a Register with address and width
    bf = BitField(name='bf', width=8)
    bfref = BitFieldRef(name='bfref', reg_offset=0)
    bfref._add(bf)
    reg = Register(name='reg', width=8)
    reg.address = 0x100
    reg._children = {bfref.name: bfref}
    bfref._parent = reg
    regmap._children = {reg.name: reg}

    start_addr, block_width = regmap._block_dimensions
    assert start_addr == 0x100
    assert block_width == 1  # width=8 bits -> 1 byte

    # Setter should set the cached __block_dimensions
    regmap._block_dimensions = (0x200, 5)
    assert regmap._block_dimensions == (0x200, 5)

def test_registermap_read_and_write(monkeypatch):
    # Create a simple RegisterMap with 2 registers
    bf1 = BitField(name='bf1', width=8)
    bfref1 = BitFieldRef(name='bfref1', reg_offset=0)
    bfref1._add(bf1)
    reg1 = Register(name='reg1', width=8)
    reg1.address = 0x10
    reg1._children = {bfref1.name: bfref1}
    bfref1._parent = reg1

    bf2 = BitField(name='bf2', width=16)
    bfref2 = BitFieldRef(name='bfref2', reg_offset=0)
    bfref2._add(bf2)
    reg2 = Register(name='reg2', width=16)
    reg2.address = 0x11
    reg2._children = {bfref2.name: bfref2}
    bfref2._parent = reg2

    regmap = RegisterMap(name='map')
    regmap._children = {reg1.name: reg1, reg2.name: reg2}

    # Provide IO functions
    regmap._block_read_func = dummy_block_read
    regmap._block_write_func = dummy_block_write

    # Should read and set register values according to dummy data
    data = regmap.read()
    assert isinstance(data, bytearray)
    assert reg1.value == data[0]
    val_reg2 = int.from_bytes(data[1:3], byteorder='little')
    assert reg2.value == val_reg2

    # Test write with None data packs register values into _data
    dummy_block_write.called = None
    regmap.write()
    assert dummy_block_write.called is not None
    addr, data_bytes = dummy_block_write.called
    assert addr == min(reg1.address, reg2.address)
    # Data length matches block width
    assert len(data_bytes) == regmap._block_dimensions[1]

def test_registermap_write_with_data_partial(monkeypatch):
    bf = BitField(name='bf', width=8)
    bfref = BitFieldRef(name='bfref', reg_offset=0)
    bfref._add(bf)
    reg = Register(name='reg', width=8)
    reg.address = 0x10
    reg._children = {bfref.name: bfref}
    bfref._parent = reg
    regmap = RegisterMap(name='map')
    regmap._children = {reg.name: reg}
    regmap._block_read_func = dummy_block_read
    regmap._block_write_func = dummy_block_write

    data = bytearray([0x55])  # Smaller than block width (1)
    # Should set reg.value from data
    regmap.write(data=data)
    assert reg.value == 0x55

def test_registermap_write_no_registers_raises():
    regmap = RegisterMap(name='map')
    regmap._children = {}
    regmap._block_write_func = dummy_block_write
    with pytest.raises(ValueError):
        regmap.write()

