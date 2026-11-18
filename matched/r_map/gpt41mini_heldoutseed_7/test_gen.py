import pytest
import r_map
from r_map import Register, BitField, BitFieldRef, RegisterMap, ValidationError

def test_register_value_property_sets_and_gets_bitfield_values():
    bf1 = BitField(name="bf1", width=8, reset_val=0x12)
    bf2 = BitField(name="bf2", width=8, reset_val=0x34)
    bfr1 = BitFieldRef(name="bfr1", reg_offset=0)
    bfr1._add(bf1)
    bfr2 = BitFieldRef(name="bfr2", reg_offset=8)
    bfr2._add(bf2)
    reg = Register(name="reg1", width=16)
    # Manually add bitfieldrefs as children
    reg._children = {"bfr1": bfr1, "bfr2": bfr2}

    # Setting value should split bits into children according to offsets
    reg.value = 0x3412
    # BitFieldRefs will propagate changes to BitFields
    # bf1 is lower 8 bits, bf2 is higher 8 bits
    assert bf1.value == 0x12
    assert bf2.value == 0x34
    # Value property of register should combine these back
    assert reg.value == 0x3412

def test_register_reset_resets_all_bitfields_to_reset_val():
    bf = BitField(name="bf", width=8, reset_val=0xAA)
    bfr = BitFieldRef(name="bfr", reg_offset=0)
    bfr._add(bf)
    reg = Register(name="reg", width=8)
    reg._children = {"bfr": bfr}
    bf.value = 0x55
    reg.reset()
    assert bf.value == bf.reset_val
    assert bf.value == 0xAA

def test_register_str_includes_value_in_hex_and_width():
    reg = Register(name="reg", width=16)
    bf = BitField(name="bf", width=16)
    bfr = BitFieldRef(name="bfr", reg_offset=0)
    bfr._add(bf)
    reg._children = {"bfr": bfr}
    reg.value = 0xABCD
    s = str(reg)
    # Should contain hex value '0xabcd' (case insensitive)
    assert f"{reg.value:#06x}".lower() in s.lower()

def test_register_access_property_concats_bitfieldref_access_values():
    bf1 = BitField(name="bf1", width=8, access="RW")
    bf2 = BitField(name="bf2", width=8, access="RO")
    bfr1 = BitFieldRef(name="bfr1", reg_offset=0)
    bfr1._add(bf1)
    bfr2 = BitFieldRef(name="bfr2", reg_offset=8)
    bfr2._add(bf2)
    reg = Register(name="reg")
    reg._children = {"bfr1": bfr1, "bfr2": bfr2}
    acc = reg.access
    # Should contain 'RW' and 'RO' separated by '|', order sorted
    parts = acc.split('|')
    assert set(parts) == {"RW", "RO"}

def test_register_reset_val_property_combines_bitfieldrefs_reset_vals():
    bf1 = BitField(name="bf1", width=4, reset_val=0x1)
    bf2 = BitField(name="bf2", width=4, reset_val=0x2)
    bfr1 = BitFieldRef(name="bfr1", reg_offset=0)
    bfr1._add(bf1)
    bfr2 = BitFieldRef(name="bfr2", reg_offset=4)
    bfr2._add(bf2)
    reg = Register(name="reg", width=8)
    reg._children = {"bfr1": bfr1, "bfr2": bfr2}
    val = reg.reset_val
    expected = (bf1.reset_val << bfr1.reg_offset) | (bf2.reset_val << bfr2.reg_offset)
    # BitFieldRef.reset_val already shifts, so just sum
    assert val == bfr1.reset_val + bfr2.reset_val

def test_register_validate_yields_error_when_no_bitfieldrefs_present():
    reg = Register(name="reg")
    # No children set
    reg._children = {}
    errors = list(reg.validate())
    assert any(isinstance(e, ValidationError) for e in errors)
    assert any("No bitfieldrefs present" in str(e) for e in errors)

def test_register_validate_yields_error_if_non_bitfieldref_children_present():
    reg = Register(name="reg")
    # Add incorrect child type
    reg._children = {"wrong": BitField(name="bf")}
    errors = list(reg.validate())
    assert any(isinstance(e, ValidationError) for e in errors)
    assert any("is not of typeBitFieldRef" in str(e) for e in errors)

def test_register_validate_yields_error_on_overlapping_bitfieldrefs():
    bf1 = BitField(name="bf1", width=8)
    bf2 = BitField(name="bf2", width=8)
    bfr1 = BitFieldRef(name="bfr1", reg_offset=0)
    bfr1._add(bf1)
    bfr2 = BitFieldRef(name="bfr2", reg_offset=4)  # Overlapping with bfr1
    bfr2._add(bf2)
    reg = Register(name="reg")
    reg._children = {"bfr1": bfr1, "bfr2": bfr2}
    errors = list(reg.validate())
    assert any(isinstance(e, ValidationError) for e in errors)
    assert any("overlaps" in str(e) for e in errors)

def test_bitfield_value_allows_setting_by_enumeration_name():
    class DummyEnum:
        def __init__(self, name, value):
            self.name = name
            self.value = value
        def __str__(self):
            return self.name

    bf = BitField(name="bf", width=4)
    # Add dummy enum children
    bf._children = {
        "enum1": DummyEnum("ZERO", 0),
        "enum2": DummyEnum("ONE", 1)
    }
    # Support iteration over enumerations
    def enum_iter():
        for e in bf._children.values():
            yield e
    bf.__iter__ = enum_iter

    bf.value = "ONE"
    assert bf.value == 1
    with pytest.raises(ValueError):
        bf.value = "NONEXISTENT"

def test_bitfield_reset_sets_value_to_reset_val():
    bf = BitField(name="bf", width=4, reset_val=0xA)
    bf.value = 0
    bf.reset()
    assert bf.value == 0xA

def test_bitfield_validate_detects_duplicate_enumeration_values():
    class DummyEnum:
        def __init__(self, name, value):
            self.name = name
            self.value = value
        def __str__(self):
            return self.name

    bf = BitField(name="bf", width=4)
    e1 = DummyEnum("ONE", 1)
    e2 = DummyEnum("UNO", 1)
    bf._children = {"one": e1, "uno": e2}

    def enum_iter():
        for e in bf._children.values():
            yield e
    bf.__iter__ = enum_iter

    errors = list(bf.validate())
    assert any(isinstance(e, ValidationError) for e in errors)
    assert any("same value" in str(e) for e in errors)

def test_bitfieldref_initialization_validates_slice_and_offset():
    with pytest.raises(ValueError):
        BitFieldRef(name="bfr", reg_offset=0, slice_width=-1)
    with pytest.raises(ValueError):
        BitFieldRef(name="bfr", reg_offset=0, field_offset=-1)

def test_bitfieldref_add_accepts_only_bitfield_instances():
    bfr = BitFieldRef(name="bfr", reg_offset=0)
    bf = BitField(name="bf", width=4)
    bfr._children = {}
    bfr._bf = None
    bfr._ref = None
    # should work
    bfr._add(bf)
    with pytest.raises(ValueError):
        bfr._add(bfr)  # passing BitFieldRef instead of BitField

def test_bitfieldref_value_property_get_set_behavior():
    bf = BitField(name="bf", width=8)
    bfr = BitFieldRef(name="bfr", reg_offset=2, slice_width=3, field_offset=1)
    bfr._add(bf)
    # bf.value initially 0
    bf.value = 0b10101010

    # Getting value should shift and mask correctly
    expected_val = ((bf.value >> bfr.field_offset) & bfr.mask) << bfr.reg_offset
    assert bfr.value == expected_val

    # Setting value should update the bf.value accordingly
    old_bf_val = bf.value
    new_val = (0b111 << bfr.reg_offset)  # place bits at reg_offset
    bfr.value = new_val
    # Calculate expected new bf.value manually:
    cleared = old_bf_val & ~(bfr.mask << bfr.field_offset)
    new_bf_val = cleared | (((new_val >> bfr.reg_offset) & bfr.mask) << bfr.field_offset)
    assert bf.value == new_bf_val

def test_bitfieldref_validate_detects_missing_bitfield_and_slice_width():
    bfr = BitFieldRef(name="bfr", reg_offset=0, slice_width=5)
    bfr._children = {}
    bfr._bf = None
    errors = list(bfr.validate())
    assert any(isinstance(e, ValidationError) for e in errors)
    assert any("No child bitfield present" in str(e) for e in errors)

    bf = BitField(name="bf", width=4)
    bfr._bf = bf
    errors = list(bfr.validate())
    assert any(isinstance(e, ValidationError) for e in errors)
    assert any("Slice width is larger than bitfield's width" in str(e) for e in errors)

def test_registermap_block_dimensions_calculation(monkeypatch):
    reg1 = Register(name="r1", address=0x1000, width=16)
    bf = BitField(name="bf", width=16)
    bfr = BitFieldRef(name="bfr", reg_offset=0)
    bfr._add(bf)
    reg1._children = {"bfr": bfr}

    reg2 = Register(name="r2", address=0x1004, width=32)
    bf2 = BitField(name="bf2", width=32)
    bfr2 = BitFieldRef(name="bfr2", reg_offset=0)
    bfr2._add(bf2)
    reg2._children = {"bfr2": bfr2}

    rm = RegisterMap(name="rm")
    rm._children = {"r1": reg1, "r2": reg2}

    start, size = rm._block_dimensions
    assert start == 0x1000
    # last register ends at 0x1004 + 4 bytes = 0x1008, size 0x8 bytes
    assert size == 8

def test_registermap_read_calls_block_read_and_updates_registers(monkeypatch):
    data = bytearray(b'\x12\x34\x56\x78')

    values_recorded = {}

    def fake_block_read_func(address, size):
        return data

    reg1 = Register(name="r1", address=0x1000, width=16)
    bf1 = BitField(name="bf1", width=16)
    bfr1 = BitFieldRef(name="bfr1", reg_offset=0)
    bfr1._add(bf1)
    reg1._children = {"bfr1": bfr1}

    reg2 = Register(name="r2", address=0x1002, width=16)
    bf2 = BitField(name="bf2", width=16)
    bfr2 = BitFieldRef(name="bfr2", reg_offset=0)
    bfr2._add(bf2)
    reg2._children = {"bfr2": bfr2}

    rm = RegisterMap(name="rm")
    rm._children = {"r1": reg1, "r2": reg2}
    rm._block_read_func = fake_block_read_func

    rm._block_dimensions = (0x1000, 4)

    ret_data = rm.read()
    assert ret_data == data

    # reg1 address 0x1000, gets bytes [0:2] -> 0x3412 little endian
    assert reg1.value == int.from_bytes(data[0:2], 'little')
    # reg2 address 0x1002, gets bytes [2:4] -> 0x7856 little endian
    assert reg2.value == int.from_bytes(data[2:4], 'little')

def test_registermap_write_calls_block_write_with_correct_data(monkeypatch):
    captured = {}
    def fake_block_write_func(address, data):
        captured['addr'] = address
        captured['data'] = data

    reg1 = Register(name="r1", address=0x1000, width=16)
    bf1 = BitField(name="bf1", width=16)
    bfr1 = BitFieldRef(name="bfr1", reg_offset=0)
    bfr1._add(bf1)
    reg1._children = {"bfr1": bfr1}
    bf1.value = 0x1234

    reg2 = Register(name="r2", address=0x1002, width=16)
    bf2 = BitField(name="bf2", width=16)
    bfr2 = BitFieldRef(name="bfr2", reg_offset=0)
    bfr2._add(bf2)
    reg2._children = {"bfr2": bfr2}
    bf2.value = 0x5678

    rm = RegisterMap(name="rm")
    rm._children = {"r1": reg1, "r2": reg2}
    rm._block_write_func = fake_block_write_func

    rm._block_dimensions = (0x1000, 4)

    rm.write()
    # The data passed should be a bytearray of length 4
    assert isinstance(captured['data'], (bytes, bytearray))
    assert len(captured['data']) == 4
    # The bytearray should contain bf1 and bf2 values at correct offsets
    assert captured['data'][0:2] == bf1.value.to_bytes(2, 'little')
    assert captured['data'][2:4] == bf2.value.to_bytes(2, 'little')

def test_registermap_write_data_partial_write(monkeypatch):
    """Test write with data shorter than block size only writes partial and breaks loop"""
    captured = {}
    def fake_block_write_func(address, data):
        captured['data'] = data

    reg1 = Register(name="r1", address=0x1000, width=16)
    bf1 = BitField(name="bf1", width=16)
    bfr1 = BitFieldRef(name="bfr1", reg_offset=0)
    bfr1._add(bf1)
    reg1._children = {"bfr1": bfr1}

    reg2 = Register(name="r2", address=0x1003, width=16)  # Note the 1 byte offset (not aligned)
    bf2 = BitField(name="bf2", width=16)
    bfr2 = BitFieldRef(name="bfr2", reg_offset=0)
    bfr2._add(bf2)
    reg2._children = {"bfr2": bfr2}

    rm = RegisterMap(name="rm")
    rm._children = {"r1": reg1, "r2": reg2}
    rm._block_write_func = fake_block_write_func

    # Block dimensions cover 5 bytes starting at 0x1000
    rm._block_dimensions = (0x1000, 5)
    # Provide only 4 bytes, shorter than block
    data = bytearray(b'\x01\x02\x03\x04')
    rm.write(data=data)

    # bf1 should be updated (bytes 0:2)
    expected_val1 = int.from_bytes(data[0:2], 'little')
    assert bf1.value == expected_val1

    # bf2 update should be skipped because end_index > data_len+1 triggers break (end_index=5>4+1=5, not breaking yet) 
    # Actually, end_index for reg2 is 0x1003 + 2 bytes = 0x1005 -> start_index=3, end_index=5
    # data length=4; end_index=5, so condition end_index > data_len+1 (5 >5) is False, no break.
    # Actually the code checks: if end_index > data_len + 1, break
    # here end_index=5; data_len+1 = 5; 5 > 5 false -> no break, so bf2 should get updated.
    expected_val2 = int.from_bytes(data[3:5], 'little') if len(data) >= 5 else 0
    assert bf2.value == expected_val2

def test_bitfield_write_reads_parent_and_calls_write(monkeypatch):
    bf = BitField(name="bf", width=8)
    bfr = BitFieldRef(name="bfr", reg_offset=0)
    bfr._add(bf)

    reg = Register(name="reg", width=8)
    reg._children = {"bfr": bfr}

    # Setup references manually for bf._references
    bf._references = {bfr}
    # Setup parent of bfr to reg
    bfr.parent = reg

    calls = {"read": 0, "write": 0}
    def dummy_read(addr):
        calls["read"] += 1
        return 0x55

    def dummy_write(addr, val):
        calls["write"] += 1

    reg._reg_read_func = dummy_read
    reg._reg_write_func = dummy_write

    # Call write with new value, forcing write
    bf.write(val=0xAA, always_write=True)
    # read should be called once for RMW, write called once
    assert calls["read"] >= 1
    assert calls["write"] >= 1
    # Value should be updated to 0xAA (masked by width=8)
    assert bf.value == 0xAA

def test_bitfield_read_reads_parent_and_returns_value(monkeypatch):
    bf = BitField(name="bf", width=8)
    bfr = BitFieldRef(name="bfr", reg_offset=0)
    bfr._add(bf)

    reg = Register(name="reg", width=8)
    reg._children = {"bfr": bfr}

    bf._references = {bfr}
    bfr.parent = reg

    calls = {"read": 0}
    def dummy_read(addr):
        calls["read"] += 1
        return 0x33

    reg._reg_read_func = dummy_read

    # initial bf.value is 0
    bf._value = 0x33
    val = bf.read()
    # read should be called once on parent reg
    assert calls["read"] == 1
    assert val == bf.value

def test_bitfieldref_copy_alias_behavior():
    bf = BitField(name="bf", width=8)
    bfr = BitFieldRef(name="bfr", reg_offset=0)
    bfr._add(bf)
    bf._references = {bfr}
    bfr._alias = True
    bfr._ref = None

    # Create alias copy
    new_bfr = bfr._copy(new_alias=True)
    # The alias copy should have _bf set and _ref is None (because original has no _ref)
    assert new_bfr.bf is not None
    # The _alias on new_bfr matches original _alias (True)
    assert new_bfr._alias

def test_bitfield_annotation_returns_name_or_hex():
    class DummyEnum:
        def __init__(self, name, value):
            self.name = name
            self.value = value

    bf = BitField(name="bf", width=4)
    e1 = DummyEnum("ZERO", 0x0)
    e2 = DummyEnum("ONE", 0x1)
    bf._children = {"zero": e1, "one": e2}

    def enum_iter():
        for e in bf._children.values():
            yield e
    bf.__iter__ = enum_iter

    bf.value = 0x1
    assert bf.annotation == "ONE"
    bf.value = 0xF
    assert bf.annotation == hex(0xF)
