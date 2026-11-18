import pytest
import random
from math import ceil
import r_map
from r_map.Register import Register
from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.RegisterMap import RegisterMap
from r_map.ValidationError import ValidationError

# Helpers for test classes that require minimal instantiation

class DummyBitField(BitField):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parent = None
        self._references = set()
        self.name = kwargs.get('name', 'dummybf')

class DummyRegister(Register):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._reg_read_func = kwargs.get('_reg_read_func', lambda addr: 1234)
        self._reg_write_func = kwargs.get('_reg_write_func', lambda addr, val: None)
        self.name = kwargs.get('name', 'dummyreg')
        self.address = kwargs.get('address', 0)
        self._children = {}
        self._children_ordered = []

    def __iter__(self):
        # Provide BitFieldRefs as children for iteration, sorted by reg_offset
        children = getattr(self, '_children_ordered', [])
        return iter(sorted(children, key=lambda c: c.reg_offset))

###########################
#        TESTS START      #
###########################

def test_register_initialization_and_str(monkeypatch):
    random.seed(0)
    r = Register(width=16, name='reg1', address=10)
    # monkeypatch _reg_read_func and _reg_write_func to avoid side effects
    monkeypatch.setattr(r, '_reg_read_func', lambda addr: 0xABCD)
    monkeypatch.setattr(r, '_reg_write_func', lambda addr, val: None)
    assert r.width == 16
    # str includes hex-formatted value (default 0)
    s = str(r)
    assert 'value' in s
    assert '0x0' in s

def test_register_value_property_set_and_get(monkeypatch):
    random.seed(0)
    bf1 = BitField(width=4, reset_val=0x3)
    bf2 = BitField(width=4, reset_val=0x5)
    # Setup BitFieldRefs for Register children
    bfr1 = BitFieldRef(reg_offset=0, slice_width=4, field_offset=0)
    bfr2 = BitFieldRef(reg_offset=4, slice_width=4, field_offset=0)
    bfr1._add(bf1)
    bfr2._add(bf2)
    reg = Register(width=8, name='r', address=1)
    # Attach children
    reg._children = {bf1.name: bfr1, bf2.name: bfr2}
    reg._children_ordered = [bfr1, bfr2]
    # Accessing value aggregates bitfields' values shifted appropriately
    # We expect reg.value to be bf1.value (lower 4 bits) | (bf2.value << 4)
    expected_val = (bf1.value << bfr1.reg_offset) | (bf2.value << bfr2.reg_offset)
    # Note: reg.value property returns sum of f.value for f in reg (f.value in BitFieldRef already includes reg_offset shifts)
    assert reg.value == expected_val
    # Setting reg.value sets underlying bitfield values accordingly
    reg.value = 0xAB
    # Each BitFieldRef shifts and masks properly
    expected_bf1_val = (0xAB >> bfr1.reg_offset) & bfr1.mask
    expected_bf2_val = (0xAB >> bfr2.reg_offset) & bfr2.mask
    assert bf1.value == expected_bf1_val
    assert bf2.value == expected_bf2_val
    # reg.value should now aggregate to same as set value masked to width
    assert reg.value == (expected_bf1_val << bfr1.reg_offset) | (expected_bf2_val << bfr2.reg_offset)

def test_register_reset_resets_all_bitfields(monkeypatch):
    random.seed(0)
    bf1 = BitField(width=8, reset_val=0xAA)
    bf2 = BitField(width=8, reset_val=0x55)
    bfr1 = BitFieldRef(reg_offset=0)
    bfr2 = BitFieldRef(reg_offset=8)
    bfr1._add(bf1)
    bfr2._add(bf2)
    reg = Register(width=16)
    reg._children = {bf1.name: bfr1, bf2.name: bfr2}
    reg._children_ordered = [bfr1, bfr2]
    # Set values different from reset_val
    bf1.value = 0
    bf2.value = 0
    reg.reset()
    # Values reset to reset_val
    assert bf1.value == bf1.reset_val
    assert bf2.value == bf2.reset_val
    # Reset val combined properly in reg.reset_val property
    combined_reset = (bf1.reset_val << bfr1.reg_offset) | (bf2.reset_val << bfr2.reg_offset)
    assert reg.reset_val == combined_reset

def test_register_access_property(monkeypatch):
    random.seed(0)
    # bf1 access RO, bf2 access WO; values should join with '|'; order sorted set
    bf1 = BitField(width=1, access='RO')
    bf2 = BitField(width=1, access='WO')
    bfr1 = BitFieldRef(reg_offset=0)
    bfr2 = BitFieldRef(reg_offset=1)
    bfr1._add(bf1)
    bfr2._add(bf2)
    reg = Register(width=2)
    reg._children = {bf1.name: bfr1, bf2.name: bfr2}
    reg._children_ordered = [bfr1, bfr2]
    access_str = reg.access
    # Should contain both 'RO' and 'WO' separated by '|', sorted sets produce deterministic order
    assert set(access_str.split('|')) == {'RO', 'WO'}

def test_register_validate_yields_errors_and_passes_for_multiple_cases(monkeypatch):
    random.seed(0)
    # Case 1: No children (no bitfieldrefs) should yield an error
    reg1 = Register(width=8)
    reg1._children = {}
    reg1._children_ordered = []
    errors = list(reg1.validate())
    assert any(isinstance(e, ValidationError) and "No bitfieldrefs" in str(e) for e in errors)
    # Case 2: one child not BitFieldRef yields an error
    reg2 = Register(width=8)
    reg2._children = {'child': BitField(width=1)}
    reg2._children_ordered = [BitField(width=1)]
    # forcibly add a child that is not BitFieldRef for iteration
    reg2._children_ordered = [r_map.BitField(width=1)]
    errors2 = list(reg2.validate())
    # Expect error about child not being BitFieldRef type
    assert any(isinstance(e, ValidationError) and "is not of type BitFieldRef" in str(e) for e in errors2)
    # Case 3: overlapping BitFieldRefs
    bf = BitField(width=4)
    bfr1 = BitFieldRef(reg_offset=0, slice_width=4)
    bfr2 = BitFieldRef(reg_offset=2, slice_width=4)
    bfr1._add(bf)
    bfr2._add(bf)
    reg3 = Register(width=8)
    reg3._children = {bf.name: bfr1, 'child2': bfr2}
    reg3._children_ordered = [bfr1, bfr2]
    errors3 = list(reg3.validate())
    assert any(isinstance(e, ValidationError) and "overlaps" in str(e) for e in errors3)

def test_register_read_and_write(monkeypatch):
    random.seed(0)
    address = 0x10
    read_val = 0xDEADBEEF
    written = {}
    def fake_read(addr):
        assert addr == address
        return read_val
    def fake_write(addr, val):
        assert addr == address
        written['val'] = val
    reg = Register(width=32, address=address)
    monkeypatch.setattr(reg, '_reg_read_func', fake_read)
    monkeypatch.setattr(reg, '_reg_write_func', fake_write)
    # Add one BitFieldRef child to avoid validation errors during value setting
    bf = BitField(width=32)
    bfr = BitFieldRef(reg_offset=0, slice_width=32)
    bfr._add(bf)
    reg._children = {bf.name: bfr}
    reg._children_ordered = [bfr]
    # Test read returns correct value and sets internal value
    val = reg.read()
    assert val == read_val
    assert reg.value == val
    # Test write sets value and calls write callback
    reg.write(0x12345678)
    assert written['val'] == 0x12345678
    # Test write with no argument uses current value
    reg.write()
    assert written['val'] == reg.value

def test_bitfield_initialization_error_on_invalid_params():
    random.seed(0)
    with pytest.raises(ValueError):
        BitField(width=0)
    with pytest.raises(ValueError):
        BitField(reset_val=-1)

def test_bitfield_value_setter_and_getter(monkeypatch):
    random.seed(0)
    bf = BitField(width=4, reset_val=0)
    # Normal int setting truncates properly with mask
    bf.value = 0xFF
    assert bf.value == 0xF  # 4 bits mask = 0xF
    # Setting with enumeration name fails without enumerations
    with pytest.raises(ValueError):
        bf.value = 'NON_EXISTENT_ENUM'
    # Setup dummy enumeration child to test setting by string
    class DummyEnum:
        def __init__(self, name, value):
            self.name = name
            self.value = value
        def __str__(self):
            return f"DummyEnum({self.name}={self.value})"
    en = DummyEnum('VALID_ENUM', 3)
    bf._children = {'valid': en}
    bf._children_ordered = [en]
    bf.value = 'VALID_ENUM'
    assert bf.value == 3
    # Setting value by BitField instance sets value from its value
    bf_enum = BitField(width=4)
    bf_enum.value = 5
    bf.value = bf_enum
    assert bf.value == 5
    # Setting value with something else raises NotImplementedError
    with pytest.raises(NotImplementedError):
        bf.value = 1.23

def test_bitfield_reset_and_annotation(monkeypatch):
    random.seed(0)
    bf = BitField(width=3, reset_val=5)
    # Setup children enumerations for annotation
    class EnumChild:
        def __init__(self, name, value):
            self.name = name
            self.value = value
        def __str__(self):
            return self.name
    enum1 = EnumChild('ZERO', 0)
    enum2 = EnumChild('FIVE', 5)
    enum3 = EnumChild('SEVEN', 7)
    bf._children = {'ZERO': enum1, 'FIVE': enum2, 'SEVEN': enum3}
    bf._children_ordered = [enum1, enum2, enum3]
    bf.value = 5
    # Annotation returns name matching current value
    assert bf.annotation == 'FIVE'
    bf.value = 1
    # No matching child, annotation returns hex string
    assert bf.annotation.startswith('0x')
    # Reset sets value to reset_val
    bf.value = 0
    bf.reset()
    assert bf.value == bf.reset_val

def test_bitfield_validate_duplicate_enumerations(monkeypatch):
    random.seed(0)
    bf = BitField(width=3)
    # Setup _children with duplicate values
    class EnumChild:
        def __init__(self, name, value):
            self.name = name
            self.value = value
        def __str__(self):
            return self.name
    enum_a1 = EnumChild('A', 1)
    enum_a2 = EnumChild('A2', 1)  # same value as enum_a1
    enum_b = EnumChild('B', 2)
    bf._children = {'A1': enum_a1, 'A2': enum_a2, 'B': enum_b}
    bf._children_ordered = [enum_a1, enum_a2, enum_b]
    errors = list(bf.validate())
    found_duplicate_error = any(isinstance(e, ValidationError) and "same value" in str(e) for e in errors)
    assert found_duplicate_error

def test_bitfield_write_read_calls_parent(monkeypatch):
    random.seed(0)
    # Create a register with read and write counters
    class DummyReg(Register):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._reg_read_func = lambda a: 7
            self._reg_write_func = lambda a, v: self._writes.append(v)
            self._writes = []
            self.address = 0
        def read(self):
            val = super().read()
            self._reads.append(True)
            return val
    reg = DummyReg()
    reg._reads = []
    bf = BitField(width=4)
    ref = BitFieldRef(reg_offset=0)
    ref._add(bf)
    reg._children = {bf.name: ref}
    reg._children_ordered = [ref]
    # Add reference from bf to parent (normally handled in add)
    bf._references = {ref}
    bf._parent = ref
    # Patch parent's read and write to record calls
    called = {'reads': 0, 'writes': 0}
    def _read():
        called['reads'] += 1
        return 10
    def _write():
        called['writes'] += 1
    reg.read = lambda: (_read() or 10)
    reg.write = lambda val=None: (_write() or None)
    # The BitField._get_parent_regs returns parents from _references
    # So set bf._references accordingly
    bf._references = {reg}
    # Call bf.write with always_write True => write always called
    bf.write(val=5, always_write=True)
    assert called['writes'] == 1
    # Call bf.write with same value and always_write False returns no write
    called['writes'] = 0
    # Setting bf.value to 6 first
    bf.value = 6
    bf.write(val=6, always_write=False)
    assert called['writes'] == 0
    # Call bf.read to trigger parents' read calls (which we simulate here)
    called['reads'] = 0
    bf.read()
    assert called['reads'] >= 1

def test_bitfieldref_init_and_value_property_errors(monkeypatch):
    random.seed(0)
    with pytest.raises(ValueError):
        BitFieldRef(reg_offset=0, slice_width=-1)
    with pytest.raises(ValueError):
        BitFieldRef(reg_offset=0, field_offset=-1)
    bf = BitField(width=8, reset_val=0xAA)
    bfr = BitFieldRef(reg_offset=1, slice_width=4, field_offset=2)
    bfr._add(bf)
    # value property gets and sets are masking and shifting
    bf.value = 0b10101010
    # Value property applied with shifts and masks
    val = bfr.value
    expected = ((bf.value >> bfr.field_offset) & bfr.mask) << bfr.reg_offset
    assert val == expected
    # Set via bfr.value setter
    old_val = bf.value
    bfr.value = (0b1111 << bfr.reg_offset)
    # Check modified bf.value matches expected bit masked and shifted version
    mask_shifted = bfr.mask << bfr.field_offset
    expected_val = (old_val & ~mask_shifted) | (( ( (0b1111 << bfr.reg_offset) >> bfr.reg_offset) & bfr.mask) << bfr.field_offset)
    assert bf.value == expected_val

def test_bitfieldref_add_duplicate_and_alias(monkeypatch):
    random.seed(0)
    bf = BitField(width=8)
    bfr1 = BitFieldRef(reg_offset=0, slice_width=8)
    bfr1._add(bf)
    # Adding same bf again does not raise error or duplicate children
    prev_len = len(bfr1._children)
    bfr1._add(bf)
    assert len(bfr1._children) == prev_len
    # Create second BitFieldRef referencing same bf to test alias logic
    bfr2 = BitFieldRef(reg_offset=0, slice_width=8)
    bfr2._add(bf)
    # The first bfr1 and bfr2 both reference same bf, one is alias
    # _alias flag set to True on bfr2 because multiple references
    assert getattr(bfr2, '_alias', False) is True
    # It keeps a _ref reference to one of the other bitfieldrefs
    assert hasattr(bfr2, '_ref') and bfr2._ref in bf._references

def test_bitfieldref_copy_behavior(monkeypatch):
    random.seed(0)
    bf = BitField(width=4)
    bfr = BitFieldRef(reg_offset=0, slice_width=4)
    bfr._add(bf)
    # Force alias to False and True to test behavioral branches
    # Normal copy (not alias)
    copy_obj = bfr._copy(new_instance=True)
    assert isinstance(copy_obj, BitFieldRef)
    # Alias copy sets new_alias True and triggers different behavior
    bfr._alias = True
    bfr._ref = bfr
    alias_copy = bfr._copy(new_instance=True, new_alias=True)
    assert isinstance(alias_copy, BitFieldRef)
    # _add should be called with _ref.bf
    # _alias True disables deep copying children

def test_registermap_block_dimensions(monkeypatch):
    random.seed(0)
    # Create RegisterMap with two DummyRegisters
    r1 = DummyRegister(address=0x10, width=16)
    r2 = DummyRegister(address=0x14, width=16)
    regmap = RegisterMap(name='testmap')
    regmap._children = {'r1': r1, 'r2': r2}
    regmap._children_ordered = [r1, r2]
    # Check that _block_dimensions property calculates correct start and size
    start, size = regmap._block_dimensions
    expected_start = r1.address
    expected_size = (r2.address + ceil(r2.width / 8)) - expected_start
    assert start == expected_start
    assert size == expected_size
    # Setter changes __block_dimensions attribute
    regmap._block_dimensions = (0, 10)
    assert regmap._block_dimensions == (0, 10)
    # Empty RegisterMap raises ValueError on _block_dimensions call
    empty_map = RegisterMap(name='empty')
    empty_map._children = {}
    empty_map._children_ordered = []
    with pytest.raises(ValueError):
        _ = empty_map._block_dimensions

def test_registermap_read_write(monkeypatch):
    random.seed(0)
    # Setup registers with addresses and known width
    r1 = DummyRegister(address=0x00, width=8)
    r2 = DummyRegister(address=0x01, width=8)
    r1.value = 0x12
    r2.value = 0x34
    regmap = RegisterMap(name='testmap')
    regmap._children = {'r1': r1, 'r2': r2}
    regmap._children_ordered = [r1, r2]
    # Setup _block_read_func and _block_write_func spies and data
    block_data = bytearray([0x12, 0x34])
    called = {'write_called': False, 'write_data': None}
    regmap._block_read_func = lambda addr, size: block_data
    regmap._block_write_func = lambda addr, data: (called.update(write_called=True, write_data=data))
    # Test read loads values from data
    data = regmap.read()
    assert data == block_data
    # Values of registers updated accordingly
    assert r1.value == 0x12
    assert r2.value == 0x34
    # Test write when data arg is None: assembles bytes from registers and writes
    regmap.write(data=None)
    assert called['write_called']
    # The data bytes written should match registers' values in order
    written_data = called['write_data']
    assert written_data[0] == r1.value
    assert written_data[1] == r2.value
    # Test write with user-supplied data fills register values accordingly
    update_data = bytearray([0xAA, 0xBB])
    regmap.write(data=update_data)
    assert r1.value == 0xAA
    assert r2.value == 0xBB
    assert called['write_called']

def test_registermap_write_with_no_registers_errors():
    random.seed(0)
    regmap = RegisterMap()
    regmap._children = {}
    regmap._children_ordered = []
    regmap._block_read_func = lambda addr, size: b'\x00'*size
    regmap._block_write_func = lambda addr, data: None
    with pytest.raises(ValueError):
        regmap.write()

