# humanize: public API (signatures and documentation)


## module `src/humanize/filesize.py`

```python
def naturalsize(value: float | str, binary: bool=False, gnu: bool=False, format: str='%.1f') -> str
```
Format a number of bytes like a human-readable filesize (e.g. 10 kB).

By default, decimal suffixes (kB, MB) are used.

Non-GNU modes are compatible with jinja2's `filesizeformat` filter.

Examples:
    ```pycon
    >>> naturalsize(3000000)
    '3.0 MB'
    >>> naturalsize(300, False, True)
    '300B'
    >>> naturalsize(3000, False, True)
    '2.9K'
    >>> naturalsize(3000, False, True, "%.3f")
    '2.930K'
    >>> naturalsize(3000, True)
    '2.9 KiB'
    >>> naturalsize(10**28)
    '10.0 RB'
    >>> naturalsize(10**34 * 3)
    '30000.0 QB'
    >>> naturalsize(-4096, True)
    '-4.0 KiB'

    ```

Args:
    value (int, float, str): Integer to convert.
    binary (bool): If `True`, uses bi ...


## module `src/humanize/i18n.py`

```python
def activate(locale: str | None, path: str | os.PathLike[str] | None=None) -> gettext_module.NullTranslations
```
Activate internationalisation.

Set `locale` as current locale. Search for locale in directory `path`.

Args:
    locale (str | None): Language name, e.g. `en_GB`. If `None`, defaults to no
        translation. Similar to calling ``deactivate()``.
    path (str | pathlib.Path): Path to search for locales.

Returns:
    dict: Translations.

Raises:
    FileNotFoundError: If humanize cannot find the locale folder.

```python
def deactivate() -> None
```
Deactivate internationalisation.

```python
def thousands_separator() -> str
```
Return the thousands separator for a locale, default to comma.

Returns:
     str: Thousands separator.

```python
def decimal_separator() -> str
```
Return the decimal separator for a locale, default to dot.

Returns:
     str: Decimal separator.


## module `src/humanize/lists.py`

```python
def natural_list(items: list[Any]) -> str
```
Natural list.

Convert a list of items into a human-readable string with commas and 'and'.

Examples:
    >>> natural_list(["one", "two", "three"])
    'one, two and three'
    >>> natural_list(["one", "two"])
    'one and two'
    >>> natural_list(["one"])
    'one'

Args:
    items (list): An iterable of items.

Returns:
    str: A string with commas and 'and' in the right places.


## module `src/humanize/number.py`

```python
def ordinal(value: NumberOrString, gender: str='male') -> str
```
Converts an integer to its ordinal as a string.

For example, 1 is "1st", 2 is "2nd", 3 is "3rd", etc. Works for any integer or
anything `int()` will turn into an integer. Anything else will return the output
of str(value).

Examples:
    ```pycon
    >>> ordinal(1)
    '1st'
    >>> ordinal(1002)
    '1002nd'
    >>> ordinal(103)
    '103rd'
    >>> ordinal(4)
    '4th'
    >>> ordinal(12)
    '12th'
    >>> ordinal(101)
    '101st'
    >>> ordinal(111)
    '111th'
    >>> ordinal("something else")
    'something else'
    >>> ordinal([1, 2, 3]) == "[1, 2, 3]"
    True

    ```

Args:
    value (int, str, float): Integer to convert.
    gender (str): Gender for translations. Accepts either ...

```python
def intcomma(value: NumberOrString, ndigits: int | None=None) -> str
```
Converts an integer to a string containing commas every three digits.

For example, 3000 becomes "3,000" and 45000 becomes "45,000". To maintain some
compatibility with Django's `intcomma`, this function also accepts floats.

Examples:
    ```pycon
    >>> intcomma(100)
    '100'
    >>> intcomma("1000")
    '1,000'
    >>> intcomma(1_000_000)
    '1,000,000'
    >>> intcomma(1_234_567.25)
    '1,234,567.25'
    >>> intcomma(1234.5454545, 2)
    '1,234.55'
    >>> intcomma(14308.40, 1)
    '14,308.4'
    >>> intcomma("14308.40", 1)
    '14,308.4'
    >>> intcomma(None)
    'None'

    ```

Args:
    value (int, float, str): Integer or float to convert.
    ndigits (int, None): Digits of prec ...

```python
def intword(value: NumberOrString, format: str='%.1f') -> str
```
Converts a large integer to a friendly text representation.

Works best for numbers over 1 million. For example, 1_000_000 becomes "1.0 million",
1_200_000 becomes "1.2 million" and "1_200_000_000" becomes "1.2 billion". Supports
up to decillion (33 digits) and googol (100 digits).

Examples:
    ```pycon
    >>> intword("100")
    '100'
    >>> intword("12400")
    '12.4 thousand'
    >>> intword("1000000")
    '1.0 million'
    >>> intword(1_200_000_000)
    '1.2 billion'
    >>> intword(8100000000000000000000000000000000)
    '8.1 decillion'
    >>> intword(None)
    'None'
    >>> intword("1234000", "%0.3f")
    '1.234 million'

    ```

Args:
    value (int, float, str): Integer to conv ...

```python
def apnumber(value: NumberOrString) -> str
```
Converts an integer to Associated Press style.

Examples:
  ```pycon
  >>> apnumber(0)
  'zero'
  >>> apnumber(5)
  'five'
  >>> apnumber(10)
  '10'
  >>> apnumber("7")
  'seven'
  >>> apnumber("foo")
  'foo'
  >>> apnumber(None)
  'None'

  ```

Args:
    value (int, float, str): Integer to convert.

Returns:
    str: For numbers 0-9, the number spelled out. Otherwise, the number. This always
        returns a string unless the value was not `int`-able, then `str(value)`
        is returned.

```python
def fractional(value: NumberOrString) -> str
```
Convert to fractional number.

There will be some cases where one might not want to show ugly decimal places for
floats and decimals.

This function returns a human-readable fractional number in form of fractions and
mixed fractions.

Pass in a string, or a number or a float, and this function returns:

* a string representation of a fraction
* or a whole number
* or a mixed fraction
* or the str output of the value, if it could not be converted

Examples:
    ```pycon
    >>> fractional(0.3)
    '3/10'
    >>> fractional(1.3)
    '1 3/10'
    >>> fractional(float(1/3))
    '1/3'
    >>> fractional(1)
    '1'
    >>> fractional("ten")
    'ten'
    >>> fractional(None)
    'None'

    ```

A ...

```python
def scientific(value: NumberOrString, precision: int=2) -> str
```
Return number in string scientific notation z.wq x 10ⁿ.

Examples:
    ```pycon
    >>> scientific(float(0.3))
    '3.00 x 10⁻¹'
    >>> scientific(int(500))
    '5.00 x 10²'
    >>> scientific(-1000)
    '-1.00 x 10³'
    >>> scientific(1000, 1)
    '1.0 x 10³'
    >>> scientific(1000, 3)
    '1.000 x 10³'
    >>> scientific("99")
    '9.90 x 10¹'
    >>> scientific("foo")
    'foo'
    >>> scientific(None)
    'None'

    ```

Args:
    value (int, float, str): Input number.
    precision (int): Number of decimal for first part of the number.

Returns:
    str: Number in scientific notation z.wq x 10ⁿ.

```python
def clamp(value: float, format: str='{:}', floor: float | None=None, ceil: float | None=None, floor_token: str='<', ceil_token: str='>') -> str
```
Returns number with the specified format, clamped between floor and ceil.

If the number is larger than ceil or smaller than floor, then the respective limit
will be returned, formatted and prepended with a token specifying as such.

Examples:
    ```pycon
    >>> clamp(123.456)
    '123.456'
    >>> clamp(0.0001, floor=0.01)
    '<0.01'
    >>> clamp(0.99, format="{:.0%}", ceil=0.99)
    '99%'
    >>> clamp(0.999, format="{:.0%}", ceil=0.99)
    '>99%'
    >>> clamp(1, format=intword, floor=1e6, floor_token="under ")
    'under 1.0 million'
    >>> clamp(None) is None
    True

    ```

Args:
    value (int, float): Input number.
    format (str OR callable): Can either be a formatting stri ...

```python
def metric(value: float, unit: str='', precision: int=3) -> str
```
Return a value with a metric SI unit-prefix appended.

Examples:
    ```pycon
    >>> metric(1500, "V")
    '1.50 kV'
    >>> metric(2e8, "W")
    '200 MW'
    >>> metric(220e-6, "F")
    '220 μF'
    >>> metric(1e-14, precision=4)
    '10.00 f'

    ```

The unit prefix is always chosen so that non-significant zero digits are required.
i.e. `123,000` will become `123k` instead of `0.123M` and `1,230,000` will become
`1.23M` instead of `1230K`. For numbers that are either too huge or too tiny to
represent without resorting to either leading or trailing zeroes, it falls back to
`scientific()`.
```pycon
>>> metric(1e40)
'1.00 x 10⁴⁰'

```

Args:
    value (int, float): Input number.
    unit ( ...


## module `src/humanize/time.py`

```python
def naturaldelta(value: dt.timedelta | float, months: bool=True, minimum_unit: str='seconds') -> str
```
Return a natural representation of a timedelta or number of seconds.

This is similar to `naturaltime`, but does not add tense to the result.

The timedelta will be rounded to the nearest unit that makes sense.

Args:
    value (datetime.timedelta, int or float): A timedelta or a number of seconds.
    months (bool): If `True`, then a number of months (based on 30.5 days) will be
        used for fuzziness between years.
    minimum_unit (str): The lowest unit that can be used.

Returns:
    str (str or `value`): A natural representation of the amount of time
        elapsed unless `value` is not datetime.timedelta or cannot be
        converted to int (cannot be float due to 'inf' or 'nan') ...

```python
def naturaltime(value: dt.datetime | dt.timedelta | float, future: bool=False, months: bool=True, minimum_unit: str='seconds', when: dt.datetime | None=None) -> str
```
Return a natural representation of a time in a resolution that makes sense.

This is more or less compatible with Django's `naturaltime` filter.

The time will be rounded to the nearest unit that makes sense.

Args:
    value (datetime.datetime, datetime.timedelta, int or float): A `datetime`, a
        `timedelta`, or a number of seconds.
    future (bool): Ignored for `datetime`s and `timedelta`s, where the tense is
        always figured out based on the current time. For integers and floats, the
        return value will be past tense by default, unless future is `True`.
    months (bool): If `True`, then a number of months (based on 30.5 days) will be
        used for fuzziness between ...

```python
def naturalday(value: dt.date | dt.datetime, format: str='%b %d') -> str
```
Return a natural day.

For date values that are tomorrow, today or yesterday compared to
present day return representing string. Otherwise, return a string
formatted according to `format`.

```python
def naturaldate(value: dt.date | dt.datetime) -> str
```
Like `naturalday`, but append a year for dates more than ~five months away.

```python
def precisedelta(value: dt.timedelta | float | None, minimum_unit: str='seconds', suppress: Iterable[str]=(), format: str='%0.2f') -> str
```
Return a precise representation of a timedelta or number of seconds.

```pycon
>>> import datetime as dt
>>> from humanize.time import precisedelta

>>> delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
>>> precisedelta(delta)
'2 days, 1 hour and 33.12 seconds'

```

A custom `format` can be specified to control how the fractional part
is represented:

```pycon
>>> precisedelta(delta, format="%0.4f")
'2 days, 1 hour and 33.1230 seconds'

```

Instead, the `minimum_unit` can be changed to have a better resolution;
the function will still readjust the unit to use the greatest of the
units that does not lose precision.

For example setting microseconds but still representing the d ...
