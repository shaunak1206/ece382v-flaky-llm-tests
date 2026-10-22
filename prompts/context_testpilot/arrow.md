# arrow: public API (signatures and documentation)


## module `arrow/api.py`

```python
def get(*, locale: str=DEFAULT_LOCALE, tzinfo: Optional[TZ_EXPR]=None, normalize_whitespace: bool=False) -> Arrow
```


```python
def get(*args: int, locale: str=DEFAULT_LOCALE, tzinfo: Optional[TZ_EXPR]=None, normalize_whitespace: bool=False) -> Arrow
```


```python
def get(__obj: Union[Arrow, datetime, date, struct_time, dt_tzinfo, int, float, str, Tuple[int, int, int]], *, locale: str=DEFAULT_LOCALE, tzinfo: Optional[TZ_EXPR]=None, normalize_whitespace: bool=False) -> Arrow
```


```python
def get(__arg1: Union[datetime, date], __arg2: TZ_EXPR, *, locale: str=DEFAULT_LOCALE, tzinfo: Optional[TZ_EXPR]=None, normalize_whitespace: bool=False) -> Arrow
```


```python
def get(__arg1: str, __arg2: Union[str, List[str]], *, locale: str=DEFAULT_LOCALE, tzinfo: Optional[TZ_EXPR]=None, normalize_whitespace: bool=False) -> Arrow
```


```python
def get(*args: Any, **kwargs: Any) -> Arrow
```
Calls the default :class:`ArrowFactory <arrow.factory.ArrowFactory>` ``get`` method.

```python
def utcnow() -> Arrow
```
Calls the default :class:`ArrowFactory <arrow.factory.ArrowFactory>` ``utcnow`` method.

```python
def now(tz: Optional[TZ_EXPR]=None) -> Arrow
```
Calls the default :class:`ArrowFactory <arrow.factory.ArrowFactory>` ``now`` method.

```python
def factory(type: Type[Arrow]) -> ArrowFactory
```
Returns an :class:`.ArrowFactory` for the specified :class:`Arrow <arrow.arrow.Arrow>`
or derived type.

:param type: the type, :class:`Arrow <arrow.arrow.Arrow>` or derived.


## module `arrow/factory.py`

```python
class ArrowFactory()
```
A factory for generating :class:`Arrow <arrow.arrow.Arrow>` objects.

:param type: (optional) the :class:`Arrow <arrow.arrow.Arrow>`-based class to construct from.
    Defaults to :class:`Arrow <arrow.arrow.Arrow>`.

```python
# method of ArrowFactory
def __init__(self, type: Type[Arrow]=Arrow) -> None
```


```python
# method of ArrowFactory
def get(self, *, locale: str=DEFAULT_LOCALE, tzinfo: Optional[TZ_EXPR]=None, normalize_whitespace: bool=False) -> Arrow
```


```python
# method of ArrowFactory
def get(self, __obj: Union[Arrow, datetime, date, struct_time, dt_tzinfo, int, float, str, Tuple[int, int, int]], *, locale: str=DEFAULT_LOCALE, tzinfo: Optional[TZ_EXPR]=None, normalize_whitespace: bool=False) -> Arrow
```


```python
# method of ArrowFactory
def get(self, __arg1: Union[datetime, date], __arg2: TZ_EXPR, *, locale: str=DEFAULT_LOCALE, tzinfo: Optional[TZ_EXPR]=None, normalize_whitespace: bool=False) -> Arrow
```


```python
# method of ArrowFactory
def get(self, __arg1: str, __arg2: Union[str, List[str]], *, locale: str=DEFAULT_LOCALE, tzinfo: Optional[TZ_EXPR]=None, normalize_whitespace: bool=False) -> Arrow
```


```python
# method of ArrowFactory
def get(self, *args: Any, **kwargs: Any) -> Arrow
```
Returns an :class:`Arrow <arrow.arrow.Arrow>` object based on flexible inputs.

:param locale: (optional) a ``str`` specifying a locale for the parser. Defaults to 'en-us'.
:param tzinfo: (optional) a :ref:`timezone expression <tz-expr>` or tzinfo object.
    Replaces the timezone unless using an input form that is explicitly UTC or specifies
    the timezone in a positional argument. Defaults to ...

```python
# method of ArrowFactory
def utcnow(self) -> Arrow
```
Returns an :class:`Arrow <arrow.arrow.Arrow>` object, representing "now" in UTC time.

Usage::

    >>> import arrow
    >>> arrow.utcnow()
    <Arrow [2013-05-08T05:19:07.018993+00:00]>

```python
# method of ArrowFactory
def now(self, tz: Optional[TZ_EXPR]=None) -> Arrow
```
Returns an :class:`Arrow <arrow.arrow.Arrow>` object, representing "now" in the given
timezone.

:param tz: (optional) A :ref:`timezone expression <tz-expr>`.  Defaults to local time.

Usage::

    >>> import arrow
    >>> arrow.now()
    <Arrow [2013-05-07T22:19:11.363410-07:00]>

    >>> arrow.now('US/Pacific')
    <Arrow [2013-05-07T22:19:15.251821-07:00]>

    >>> arrow.now('+02:00')
    <Arro ...


## module `arrow/arrow.py`

```python
class Arrow()
```
An :class:`Arrow <arrow.arrow.Arrow>` object.

Implements the ``datetime`` interface, behaving as an aware ``datetime`` while implementing
additional functionality.

:param year: the calendar year.
:param month: the calendar month.
:param day: the calendar day.
:param hour: (optional) the hour. Defaults to 0.
:param minute: (optional) the minute, Defaults to 0.
:param second: (optional) the second, Defaults to 0.
:param microsecond: (optional) the microsecond. Defaults to 0.
:param tzinfo: (optional) A timezone expression.  Defaults to UTC.
:param fold: (optional) 0 or 1, used to disambiguate repeated wall times. Defaults to 0.

.. _tz-expr:

Recognized timezone expressions:

    - A ``tzinf ...

```python
# method of Arrow
def __init__(self, year: int, month: int, day: int, hour: int=0, minute: int=0, second: int=0, microsecond: int=0, tzinfo: Optional[TZ_EXPR]=None, **kwargs: Any) -> None
```


```python
# method of Arrow
def now(cls, tzinfo: Optional[dt_tzinfo]=None) -> 'Arrow'
```
Constructs an :class:`Arrow <arrow.arrow.Arrow>` object, representing "now" in the given
timezone.

:param tzinfo: (optional) a ``tzinfo`` object. Defaults to local time.

Usage::

    >>> arrow.now('Asia/Baku')
    <Arrow [2019-01-24T20:26:31.146412+04:00]>

```python
# method of Arrow
def utcnow(cls) -> 'Arrow'
```
Constructs an :class:`Arrow <arrow.arrow.Arrow>` object, representing "now" in UTC
time.

Usage::

    >>> arrow.utcnow()
    <Arrow [2019-01-24T16:31:40.651108+00:00]>

```python
# method of Arrow
def fromtimestamp(cls, timestamp: Union[int, float, str], tzinfo: Optional[TZ_EXPR]=None) -> 'Arrow'
```
Constructs an :class:`Arrow <arrow.arrow.Arrow>` object from a timestamp, converted to
the given timezone.

:param timestamp: an ``int`` or ``float`` timestamp, or a ``str`` that converts to either.
:param tzinfo: (optional) a ``tzinfo`` object.  Defaults to local time.

```python
# method of Arrow
def utcfromtimestamp(cls, timestamp: Union[int, float, str]) -> 'Arrow'
```
Constructs an :class:`Arrow <arrow.arrow.Arrow>` object from a timestamp, in UTC time.

:param timestamp: an ``int`` or ``float`` timestamp, or a ``str`` that converts to either.

```python
# method of Arrow
def fromdatetime(cls, dt: dt_datetime, tzinfo: Optional[TZ_EXPR]=None) -> 'Arrow'
```
Constructs an :class:`Arrow <arrow.arrow.Arrow>` object from a ``datetime`` and
optional replacement timezone.

:param dt: the ``datetime``
:param tzinfo: (optional) A :ref:`timezone expression <tz-expr>`.  Defaults to ``dt``'s
    timezone, or UTC if naive.

Usage::

    >>> dt
    datetime.datetime(2021, 4, 7, 13, 48, tzinfo=tzfile('/usr/share/zoneinfo/US/Pacific'))
    >>> arrow.Arrow.fromdatet ...

```python
# method of Arrow
def fromdate(cls, date: dt_date, tzinfo: Optional[TZ_EXPR]=None) -> 'Arrow'
```
Constructs an :class:`Arrow <arrow.arrow.Arrow>` object from a ``date`` and optional
replacement timezone.  All time values are set to 0.

:param date: the ``date``
:param tzinfo: (optional) A :ref:`timezone expression <tz-expr>`.  Defaults to UTC.

```python
# method of Arrow
def strptime(cls, date_str: str, fmt: str, tzinfo: Optional[TZ_EXPR]=None) -> 'Arrow'
```
Constructs an :class:`Arrow <arrow.arrow.Arrow>` object from a date string and format,
in the style of ``datetime.strptime``.  Optionally replaces the parsed timezone.

:param date_str: the date string.
:param fmt: the format string using datetime format codes.
:param tzinfo: (optional) A :ref:`timezone expression <tz-expr>`.  Defaults to the parsed
    timezone if ``fmt`` contains a timezone dire ...

```python
# method of Arrow
def fromordinal(cls, ordinal: int) -> 'Arrow'
```
Constructs an :class:`Arrow <arrow.arrow.Arrow>` object corresponding
    to the Gregorian Ordinal.

:param ordinal: an ``int`` corresponding to a Gregorian Ordinal.

Usage::

    >>> arrow.fromordinal(737741)
    <Arrow [2020-11-12T00:00:00+00:00]>

```python
# method of Arrow
def range(cls, frame: _T_FRAMES, start: Union['Arrow', dt_datetime], end: Union['Arrow', dt_datetime, None]=None, tz: Optional[TZ_EXPR]=None, limit: Optional[int]=None) -> Generator['Arrow', None, None]
```
Returns an iterator of :class:`Arrow <arrow.arrow.Arrow>` objects, representing
points in time between two inputs.

:param frame: The timeframe.  Can be any ``datetime`` property (day, hour, minute...).
:param start: A datetime expression, the start of the range.
:param end: (optional) A datetime expression, the end of the range.
:param tz: (optional) A :ref:`timezone expression <tz-expr>`.  Defau ...

```python
# method of Arrow
def span(self, frame: _T_FRAMES, count: int=1, bounds: _BOUNDS='[)', exact: bool=False, week_start: int=1) -> Tuple['Arrow', 'Arrow']
```
Returns a tuple of two new :class:`Arrow <arrow.arrow.Arrow>` objects, representing the timespan
of the :class:`Arrow <arrow.arrow.Arrow>` object in a given timeframe.

:param frame: the timeframe.  Can be any ``datetime`` property (day, hour, minute...).
:param count: (optional) the number of frames to span.
:param bounds: (optional) a ``str`` of either '()', '(]', '[)', or '[]' that specifies ...

```python
# method of Arrow
def floor(self, frame: _T_FRAMES, **kwargs: Any) -> 'Arrow'
```
Returns a new :class:`Arrow <arrow.arrow.Arrow>` object, representing the "floor"
of the timespan of the :class:`Arrow <arrow.arrow.Arrow>` object in a given timeframe.
Equivalent to the first element in the 2-tuple returned by
:func:`span <arrow.arrow.Arrow.span>`.

:param frame: the timeframe.  Can be any ``datetime`` property (day, hour, minute...).
:param week_start: (optional) only used in co ...

```python
# method of Arrow
def ceil(self, frame: _T_FRAMES, **kwargs: Any) -> 'Arrow'
```
Returns a new :class:`Arrow <arrow.arrow.Arrow>` object, representing the "ceiling"
of the timespan of the :class:`Arrow <arrow.arrow.Arrow>` object in a given timeframe.
Equivalent to the second element in the 2-tuple returned by
:func:`span <arrow.arrow.Arrow.span>`.

:param frame: the timeframe.  Can be any ``datetime`` property (day, hour, minute...).
:param week_start: (optional) only used in ...

```python
# method of Arrow
def span_range(cls, frame: _T_FRAMES, start: dt_datetime, end: dt_datetime, tz: Optional[TZ_EXPR]=None, limit: Optional[int]=None, bounds: _BOUNDS='[)', exact: bool=False) -> Iterable[Tuple['Arrow', 'Arrow']]
```
Returns an iterator of tuples, each :class:`Arrow <arrow.arrow.Arrow>` objects,
representing a series of timespans between two inputs.

:param frame: The timeframe.  Can be any ``datetime`` property (day, hour, minute...).
:param start: A datetime expression, the start of the range.
:param end: (optional) A datetime expression, the end of the range.
:param tz: (optional) A :ref:`timezone expressio ...

```python
# method of Arrow
def interval(cls, frame: _T_FRAMES, start: dt_datetime, end: dt_datetime, interval: int=1, tz: Optional[TZ_EXPR]=None, bounds: _BOUNDS='[)', exact: bool=False) -> Iterable[Tuple['Arrow', 'Arrow']]
```
Returns an iterator of tuples, each :class:`Arrow <arrow.arrow.Arrow>` objects,
representing a series of intervals between two inputs.

:param frame: The timeframe.  Can be any ``datetime`` property (day, hour, minute...).
:param start: A datetime expression, the start of the range.
:param end: (optional) A datetime expression, the end of the range.
:param interval: (optional) Time interval for th ...

```python
# method of Arrow
def tzinfo(self) -> dt_tzinfo
```
Gets the ``tzinfo`` of the :class:`Arrow <arrow.arrow.Arrow>` object.

Usage::

    >>> arw=arrow.utcnow()
    >>> arw.tzinfo
    tzutc()

```python
# method of Arrow
def datetime(self) -> dt_datetime
```
Returns a datetime representation of the :class:`Arrow <arrow.arrow.Arrow>` object.

Usage::

    >>> arw=arrow.utcnow()
    >>> arw.datetime
    datetime.datetime(2019, 1, 24, 16, 35, 27, 276649, tzinfo=tzutc())

```python
# method of Arrow
def naive(self) -> dt_datetime
```
Returns a naive datetime representation of the :class:`Arrow <arrow.arrow.Arrow>`
object.

Usage::

    >>> nairobi = arrow.now('Africa/Nairobi')
    >>> nairobi
    <Arrow [2019-01-23T19:27:12.297999+03:00]>
    >>> nairobi.naive
    datetime.datetime(2019, 1, 23, 19, 27, 12, 297999)

```python
# method of Arrow
def timestamp(self) -> float
```
Returns a timestamp representation of the :class:`Arrow <arrow.arrow.Arrow>` object, in
UTC time.

Usage::

    >>> arrow.utcnow().timestamp()
    1616882340.256501

```python
# method of Arrow
def int_timestamp(self) -> int
```
Returns an integer timestamp representation of the :class:`Arrow <arrow.arrow.Arrow>` object, in
UTC time.

Usage::

    >>> arrow.utcnow().int_timestamp
    1548260567

```python
# method of Arrow
def float_timestamp(self) -> float
```
Returns a floating-point timestamp representation of the :class:`Arrow <arrow.arrow.Arrow>`
object, in UTC time.

Usage::

    >>> arrow.utcnow().float_timestamp
    1548260516.830896

```python
# method of Arrow
def fold(self) -> int
```
Returns the ``fold`` value of the :class:`Arrow <arrow.arrow.Arrow>` object.

```python
# method of Arrow
def ambiguous(self) -> bool
```
Indicates whether the :class:`Arrow <arrow.arrow.Arrow>` object is a repeated wall time in the current
timezone.

```python
# method of Arrow
def imaginary(self) -> bool
```
Indicates whether the :class: `Arrow <arrow.arrow.Arrow>` object exists in the current timezone.

```python
# method of Arrow
def clone(self) -> 'Arrow'
```
Returns a new :class:`Arrow <arrow.arrow.Arrow>` object, cloned from the current one.

Usage:

    >>> arw = arrow.utcnow()
    >>> cloned = arw.clone()

```python
# method of Arrow
def replace(self, **kwargs: Any) -> 'Arrow'
```
Returns a new :class:`Arrow <arrow.arrow.Arrow>` object with attributes updated
according to inputs.

Use property names to set their value absolutely::

    >>> import arrow
    >>> arw = arrow.utcnow()
    >>> arw
    <Arrow [2013-05-11T22:27:34.787885+00:00]>
    >>> arw.replace(year=2014, month=6)
    <Arrow [2014-06-11T22:27:34.787885+00:00]>

You can also replace the timezone without convers ...

```python
# method of Arrow
def shift(self, check_imaginary: bool=True, **kwargs: Any) -> 'Arrow'
```
Returns a new :class:`Arrow <arrow.arrow.Arrow>` object with attributes updated
according to inputs.

Parameters:
check_imaginary (bool): If True (default), will check for and resolve
imaginary times (like during DST transitions). If False, skips this check.


Use pluralized property names to relatively shift their current value:

>>> import arrow
>>> arw = arrow.utcnow()
>>> arw
<Arrow [2013-05-1 ...

```python
# method of Arrow
def to(self, tz: TZ_EXPR) -> 'Arrow'
```
Returns a new :class:`Arrow <arrow.arrow.Arrow>` object, converted
to the target timezone.

:param tz: A :ref:`timezone expression <tz-expr>`.

Usage::

    >>> utc = arrow.utcnow()
    >>> utc
    <Arrow [2013-05-09T03:49:12.311072+00:00]>

    >>> utc.to('US/Pacific')
    <Arrow [2013-05-08T20:49:12.311072-07:00]>

    >>> utc.to(tz.tzlocal())
    <Arrow [2013-05-08T20:49:12.311072-07:00]> ...

```python
# method of Arrow
def format(self, fmt: str='YYYY-MM-DD HH:mm:ssZZ', locale: str=DEFAULT_LOCALE) -> str
```
Returns a string representation of the :class:`Arrow <arrow.arrow.Arrow>` object,
formatted according to the provided format string. For a list of formatting values,
see :ref:`supported-tokens`

:param fmt: the format string.
:param locale: the locale to format.

Usage::

    >>> arrow.utcnow().format('YYYY-MM-DD HH:mm:ss ZZ')
    '2013-05-09 03:56:47 -00:00'

    >>> arrow.utcnow().format('X') ...

```python
# method of Arrow
def humanize(self, other: Union['Arrow', dt_datetime, None]=None, locale: str=DEFAULT_LOCALE, only_distance: bool=False, granularity: Union[_GRANULARITY, List[_GRANULARITY]]='auto') -> str
```
Returns a localized, humanized representation of a relative difference in time.

:param other: (optional) an :class:`Arrow <arrow.arrow.Arrow>` or ``datetime`` object.
    Defaults to now in the current :class:`Arrow <arrow.arrow.Arrow>` object's timezone.
:param locale: (optional) a ``str`` specifying a locale.  Defaults to 'en-us'.
:param only_distance: (optional) returns only time difference eg ...

```python
# method of Arrow
def dehumanize(self, input_string: str, locale: str='en_us') -> 'Arrow'
```
Returns a new :class:`Arrow <arrow.arrow.Arrow>` object, that represents
the time difference relative to the attributes of the
:class:`Arrow <arrow.arrow.Arrow>` object.

:param timestring: a ``str`` representing a humanized relative time.
:param locale: (optional) a ``str`` specifying a locale.  Defaults to 'en-us'.

Usage::

        >>> arw = arrow.utcnow()
        >>> arw
        <Arrow [2021-0 ...

```python
# method of Arrow
def is_between(self, start: 'Arrow', end: 'Arrow', bounds: _BOUNDS='()') -> bool
```
Returns a boolean denoting whether the :class:`Arrow <arrow.arrow.Arrow>` object is between
the start and end limits.

:param start: an :class:`Arrow <arrow.arrow.Arrow>` object.
:param end: an :class:`Arrow <arrow.arrow.Arrow>` object.
:param bounds: (optional) a ``str`` of either '()', '(]', '[)', or '[]' that specifies
    whether to include or exclude the start and end values in the range. '(' ...

```python
# method of Arrow
def date(self) -> dt_date
```
Returns a ``date`` object with the same year, month and day.

Usage::

    >>> arrow.utcnow().date()
    datetime.date(2019, 1, 23)

```python
# method of Arrow
def time(self) -> dt_time
```
Returns a ``time`` object with the same hour, minute, second, microsecond.

Usage::

    >>> arrow.utcnow().time()
    datetime.time(12, 15, 34, 68352)

```python
# method of Arrow
def timetz(self) -> dt_time
```
Returns a ``time`` object with the same hour, minute, second, microsecond and
tzinfo.

Usage::

    >>> arrow.utcnow().timetz()
    datetime.time(12, 5, 18, 298893, tzinfo=tzutc())

```python
# method of Arrow
def astimezone(self, tz: Optional[dt_tzinfo]) -> dt_datetime
```
Returns a ``datetime`` object, converted to the specified timezone.

:param tz: a ``tzinfo`` object.

Usage::

    >>> pacific=arrow.now('US/Pacific')
    >>> nyc=arrow.now('America/New_York').tzinfo
    >>> pacific.astimezone(nyc)
    datetime.datetime(2019, 1, 20, 10, 24, 22, 328172, tzinfo=tzfile('/usr/share/zoneinfo/America/New_York'))

```python
# method of Arrow
def utcoffset(self) -> Optional[timedelta]
```
Returns a ``timedelta`` object representing the whole number of minutes difference from
UTC time.

Usage::

    >>> arrow.now('US/Pacific').utcoffset()
    datetime.timedelta(-1, 57600)

```python
# method of Arrow
def dst(self) -> Optional[timedelta]
```
Returns the daylight savings time adjustment.

Usage::

    >>> arrow.utcnow().dst()
    datetime.timedelta(0)

```python
# method of Arrow
def timetuple(self) -> struct_time
```
Returns a ``time.struct_time``, in the current timezone.

Usage::

    >>> arrow.utcnow().timetuple()
    time.struct_time(tm_year=2019, tm_mon=1, tm_mday=20, tm_hour=15, tm_min=17, tm_sec=8, tm_wday=6, tm_yday=20, tm_isdst=0)

```python
# method of Arrow
def utctimetuple(self) -> struct_time
```
Returns a ``time.struct_time``, in UTC time.

Usage::

    >>> arrow.utcnow().utctimetuple()
    time.struct_time(tm_year=2019, tm_mon=1, tm_mday=19, tm_hour=21, tm_min=41, tm_sec=7, tm_wday=5, tm_yday=19, tm_isdst=0)

```python
# method of Arrow
def toordinal(self) -> int
```
Returns the proleptic Gregorian ordinal of the date.

Usage::

    >>> arrow.utcnow().toordinal()
    737078

```python
# method of Arrow
def weekday(self) -> int
```
Returns the day of the week as an integer (0-6).

Usage::

    >>> arrow.utcnow().weekday()
    5

```python
# method of Arrow
def isoweekday(self) -> int
```
Returns the ISO day of the week as an integer (1-7).

Usage::

    >>> arrow.utcnow().isoweekday()
    6

```python
# method of Arrow
def isocalendar(self) -> Tuple[int, int, int]
```
Returns a 3-tuple, (ISO year, ISO week number, ISO weekday).

Usage::

    >>> arrow.utcnow().isocalendar()
    (2019, 3, 6)

```python
# method of Arrow
def isoformat(self, sep: str='T', timespec: str='auto') -> str
```
Returns an ISO 8601 formatted representation of the date and time.

Usage::

    >>> arrow.utcnow().isoformat()
    '2019-01-19T18:30:52.442118+00:00'

```python
# method of Arrow
def ctime(self) -> str
```
Returns a ctime formatted representation of the date and time.

Usage::

    >>> arrow.utcnow().ctime()
    'Sat Jan 19 18:26:50 2019'

```python
# method of Arrow
def strftime(self, format: str) -> str
```
Formats in the style of ``datetime.strftime``.

:param format: the format string.

Usage::

    >>> arrow.utcnow().strftime('%d-%m-%Y %H:%M:%S')
    '23-01-2019 12:28:17'

```python
# method of Arrow
def for_json(self) -> str
```
Serializes for the ``for_json`` protocol of simplejson.

Usage::

    >>> arrow.utcnow().for_json()
    '2019-01-19T18:25:36.760079+00:00'


## module `arrow/parser.py`

```python
class ParserError(ValueError)
```
A custom exception class for handling parsing errors in the parser.

Notes:
    This class inherits from the built-in `ValueError` class and is used to raise exceptions
    when an error occurs during the parsing process.

```python
class ParserMatchError(ParserError)
```
This class is a subclass of the ParserError class and is used to raise errors that occur during the matching process.

Notes:
    This class is part of the Arrow parser and is used to provide error handling when a parsing match fails.

```python
class DateTimeParser()
```
A :class:`DateTimeParser <arrow.arrow.parser>` object

Contains the regular expressions and functions to parse and split the input strings into tokens and eventually
produce a datetime that is used by :class:`Arrow <arrow.arrow.Arrow>` internally.

:param locale: the locale string
:param cache_size: the size of the LRU cache used for regular expressions. Defaults to 0.

```python
# method of DateTimeParser
def __init__(self, locale: str=DEFAULT_LOCALE, cache_size: int=0) -> None
```
Contains the regular expressions and functions to parse and split the input strings into tokens and eventually
produce a datetime that is used by :class:`Arrow <arrow.arrow.Arrow>` internally.

:param locale: the locale string
:type locale: str
:param cache_size: the size of the LRU cache used for regular expressions. Defaults to 0.
:type cache_size: int

```python
# method of DateTimeParser
def parse_iso(self, datetime_string: str, normalize_whitespace: bool=False) -> datetime
```
Parses a datetime string using a ISO 8601-like format.

:param datetime_string: The datetime string to parse.
:param normalize_whitespace: Whether to normalize whitespace in the datetime string (default is False).
:type datetime_string: str
:type normalize_whitespace: bool
:returns: The parsed datetime object.
:rtype: datetime
:raises ParserError: If the datetime string is not in a valid ISO 8601- ...

```python
# method of DateTimeParser
def parse(self, datetime_string: str, fmt: Union[List[str], str], normalize_whitespace: bool=False) -> datetime
```
Parses a datetime string using a specified format.

:param datetime_string: The datetime string to parse.
:param fmt: The format string or list of format strings to use for parsing.
:param normalize_whitespace: Whether to normalize whitespace in the datetime string (default is False).
:type datetime_string: str
:type fmt: Union[List[str], str]
:type normalize_whitespace: bool
:returns: The parsed ...

```python
class TzinfoParser()
```
Parser for timezone information.

```python
# method of TzinfoParser
def parse(cls, tzinfo_string: str) -> dt_tzinfo
```
Parse a timezone string and return a datetime timezone object.

:param tzinfo_string: The timezone string to parse.
:type tzinfo_string: str
:returns: The parsed datetime timezone object.
:rtype: datetime.timezone
:raises ParserError: If the timezone string cannot be parsed.


## module `arrow/formatter.py`

```python
class DateTimeFormatter()
```


```python
# method of DateTimeFormatter
def __init__(self, locale: str=DEFAULT_LOCALE) -> None
```


```python
# method of DateTimeFormatter
def format(cls, dt: datetime, fmt: str) -> str
```



# Usage examples from the project README

```python
>>> import arrow
>>> arrow.get('2013-05-11T21:23:58.970460+07:00')
<Arrow [2013-05-11T21:23:58.970460+07:00]>

>>> utc = arrow.utcnow()
>>> utc
<Arrow [2013-05-11T21:23:58.970460+00:00]>

>>> utc = utc.shift(hours=-1)
>>> utc
<Arrow [2013-05-11T20:23:58.970460+00:00]>

>>> local = utc.to('US/Pacific')
>>> local
<Arrow [2013-05-11T13:23:58.970460-07:00]>

>>> local.timestamp()
1368303838.970460

>>> local.format()
'2013-05-11 13:23:58 -07:00'

>>> local.format('YYYY-MM-DD HH:mm:ss ZZ')
'2013-05-11 13:23:58 -07:00'

>>> local.humanize()
'an hour ago'

>>> local.humanize(locale='ko-kr')
'한시간 전'
```