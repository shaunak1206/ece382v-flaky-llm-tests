# tabulate: public API (signatures and documentation)


## module `tabulate/__init__.py`

```python
def simple_separated_format(separator)
```
Construct a simple TableFormat with columns separated by a separator.

>>> tsv = simple_separated_format("\t") ;         tabulate([["foo", 1], ["spam", 23]], tablefmt=tsv) == 'foo \t 1\nspam\t23'
True

```python
def tabulate(tabular_data, headers=(), tablefmt='simple', floatfmt=_DEFAULT_FLOATFMT, intfmt=_DEFAULT_INTFMT, numalign=_DEFAULT_ALIGN, stralign=_DEFAULT_ALIGN, missingval=_DEFAULT_MISSINGVAL, showindex='default', disable_numparse=False, colglobalalign=None, colalign=None, preserve_whitespace=False, maxcolwidths=None, headersglobalalign=None, headersalign=None, rowalign=None, maxheadercolwidths=None, break_long_words=_BREAK_LONG_WORDS, break_on_hyphens=_BREAK_ON_HYPHENS)
```
Format a fixed width table for pretty printing.

>>> print(tabulate([[1, 2.34], [-56, "8.999"], ["2", "10001"]]))
---  ---------
  1      2.34
-56      8.999
  2  10001
---  ---------

The first required argument (`tabular_data`) can be a
list-of-lists (or another iterable of iterables), a list of named
tuples, a dictionary of iterables, an iterable of dictionaries,
an iterable of dataclasses, a two-dimensional NumPy array,
NumPy record array, or a Pandas' dataframe.


Table headers
-------------

To print nice column headers, supply the second argument (`headers`):

  - `headers` can be an explicit list of column headers
  - if `headers="firstrow"`, then the first row of data is used
  - if ...


# Usage examples from the project README

```python
The command line utility will be installed as `tabulate` to `bin` on
Linux (e.g. `/usr/bin`); or as `tabulate.exe` to `Scripts` in your
Python installation on Windows (e.g. `C:\Python39\Scripts\tabulate.exe`).

You may consider installing the library only for the current user:
```
```python
Build status
------------

[![python-tabulate](https://github.com/astanin/python-tabulate/actions/workflows/tabulate.yml/badge.svg)](https://github.com/astanin/python-tabulate/actions/workflows/tabulate.yml) [![codecov](https://codecov.io/github/astanin/python-tabulate/graph/badge.svg?token=Aa6wexP5wq)](https://codecov.io/github/astanin/python-tabulate)

Library usage
-------------

The module provides just one function, `tabulate`, which takes a list of
lists or another tabular data type as the first argument, and outputs a
nicely formatted plain-text table:
```
```python
The following tabular data types are supported:

-   list of lists or another iterable of iterables
-   list or another iterable of dicts (keys as columns)
-   dict of iterables (keys as columns)
-   list of dataclasses (field names as columns)
-   two-dimensional NumPy array
-   NumPy record arrays (names as columns)
-   pandas.DataFrame

Tabulate is a Python3 library.

### Headers

The second optional argument named `headers` defines a list of column
headers to be used:
```
```python
### Row Indices

By default, only pandas.DataFrame tables have an additional column
called row index. To add a similar column to any other type of table,
pass `showindex="always"` or `showindex=True` argument to `tabulate()`.
To suppress row indices for all types of data, pass `showindex="never"`
or `showindex=False`. To add a custom row index column, pass
`showindex=rowIDs`, where `rowIDs` is some iterable:
```
```python
`simple` is the default format (the default may change in future
versions). It corresponds to `simple_tables` in [Pandoc Markdown
extensions](http://johnmacfarlane.net/pandoc/README.html#tables):
```
```python
`grid` is like tables formatted by Emacs'
[table.el](http://table.sourceforge.net/) package. It corresponds to
`grid_tables` in Pandoc Markdown extensions:
```
```python
`simple_grid` draws a grid using single-line box-drawing characters:

    >>> print(tabulate(table, headers, tablefmt="simple_grid"))
    ┌────────┬───────┐
    │ item   │   qty │
    ├────────┼───────┤
    │ spam   │    42 │
    ├────────┼───────┤
    │ eggs   │   451 │
    ├────────┼───────┤
    │ bacon  │     0 │
    └────────┴───────┘

`rounded_grid` draws a grid using single-line box-drawing characters with rounded corners:

    >>> print(tabulate(table, headers, tablefmt="rounded_grid"))
    ╭────────┬───────╮
    │ item   │   qty │
    ├────────┼───────┤
    │ spam   │    42 │
    ├────────┼───────┤
    │ eggs   │   451 │
    ├────────┼───────┤
    │ bacon  │     0 │
    ╰────────┴───────╯

`heavy_grid` draws a grid using bold (thick) single-line box-drawing characters:

    >>> print(tabulate(table, headers, tablefmt="heavy_grid"))
    ┏━━━━━━━━┳━━━━━━━┓
    ┃ item   ┃   qty ┃
    ┣━━━━━━━━╋━━━━━━━┫
    ┃ spam   ┃    42 ┃
    ┣━━━━━━━━╋━━━━━━━┫
    ┃ eggs   ┃   451 ┃
    ┣━━━━━━━━╋━━━━━━━┫
    ┃ bacon  ┃     0 ┃
    ┗━━━━━━━━┻━━━━━━━┛

`mixed_grid` draws a grid using a mix of light (thin) and heavy (thick) lines box-drawing characters:

    >>> print(tabulate(table, headers, tablefmt="mixed_grid"))
    ┍━━━━━━━━┯━━━━━━━┑
    │ item   │   qty │
    ┝━━━━━━━━┿━━━━━━━┥
    │ spam   │    42 │
    ├────────┼───────┤
    │ eggs   │   451 │
    ├────────┼───────┤
    │ bacon  │     0 │
    ┕━━━━━━━━┷━━━━━━━┙

`double_grid` draws a grid using double-line box-drawing characters:

    >>> print(tabulate(table, headers, tablefmt="double_grid"))
    ╔════════╦═══════╗
    ║ item   ║   qty ║
    ╠════════╬═══════╣
    ║ spam   ║    42 ║
    ╠════════╬═══════╣
    ║ eggs   ║   451 ║
    ╠════════╬═══════╣
    ║ bacon  ║     0 ║
    ╚════════╩═══════╝

`fancy_grid` draws a grid using a mix of single and
    double-line box-drawing characters:
```