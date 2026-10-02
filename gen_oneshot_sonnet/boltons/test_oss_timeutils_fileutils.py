"""Tests for boltons.timeutils and boltons.fileutils."""

import itertools
import os
import stat
import time
from datetime import date, datetime, timedelta, timezone

import pytest
from freezegun import freeze_time

from boltons.fileutils import (
    FilePerms,
    atomic_save,
    copytree,
    iter_find_files,
    mkdir_p,
    rotate_file,
)
from boltons.timeutils import (
    UTC,
    Central,
    ConstantTZInfo,
    Eastern,
    Mountain,
    Pacific,
    daterange,
    decimal_relative_time,
    dt_to_timestamp,
    isoparse,
    parse_td,
    parse_timedelta,
    relative_time,
    strpdate,
)

# ---------------------------------------------------------------------------
# timeutils
# ---------------------------------------------------------------------------


def test_isoparse():
    assert isoparse("1970-01-01T00:00:00") == datetime(1970, 1, 1)
    assert isoparse("1970-01-01T00:00:00.851") == datetime(1970, 1, 1, 0, 0, 0, 851000)
    assert isoparse("2020-05-17") == datetime(2020, 5, 17)
    stamp = datetime(2020, 5, 17, 13, 45, 59, 123456)
    assert isoparse(stamp.isoformat()) == stamp
    with pytest.raises(ValueError):
        isoparse("not a date")


def test_parse_timedelta():
    assert parse_td("1d 2h 3.5m 0s") == timedelta(days=1, seconds=7410)
    assert parse_td("2 weeks 1 day") == timedelta(days=15)
    assert parse_td("-1.5 weeks 3m 20s") == timedelta(days=-11, seconds=43400)
    assert parse_timedelta("90 minutes") == timedelta(hours=1, minutes=30)
    assert parse_timedelta("1.5h") == timedelta(minutes=90)
    assert parse_timedelta("") == timedelta(0)
    assert parse_td is parse_timedelta
    with pytest.raises(ValueError):
        parse_timedelta("5 y")


def test_relative_time_between_two_fixed_datetimes():
    base = datetime(2024, 1, 1, 12, 0, 0)
    assert relative_time(base, base) == "0 seconds ago"
    assert relative_time(base - timedelta(seconds=30), base) == "30 seconds ago"
    assert relative_time(base - timedelta(hours=3), base) == "3 hours ago"
    assert relative_time(base - timedelta(days=1), base) == "1 day ago"
    assert relative_time(base + timedelta(days=7), base, ndigits=1) == "1 week from now"
    assert relative_time(base - timedelta(days=90), base) == "3 months ago"
    assert relative_time(base - timedelta(days=400), base) == "1 year ago"
    assert relative_time(base - timedelta(minutes=90), base) == "2 hours ago"
    assert relative_time(base - timedelta(minutes=90), base, ndigits=1) == "1.5 hours ago"
    assert relative_time(base - timedelta(days=1, seconds=36000), base, ndigits=1) == "1.4 days ago"


def test_decimal_relative_time():
    now = datetime(2024, 6, 1, 12, 0, 0)
    assert decimal_relative_time(now - timedelta(days=1, seconds=3600), now) == (1.0, "day")
    assert decimal_relative_time(now - timedelta(seconds=0.002), now, ndigits=5) == (0.002, "seconds")
    assert decimal_relative_time(now, now - timedelta(days=900), ndigits=1) == (-2.5, "years")
    assert decimal_relative_time(now - timedelta(days=2), now) == (2.0, "days")
    assert decimal_relative_time(now - timedelta(days=2), now, cardinalize=False) == (2.0, "day")


def test_relative_time_against_the_current_time():
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    assert relative_time(now) == "0 seconds ago"
    assert relative_time(now - timedelta(minutes=10)) == "10 minutes ago"
    assert relative_time(now - timedelta(hours=2)) == "2 hours ago"
    assert relative_time(now - timedelta(days=3)) == "3 days ago"
    assert relative_time(now + timedelta(hours=5)) == "5 hours from now"

    aware = datetime.now(timezone.utc)
    assert relative_time(aware - timedelta(minutes=10)) == "10 minutes ago"


def test_relative_time_with_a_frozen_clock():
    with freeze_time("2024-06-01 12:00:00"):
        assert relative_time(datetime(2024, 6, 1, 10, 0, 0)) == "2 hours ago"
        assert relative_time(datetime(2024, 6, 1, 12, 0, 0)) == "0 seconds ago"
        assert relative_time(datetime(2024, 6, 8, 12, 0, 0), ndigits=1) == "1 week from now"


def test_daterange_steps_and_bounds():
    christmas = date(2015, 12, 25)
    boxing_day = date(2015, 12, 26)
    new_year = date(2016, 1, 1)

    days = list(daterange(christmas, new_year))
    assert len(days) == 7
    assert days[0] == christmas
    assert days[-1] == date(2015, 12, 31)
    assert list(daterange(christmas, boxing_day)) == [christmas]
    assert list(daterange(christmas, new_year, inclusive=True))[-1] == new_year

    monthly = daterange(date(2017, 5, 1), date(2017, 8, 1), step=(0, 1, 0), inclusive=True)
    assert list(monthly) == [date(2017, 5, 1), date(2017, 6, 1), date(2017, 7, 1), date(2017, 8, 1)]

    every_other = daterange(date(2020, 1, 1), date(2020, 1, 5), step=timedelta(days=2))
    assert list(every_other) == [date(2020, 1, 1), date(2020, 1, 3)]
    weekly = daterange(date(2020, 1, 1), date(2020, 1, 15), step=7)
    assert list(weekly) == [date(2020, 1, 1), date(2020, 1, 8)]

    backwards = daterange(date(2020, 1, 5), date(2020, 1, 1), step=-1)
    assert list(backwards) == [date(2020, 1, 5), date(2020, 1, 4), date(2020, 1, 3), date(2020, 1, 2)]
    backwards_inclusive = daterange(date(2020, 1, 5), date(2020, 1, 1), step=-1, inclusive=True)
    assert list(backwards_inclusive)[-1] == date(2020, 1, 1)


def test_daterange_infinite_and_invalid_arguments():
    forever = daterange(date(2020, 1, 1), None)
    assert list(itertools.islice(forever, 3)) == [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]

    # a step pointing away from stop yields nothing, like range(1, 5, -1)
    assert list(daterange(date(2020, 1, 1), date(2020, 1, 5), step=-1)) == []

    with pytest.raises(ValueError):
        list(daterange(date(2020, 1, 1), date(2020, 1, 5), step=0))
    with pytest.raises(ValueError):
        list(daterange(date(2020, 1, 1), date(2020, 3, 1), step=(0, 1, -31)))
    with pytest.raises(TypeError):
        list(daterange("2020-01-01", date(2020, 1, 2)))
    with pytest.raises(TypeError):
        list(daterange(date(2020, 1, 1), "2020-01-02"))


def test_strpdate_and_dt_to_timestamp():
    assert strpdate("2016-02-14", "%Y-%m-%d") == date(2016, 2, 14)
    assert strpdate("26/12 (2015)", "%d/%m (%Y)") == date(2015, 12, 26)
    assert strpdate("20151231 23:59:59", "%Y%m%d %H:%M:%S") == date(2015, 12, 31)

    assert dt_to_timestamp(datetime(1970, 1, 1)) == 0.0
    assert dt_to_timestamp(datetime(1970, 1, 2)) == 86400.0
    assert dt_to_timestamp(datetime(1970, 1, 1, 1, tzinfo=timezone.utc)) == 3600.0
    plus_one = ConstantTZInfo("PLUS1", timedelta(hours=1))
    assert dt_to_timestamp(datetime(1970, 1, 1, 1, tzinfo=plus_one)) == 0.0

    timestamp = int(time.time())
    utc_dt = datetime.fromtimestamp(timestamp, timezone.utc)
    assert dt_to_timestamp(utc_dt) == timestamp


def test_constant_and_us_timezones():
    assert UTC.utcoffset(None) == timedelta(0)
    est = ConstantTZInfo("EST", timedelta(hours=-5))
    assert est.tzname(None) == "EST"
    assert est.utcoffset_hours == -5.0
    assert est.dst(None) == timedelta(0)
    assert repr(est).startswith("ConstantTZInfo(name='EST'")

    summer = datetime(2023, 7, 1, 12, 0, tzinfo=Eastern)
    winter = datetime(2023, 1, 1, 12, 0, tzinfo=Eastern)
    assert summer.utcoffset() == timedelta(hours=-4)
    assert summer.tzname() == "EDT"
    assert winter.utcoffset() == timedelta(hours=-5)
    assert winter.tzname() == "EST"
    assert datetime(2023, 7, 1, 12, tzinfo=Pacific).utcoffset() == timedelta(hours=-7)
    assert datetime(2023, 1, 1, 12, tzinfo=Central).utcoffset() == timedelta(hours=-6)
    assert datetime(2023, 7, 1, 12, tzinfo=Mountain).tzname() == "MDT"

    assert datetime(2023, 7, 1, 16, 0, tzinfo=timezone.utc).astimezone(Eastern).hour == 12
    assert datetime(2023, 1, 1, 16, 0, tzinfo=timezone.utc).astimezone(Eastern).hour == 11


# ---------------------------------------------------------------------------
# fileutils
# ---------------------------------------------------------------------------


def test_mkdir_p_creates_nested_directories_and_tolerates_existing(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    mkdir_p(str(target))
    assert target.is_dir()
    mkdir_p(str(target))  # already there: no error
    blocker = tmp_path / "file.txt"
    blocker.write_text("x")
    with pytest.raises(OSError):
        mkdir_p(str(blocker))


def test_atomic_save_writes_bytes_text_and_replaces_existing_files(tmp_path):
    dest = tmp_path / "out.bin"
    part = tmp_path / "out.bin.part"
    with atomic_save(str(dest)) as f:
        f.write(b"hello")
        assert part.exists()
    assert dest.read_bytes() == b"hello"
    assert not part.exists()

    text_dest = tmp_path / "out.txt"
    with atomic_save(text_dest, text_mode=True) as f:
        f.write("line one\n")
    assert text_dest.read_text() == "line one\n"

    with atomic_save(str(dest)) as f:
        f.write(b"replaced")
    assert dest.read_bytes() == b"replaced"


def test_atomic_save_keeps_the_original_when_the_block_fails(tmp_path):
    dest = tmp_path / "data.txt"
    part = tmp_path / "data.txt.part"
    dest.write_text("original")

    with pytest.raises(ZeroDivisionError):
        with atomic_save(str(dest), text_mode=True) as f:
            f.write("partial")
            1 / 0
    assert dest.read_text() == "original"
    assert not part.exists()

    with pytest.raises(RuntimeError):
        with atomic_save(str(dest), text_mode=True, rm_part_on_exc=False) as f:
            f.write("partial")
            raise RuntimeError("boom")
    assert dest.read_text() == "original"
    assert part.read_text() == "partial"

    # a leftover part file blocks the next save unless overwrite_part is set
    with pytest.raises(OSError):
        with atomic_save(str(dest), text_mode=True) as f:
            f.write("blocked")
    with atomic_save(str(dest), text_mode=True, overwrite_part=True) as f:
        f.write("fresh")
    assert dest.read_text() == "fresh"
    assert not part.exists()


def test_atomic_save_overwrite_flag_permissions_and_part_file_name(tmp_path):
    existing = tmp_path / "exists.txt"
    existing.write_text("keep")
    with pytest.raises(OSError):
        with atomic_save(str(existing), overwrite=False) as f:
            f.write(b"new")
    assert existing.read_text() == "keep"

    perm_dest = tmp_path / "perm.txt"
    with atomic_save(str(perm_dest), file_perms=0o600) as f:
        f.write(b"x")
    assert stat.S_IMODE(os.stat(perm_dest).st_mode) == 0o600

    # permissions of the file being replaced are carried over
    os.chmod(perm_dest, 0o640)
    with atomic_save(str(perm_dest)) as f:
        f.write(b"y")
    assert stat.S_IMODE(os.stat(perm_dest).st_mode) == 0o640
    assert perm_dest.read_bytes() == b"y"

    named = tmp_path / "named.txt"
    with atomic_save(str(named), part_file="custom.tmp") as f:
        assert (tmp_path / "custom.tmp").exists()
        f.write(b"z")
    assert named.read_bytes() == b"z"
    assert not (tmp_path / "custom.tmp").exists()


def test_file_perms(tmp_path):
    perms = FilePerms(user="rwx", group="xrw", other="wxr")
    assert repr(perms) == "FilePerms(user='rwx', group='rwx', other='rwx')"
    assert int(perms) == 0o777
    assert int(FilePerms("r", "r", "")) == 288
    assert repr(FilePerms.from_int(0o644)) == "FilePerms(user='rw', group='r', other='r')"
    assert repr(FilePerms.from_int(0o755)) == "FilePerms(user='rwx', group='rx', other='rx')"
    with pytest.raises(ValueError):
        FilePerms(user="rwz")

    path = tmp_path / "f.txt"
    path.write_text("x")
    os.chmod(path, 0o640)
    assert repr(FilePerms.from_path(str(path))) == "FilePerms(user='rw', group='r', other='')"


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "sub" / "deep").mkdir(parents=True)
    for name in ["a.py", "b.txt", ".#lock.py", "sub/c.py", "sub/deep/d.py", "sub/e.txt"]:
        (tmp_path / name).write_text("x")
    return tmp_path


def _relative(paths, root):
    return sorted(os.path.relpath(p, str(root)) for p in paths)


def test_iter_find_files_patterns_and_ignored(tree):
    py_files = _relative(iter_find_files(str(tree), "*.py"), tree)
    assert py_files == [
        ".#lock.py",
        "a.py",
        os.path.join("sub", "c.py"),
        os.path.join("sub", "deep", "d.py"),
    ]
    both = _relative(iter_find_files(str(tree), ["*.py", "*.txt"]), tree)
    assert len(both) == 6
    without_lock = _relative(iter_find_files(str(tree), "*.py", ignored=".#*"), tree)
    assert ".#lock.py" not in without_lock
    assert len(without_lock) == 3
    assert _relative(iter_find_files(str(tree), "*.md"), tree) == []


def test_iter_find_files_include_dirs_and_max_depth(tree):
    assert _relative(iter_find_files(str(tree), "sub", include_dirs=True), tree) == ["sub"]
    assert _relative(iter_find_files(str(tree), "sub"), tree) == []
    assert _relative(iter_find_files(str(tree), "*.py", max_depth=0), tree) == [".#lock.py", "a.py"]
    assert _relative(iter_find_files(str(tree), "*.py", max_depth=1), tree) == [
        ".#lock.py",
        "a.py",
        os.path.join("sub", "c.py"),
    ]


def test_copytree_into_existing_destination_and_rotate_file(tmp_path):
    src = tmp_path / "src"
    (src / "inner").mkdir(parents=True)
    (src / "top.txt").write_text("top")
    (src / "inner" / "leaf.txt").write_text("leaf")
    dst = tmp_path / "dst"
    dst.mkdir()

    copytree(str(src), str(dst))
    assert (dst / "top.txt").read_text() == "top"
    assert (dst / "inner" / "leaf.txt").read_text() == "leaf"
    copytree(str(src), str(dst))  # copying over an existing tree does not raise

    log = tmp_path / "app.log"
    log.write_text("one")
    rotate_file(str(log))
    assert not log.exists()
    assert (tmp_path / "app.1.log").read_text() == "one"

    log.write_text("two")
    rotate_file(str(log))
    assert (tmp_path / "app.1.log").read_text() == "two"
    assert (tmp_path / "app.2.log").read_text() == "one"

    rotate_file(str(tmp_path / "missing.log"))  # nothing to rotate
    with pytest.raises(ValueError):
        rotate_file(str(log), keep=0)
