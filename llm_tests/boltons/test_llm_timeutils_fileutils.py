"""Unit tests for boltons.timeutils and boltons.fileutils."""
import os
import stat
from datetime import date, datetime, timedelta, timezone
from itertools import islice

import pytest

from boltons.fileutils import (
    AtomicSaver, DummyFile, FilePerms, atomic_save, copy_tree,
    iter_find_files, mkdir_p, rotate_file)
from boltons.timeutils import (
    ConstantTZInfo, UTC, daterange, decimal_relative_time, dt_to_timestamp,
    isoparse, parse_timedelta, relative_time, strpdate)


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ------------------------------------------------------------------ timeutils

def test_dt_to_timestamp_naive_is_treated_as_utc():
    assert dt_to_timestamp(datetime(1970, 1, 1)) == 0.0
    assert dt_to_timestamp(datetime(1970, 1, 2)) == 86400.0


def test_dt_to_timestamp_aware_respects_offset():
    plus_two = ConstantTZInfo('plus2', timedelta(hours=2))
    dt = datetime(1970, 1, 1, 2, 0, 0, tzinfo=plus_two)
    assert dt_to_timestamp(dt) == 0.0
    assert dt_to_timestamp(datetime(2000, 1, 1, tzinfo=UTC)) == 946684800.0


def test_isoparse_roundtrips_isoformat():
    dt = datetime(2021, 3, 4, 5, 6, 7)
    assert isoparse(dt.isoformat()) == dt
    now = _utcnow()
    assert isoparse(now.isoformat()) == now


def test_isoparse_fractional_seconds_scale_to_microseconds():
    assert isoparse('1970-01-01T00:00:00.851') == datetime(1970, 1, 1, 0, 0, 0, 851000)
    assert isoparse('2020-02-29T12:30:45.000123') == datetime(2020, 2, 29, 12, 30, 45, 123)


@pytest.mark.parametrize('text, expected', [
    ('1d 2h 3.5m 0s', timedelta(days=1, seconds=7410)),
    ('2 weeks 1 day', timedelta(days=15)),
    ('-1.5 weeks 3m 20s', timedelta(days=-11, seconds=43400)),
    ('90 seconds', timedelta(seconds=90)),
    ('1h 30m', timedelta(hours=1, minutes=30)),
    ('', timedelta(0)),
])
def test_parse_timedelta_accepts_varied_formats(text, expected):
    assert parse_timedelta(text) == expected


def test_parse_timedelta_rejects_unknown_unit():
    with pytest.raises(ValueError):
        parse_timedelta('5 parsecs')


def test_decimal_relative_time_between_fixed_datetimes():
    now = datetime(2020, 6, 15, 12, 0, 0)
    assert decimal_relative_time(now - timedelta(days=1, seconds=3600), now) == (1.0, 'day')
    assert decimal_relative_time(now - timedelta(seconds=0.002), now, ndigits=5) == (0.002, 'seconds')
    assert decimal_relative_time(now, now - timedelta(days=900), ndigits=1) == (-2.5, 'years')


def test_decimal_relative_time_without_cardinalizing():
    now = datetime(2020, 6, 15, 12, 0, 0)
    value, unit = decimal_relative_time(now - timedelta(hours=5), now, cardinalize=False)
    assert (value, unit) == (5.0, 'hour')


def test_relative_time_future_and_past_with_explicit_other():
    now = datetime(2020, 6, 15, 12, 0, 0)
    assert relative_time(now + timedelta(days=7), now) == '1 week from now'
    assert relative_time(now - timedelta(days=1, seconds=36000), now, ndigits=1) == '1.4 days ago'
    assert relative_time(now - timedelta(minutes=1), now) == '1 minute ago'
    assert relative_time(now, now) == '0 seconds ago'


def test_relative_time_against_current_time_past():
    assert relative_time(_utcnow() - timedelta(hours=3)) == '3 hours ago'
    assert relative_time(_utcnow() - timedelta(days=2)) == '2 days ago'


def test_relative_time_against_current_time_future():
    assert relative_time(_utcnow() + timedelta(days=2)) == '2 days from now'
    assert relative_time(_utcnow() + timedelta(minutes=30)) == '30 minutes from now'


def test_relative_time_just_now_is_zero_seconds():
    assert relative_time(_utcnow(), ndigits=1) == '0 seconds ago'


def test_relative_time_with_timezone_aware_datetime():
    aware = datetime.now(timezone.utc) - timedelta(minutes=5)
    assert relative_time(aware) == '5 minutes ago'


def test_strpdate_ignores_time_fields():
    assert strpdate('2016-02-14', '%Y-%m-%d') == date(2016, 2, 14)
    assert strpdate('20151231 23:59:59', '%Y%m%d %H:%M:%S') == date(2015, 12, 31)
    with pytest.raises(ValueError):
        strpdate('not a date', '%Y-%m-%d')


def test_daterange_default_excludes_stop():
    days = list(daterange(date(2015, 12, 25), date(2016, 1, 1)))
    assert len(days) == 7
    assert days[0] == date(2015, 12, 25)
    assert days[-1] == date(2015, 12, 31)


def test_daterange_inclusive_and_tuple_steps():
    assert list(daterange(date(2015, 12, 25), date(2015, 12, 27), inclusive=True)) == [
        date(2015, 12, 25), date(2015, 12, 26), date(2015, 12, 27)]
    months = list(daterange(date(2017, 5, 1), date(2017, 8, 1),
                            step=(0, 1, 0), inclusive=True))
    assert months == [date(2017, 5, 1), date(2017, 6, 1),
                      date(2017, 7, 1), date(2017, 8, 1)]


def test_daterange_negative_step_and_timedelta_step():
    back = list(daterange(date(2020, 1, 5), date(2020, 1, 1), step=-2))
    assert back == [date(2020, 1, 5), date(2020, 1, 3)]
    weekly = list(daterange(date(2020, 1, 1), date(2020, 1, 20), step=timedelta(days=7)))
    assert weekly == [date(2020, 1, 1), date(2020, 1, 8), date(2020, 1, 15)]


def test_daterange_infinite_with_stop_none():
    days = list(islice(daterange(date(2020, 2, 27), None), 4))
    assert days == [date(2020, 2, 27), date(2020, 2, 28),
                    date(2020, 2, 29), date(2020, 3, 1)]


def test_daterange_validation_errors():
    with pytest.raises(TypeError):
        list(daterange('2020-01-01', date(2020, 1, 2)))
    with pytest.raises(ValueError):
        list(daterange(date(2020, 1, 1), date(2020, 1, 5), step=0))
    assert list(daterange(date(2020, 1, 1), date(2020, 1, 5), step=-1)) == []


def test_constant_tzinfo_reports_offset():
    tz = ConstantTZInfo('EST-ish', timedelta(hours=-5))
    assert tz.utcoffset(None) == timedelta(hours=-5)
    assert tz.utcoffset_hours == -5.0
    assert tz.tzname(None) == 'EST-ish'
    assert tz.dst(None) == timedelta(0)


# ------------------------------------------------------------------ fileutils

def test_mkdir_p_creates_nested_directories(tmp_path):
    target = tmp_path / 'a' / 'b' / 'c'
    mkdir_p(str(target))
    assert target.is_dir()


def test_mkdir_p_is_idempotent(tmp_path):
    target = tmp_path / 'x' / 'y'
    mkdir_p(str(target))
    mkdir_p(str(target))
    assert target.is_dir()


def test_mkdir_p_raises_when_path_is_a_file(tmp_path):
    f = tmp_path / 'file.txt'
    f.write_text('hi')
    with pytest.raises(OSError):
        mkdir_p(str(f))


def test_atomic_save_writes_binary_file(tmp_path):
    dest = tmp_path / 'out.bin'
    with atomic_save(str(dest)) as fo:
        fo.write(b'\x00\x01\x02')
    assert dest.read_bytes() == b'\x00\x01\x02'
    assert not (tmp_path / 'out.bin.part').exists()


def test_atomic_save_text_mode_overwrites_existing(tmp_path):
    dest = tmp_path / 'notes.txt'
    dest.write_text('old')
    with atomic_save(str(dest), text_mode=True) as fo:
        fo.write('new content')
    assert dest.read_text() == 'new content'


def test_atomic_save_keeps_original_when_exception_raised(tmp_path):
    dest = tmp_path / 'keep.txt'
    dest.write_text('original')
    with pytest.raises(ZeroDivisionError):
        with atomic_save(str(dest), text_mode=True) as fo:
            fo.write('partial')
            1 / 0
    assert dest.read_text() == 'original'
    assert not (tmp_path / 'keep.txt.part').exists()


def test_atomic_save_overwrite_false_refuses_existing_file(tmp_path):
    dest = tmp_path / 'exists.txt'
    dest.write_text('here')
    with pytest.raises(OSError):
        with atomic_save(str(dest), overwrite=False):
            pass
    assert dest.read_text() == 'here'


def test_atomic_save_refuses_stale_part_file_unless_overwrite_part(tmp_path):
    dest = tmp_path / 'data.txt'
    (tmp_path / 'data.txt.part').write_text('stale')
    with pytest.raises(OSError):
        with atomic_save(str(dest), text_mode=True) as fo:
            fo.write('x')
    with atomic_save(str(dest), text_mode=True, overwrite_part=True) as fo:
        fo.write('fresh')
    assert dest.read_text() == 'fresh'


def test_atomic_save_custom_permissions(tmp_path):
    dest = tmp_path / 'secret.txt'
    with atomic_save(str(dest), file_perms=0o600, text_mode=True) as fo:
        fo.write('s3cret')
    assert stat.S_IMODE(os.stat(dest).st_mode) == 0o600


def test_atomic_saver_rejects_unknown_kwargs(tmp_path):
    with pytest.raises(TypeError):
        AtomicSaver(str(tmp_path / 'x'), bogus=1)


def test_file_perms_from_int_and_back():
    perms = FilePerms.from_int(0o644)
    assert (perms.user, perms.group, perms.other) == ('rw', 'r', 'r')
    assert int(perms) == 0o644
    assert int(FilePerms('r', 'r', '')) == 0o440


def test_file_perms_normalizes_character_order_and_validates():
    perms = FilePerms(user='xwr', group='', other='x')
    assert perms.user == 'rwx'
    assert 'user=' in repr(perms)
    with pytest.raises(ValueError):
        FilePerms(user='rwz')


def test_file_perms_from_path_matches_chmod(tmp_path):
    f = tmp_path / 'p.txt'
    f.write_text('x')
    os.chmod(f, 0o640)
    perms = FilePerms.from_path(str(f))
    assert int(perms) == 0o640
    assert perms.group == 'r'


@pytest.fixture
def sample_tree(tmp_path):
    (tmp_path / 'a.py').write_text('1')
    (tmp_path / 'b.txt').write_text('2')
    (tmp_path / '.#lock.py').write_text('3')
    sub = tmp_path / 'sub'
    sub.mkdir()
    (sub / 'c.py').write_text('4')
    deep = sub / 'deeper'
    deep.mkdir()
    (deep / 'd.py').write_text('5')
    return tmp_path


def _names(paths):
    return sorted(os.path.basename(p) for p in paths)


def test_iter_find_files_matches_glob_pattern_recursively(sample_tree):
    found = iter_find_files(str(sample_tree), '*.py')
    assert _names(found) == ['.#lock.py', 'a.py', 'c.py', 'd.py']


def test_iter_find_files_multiple_patterns_and_ignored(sample_tree):
    found = iter_find_files(str(sample_tree), ['*.py', '*.txt'], ignored='.#*')
    assert _names(found) == ['a.py', 'b.txt', 'c.py', 'd.py']


def test_iter_find_files_max_depth_limits_recursion(sample_tree):
    top_only = iter_find_files(str(sample_tree), '*.py', max_depth=0)
    assert _names(top_only) == ['.#lock.py', 'a.py']
    one_level = iter_find_files(str(sample_tree), '*.py', max_depth=1)
    assert _names(one_level) == ['.#lock.py', 'a.py', 'c.py']


def test_iter_find_files_include_dirs(sample_tree):
    found = iter_find_files(str(sample_tree), 'sub*', include_dirs=True)
    assert _names(found) == ['sub']
    assert not list(iter_find_files(str(sample_tree), 'sub*'))


def test_copy_tree_tolerates_existing_destination(tmp_path):
    src = tmp_path / 'src'
    (src / 'nested').mkdir(parents=True)
    (src / 'one.txt').write_text('1')
    (src / 'nested' / 'two.txt').write_text('2')
    dst = tmp_path / 'dst'
    dst.mkdir()
    copy_tree(str(src), str(dst))
    copy_tree(str(src), str(dst))       # second copy must not raise
    assert (dst / 'one.txt').read_text() == '1'
    assert (dst / 'nested' / 'two.txt').read_text() == '2'


def test_rotate_file_shifts_existing_backups(tmp_path):
    log = tmp_path / 'app.log'
    log.write_text('v1')
    rotate_file(str(log))
    assert not log.exists()
    assert (tmp_path / 'app.1.log').read_text() == 'v1'
    log.write_text('v2')
    rotate_file(str(log))
    assert (tmp_path / 'app.1.log').read_text() == 'v2'
    assert (tmp_path / 'app.2.log').read_text() == 'v1'


def test_rotate_file_validates_arguments(tmp_path):
    log = tmp_path / 'x.txt'
    log.write_text('data')
    with pytest.raises(ValueError):
        rotate_file(str(log), keep=0)
    rotate_file(str(tmp_path / 'missing.txt'))      # silently does nothing
    with pytest.raises(ValueError):
        rotate_file(str(tmp_path))


def test_dummy_file_discards_writes_and_reads_empty():
    df = DummyFile('/dev/null', 'w')
    df.write('ignored')
    assert df.read() == ''
    assert df.readlines() == []
    df.close()
    with pytest.raises(ValueError):
        df.write('too late')
