import datetime
import logging
import multiprocessing
from unittest import mock

import pytest
import zmq
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from reactobus import db as db_module
from reactobus.db import Base, DB, Message


class _StopLoop(Exception):
    """Raised from a mocked poll() to break out of the infinite run loop."""


VALID_FRAMES = [
    b"lava.job.start",
    b"uuid-1",
    b"2016-05-10T12:30:45.123456",
    b"alice",
    b'{"a": 1}',
]


def _loop_db():
    """A DB whose setup() is a no-op and whose zmq objects are mocks."""
    d = DB({"url": "sqlite://"}, "tcp://127.0.0.1:1")
    d.setup = lambda: None
    d.sub = mock.MagicMock(name="sub")
    d.poller = mock.MagicMock(name="poller")
    return d


@pytest.fixture
def fake_time(monkeypatch):
    fake = mock.Mock()
    fake.time.return_value = 100.0
    monkeypatch.setattr(db_module, "time", fake)
    return fake


@pytest.fixture
def real_db(tmp_path):
    d = DB({"url": "sqlite:///" + str(tmp_path / "rob.db")}, "tcp://127.0.0.1:1")
    d.engine = create_engine(d.options["url"])
    Base.metadata.create_all(d.engine)
    d.sessions = sessionmaker(bind=d.engine)
    yield d
    d.engine.dispose()


# ---------------------------------------------------------------- Message


def test_message_tablename_and_base():
    assert Message.__tablename__ == "messages"
    assert issubclass(Message, Base)


def test_message_columns_present():
    assert set(Message.__table__.columns.keys()) == {
        "id",
        "topic",
        "uuid",
        "datetime",
        "username",
        "data",
    }


def test_message_id_is_primary_key():
    assert list(Message.__table__.primary_key.columns.keys()) == ["id"]


def test_message_indexed_columns():
    cols = Message.__table__.c
    for name in ("topic", "uuid", "datetime", "username"):
        assert cols[name].index is True, name
    assert cols["data"].index is not True


def test_message_transient_attributes_default_to_none():
    m = Message(topic="x")
    assert m.topic == "x"
    assert m.id is None
    assert m.data is None


def test_message_roundtrip_through_database(real_db):
    dt = datetime.datetime(2016, 5, 10, 12, 30, 45, 123456)
    session = real_db.sessions()
    session.add(
        Message(topic="t", uuid="u-1", datetime=dt, username="bob", data="payload")
    )
    session.commit()
    session.close()

    session = real_db.sessions()
    row = session.query(Message).one()
    session.close()
    assert row.id is not None
    assert row.topic == "t"
    assert row.uuid == "u-1"
    assert row.username == "bob"
    assert row.data == "payload"


# ---------------------------------------------------------------- DB init


def test_db_init_defaults():
    options = {"url": "sqlite://"}
    d = DB(options, "tcp://127.0.0.1:5555")
    assert d.options is options
    assert d.outbound == "tcp://127.0.0.1:5555"
    assert d.engine is None
    assert d.sessions is None
    assert d.messages == []
    assert d.pid is None


def test_db_is_a_process():
    assert issubclass(DB, multiprocessing.Process)


# ---------------------------------------------------------------- setup()


def test_setup_connects_subscriber_to_outbound(tmp_path, monkeypatch):
    fake_zmq = mock.MagicMock()
    monkeypatch.setattr(db_module, "zmq", fake_zmq)
    monkeypatch.setattr(db_module, "setproctitle", mock.Mock())
    d = DB({"url": "sqlite:///" + str(tmp_path / "s.db")}, "tcp://example:5556")

    d.setup()

    context = fake_zmq.Context.instance.return_value
    context.socket.assert_called_once_with(fake_zmq.SUB)
    sub = context.socket.return_value
    sub.connect.assert_called_once_with("tcp://example:5556")
    d.poller.register.assert_called_once_with(sub, fake_zmq.POLLIN)
    d.engine.dispose()


def test_setup_subscribes_to_all_topics(tmp_path, monkeypatch):
    fake_zmq = mock.MagicMock()
    monkeypatch.setattr(db_module, "zmq", fake_zmq)
    monkeypatch.setattr(db_module, "setproctitle", mock.Mock())
    d = DB({"url": "sqlite:///" + str(tmp_path / "s.db")}, "tcp://example:5556")

    d.setup()

    sub = fake_zmq.Context.instance.return_value.socket.return_value
    sub.setsockopt.assert_called_once_with(fake_zmq.SUBSCRIBE, b"")
    d.engine.dispose()


def test_setup_sets_process_title(tmp_path, monkeypatch):
    fake_zmq = mock.MagicMock()
    fake_title = mock.Mock()
    monkeypatch.setattr(db_module, "zmq", fake_zmq)
    monkeypatch.setattr(db_module, "setproctitle", fake_title)
    d = DB({"url": "sqlite:///" + str(tmp_path / "s.db")}, "tcp://example:5556")

    d.setup()

    fake_title.assert_called_once_with("ReactOBus [db]")
    d.engine.dispose()


def test_setup_creates_schema_and_sessionmaker(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "zmq", mock.MagicMock())
    monkeypatch.setattr(db_module, "setproctitle", mock.Mock())
    d = DB({"url": "sqlite:///" + str(tmp_path / "s.db")}, "tcp://example:5556")

    d.setup()

    assert d.engine is not None
    session = d.sessions()
    try:
        assert session.query(Message).count() == 0
    finally:
        session.close()
        d.engine.dispose()


# ---------------------------------------------------------------- save_to_db()


def test_save_to_db_persists_and_clears_messages(real_db):
    dt = datetime.datetime(2016, 1, 1, 0, 0, 0)
    real_db.messages = [
        Message(topic="t", uuid="u", datetime=dt, username="x", data="d")
        for _ in range(3)
    ]

    real_db.save_to_db()

    assert real_db.messages == []
    session = real_db.sessions()
    try:
        assert session.query(Message).count() == 3
    finally:
        session.close()


def test_save_to_db_with_no_messages_is_harmless(real_db):
    real_db.messages = []

    real_db.save_to_db()

    assert real_db.messages == []
    session = real_db.sessions()
    try:
        assert session.query(Message).count() == 0
    finally:
        session.close()


def test_save_to_db_retries_then_succeeds():
    session = mock.MagicMock()
    session.bulk_save_objects.side_effect = [SQLAlchemyError("boom"), None]
    d = DB({"url": "sqlite://"}, "tcp://x")
    d.sessions = mock.MagicMock(return_value=session)
    d.messages = [Message(topic="t")]

    d.save_to_db()

    assert session.bulk_save_objects.call_count == 2
    session.commit.assert_called_once_with()
    assert d.messages == []


def test_save_to_db_gives_up_after_three_attempts(caplog):
    session = mock.MagicMock()
    session.bulk_save_objects.side_effect = SQLAlchemyError("disk full")
    d = DB({"url": "sqlite://"}, "tcp://x")
    d.sessions = mock.MagicMock(return_value=session)
    pending = [Message(topic="t")]
    d.messages = pending

    caplog.set_level(logging.ERROR, logger="ROB.DB")
    d.save_to_db()

    assert session.bulk_save_objects.call_count == 3
    session.commit.assert_not_called()
    assert d.messages is pending
    assert "Unable to commit to the database" in caplog.text
    assert "disk full" in caplog.text


# ---------------------------------------------------------------- run()


def test_run_ignores_poll_without_pollin(fake_time):
    d = _loop_db()
    d.sessions = mock.MagicMock()
    d.poller.poll.side_effect = [[], _StopLoop()]

    with pytest.raises(_StopLoop):
        d.run()

    d.sub.recv_multipart.assert_not_called()
    assert d.messages == []


def test_run_polls_with_infinite_timeout_when_empty_and_1s_when_pending(fake_time):
    d = _loop_db()
    d.sessions = mock.MagicMock()
    d.poller.poll.side_effect = [[(d.sub, zmq.POLLIN)], _StopLoop()]
    d.sub.recv_multipart.return_value = VALID_FRAMES

    with pytest.raises(_StopLoop):
        d.run()

    assert d.poller.poll.call_args_list[0] == mock.call(-1)
    assert d.poller.poll.call_args_list[1] == mock.call(1000)


def test_run_stores_valid_message(fake_time):
    d = _loop_db()
    d.sessions = mock.MagicMock()
    d.poller.poll.side_effect = [[(d.sub, zmq.POLLIN)], _StopLoop()]
    d.sub.recv_multipart.return_value = VALID_FRAMES

    with pytest.raises(_StopLoop):
        d.run()

    assert len(d.messages) == 1
    m = d.messages[0]
    assert m.topic == "lava.job.start"
    assert m.uuid == "uuid-1"
    assert m.datetime == datetime.datetime(2016, 5, 10, 12, 30, 45, 123456)
    assert m.username == "alice"
    assert m.data == '{"a": 1}'


def test_run_skips_message_with_invalid_datetime(fake_time, caplog):
    d = _loop_db()
    d.sessions = mock.MagicMock()
    bad = list(VALID_FRAMES)
    bad[2] = b"not-a-date"
    d.poller.poll.side_effect = [[(d.sub, zmq.POLLIN)], _StopLoop()]
    d.sub.recv_multipart.return_value = bad

    caplog.set_level(logging.ERROR, logger="ROB.DB")
    with pytest.raises(_StopLoop):
        d.run()

    assert d.messages == []
    assert "Invalid message" in caplog.text


def test_run_skips_message_with_wrong_frame_count(fake_time, caplog):
    d = _loop_db()
    d.sessions = mock.MagicMock()
    d.poller.poll.side_effect = [[(d.sub, zmq.POLLIN)], _StopLoop()]
    d.sub.recv_multipart.return_value = [b"only", b"two"]

    caplog.set_level(logging.ERROR, logger="ROB.DB")
    with pytest.raises(_StopLoop):
        d.run()

    assert d.messages == []
    assert "Invalid message" in caplog.text


def test_run_saves_after_1000_messages(fake_time):
    d = _loop_db()
    session = mock.MagicMock()
    d.sessions = mock.MagicMock(return_value=session)
    d.poller.poll.side_effect = [[(d.sub, zmq.POLLIN)] for _ in range(1000)] + [
        _StopLoop()
    ]
    d.sub.recv_multipart.return_value = VALID_FRAMES

    with pytest.raises(_StopLoop):
        d.run()

    assert session.bulk_save_objects.call_count == 1
    saved = session.bulk_save_objects.call_args[0][0]
    assert len(saved) == 1000
    assert d.messages == []


def test_run_saves_when_more_than_one_second_has_passed(fake_time):
    d = _loop_db()
    session = mock.MagicMock()
    d.sessions = mock.MagicMock(return_value=session)
    # First value: last_save at start of run; second: "now" after one message.
    fake_time.time.side_effect = [0.0, 2.0]
    d.poller.poll.side_effect = [[(d.sub, zmq.POLLIN)], _StopLoop()]
    d.sub.recv_multipart.return_value = VALID_FRAMES

    with pytest.raises(_StopLoop):
        d.run()

    assert session.bulk_save_objects.call_count == 1
    assert len(session.bulk_save_objects.call_args[0][0]) == 1
    assert d.messages == []
