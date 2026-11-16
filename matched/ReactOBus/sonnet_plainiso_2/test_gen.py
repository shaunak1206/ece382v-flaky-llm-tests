import datetime
import logging
from types import SimpleNamespace
from unittest import mock

import pytest
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
import zmq

import reactobus.db as db
from reactobus.db import DB, Message, Base


class StopLoop(Exception):
    pass


@pytest.fixture
def sqlite_sessions(tmp_path):
    engine = create_engine("sqlite:///%s" % (tmp_path / "test.sqlite3"))
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def make_message(i=0):
    return Message(
        topic="org.test.%d" % i,
        uuid="uuid-%d" % i,
        datetime=datetime.datetime(2016, 1, 2, 3, 4, 5, 678),
        username="user",
        data='{"n": %d}' % i,
    )


def test_message_table_definition():
    table = Message.__table__
    assert table.name == "messages"
    assert set(table.c.keys()) == {
        "id", "topic", "uuid", "datetime", "username", "data"
    }
    assert table.c.id.primary_key


def test_message_indexed_columns():
    table = Message.__table__
    for name in ("topic", "uuid", "datetime", "username"):
        assert table.c[name].index, name
    assert not table.c.data.index


def test_message_attributes():
    dt = datetime.datetime(2016, 1, 1, 12, 0, 0)
    msg = Message(topic="t", uuid="u", datetime=dt, username="bob", data="d")
    assert msg.topic == "t"
    assert msg.uuid == "u"
    assert msg.datetime == dt
    assert msg.username == "bob"
    assert msg.data == "d"
    assert msg.id is None


def test_message_roundtrip(sqlite_sessions):
    session = sqlite_sessions()
    session.add(make_message(3))
    session.commit()
    row = session.query(Message).one()
    assert row.id is not None
    assert row.topic == "org.test.3"
    assert row.uuid == "uuid-3"
    assert row.username == "user"
    assert row.data == '{"n": 3}'
    assert row.datetime == datetime.datetime(2016, 1, 2, 3, 4, 5, 678)
    session.close()


def test_db_init():
    d = DB({"url": "sqlite://"}, "tcp://127.0.0.1:5555")
    assert d.options == {"url": "sqlite://"}
    assert d.outbound == "tcp://127.0.0.1:5555"
    assert d.engine is None
    assert d.sessions is None
    assert d.messages == []


def test_db_is_process():
    import multiprocessing

    assert issubclass(DB, multiprocessing.Process)


def test_setup_creates_engine_and_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "setproctitle", mock.MagicMock())
    url = "sqlite:///%s" % (tmp_path / "setup.sqlite3")
    d = DB({"url": url}, "tcp://127.0.0.1:59871")
    d.setup()
    try:
        assert d.engine is not None
        assert d.sessions is not None
        names = sqlalchemy.inspect(d.engine).get_table_names()
        assert "messages" in names
        assert d.sub.socket_type == zmq.SUB
        db.setproctitle.assert_called_once_with("ReactOBus [db]")
    finally:
        d.sub.close(linger=0)


def test_save_to_db_success(sqlite_sessions):
    d = DB({}, "x")
    d.sessions = sqlite_sessions
    d.messages = [make_message(i) for i in range(5)]
    d.save_to_db()
    assert d.messages == []
    session = sqlite_sessions()
    assert session.query(Message).count() == 5
    session.close()


def test_save_to_db_empty_list(sqlite_sessions):
    d = DB({}, "x")
    d.sessions = sqlite_sessions
    d.save_to_db()
    assert d.messages == []
    session = sqlite_sessions()
    assert session.query(Message).count() == 0
    session.close()


def test_save_to_db_retries_then_succeeds():
    session = mock.MagicMock()
    session.bulk_save_objects.side_effect = [SQLAlchemyError("boom"), None]
    d = DB({}, "x")
    d.sessions = mock.MagicMock(return_value=session)
    d.messages = [make_message()]
    d.save_to_db()
    assert session.bulk_save_objects.call_count == 2
    assert d.messages == []


def test_save_to_db_gives_up_after_three_failures(caplog):
    session = mock.MagicMock()
    session.bulk_save_objects.side_effect = SQLAlchemyError("boom")
    d = DB({}, "x")
    d.sessions = mock.MagicMock(return_value=session)
    msgs = [make_message()]
    d.messages = list(msgs)
    with caplog.at_level(logging.ERROR, logger="ROB.DB"):
        d.save_to_db()
    assert session.bulk_save_objects.call_count == 3
    assert d.messages == msgs
    text = caplog.text
    assert "Unable to commit to the database" in text
    assert "boom" in text


def _prepare_run(d, poll_results):
    d.setup = mock.MagicMock()
    d.sub = mock.MagicMock()
    d.poller = mock.MagicMock()
    d.poller.poll.side_effect = poll_results
    d.save_to_db = mock.MagicMock()


def test_run_receives_valid_message(monkeypatch):
    d = DB({}, "x")
    _prepare_run(d, None)
    d.poller.poll.side_effect = [[(d.sub, zmq.POLLIN)], StopLoop()]
    d.sub.recv_multipart.return_value = [
        b"org.test", b"uuid-1", b"2016-01-01T10:20:30.123456", b"user", b'{"a": 1}'
    ]
    with pytest.raises(StopLoop):
        d.run()
    d.setup.assert_called_once_with()
    assert len(d.messages) == 1
    m = d.messages[0]
    assert isinstance(m, Message)
    assert m.topic == "org.test"
    assert m.uuid == "uuid-1"
    assert m.username == "user"
    assert m.data == '{"a": 1}'
    assert m.datetime == datetime.datetime(2016, 1, 1, 10, 20, 30, 123456)
    d.save_to_db.assert_not_called()


def test_run_poll_timeouts():
    d = DB({}, "x")
    _prepare_run(d, None)
    d.poller.poll.side_effect = [[(d.sub, zmq.POLLIN)], StopLoop()]
    d.sub.recv_multipart.return_value = [
        b"t", b"u", b"2016-01-01T10:20:30.000001", b"user", b"d"
    ]
    with pytest.raises(StopLoop):
        d.run()
    timeouts = [c[0][0] for c in d.poller.poll.call_args_list]
    # No message pending: wait forever; then wait 1000ms
    assert timeouts == [-1, 1000]


def test_run_invalid_date_is_dropped(caplog):
    d = DB({}, "x")
    _prepare_run(d, None)
    d.poller.poll.side_effect = [[(d.sub, zmq.POLLIN)], StopLoop()]
    d.sub.recv_multipart.return_value = [
        b"t", b"u", b"not-a-date", b"user", b"d"
    ]
    with caplog.at_level(logging.ERROR, logger="ROB.DB"):
        with pytest.raises(StopLoop):
            d.run()
    assert d.messages == []
    assert "Invalid message" in caplog.text


def test_run_wrong_field_count_is_dropped(caplog):
    d = DB({}, "x")
    _prepare_run(d, None)
    d.poller.poll.side_effect = [[(d.sub, zmq.POLLIN)], StopLoop()]
    d.sub.recv_multipart.return_value = [b"t", b"u"]
    with caplog.at_level(logging.ERROR, logger="ROB.DB"):
        with pytest.raises(StopLoop):
            d.run()
    assert d.messages == []
    assert "Invalid message" in caplog.text


def test_run_saves_after_one_second(monkeypatch):
    d = DB({}, "x")
    _prepare_run(d, None)
    d.poller.poll.side_effect = [[(d.sub, zmq.POLLIN)], StopLoop()]
    d.sub.recv_multipart.return_value = [
        b"t", b"u", b"2016-01-01T10:20:30.000001", b"user", b"d"
    ]
    values = iter([100.0, 105.0, 106.0, 107.0])
    monkeypatch.setattr(db, "time", SimpleNamespace(time=lambda: next(values)))
    with pytest.raises(StopLoop):
        d.run()
    d.save_to_db.assert_called_once_with()


def test_run_saves_when_1000_messages(monkeypatch):
    d = DB({}, "x")
    _prepare_run(d, None)
    d.poller.poll.side_effect = [[(d.sub, zmq.POLLIN)]] * 1000 + [StopLoop()]
    d.sub.recv_multipart.return_value = [
        b"t", b"u", b"2016-01-01T10:20:30.000001", b"user", b"d"
    ]
    # time never advances, so only the count threshold can trigger the save
    monkeypatch.setattr(db, "time", SimpleNamespace(time=lambda: 50.0))
    with pytest.raises(StopLoop):
        d.run()
    assert len(d.messages) == 1000
    d.save_to_db.assert_called_once_with()
