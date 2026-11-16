import datetime
from unittest import mock

import multiprocessing
import pytest
import zmq
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from reactobus.db import DB, Base, Message


class StopLoop(Exception):
    pass


@pytest.fixture
def sqlite_sessions():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def make_db():
    return DB({"url": "sqlite://"}, "ipc:///tmp/outbound-test")


def make_message(i=0):
    return Message(
        topic="org.example.topic%d" % i,
        uuid="uuid-%d" % i,
        datetime=datetime.datetime(2016, 1, 2, 3, 4, 5, 678),
        username="user",
        data='{"k": %d}' % i,
    )


# ---------- Message ----------

def test_message_tablename():
    assert Message.__tablename__ == "messages"


def test_message_columns():
    cols = Message.__table__.columns
    assert set(cols.keys()) == {"id", "topic", "uuid", "datetime", "username", "data"}
    assert cols["id"].primary_key


def test_message_indexed_columns():
    cols = Message.__table__.columns
    for name in ("topic", "uuid", "datetime", "username"):
        assert cols[name].index
    assert not cols["data"].index


def test_message_constructor_sets_attributes():
    msg = make_message(1)
    assert msg.topic == "org.example.topic1"
    assert msg.uuid == "uuid-1"
    assert msg.username == "user"
    assert msg.data == '{"k": 1}'
    assert msg.datetime == datetime.datetime(2016, 1, 2, 3, 4, 5, 678)
    assert msg.id is None


def test_message_persisted_and_queried(sqlite_sessions):
    session = sqlite_sessions()
    session.add(make_message(1))
    session.commit()
    row = session.query(Message).one()
    assert row.id == 1
    assert row.topic == "org.example.topic1"
    assert row.datetime.year == 2016
    assert row.data == '{"k": 1}'


# ---------- DB ----------

def test_db_init():
    db = make_db()
    assert isinstance(db, multiprocessing.Process)
    assert db.options == {"url": "sqlite://"}
    assert db.outbound == "ipc:///tmp/outbound-test"
    assert db.engine is None
    assert db.sessions is None
    assert db.messages == []


def test_db_setup_creates_engine_and_tables():
    db = make_db()
    with mock.patch("reactobus.db.setproctitle") as title, mock.patch(
        "reactobus.db.zmq.Context"
    ) as ctx_cls:
        db.setup()
    title.assert_called_once_with("ReactOBus [db]")
    sub = ctx_cls.instance.return_value.socket.return_value
    ctx_cls.instance.return_value.socket.assert_called_once_with(zmq.SUB)
    sub.setsockopt.assert_called_once_with(zmq.SUBSCRIBE, b"")
    sub.connect.assert_called_once_with("ipc:///tmp/outbound-test")
    assert db.engine is not None
    assert db.engine.dialect.has_table(db.engine.connect(), "messages")
    session = db.sessions()
    assert session.query(Message).count() == 0


def test_save_to_db_success_clears_messages(sqlite_sessions):
    db = make_db()
    db.sessions = sqlite_sessions
    db.messages = [make_message(i) for i in range(3)]
    db.save_to_db()
    assert db.messages == []
    assert sqlite_sessions().query(Message).count() == 3


def test_save_to_db_retries_then_succeeds():
    db = make_db()
    session = mock.MagicMock()
    session.bulk_save_objects.side_effect = [SQLAlchemyError("a"), SQLAlchemyError("b"), None]
    db.sessions = mock.MagicMock(return_value=session)
    db.messages = [make_message()]
    db.save_to_db()
    assert session.bulk_save_objects.call_count == 3
    assert session.commit.call_count == 1
    assert db.messages == []


def test_save_to_db_gives_up_after_three_failures(caplog):
    db = make_db()
    session = mock.MagicMock()
    session.commit.side_effect = SQLAlchemyError("boom")
    db.sessions = mock.MagicMock(return_value=session)
    msgs = [make_message()]
    db.messages = msgs
    with caplog.at_level("ERROR", logger="ROB.DB"):
        db.save_to_db()
    assert session.bulk_save_objects.call_count == 3
    assert db.messages == msgs
    assert "Unable to commit to the database" in caplog.text
    assert "boom" in caplog.text


def _prepare_run(db, recv_messages, poll_results):
    db.sub = mock.MagicMock()
    db.poller = mock.MagicMock()
    db.sub.recv_multipart.side_effect = recv_messages
    results = [{db.sub: zmq.POLLIN} if r else {} for r in poll_results]
    db.poller.poll.side_effect = results + [StopLoop()]


def test_run_valid_message_saved_when_time_elapsed():
    db = make_db()
    _prepare_run(
        db,
        [[b"topic", b"uuid", b"2016-01-02T03:04:05.000678", b"user", b"data"]],
        [True],
    )
    saved = []
    with mock.patch.object(db, "setup"), mock.patch(
        "reactobus.db.time"
    ) as fake_time, mock.patch.object(
        db, "save_to_db", side_effect=lambda: saved.extend(db.messages)
    ):
        fake_time.time.side_effect = [0, 10]
        with pytest.raises(StopLoop):
            db.run()
    assert len(saved) == 1
    m = saved[0]
    assert isinstance(m, Message)
    assert m.topic == "topic"
    assert m.uuid == "uuid"
    assert m.username == "user"
    assert m.data == "data"
    assert m.datetime == datetime.datetime(2016, 1, 2, 3, 4, 5, 678)


def test_run_invalid_date_is_dropped():
    db = make_db()
    _prepare_run(db, [[b"t", b"u", b"not-a-date", b"user", b"d"]], [True])
    with mock.patch.object(db, "setup"), mock.patch.object(
        db, "save_to_db"
    ) as save:
        with pytest.raises(StopLoop):
            db.run()
    assert db.messages == []
    save.assert_not_called()


def test_run_wrong_field_count_is_dropped(caplog):
    db = make_db()
    _prepare_run(db, [[b"only", b"two"]], [True])
    with mock.patch.object(db, "setup"), mock.patch.object(db, "save_to_db"):
        with caplog.at_level("ERROR", logger="ROB.DB"):
            with pytest.raises(StopLoop):
                db.run()
    assert db.messages == []
    assert "Invalid message" in caplog.text


def test_run_poll_timeout_depends_on_pending_messages():
    db = make_db()
    _prepare_run(
        db,
        [[b"t", b"u", b"2016-01-02T03:04:05.000001", b"user", b"d"]],
        [True],
    )
    with mock.patch.object(db, "setup"), mock.patch(
        "reactobus.db.time"
    ) as fake_time, mock.patch.object(db, "save_to_db"):
        fake_time.time.return_value = 0
        with pytest.raises(StopLoop):
            db.run()
    assert db.poller.poll.call_args_list == [mock.call(-1), mock.call(1000)]
    assert len(db.messages) == 1


def test_run_does_not_save_before_one_second():
    db = make_db()
    _prepare_run(
        db,
        [[b"t", b"u", b"2016-01-02T03:04:05.000001", b"user", b"d"]],
        [True],
    )
    with mock.patch.object(db, "setup"), mock.patch(
        "reactobus.db.time"
    ) as fake_time, mock.patch.object(db, "save_to_db") as save:
        fake_time.time.return_value = 0.5
        with pytest.raises(StopLoop):
            db.run()
    save.assert_not_called()
