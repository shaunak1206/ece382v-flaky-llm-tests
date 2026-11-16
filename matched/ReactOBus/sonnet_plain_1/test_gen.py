import datetime
import logging
from unittest import mock

import pytest
import zmq
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

import reactobus.db as rdb
from reactobus.db import DB, Base, Message


class StopLoop(Exception):
    pass


def make_sqlite_db():
    db = DB({"url": "sqlite://"}, "ipc:///tmp/unused")
    db.engine = create_engine("sqlite://")
    Base.metadata.create_all(db.engine)
    db.sessions = sessionmaker(bind=db.engine)
    return db


def make_msg(topic="org.test", uuid="u1", dt=None, username="user", data="{}"):
    return Message(
        topic=topic,
        uuid=uuid,
        datetime=dt or datetime.datetime(2016, 1, 2, 3, 4, 5),
        username=username,
        data=data,
    )


def raw_message(dt="2016-01-02T03:04:05.123456"):
    return [b"org.test.topic", b"uuid-1", dt.encode(), b"someone", b'{"a": 1}']


def make_run_db(poll_side_effect, recv_return=None):
    db = DB({"url": "sqlite://"}, "ipc:///tmp/unused")
    db.sub = mock.MagicMock()
    if recv_return is not None:
        db.sub.recv_multipart.return_value = recv_return
    db.poller = mock.MagicMock()
    db.poller.poll.side_effect = poll_side_effect
    return db


# ---------- Message ----------

def test_message_table_name():
    assert Message.__tablename__ == "messages"


def test_message_columns():
    cols = set(Message.__table__.columns.keys())
    assert cols == {"id", "topic", "uuid", "datetime", "username", "data"}


def test_message_primary_key_and_indexes():
    table = Message.__table__
    assert table.c.id.primary_key
    for name in ("topic", "uuid", "datetime", "username"):
        assert table.c[name].index is True
    assert not table.c.data.index


def test_message_constructor_sets_attributes():
    dt = datetime.datetime(2020, 5, 6, 7, 8, 9)
    m = Message(topic="t", uuid="u", datetime=dt, username="name", data="d")
    assert (m.topic, m.uuid, m.datetime, m.username, m.data) == (
        "t", "u", dt, "name", "d")
    assert m.id is None


def test_message_persist_and_query_roundtrip():
    db = make_sqlite_db()
    dt = datetime.datetime(2016, 1, 2, 3, 4, 5, 123456)
    session = db.sessions()
    session.add(make_msg(topic="a.b", uuid="xyz", dt=dt, username="bob", data="payload"))
    session.commit()

    row = db.sessions().query(Message).filter_by(uuid="xyz").one()
    assert row.id is not None
    assert row.topic == "a.b"
    assert row.username == "bob"
    assert row.data == "payload"
    assert row.datetime == dt


# ---------- DB ----------

def test_db_init_attributes():
    options = {"url": "sqlite://"}
    db = DB(options, "tcp://127.0.0.1:5555")
    assert db.options is options
    assert db.outbound == "tcp://127.0.0.1:5555"
    assert db.engine is None
    assert db.sessions is None
    assert db.messages == []


def test_db_is_a_process():
    import multiprocessing
    assert issubclass(DB, multiprocessing.Process)


def test_setup_configures_zmq_and_sqlalchemy():
    db = DB({"url": "sqlite://"}, "ipc:///tmp/unused")
    with mock.patch.object(rdb, "setproctitle") as title, \
            mock.patch.object(rdb, "zmq") as zmq_mock:
        db.setup()

    title.assert_called_once_with("ReactOBus [db]")
    sock = zmq_mock.Context.instance.return_value.socket.return_value
    sock.setsockopt.assert_called_once_with(zmq_mock.SUBSCRIBE, b"")
    sock.connect.assert_called_once_with("ipc:///tmp/unused")
    db.poller.register.assert_called_once_with(sock, zmq_mock.POLLIN)
    assert db.engine is not None
    assert db.sessions is not None
    # The table must have been created
    assert "messages" in db.engine.table_names()


def test_save_to_db_success_clears_messages():
    db = make_sqlite_db()
    db.messages = [make_msg(uuid="1"), make_msg(uuid="2")]
    db.save_to_db()

    assert db.messages == []
    rows = db.sessions().query(Message).all()
    assert sorted(r.uuid for r in rows) == ["1", "2"]


def test_save_to_db_empty_list_is_ok():
    db = make_sqlite_db()
    db.save_to_db()
    assert db.messages == []
    assert db.sessions().query(Message).count() == 0


def test_save_to_db_retries_three_times_and_keeps_messages(caplog):
    db = DB({"url": "sqlite://"}, "ipc:///tmp/unused")
    session = mock.MagicMock()
    session.bulk_save_objects.side_effect = SQLAlchemyError("boom")
    db.sessions = mock.MagicMock(return_value=session)
    msgs = [make_msg()]
    db.messages = list(msgs)

    with caplog.at_level(logging.ERROR, logger="ROB.DB"):
        db.save_to_db()

    assert session.bulk_save_objects.call_count == 3
    session.commit.assert_not_called()
    assert db.messages == msgs
    text = caplog.text
    assert "Unable to commit to the database" in text
    assert "boom" in text


def test_save_to_db_succeeds_after_transient_failure():
    db = DB({"url": "sqlite://"}, "ipc:///tmp/unused")
    session = mock.MagicMock()
    session.bulk_save_objects.side_effect = [SQLAlchemyError("once"), None]
    db.sessions = mock.MagicMock(return_value=session)
    db.messages = [make_msg()]

    db.save_to_db()

    assert session.bulk_save_objects.call_count == 2
    session.commit.assert_called_once_with()
    assert db.messages == []


def test_run_parses_valid_message():
    db = make_run_db(
        [[(None, 0)], StopLoop()], recv_return=raw_message())
    # Make the poll result refer to the subscriber socket
    db.poller.poll.side_effect = [[(db.sub, zmq.POLLIN)], StopLoop()]

    with mock.patch.object(DB, "setup"), \
            mock.patch.object(db, "save_to_db") as save:
        with pytest.raises(StopLoop):
            db.run()

    assert save.call_count == 0
    assert len(db.messages) == 1
    m = db.messages[0]
    assert m.topic == "org.test.topic"
    assert m.uuid == "uuid-1"
    assert m.username == "someone"
    assert m.data == '{"a": 1}'
    assert m.datetime == datetime.datetime(2016, 1, 2, 3, 4, 5, 123456)


def test_run_poll_timeout_depends_on_pending_messages():
    db = make_run_db(None, recv_return=raw_message())
    db.poller.poll.side_effect = [[(db.sub, zmq.POLLIN)], StopLoop()]

    with mock.patch.object(DB, "setup"), \
            mock.patch.object(db, "save_to_db"):
        with pytest.raises(StopLoop):
            db.run()

    timeouts = [c[0][0] for c in db.poller.poll.call_args_list]
    # No pending messages at first: wait forever; then 1s
    assert timeouts == [-1, 1000]


def test_run_invalid_date_is_dropped(caplog):
    db = make_run_db(None, recv_return=raw_message(dt="not-a-date"))
    db.poller.poll.side_effect = [[(db.sub, zmq.POLLIN)], StopLoop()]

    with mock.patch.object(DB, "setup"), \
            mock.patch.object(db, "save_to_db"):
        with caplog.at_level(logging.ERROR, logger="ROB.DB"):
            with pytest.raises(StopLoop):
                db.run()

    assert db.messages == []
    assert "Invalid message" in caplog.text


def test_run_wrong_number_of_fields_is_dropped(caplog):
    db = make_run_db(None, recv_return=[b"topic", b"uuid"])
    db.poller.poll.side_effect = [[(db.sub, zmq.POLLIN)], StopLoop()]

    with mock.patch.object(DB, "setup"), \
            mock.patch.object(db, "save_to_db"):
        with caplog.at_level(logging.ERROR, logger="ROB.DB"):
            with pytest.raises(StopLoop):
                db.run()

    assert db.messages == []
    assert "Invalid message" in caplog.text


def test_run_saves_when_more_than_one_second_elapsed():
    db = make_run_db(None, recv_return=raw_message())
    db.poller.poll.side_effect = [[(db.sub, zmq.POLLIN)], StopLoop()]

    with mock.patch.object(DB, "setup"), \
            mock.patch.object(db, "save_to_db") as save, \
            mock.patch.object(rdb, "time") as time_mock:
        time_mock.time.side_effect = [100.0, 105.0]
        with pytest.raises(StopLoop):
            db.run()

    save.assert_called_once_with()


def test_run_saves_when_1000_messages_reached():
    db = make_run_db(None, recv_return=raw_message())
    db.poller.poll.side_effect = [[(db.sub, zmq.POLLIN)], StopLoop()]

    def fake_setup():
        pass

    with mock.patch.object(DB, "setup", side_effect=fake_setup), \
            mock.patch.object(db, "save_to_db") as save, \
            mock.patch.object(rdb, "time") as time_mock:
        time_mock.time.side_effect = [100.0, 100.1]
        # run() resets messages at start, so seed through the first receive
        original_append = list.append
        seeded = {"done": False}

        class SeedList(list):
            def append(self, item):
                if not seeded["done"]:
                    seeded["done"] = True
                    for _ in range(999):
                        original_append(self, item)
                original_append(self, item)

        with mock.patch.object(rdb, "Message", side_effect=lambda **kw: kw):
            # Replace the list created in run() with a seeded one
            with mock.patch.object(DB, "messages", create=True):
                pass
            db.sub.recv_multipart.return_value = raw_message()
            # Use a custom poll to swap the list in right before first receive
            def poll(timeout):
                if not seeded["done"] and not isinstance(db.messages, SeedList):
                    db.messages = SeedList()
                    return [(db.sub, zmq.POLLIN)]
                raise StopLoop()
            db.poller.poll.side_effect = poll
            with pytest.raises(StopLoop):
                db.run()

    assert len(db.messages) == 1000
    save.assert_called_once_with()
