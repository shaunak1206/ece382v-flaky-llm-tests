import datetime
from unittest import mock

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

import reactobus.db as db_module
from reactobus.db import Base, DB, Message


class StopLoop(Exception):
    pass


@pytest.fixture
def sqlite_session_factory():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def make_msg(i=0):
    return Message(
        topic="org.example.%d" % i,
        uuid="uuid-%d" % i,
        datetime=datetime.datetime(2016, 1, 2, 3, 4, 5, 6),
        username="user%d" % i,
        data='{"a": %d}' % i,
    )


def make_db_for_run(poll_side_effect, recv_values):
    obj = DB({"url": "sqlite://"}, "ipc:///tmp/unused")
    obj.setup = lambda: None
    obj.sub = mock.MagicMock()
    obj.sub.recv_multipart.side_effect = recv_values
    obj.poller = mock.MagicMock()
    obj.poller.poll.side_effect = poll_side_effect
    return obj


def test_message_tablename_and_columns():
    assert Message.__tablename__ == "messages"
    cols = set(Message.__table__.columns.keys())
    assert cols == {"id", "topic", "uuid", "datetime", "username", "data"}


def test_message_primary_key_is_id():
    pks = [c.name for c in Message.__table__.primary_key.columns]
    assert pks == ["id"]


def test_message_indexed_columns():
    table = Message.__table__
    for name in ("topic", "uuid", "datetime", "username"):
        assert table.columns[name].index is True
    assert not table.columns["data"].index


def test_message_constructor_sets_attributes():
    m = make_msg(3)
    assert m.topic == "org.example.3"
    assert m.uuid == "uuid-3"
    assert m.username == "user3"
    assert m.data == '{"a": 3}'
    assert m.datetime == datetime.datetime(2016, 1, 2, 3, 4, 5, 6)
    assert m.id is None


def test_message_roundtrip_in_sqlite(sqlite_session_factory):
    session = sqlite_session_factory()
    session.add(make_msg(1))
    session.commit()
    row = session.query(Message).one()
    assert row.id is not None
    assert row.topic == "org.example.1"
    assert row.uuid == "uuid-1"
    assert row.username == "user1"
    assert row.data == '{"a": 1}'
    assert row.datetime == datetime.datetime(2016, 1, 2, 3, 4, 5, 6)


def test_db_init_defaults():
    obj = DB({"url": "sqlite://"}, "tcp://127.0.0.1:5555")
    assert obj.options == {"url": "sqlite://"}
    assert obj.outbound == "tcp://127.0.0.1:5555"
    assert obj.engine is None
    assert obj.sessions is None
    assert obj.messages == []


def test_setup_creates_tables_and_sessions(tmp_path):
    url = "sqlite:///" + str(tmp_path / "test.db")
    obj = DB({"url": url}, "ipc:///tmp/unused")
    with mock.patch.object(db_module, "zmq") as zmq_mock, mock.patch.object(
        db_module, "setproctitle"
    ) as title:
        obj.setup()
    title.assert_called_once_with("ReactOBus [db]")
    sub = zmq_mock.Context.instance.return_value.socket.return_value
    sub.connect.assert_called_once_with("ipc:///tmp/unused")
    obj.poller.register.assert_called_once()
    assert obj.engine is not None
    assert "messages" in inspect(obj.engine).get_table_names()
    session = obj.sessions()
    assert session.query(Message).count() == 0


def test_save_to_db_success_clears_messages(sqlite_session_factory):
    obj = DB({"url": "sqlite://"}, "x")
    obj.sessions = sqlite_session_factory
    obj.messages = [make_msg(i) for i in range(5)]
    obj.save_to_db()
    assert obj.messages == []
    assert sqlite_session_factory().query(Message).count() == 5


def test_save_to_db_empty_list(sqlite_session_factory):
    obj = DB({"url": "sqlite://"}, "x")
    obj.sessions = sqlite_session_factory
    obj.messages = []
    obj.save_to_db()
    assert obj.messages == []
    assert sqlite_session_factory().query(Message).count() == 0


def test_save_to_db_retries_three_times_and_keeps_messages(caplog):
    session = mock.MagicMock()
    session.bulk_save_objects.side_effect = SQLAlchemyError("boom")
    obj = DB({"url": "sqlite://"}, "x")
    obj.sessions = mock.MagicMock(return_value=session)
    msgs = [make_msg(1), make_msg(2)]
    obj.messages = list(msgs)
    with caplog.at_level("ERROR", logger="ROB.DB"):
        obj.save_to_db()
    assert session.bulk_save_objects.call_count == 3
    session.commit.assert_not_called()
    assert obj.messages == msgs
    assert "Unable to commit to the database" in caplog.text
    assert "boom" in caplog.text


def test_save_to_db_succeeds_after_transient_failure():
    session = mock.MagicMock()
    session.bulk_save_objects.side_effect = [SQLAlchemyError("x"), None]
    obj = DB({"url": "sqlite://"}, "x")
    obj.sessions = mock.MagicMock(return_value=session)
    obj.messages = [make_msg(1)]
    obj.save_to_db()
    assert session.bulk_save_objects.call_count == 2
    session.commit.assert_called_once()
    assert obj.messages == []


def test_run_parses_valid_message():
    raw = [
        b"org.example.topic",
        b"1234-uuid",
        b"2016-05-06T07:08:09.123456",
        b"someuser",
        b'{"k": "v"}',
    ]
    obj = make_db_for_run(
        [[(None, 1)], StopLoop()], [raw]
    )
    # Make the dict lookup of self.sub work
    obj.poller.poll.side_effect = [[(obj.sub, db_module.zmq.POLLIN)], StopLoop()]
    with mock.patch.object(db_module, "time") as t:
        t.time.side_effect = [0, 0.1]
        with pytest.raises(StopLoop):
            obj.run()
    assert len(obj.messages) == 1
    m = obj.messages[0]
    assert m.topic == "org.example.topic"
    assert m.uuid == "1234-uuid"
    assert m.username == "someuser"
    assert m.data == '{"k": "v"}'
    assert m.datetime == datetime.datetime(2016, 5, 6, 7, 8, 9, 123456)


def test_run_invalid_date_is_dropped(caplog):
    raw = [b"t", b"u", b"not-a-date", b"user", b"data"]
    obj = make_db_for_run(StopLoop(), [raw])
    obj.poller.poll.side_effect = [[(obj.sub, db_module.zmq.POLLIN)], StopLoop()]
    with mock.patch.object(db_module, "time") as t:
        t.time.side_effect = [0, 0.1]
        with caplog.at_level("ERROR", logger="ROB.DB"):
            with pytest.raises(StopLoop):
                obj.run()
    assert obj.messages == []
    assert "Invalid message" in caplog.text


def test_run_wrong_field_count_is_dropped():
    raw = [b"only", b"three", b"fields"]
    obj = make_db_for_run(StopLoop(), [raw])
    obj.poller.poll.side_effect = [[(obj.sub, db_module.zmq.POLLIN)], StopLoop()]
    with mock.patch.object(db_module, "time") as t:
        t.time.side_effect = [0, 0.1]
        with pytest.raises(StopLoop):
            obj.run()
    assert obj.messages == []


def test_run_saves_when_more_than_one_second_elapsed():
    raw = [b"t", b"u", b"2016-05-06T07:08:09.000001", b"user", b"data"]
    obj = make_db_for_run(StopLoop(), [raw])
    obj.poller.poll.side_effect = [[(obj.sub, db_module.zmq.POLLIN)], StopLoop()]
    obj.save_to_db = mock.MagicMock()
    with mock.patch.object(db_module, "time") as t:
        t.time.side_effect = [0, 10]
        with pytest.raises(StopLoop):
            obj.run()
    obj.save_to_db.assert_called_once_with()


def test_run_poll_timeout_depends_on_pending_messages():
    raw = [b"t", b"u", b"2016-05-06T07:08:09.000001", b"user", b"data"]
    obj = make_db_for_run(StopLoop(), [raw])
    obj.poller.poll.side_effect = [
        [(obj.sub, db_module.zmq.POLLIN)],
        [],
        StopLoop(),
    ]
    obj.save_to_db = mock.MagicMock()
    with mock.patch.object(db_module, "time") as t:
        t.time.side_effect = [0, 0.1, 0.2]
        with pytest.raises(StopLoop):
            obj.run()
    timeouts = [c[0][0] for c in obj.poller.poll.call_args_list]
    assert timeouts == [-1, 1000, 1000]
