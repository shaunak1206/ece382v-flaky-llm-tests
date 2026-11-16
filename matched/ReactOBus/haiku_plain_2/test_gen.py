import datetime
import logging
import multiprocessing
from unittest import mock

import pytest
from sqlalchemy import DateTime, create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

import reactobus.db as dbmod
from reactobus.db import Base, DB, Message


VALID_MSG = [
    b"topic.name",
    b"uuid-1",
    b"2016-01-01T10:00:00.000000",
    b"alice",
    b'{"a": 1}',
]


class _StopLoop(Exception):
    """Raised by a mocked poller to break out of DB.run()."""


def make_message(**kwargs):
    values = dict(
        topic="topic.a",
        uuid="uuid-a",
        datetime=datetime.datetime(2016, 1, 1, 10, 0, 0),
        username="alice",
        data="{}",
    )
    values.update(kwargs)
    return Message(**values)


@pytest.fixture
def engine(tmp_path):
    eng = create_engine("sqlite:///" + str(tmp_path / "rob.db"))
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


# ---------------------------------------------------------------- Message


def test_message_tablename():
    assert Message.__tablename__ == "messages"


def test_message_has_expected_columns():
    assert set(Message.__table__.columns.keys()) == {
        "id",
        "topic",
        "uuid",
        "datetime",
        "username",
        "data",
    }


def test_message_primary_key_is_id():
    pk_names = [c.name for c in Message.__table__.primary_key.columns]
    assert pk_names == ["id"]


def test_message_indexed_columns():
    for name in ("topic", "uuid", "datetime", "username"):
        assert Message.__table__.c[name].index is True
    assert Message.__table__.c["data"].index in (None, False)


def test_message_datetime_column_is_timezone_aware_type():
    col_type = Message.__table__.c.datetime.type
    assert isinstance(col_type, DateTime)
    assert col_type.timezone is True


def test_message_constructor_sets_attributes():
    when = datetime.datetime(2016, 5, 6, 7, 8, 9)
    msg = Message(
        topic="t",
        uuid="u",
        datetime=when,
        username="bob",
        data="payload",
    )
    assert msg.topic == "t"
    assert msg.uuid == "u"
    assert msg.datetime == when
    assert msg.username == "bob"
    assert msg.data == "payload"


def test_message_unset_attributes_are_none():
    msg = Message()
    assert msg.id is None
    assert msg.topic is None
    assert msg.uuid is None
    assert msg.datetime is None
    assert msg.username is None
    assert msg.data is None


def test_message_roundtrip_through_database(engine):
    session = sessionmaker(bind=engine)()
    try:
        session.add(make_message(topic="roundtrip", data="xyz"))
        session.commit()
    finally:
        session.close()

    session = sessionmaker(bind=engine)()
    try:
        row = session.query(Message).filter_by(topic="roundtrip").one()
        assert row.id is not None
        assert row.data == "xyz"
        assert row.datetime == datetime.datetime(2016, 1, 1, 10, 0, 0)
    finally:
        session.close()


# ---------------------------------------------------------------- DB init


def test_db_init_stores_options_and_outbound():
    options = {"url": "sqlite://"}
    db = DB(options, "tcp://127.0.0.1:5001")
    assert db.options is options
    assert db.outbound == "tcp://127.0.0.1:5001"
    assert db.engine is None
    assert db.sessions is None
    assert db.messages == []


def test_db_is_a_multiprocessing_process():
    db = DB({"url": "sqlite://"}, "tcp://127.0.0.1:5001")
    assert isinstance(db, multiprocessing.Process)


def test_db_instances_do_not_share_message_lists():
    first = DB({"url": "sqlite://"}, "tcp://a")
    second = DB({"url": "sqlite://"}, "tcp://b")
    first.messages.append(make_message())
    assert second.messages == []


# ---------------------------------------------------------------- DB setup


def test_setup_sets_process_title():
    db = DB({"url": "sqlite://"}, "tcp://127.0.0.1:5002")
    with mock.patch.object(dbmod, "setproctitle") as spt, mock.patch.object(
        dbmod, "zmq"
    ):
        db.setup()
    spt.assert_called_once_with("ReactOBus [db]")


def test_setup_subscribes_to_outbound():
    db = DB({"url": "sqlite://"}, "tcp://127.0.0.1:5003")
    with mock.patch.object(dbmod, "setproctitle"), mock.patch.object(
        dbmod, "zmq"
    ) as zmq_mock:
        db.setup()

    sub = zmq_mock.Context.instance.return_value.socket.return_value
    zmq_mock.Context.instance.return_value.socket.assert_called_once_with(
        zmq_mock.SUB
    )
    sub.setsockopt.assert_called_once_with(zmq_mock.SUBSCRIBE, b"")
    sub.connect.assert_called_once_with("tcp://127.0.0.1:5003")
    zmq_mock.Poller.return_value.register.assert_called_once_with(
        sub, zmq_mock.POLLIN
    )
    assert db.sub is sub


def test_setup_creates_engine_and_tables(tmp_path):
    url = "sqlite:///" + str(tmp_path / "setup.db")
    db = DB({"url": url}, "tcp://127.0.0.1:5004")
    with mock.patch.object(dbmod, "setproctitle"), mock.patch.object(
        dbmod, "zmq"
    ):
        db.setup()
    try:
        assert db.engine is not None
        assert db.sessions is not None
        assert "messages" in inspect(db.engine).get_table_names()
    finally:
        db.engine.dispose()


# ---------------------------------------------------------------- save_to_db


def test_save_to_db_persists_and_clears_messages(engine):
    db = DB({"url": "sqlite://"}, "tcp://x")
    db.engine = engine
    db.sessions = sessionmaker(bind=engine)
    db.messages = [make_message(uuid="u1"), make_message(uuid="u2")]

    db.save_to_db()

    assert db.messages == []
    session = db.sessions()
    try:
        assert session.query(Message).count() == 2
    finally:
        session.close()


def test_save_to_db_retries_then_succeeds():
    db = DB({"url": "sqlite://"}, "tcp://x")
    session = mock.MagicMock()
    session.bulk_save_objects.side_effect = [
        SQLAlchemyError("first"),
        SQLAlchemyError("second"),
        None,
    ]
    db.sessions = mock.MagicMock(return_value=session)
    db.messages = [make_message()]

    db.save_to_db()

    assert session.bulk_save_objects.call_count == 3
    session.commit.assert_called_once_with()
    assert db.messages == []


def test_save_to_db_gives_up_after_three_attempts(caplog):
    db = DB({"url": "sqlite://"}, "tcp://x")
    session = mock.MagicMock()
    session.bulk_save_objects.side_effect = SQLAlchemyError("disk on fire")
    db.sessions = mock.MagicMock(return_value=session)
    message = make_message()
    db.messages = [message]

    with caplog.at_level(logging.ERROR, logger="ROB.DB"):
        db.save_to_db()

    assert session.bulk_save_objects.call_count == 3
    session.commit.assert_not_called()
    assert db.messages == [message]
    assert "Unable to commit to the database" in caplog.text
    assert "disk on fire" in caplog.text


def test_save_to_db_with_no_messages_does_not_fail(engine):
    db = DB({"url": "sqlite://"}, "tcp://x")
    db.engine = engine
    db.sessions = sessionmaker(bind=engine)
    db.messages = []

    db.save_to_db()

    assert db.messages == []
    session = db.sessions()
    try:
        assert session.query(Message).count() == 0
    finally:
        session.close()


# ---------------------------------------------------------------- run loop


def _make_running_db(poll_side_effect, recv_value=None):
    db = DB({"url": "sqlite://"}, "tcp://x")
    db.sub = mock.MagicMock(name="sub")
    db.poller = mock.MagicMock(name="poller")
    db.poller.poll.side_effect = poll_side_effect
    db.sub.recv_multipart.return_value = recv_value
    return db


def test_run_stores_valid_message():
    db = _make_running_db(
        [[(None, dbmod.zmq.POLLIN)], _StopLoop()], recv_value=VALID_MSG
    )
    db.poller.poll.side_effect = [[(db.sub, dbmod.zmq.POLLIN)], _StopLoop()]

    with mock.patch.object(DB, "setup"), mock.patch.object(
        DB, "save_to_db"
    ) as save, mock.patch.object(dbmod, "time") as fake_time:
        fake_time.time.return_value = 100.0
        with pytest.raises(_StopLoop):
            db.run()

    assert len(db.messages) == 1
    msg = db.messages[0]
    assert msg.topic == "topic.name"
    assert msg.uuid == "uuid-1"
    assert msg.datetime == datetime.datetime(2016, 1, 1, 10, 0, 0)
    assert msg.username == "alice"
    assert msg.data == '{"a": 1}'
    save.assert_not_called()


def test_run_skips_invalid_message(caplog):
    bad = [b"topic", b"uuid", b"not-a-date", b"alice", b"data"]
    db = _make_running_db(None, recv_value=bad)
    db.poller.poll.side_effect = [[(db.sub, dbmod.zmq.POLLIN)], _StopLoop()]

    with mock.patch.object(DB, "setup"), mock.patch.object(
        DB, "save_to_db"
    ), mock.patch.object(dbmod, "time") as fake_time:
        fake_time.time.return_value = 100.0
        with caplog.at_level(logging.ERROR, logger="ROB.DB"):
            with pytest.raises(_StopLoop):
                db.run()

    assert db.messages == []
    assert "Invalid message" in caplog.text


def test_run_saves_when_batch_reaches_1000_messages():
    db = _make_running_db(None, recv_value=VALID_MSG)
    poll_results = [[(db.sub, dbmod.zmq.POLLIN)]] * 1000 + [_StopLoop()]
    db.poller.poll.side_effect = poll_results

    with mock.patch.object(DB, "setup"), mock.patch.object(
        DB, "save_to_db"
    ) as save, mock.patch.object(dbmod, "time") as fake_time:
        fake_time.time.return_value = 100.0
        with pytest.raises(_StopLoop):
            db.run()

    assert save.call_count == 1
    assert len(db.messages) == 1000


def test_run_saves_after_one_second_elapsed():
    db = _make_running_db(
        [[(None, dbmod.zmq.POLLIN)], _StopLoop()], recv_value=VALID_MSG
    )
    db.poller.poll.side_effect = [[(db.sub, dbmod.zmq.POLLIN)], _StopLoop()]

    with mock.patch.object(DB, "setup"), mock.patch.object(
        DB, "save_to_db"
    ) as save, mock.patch.object(dbmod, "time") as fake_time:
        # last_save = 0.0, then "now" = 5.0 (more than one second later)
        fake_time.time.side_effect = [0.0, 5.0, 5.0, 5.0]
        with pytest.raises(_StopLoop):
            db.run()

    assert save.call_count == 1


def test_run_polls_with_timeout_only_when_messages_pending():
    db = _make_running_db(None, recv_value=VALID_MSG)
    db.poller.poll.side_effect = [[(db.sub, dbmod.zmq.POLLIN)], _StopLoop()]

    with mock.patch.object(DB, "setup"), mock.patch.object(
        DB, "save_to_db"
    ), mock.patch.object(dbmod, "time") as fake_time:
        fake_time.time.return_value = 100.0
        with pytest.raises(_StopLoop):
            db.run()

    assert db.poller.poll.call_args_list == [
        mock.call(-1),
        mock.call(1000),
    ]
