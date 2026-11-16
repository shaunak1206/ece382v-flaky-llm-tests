import datetime
import logging
import multiprocessing
import sqlite3
from unittest.mock import MagicMock, call, patch

import pytest
import zmq
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from reactobus.db import DB, Base, Message


class StopLoop(Exception):
    """Raised by a mocked poller to break out of DB.run()."""


VALID_FRAMES = [
    b"lava-test",
    b"uuid-1",
    b"2016-05-12T10:20:30.123456",
    b"bob",
    b'{"result": "pass"}',
]


@pytest.fixture
def runtime_mocks():
    """Mock setproctitle and zmq so setup() does not touch the system."""
    with patch("reactobus.db.setproctitle") as proctitle, patch(
        "reactobus.db.zmq"
    ) as mock_zmq:
        yield proctitle, mock_zmq


@pytest.fixture
def no_setup():
    """Replace DB.setup so run() can be tested without zmq or a database."""
    with patch.object(DB, "setup") as mock_setup:
        yield mock_setup


def _setup_db(tmp_path):
    db_file = tmp_path / "reactobus.db"
    db = DB({"url": "sqlite:///" + str(db_file)}, "ipc://outbound")
    db.setup()
    return db, db_file


def _make_run_db():
    db = DB({"url": "sqlite:///:memory:"}, "ipc://outbound")
    db.sub = MagicMock(name="sub")
    db.sub.recv_multipart.return_value = list(VALID_FRAMES)
    db.poller = MagicMock(name="poller")
    return db


def _ready(db):
    return [(db.sub, zmq.POLLIN)]


# ---------------------------------------------------------------- Message


def test_message_tablename_and_primary_key():
    assert issubclass(Message, Base)
    assert Message.__tablename__ == "messages"
    assert [c.name for c in Message.__table__.primary_key.columns] == ["id"]


def test_message_has_expected_columns():
    names = {c.name for c in Message.__table__.columns}
    assert names == {"id", "topic", "uuid", "datetime", "username", "data"}


def test_message_indexed_columns():
    indexed = {c.name for c in Message.__table__.columns if c.index}
    assert indexed == {"topic", "uuid", "datetime", "username"}


def test_message_datetime_column_is_timezone_aware():
    assert Message.__table__.c.datetime.type.timezone is True


def test_message_init_sets_attributes_and_leaves_id_unset():
    dt = datetime.datetime(2016, 5, 12, 10, 20, 30, 123456)
    msg = Message(
        topic="lava-test", uuid="uuid-1", datetime=dt, username="bob", data="{}"
    )
    assert msg.topic == "lava-test"
    assert msg.uuid == "uuid-1"
    assert msg.datetime == dt
    assert msg.username == "bob"
    assert msg.data == "{}"
    assert msg.id is None


def test_message_roundtrip_through_sqlite(tmp_path):
    engine = create_engine("sqlite:///" + str(tmp_path / "message.db"))
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    dt = datetime.datetime(2016, 5, 12, 10, 20, 30, 123456)
    session = Session()
    session.add(
        Message(topic="t", uuid="u", datetime=dt, username="bob", data="payload")
    )
    session.commit()
    session.close()

    session = Session()
    row = session.query(Message).one()
    assert row.id is not None
    assert row.topic == "t"
    assert row.uuid == "u"
    assert row.datetime == dt
    assert row.username == "bob"
    assert row.data == "payload"
    session.close()
    engine.dispose()


# ---------------------------------------------------------------- DB init


def test_db_init_stores_options_and_defaults():
    options = {"url": "sqlite:///:memory:"}
    db = DB(options, "ipc://outbound")
    assert db.options is options
    assert db.outbound == "ipc://outbound"
    assert db.engine is None
    assert db.sessions is None
    assert db.messages == []


def test_db_is_a_process():
    assert isinstance(DB({}, "ipc://x"), multiprocessing.Process)


def test_db_message_buffers_are_not_shared():
    first = DB({}, "ipc://a")
    second = DB({}, "ipc://b")
    first.messages.append("something")
    assert second.messages == []


# ---------------------------------------------------------------- setup


def test_setup_subscribes_to_outbound(runtime_mocks, tmp_path):
    _, mock_zmq = runtime_mocks
    db, _ = _setup_db(tmp_path)

    context = mock_zmq.Context.instance.return_value
    socket = context.socket.return_value
    context.socket.assert_called_once_with(mock_zmq.SUB)
    socket.setsockopt.assert_called_once_with(mock_zmq.SUBSCRIBE, b"")
    socket.connect.assert_called_once_with("ipc://outbound")
    mock_zmq.Poller.return_value.register.assert_called_once_with(
        socket, mock_zmq.POLLIN
    )
    assert db.sub is socket


def test_setup_sets_process_title(runtime_mocks, tmp_path):
    proctitle, _ = runtime_mocks
    _setup_db(tmp_path)
    proctitle.assert_called_once_with("ReactOBus [db]")


def test_setup_creates_tables_and_session_factory(runtime_mocks, tmp_path):
    _, mock_zmq = runtime_mocks
    db, _ = _setup_db(tmp_path)

    assert db.engine is not None
    assert "messages" in inspect(db.engine).get_table_names()

    session = db.sessions()
    assert session.bind is db.engine
    session.close()


# ---------------------------------------------------------------- save_to_db


def test_save_to_db_writes_rows_and_clears_buffer(runtime_mocks, tmp_path):
    db, db_file = _setup_db(tmp_path)
    dt = datetime.datetime(2016, 5, 12, 10, 20, 30)
    db.messages = [
        Message(topic="a", uuid="1", datetime=dt, username="u", data="d1"),
        Message(topic="b", uuid="2", datetime=dt, username="u", data="d2"),
    ]

    db.save_to_db()

    assert db.messages == []
    conn = sqlite3.connect(str(db_file))
    try:
        count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    finally:
        conn.close()
    assert count == 2


def test_save_to_db_with_empty_buffer_is_harmless(runtime_mocks, tmp_path):
    db, db_file = _setup_db(tmp_path)
    db.save_to_db()
    assert db.messages == []
    conn = sqlite3.connect(str(db_file))
    try:
        count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    finally:
        conn.close()
    assert count == 0


def test_save_to_db_retries_then_succeeds():
    db = DB({}, "ipc://x")
    messages = [Message(topic="t", uuid="u", data="d")]
    db.messages = messages
    session = MagicMock()
    db.sessions = MagicMock(return_value=session)
    session.bulk_save_objects.side_effect = [
        SQLAlchemyError("one"),
        SQLAlchemyError("two"),
        None,
    ]

    db.save_to_db()

    assert session.bulk_save_objects.call_count == 3
    session.commit.assert_called_once_with()
    assert db.messages == []


def test_save_to_db_gives_up_after_three_attempts(caplog):
    db = DB({}, "ipc://x")
    messages = [Message(topic="t", uuid="u", data="d")]
    db.messages = messages
    session = MagicMock()
    db.sessions = MagicMock(return_value=session)
    session.bulk_save_objects.side_effect = SQLAlchemyError("boom")

    with caplog.at_level(logging.ERROR, logger="ROB.DB"):
        db.save_to_db()

    assert session.bulk_save_objects.call_count == 3
    session.commit.assert_not_called()
    # Messages are kept so they can be retried later
    assert db.messages is messages
    assert len(db.messages) == 1

    logged = [r.getMessage() for r in caplog.records if r.name == "ROB.DB"]
    assert "Unable to commit to the database, retrying later on" in logged
    assert "Database error: boom" in logged


# ---------------------------------------------------------------- run


def test_run_stores_valid_message_and_polls_with_timeouts(no_setup):
    db = _make_run_db()
    db.poller.poll.side_effect = [_ready(db), StopLoop()]

    with patch("reactobus.db.time") as mock_time, patch.object(
        db, "save_to_db"
    ) as save, pytest.raises(StopLoop):
        mock_time.time.return_value = 0.0
        db.run()

    no_setup.assert_called_once_with()
    save.assert_not_called()
    # Wait forever while empty, then wait up to 1000ms once a message is buffered
    assert db.poller.poll.call_args_list == [call(-1), call(1000)]

    assert len(db.messages) == 1
    msg = db.messages[0]
    assert msg.topic == "lava-test"
    assert msg.uuid == "uuid-1"
    assert msg.datetime == datetime.datetime(2016, 5, 12, 10, 20, 30, 123456)
    assert msg.username == "bob"
    assert msg.data == '{"result": "pass"}'


def test_run_logs_and_skips_invalid_datetime(no_setup, caplog):
    db = _make_run_db()
    db.sub.recv_multipart.return_value = [b"t", b"u", b"not-a-date", b"bob", b"d"]
    db.poller.poll.side_effect = [_ready(db), StopLoop()]

    with caplog.at_level(logging.ERROR, logger="ROB.DB"), patch(
        "reactobus.db.time"
    ) as mock_time, pytest.raises(StopLoop):
        mock_time.time.return_value = 0.0
        db.run()

    assert db.messages == []
    logged = [r.getMessage() for r in caplog.records if r.name == "ROB.DB"]
    assert any(text.startswith("Invalid message") for text in logged)


def test_run_logs_row_build_failure_and_continues(no_setup, caplog):
    db = _make_run_db()
    db.poller.poll.side_effect = [_ready(db), StopLoop()]

    with caplog.at_level(logging.ERROR, logger="ROB.DB"), patch(
        "reactobus.db.Message", side_effect=SQLAlchemyError("bad row")
    ), patch("reactobus.db.time") as mock_time, pytest.raises(StopLoop):
        mock_time.time.return_value = 0.0
        db.run()

    assert db.messages == []
    logged = [r.getMessage() for r in caplog.records if r.name == "ROB.DB"]
    assert "Unable to build the new message row: bad row" in logged


def test_run_saves_when_buffer_reaches_1000(no_setup):
    db = _make_run_db()
    db.poller.poll.side_effect = [_ready(db)] * 1000 + [StopLoop()]
    sizes_at_save = []

    with patch("reactobus.db.time") as mock_time, patch.object(
        db, "save_to_db", side_effect=lambda: sizes_at_save.append(len(db.messages))
    ), pytest.raises(StopLoop):
        mock_time.time.return_value = 0.0
        db.run()

    assert sizes_at_save == [1000]


def test_run_saves_after_one_second(no_setup):
    db = _make_run_db()
    db.poller.poll.side_effect = [_ready(db), StopLoop()]

    with patch("reactobus.db.time") as mock_time, patch.object(
        db, "save_to_db"
    ) as save, pytest.raises(StopLoop):
        # last_save is set to 0.0, then "now" is 2.0 (more than 1s later)
        mock_time.time.side_effect = [0.0, 2.0]
        db.run()

    save.assert_called_once_with()
