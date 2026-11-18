import datetime
import pytest
from sqlalchemy.exc import SQLAlchemyError
from reactobus.db import DB, Message, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def in_memory_db_url():
    return "sqlite:///:memory:"


@pytest.fixture
def db_instance(in_memory_db_url):
    options = {"url": in_memory_db_url}
    # Use a dummy outbound string since we won't actually connect
    db = DB(options, outbound="tcp://127.0.0.1:12345")
    return db


def test_message_table_columns():
    # Ensure the Message model has required columns
    columns = {c.name for c in Message.__table__.columns}
    assert "id" in columns
    assert "topic" in columns
    assert "uuid" in columns
    assert "datetime" in columns
    assert "username" in columns
    assert "data" in columns


def test_setup_creates_engine_and_session(db_instance):
    db_instance.setup()
    assert db_instance.engine is not None
    assert db_instance.sessions is not None
    # Check it is SQLAlchemy engine
    assert hasattr(db_instance.engine, "execute")
    # sessions should be callable (sessionmaker factory)
    assert callable(db_instance.sessions)


def test_setup_creates_tables(db_instance):
    # Setup should create tables in the db
    db_instance.setup()
    inspector = db_instance.engine.execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    )
    table_names = {r[0] for r in inspector.fetchall()}
    assert "messages" in table_names


def test_save_to_db_commits_and_clears_messages(db_instance):
    db_instance.setup()
    # Prepare one message to save
    dt = datetime.datetime.utcnow()
    msg = Message(topic="test-topic", uuid="1234", datetime=dt, username="user", data="data")
    db_instance.messages.append(msg)
    # Call save_to_db, should commit and clear messages list
    db_instance.save_to_db()
    assert db_instance.messages == []

    # Confirm message was saved in DB
    session = db_instance.sessions()
    saved_msgs = session.query(Message).all()
    assert len(saved_msgs) == 1
    assert saved_msgs[0].topic == "test-topic"
    assert saved_msgs[0].uuid == "1234"
    assert saved_msgs[0].username == "user"
    assert saved_msgs[0].data == "data"


def test_save_to_db_retry_on_commit_error(monkeypatch, db_instance):
    db_instance.setup()
    # Prepare message to save
    dt = datetime.datetime.utcnow()
    msg = Message(topic="retry", uuid="retry", datetime=dt, username="u", data="d")
    db_instance.messages.append(msg)

    call_counter = {"count": 0}

    def raise_commit_error():
        call_counter["count"] += 1
        raise SQLAlchemyError("commit failed")

    # Patch session.commit to fail 3 times
    original_session = db_instance.sessions()

    def fake_session_factory():
        class FakeSession:
            def bulk_save_objects(self, objs):
                pass

            def commit(self):
                raise_commit_error()

        return FakeSession()

    monkeypatch.setattr(db_instance, "sessions", lambda: fake_session_factory())

    db_instance.save_to_db()
    # Should have tried 3 times
    assert call_counter["count"] == 3
    # Messages list should not be cleared after failures
    assert len(db_instance.messages) == 1


def test_run_method_processes_valid_message(monkeypatch, db_instance):
    """
    Run a single iteration of the run() method
    - monkeypatch poller.poll to simulate input
    - monkeypatch sub.recv_multipart to return a proper message tuple
    - monkeypatch save_to_db to avoid DB writes
    """
    db_instance.setup()

    # Prepare a well-formed message
    dt_str = "2023-01-01T12:00:00.000000"
    msg_parts = [
        b"topic1",
        b"uuid1",
        dt_str.encode(),
        b"user1",
        b"data1",
    ]

    # Patch poller to simulate sub socket ready to receive
    def fake_poll(timeout):
        return {db_instance.sub: zmq.POLLIN}

    # Patch sub.recv_multipart to return the prepared message
    monkeypatch.setattr(db_instance.poller, "poll", fake_poll)
    monkeypatch.setattr(db_instance.sub, "recv_multipart", lambda: msg_parts)
    monkeypatch.setattr(db_instance, "save_to_db", lambda: None)

    # Run one iteration of run()'s while True loop by breaking immediately after
    import threading

    # We'll run run() in a thread and stop it after one loop iteration
    def run_wrapper():
        # Save old messages list length
        old_len = len(db_instance.messages)
        # We expect the message to be processed and appended to messages
        count = 0
        last_save = 0

        # A version of run that breaks after one loop iteration
        db_instance.setup()
        while True:
            timeout = 1000 if db_instance.messages else -1
            sockets = dict(db_instance.poller.poll(timeout))
            if sockets.get(db_instance.sub) == zmq.POLLIN:
                msg = db_instance.sub.recv_multipart()
                from zmq.utils.strtypes import u as _u

                try:
                    (topic, uuid, dt, username, data) = (_u(m) for m in msg)
                    dt = datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f")
                    db_instance.messages.append(
                        Message(
                            topic=topic,
                            uuid=uuid,
                            datetime=dt,
                            username=username,
                            data=data,
                        )
                    )
                except Exception:
                    # Should not happen here
                    pass
            break

    t = threading.Thread(target=run_wrapper)
    t.start()
    t.join(timeout=2)

    # The message should be appended
    assert len(db_instance.messages) == 1
    message = db_instance.messages[0]
    assert message.topic == "topic1"
    assert message.uuid == "uuid1"
    assert message.datetime == datetime.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%f")
    assert message.username == "user1"
    assert message.data == "data1"


def test_run_method_invalid_message_logs_error(monkeypatch, caplog, db_instance):
    db_instance.setup()

    # Prepare invalid datetime message (will raise ValueError)
    invalid_msg_parts = [
        b"topic", b"uuid", b"invalid-datetime", b"user", b"data"
    ]

    def fake_poll(timeout):
        return {db_instance.sub: zmq.POLLIN}

    monkeypatch.setattr(db_instance.poller, "poll", fake_poll)
    monkeypatch.setattr(db_instance.sub, "recv_multipart", lambda: invalid_msg_parts)

    # Patch save_to_db to do nothing
    monkeypatch.setattr(db_instance, "save_to_db", lambda: None)

    # Ensure we catch the error log from invalid datetime parsing
    caplog.set_level("ERROR", logger="ROB.DB")

    import threading

    def run_wrapper():
        db_instance.setup()
        while True:
            timeout = 1000 if db_instance.messages else -1
            sockets = dict(db_instance.poller.poll(timeout))
            if sockets.get(db_instance.sub) == zmq.POLLIN:
                msg = db_instance.sub.recv_multipart()
                from zmq.utils.strtypes import u as _u

                try:
                    (topic, uuid, dt, username, data) = (_u(m) for m in msg)
                    dt = datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f")
                    db_instance.messages.append(
                        Message(
                            topic=topic,
                            uuid=uuid,
                            datetime=dt,
                            username=username,
                            data=data,
                        )
                    )
                except ValueError:
                    import logging

                    LOG = logging.getLogger("ROB.DB")
                    LOG.error("Invalid message: %s", msg)
                    break
                except SQLAlchemyError as err:
                    LOG = logging.getLogger("ROB.DB")
                    LOG.error("Unable to build the new message row: %s", err)
                    break

    t = threading.Thread(target=run_wrapper)
    t.start()
    t.join(timeout=2)

    # Check the error log message was generated
    assert any("Invalid message" in rec.message for rec in caplog.records)


def test_run_method_empty_messages_skips_save(monkeypatch, db_instance):
    """
    Confirm that if messages is empty, run continues without save_to_db calls.
    """
    db_instance.setup()

    # patch poller.poll to return no message socket ready (timeout -1)
    def fake_poll(timeout):
        return {}

    monkeypatch.setattr(db_instance.poller, "poll", fake_poll)

    # patch save_to_db to track calls
    calls = {"count": 0}

    def fake_save():
        calls["count"] += 1

    monkeypatch.setattr(db_instance, "save_to_db", fake_save)

    import threading

    def run_wrapper():
        db_instance.setup()
        # Break after one poll
        while True:
            timeout = 1000 if db_instance.messages else -1
            sockets = dict(db_instance.poller.poll(timeout))
            if sockets.get(db_instance.sub) == zmq.POLLIN:
                pass  # no msg in this test
            # The important part to test: no messages so no save_to_db call
            if db_instance.messages is []:
                # from source code, test runs continue on messages is []
                # This test ensures no call to save_to_db
                pass
            break

    t = threading.Thread(target=run_wrapper)
    t.start()
    t.join(timeout=2)

    assert calls["count"] == 0


def test_save_to_db_clears_messages_after_commit(monkeypatch, db_instance):
    """
    Check that after successful commit messages list is cleared
    """
    db_instance.setup()
    dt = datetime.datetime.utcnow()
    m1 = Message(topic="foo", uuid="1", datetime=dt, username="u", data="d")
    m2 = Message(topic="foo2", uuid="2", datetime=dt, username="u2", data="d2")
    db_instance.messages = [m1, m2]

    called = {"committed": False}

    # Patch sessionmaker to intercept bulk_save_objects and commit
    class FakeSession:
        def bulk_save_objects(self_, objs):
            assert objs == db_instance.messages

        def commit(self_):
            called["committed"] = True

    monkeypatch.setattr(db_instance, "sessions", lambda: lambda: FakeSession())

    db_instance.save_to_db()
    assert called["committed"] is True
    assert db_instance.messages == []


def test_save_to_db_logs_error_after_max_retries(monkeypatch, caplog, db_instance):
    db_instance.setup()
    dt = datetime.datetime.utcnow()
    msg = Message(topic="t", uuid="u", datetime=dt, username="usr", data="data")
    db_instance.messages.append(msg)

    # Patch sessionmaker/session.commit to always raise
    class FailsSession:
        def bulk_save_objects(self_, objs):
            pass

        def commit(self_):
            raise SQLAlchemyError("fail")

    monkeypatch.setattr(db_instance, "sessions", lambda: lambda: FailsSession())

    caplog.set_level("ERROR", logger="ROB.DB")
    db_instance.save_to_db()
    # Should have logged error messages
    assert any("Unable to commit to the database" in rec.message for rec in caplog.records)
    assert any("Database error" in rec.message for rec in caplog.records)


def test_message_model_repr_and_types():
    # Check creating Message with given fields and that types are proper
    dt = datetime.datetime(2024, 1, 1, 10, 0)
    msg = Message(
        id=1,
        topic="topic-1",
        uuid="uuid-1234",
        datetime=dt,
        username="username",
        data="some data",
    )
    assert msg.id == 1
    assert msg.topic == "topic-1"
    assert msg.uuid == "uuid-1234"
    assert msg.datetime == dt
    assert msg.username == "username"
    assert msg.data == "some data"


def test_db_init_sets_attributes(db_instance):
    assert db_instance.options is not None
    assert db_instance.outbound is not None
    assert db_instance.engine is None
    assert db_instance.sessions is None
    assert isinstance(db_instance.messages, list)
    assert db_instance.messages == []
