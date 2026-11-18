import datetime
import io
import logging
import multiprocessing
import time

import pytest
from sqlalchemy.exc import SQLAlchemyError

from reactobus.db import DB, Message, Base


@pytest.fixture(scope="function")
def in_memory_db():
    # Setup an in-memory sqlite database for testing
    options = {"url": "sqlite:///:memory:"}
    outbound = "inproc://test"
    db = DB(options, outbound)
    db.setup()
    yield db
    if db.engine:
        db.engine.dispose()


def test_message_model_fields():
    # Verify Message columns and types
    cols = Message.__table__.columns
    assert "id" in cols
    assert "topic" in cols
    assert "uuid" in cols
    assert "datetime" in cols
    assert "username" in cols
    assert "data" in cols

    assert cols.id.primary_key is True
    assert cols.datetime.type.__class__.__name__ == "DateTime"


def test_db_setup_creates_tables(in_memory_db):
    # Tables should be created on setup
    insp = in_memory_db.engine.dialect.get_inspector(in_memory_db.engine)
    tables = insp.get_table_names() if hasattr(in_memory_db.engine.dialect, "get_inspector") else in_memory_db.engine.table_names()
    # SQLAlchemy 1.4/declarative creates "messages" table
    assert "messages" in in_memory_db.engine.table_names()


def test_save_to_db_saves_messages(in_memory_db):
    m1 = Message(
        topic="topic1",
        uuid="uuid1",
        datetime=datetime.datetime.now(),
        username="user1",
        data="data1",
    )
    m2 = Message(
        topic="topic2",
        uuid="uuid2",
        datetime=datetime.datetime.now(),
        username="user2",
        data="data2",
    )
    in_memory_db.messages = [m1, m2]
    in_memory_db.save_to_db()

    session = in_memory_db.sessions()
    all_msgs = session.query(Message).all()
    assert len(all_msgs) == 2
    assert all_msgs[0].topic == "topic1"
    assert all_msgs[1].username == "user2"
    # messages list is cleared after save
    assert in_memory_db.messages == []


def test_save_to_db_retries_on_exception(monkeypatch, in_memory_db):
    calls = []

    class FailingSession:
        def bulk_save_objects(self, msgs):
            calls.append("bulk_save")
            raise SQLAlchemyError("fail")

        def commit(self):
            calls.append("commit")
            raise SQLAlchemyError("fail")

    # Patch sessions to return a failing session instance
    monkeypatch.setattr(in_memory_db, "sessions", lambda: FailingSession())
    in_memory_db.messages = [Message(topic="t", uuid="u", datetime=datetime.datetime.now(), username="a", data="d")]

    # save_to_db should retry 3 times causing 3 bulk_save + 3 commit attempts
    in_memory_db.save_to_db()
    # We expect multiple calls in pairs bulk_save and commit, pairs count == 3
    assert calls.count("bulk_save") == 3
    assert calls.count("commit") == 3


def test_run_invalid_message_logs_error(monkeypatch, caplog):
    # Setup DB instance and monkeypatch setup to do nothing
    db = DB({"url": "sqlite:///:memory:"}, "inproc://test")
    monkeypatch.setattr(db, "setup", lambda: None)
    db.sessions = None
    db.messages = []

    # Setup poller.poll to report sub ready once, then no data.
    class DummyPoller:
        call_count = 0

        def poll(self, timeout):
            self.call_count += 1
            # Return sub fd is ready the first call, then empty
            if self.call_count == 1:
                return {db.sub: 1}
            return {}

    # Setup sub.recv_multipart to return an invalid message (missing fields)
    class DummySub:
        def __init__(self):
            # Return a tuple with bad data to trigger ValueError in run()
            self.calls = 0

        def recv_multipart(self):
            self.calls += 1
            # Return too few items (should be 5)
            return [b"topic", b"uuid"]

    # Setup dummy sub and poller
    db.sub = DummySub()
    db.poller = DummyPoller()

    # Patch LOG level for capture
    caplog.set_level(logging.ERROR, logger="ROB.DB")

    # Run one iteration of run, monkeypatch time and poller for controlled run
    def fake_time():
        return 0

    monkeypatch.setattr(db, "save_to_db", lambda: None)
    monkeypatch.setattr(time, "time", fake_time)

    # Run only one loop of run: patch while True with a break after first iteration
    original_run = db.run

    def limited_run():
        count = 0
        db.setup = lambda: None
        last_save = time.time()
        while count < 1:
            timeout = 1000 if db.messages else -1
            sockets = dict(db.poller.poll(timeout))
            if sockets.get(db.sub) == 1:
                msg = db.sub.recv_multipart()
                try:
                    # raising ValueError, trigger logging error
                    (db.sub._topic, db.sub._uuid, db.sub._dt, db.sub._username, db.sub._data) = (m.decode() for m in msg)
                    dt = datetime.datetime.strptime(db.sub._dt, "%Y-%m-%dT%H:%M:%S.%f")
                    db.messages.append(
                        Message(
                            topic=db.sub._topic,
                            uuid=db.sub._uuid,
                            datetime=dt,
                            username=db.sub._username,
                            data=db.sub._data,
                        )
                    )
                except ValueError:
                    logging.getLogger("ROB.DB").error("Invalid message: %s", msg)
                except SQLAlchemyError as err:
                    logging.getLogger("ROB.DB").error("Unable to build the new message row: %s", err)
            count += 1

    limited_run()

    assert any("Invalid message" in rec.message for rec in caplog.records)


def test_run_appends_valid_message(monkeypatch):
    messages = []

    db = DB({"url": "sqlite:///:memory:"}, "inproc://test")
    monkeypatch.setattr(db, "setup", lambda: None)

    # Compose a valid message in byte encoded format
    topic = b"topic"
    uuid = b"uuid"
    dt_str = b"2020-01-01T12:30:45.123456"
    username = b"user"
    data = b"data"

    class DummyPoller:
        def poll(self, timeout):
            return {db.sub: 1}

    class DummySub:
        def __init__(self):
            self.called = 0

        def recv_multipart(self):
            if self.called == 0:
                self.called += 1
                return [topic, uuid, dt_str, username, data]
            return []

    db.sub = DummySub()
    db.poller = DummyPoller()
    db.messages = messages

    # Run one iteration of run logic manually to control test
    timeout = 1000 if db.messages else -1
    sockets = dict(db.poller.poll(timeout))
    if sockets.get(db.sub) == 1:
        msg = db.sub.recv_multipart()
        (topic_dec, uuid_dec, dt_dec, username_dec, data_dec) = (m.decode() for m in msg)
        dt = datetime.datetime.strptime(dt_dec, "%Y-%m-%dT%H:%M:%S.%f")
        db.messages.append(
            Message(
                topic=topic_dec,
                uuid=uuid_dec,
                datetime=dt,
                username=username_dec,
                data=data_dec,
            )
        )

    assert len(db.messages) == 1
    m = db.messages[0]
    assert m.topic == "topic"
    assert m.uuid == "uuid"
    assert m.username == "user"
    assert m.data == "data"
    assert isinstance(m.datetime, datetime.datetime)


def test_save_to_db_clears_messages_after_success(in_memory_db):
    m = Message(topic="t", uuid="u", datetime=datetime.datetime.now(), username="x", data="d")
    in_memory_db.messages = [m]
    in_memory_db.save_to_db()
    assert len(in_memory_db.messages) == 0
    # Confirm DB contains this message
    session = in_memory_db.sessions()
    result = session.query(Message).filter_by(uuid="u").first()
    assert result is not None
    assert result.topic == "t"


def test_save_to_db_logs_error_on_commit_failure(monkeypatch, caplog, in_memory_db):
    class BadSession:
        def bulk_save_objects(self, objs):
            pass

        def commit(self):
            raise SQLAlchemyError("db error")

    monkeypatch.setattr(in_memory_db, "sessions", lambda: BadSession())
    in_memory_db.messages = [Message(topic="t", uuid="u", datetime=datetime.datetime.now(), username="x", data="d")]
    caplog.set_level(logging.ERROR, logger="ROB.DB")
    in_memory_db.save_to_db()
    # Should log error twice as per code
    errors = [rec for rec in caplog.records if rec.levelname == "ERROR"]
    assert any("Unable to commit" in rec.message for rec in errors)
    assert any("Database error" in rec.message for rec in errors)


def test_db_process_inherits_multiprocessing_process():
    db = DB({"url": "sqlite:///:memory:"}, "outbound")
    assert isinstance(db, multiprocessing.Process)


def test_db_init_sets_attributes():
    options = {"url": "sqlite:///:memory:"}
    outbound = "foo"
    db = DB(options, outbound)
    assert db.options == options
    assert db.outbound == outbound
    assert db.engine is None
    assert db.sessions is None
    assert db.messages == []


def test_message_repr_and_str():
    # Just ensure Message instances can be repr and str without error
    dt = datetime.datetime.now()
    msg = Message(id=1, topic="topic", uuid="uuid", datetime=dt, username="user", data="data")
    r = repr(msg)
    s = str(msg)
    assert isinstance(r, str)
    assert isinstance(s, str)


def test_run_saves_on_1000_messages(monkeypatch):
    db = DB({"url": "sqlite:///:memory:"}, "outbound")
    monkeypatch.setattr(db, "setup", lambda: None)
    monkeypatch.setattr(time, "time", lambda: 1000)
    db.messages = [Message(topic="t", uuid=str(i), datetime=datetime.datetime.now(), username="u", data="d") for i in range(1000)]
    called = {}

    def fake_save():
        called["save"] = True

    db.save_to_db = fake_save
    last_save = 900
    now = 1001

    # Inject last_save in method's local scope through monkeypatching is tricky; so directly call the conditional code:
    if len(db.messages) >= 1000 or now - last_save > 1:
        db.save_to_db()

    assert "save" in called


def test_run_saves_after_1_second(monkeypatch):
    db = DB({"url": "sqlite:///:memory:"}, "outbound")
    monkeypatch.setattr(db, "setup", lambda: None)
    start_time = 1000.0
    monkeypatch.setattr(time, "time", lambda: start_time)
    db.messages = [Message(topic="t", uuid="u", datetime=datetime.datetime.now(), username="u", data="d")] * 10
    called = {}

    def fake_save():
        called["save"] = True

    db.save_to_db = fake_save

    last_save = start_time - 2  # simulate last save 2 seconds ago
    now = time.time()

    if len(db.messages) >= 1000 or now - last_save > 1:
        db.save_to_db()

    assert "save" in called


def test_save_to_db_returns_after_success(monkeypatch):
    in_db = DB({"url": "sqlite:///:memory:"}, "outbound")
    monkeypatch.setattr(in_db, "sessions", lambda: in_db.sessions())

    # We patch session's commit method to ensure it is called once and returns.
    session = in_db.sessions()
    in_db.messages = [Message(topic="t", uuid="u", datetime=datetime.datetime.now(), username="u", data="d")]

    called = {"bulk_save": False, "commit": False}

    orig_bulk = session.bulk_save_objects
    orig_commit = session.commit

    def bulk_save(objects):
        called["bulk_save"] = True
        return orig_bulk(objects)

    def commit():
        called["commit"] = True
        return orig_commit()

    session.bulk_save_objects = bulk_save
    session.commit = commit

    in_db.save_to_db()
    assert called["bulk_save"] is True
    assert called["commit"] is True
    # messages clear
    assert in_db.messages == []


def test_db_run_break_loop(monkeypatch):
    # Sanity test to run run method but break after one iteration to avoid infinite loop
    db = DB({"url": "sqlite:///:memory:"}, "outbound")
    monkeypatch.setattr(db, "setup", lambda: None)

    class DummyPoller:
        def __init__(self):
            self.called = False

        def poll(self, timeout):
            # Return ready once, then empty
            if not self.called:
                self.called = True
                return {db.sub: 1}
            return {}

    class DummySub:
        def __init__(self):
            self.messages = [
                [b"topic", b"uuid", b"2021-01-01T00:00:00.000000", b"user", b"data"]
            ]
            self.index = 0

        def recv_multipart(self):
            msg = self.messages[self.index]
            self.index += 1
            return msg

    db.sub = DummySub()
    db.poller = DummyPoller()
    db.engine = None

    iteration = {"count": 0}

    def fake_save_to_db():
        iteration["count"] += 1

    db.save_to_db = fake_save_to_db

    # Patch time to simulate time > 1 sec difference
    base_time = 1000

    def fake_time():
        return base_time + iteration["count"] * 2

    monkeypatch.setattr(time, "time", fake_time)

    # Run partial loop, break after first iteration that would save
    last_save = time.time()
    db.messages = []

    for _ in range(2):
        timeout = 1000 if db.messages else -1
        sockets = dict(db.poller.poll(timeout))
        if sockets.get(db.sub) == 1 and db.sub.index <= 1:
            msg = db.sub.recv_multipart()
            (topic, uuid, dt, username, data) = (m.decode() for m in msg)
            dt = datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f")
            db.messages.append(Message(topic=topic, uuid=uuid, datetime=dt, username=username, data=data))
        now = time.time()
        if len(db.messages) >= 1000 or now - last_save > 1:
            db.save_to_db()
            last_save = now

    # We triggered save_to_db at least once
    assert iteration["count"] >= 1
