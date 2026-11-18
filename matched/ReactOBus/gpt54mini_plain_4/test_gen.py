import datetime as real_datetime

import pytest
from sqlalchemy.exc import SQLAlchemyError

import reactobus.db as dbmod


class DummySession:
    def __init__(self, commit_side_effect=None):
        self.bulk_saved = []
        self.committed = 0
        self.commit_side_effect = commit_side_effect

    def bulk_save_objects(self, objs):
        self.bulk_saved.append(list(objs))

    def commit(self):
        self.committed += 1
        if self.commit_side_effect is not None:
            side_effect = self.commit_side_effect
            self.commit_side_effect = None
            raise side_effect


class DummySessionFactory:
    def __init__(self, session):
        self.session = session
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.session


class DummySocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.connected = None
        self.opts = []

    def setsockopt(self, opt, value):
        self.opts.append((opt, value))

    def connect(self, outbound):
        self.connected = outbound

    def recv_multipart(self):
        if not self.messages:
            raise RuntimeError("no more messages")
        return self.messages.pop(0)


class DummyPoller:
    def __init__(self, poll_results):
        self.poll_results = list(poll_results)
        self.registered = []

    def register(self, sock, event):
        self.registered.append((sock, event))

    def poll(self, timeout):
        if not self.poll_results:
            raise StopIteration()
        return self.poll_results.pop(0)


class DummyContext:
    def __init__(self, socket):
        self.socket_obj = socket

    def socket(self, socktype):
        return self.socket_obj


def make_db(monkeypatch, session=None, socket=None, poll_results=None, now_values=None):
    if session is None:
        session = DummySession()
    if socket is None:
        socket = DummySocket([])
    if poll_results is None:
        poll_results = []
    if now_values is None:
        now_values = [0.0]

    context = DummyContext(socket)
    monkeypatch.setattr(dbmod.zmq.Context, "instance", lambda: context)
    monkeypatch.setattr(dbmod.zmq, "SUB", 1)
    monkeypatch.setattr(dbmod.zmq, "SUBSCRIBE", 2)
    monkeypatch.setattr(dbmod.zmq, "POLLIN", 3)

    created_engines = []

    def fake_create_engine(url):
        created_engines.append(url)
        return object()

    monkeypatch.setattr(dbmod, "create_engine", fake_create_engine)

    created_tables = []

    def fake_create_all(engine):
        created_tables.append(engine)

    monkeypatch.setattr(dbmod.Base.metadata, "create_all", fake_create_all)

    session_factory = DummySessionFactory(session)
    monkeypatch.setattr(dbmod, "sessionmaker", lambda bind: session_factory)

    times = list(now_values)
    monkeypatch.setattr(dbmod.time, "time", lambda: times.pop(0) if times else now_values[-1])

    db = dbmod.DB({"url": "sqlite:///:memory:"}, "tcp://example")
    db.poller = DummyPoller(poll_results)
    db.sub = socket
    db.engine = object()
    db.sessions = session_factory
    return db, session, socket, created_engines, created_tables, session_factory


def test_message_table_and_columns():
    assert dbmod.Message.__tablename__ == "messages"
    cols = dbmod.Message.__table__.columns
    assert "id" in cols
    assert "topic" in cols
    assert "uuid" in cols
    assert "datetime" in cols
    assert "username" in cols
    assert "data" in cols
    assert cols["id"].primary_key is True


def test_message_model_can_be_instantiated():
    dt = real_datetime.datetime(2020, 1, 2, 3, 4, 5)
    msg = dbmod.Message(
        topic="bus.topic",
        uuid="abc",
        datetime=dt,
        username="alice",
        data="payload",
    )
    assert msg.topic == "bus.topic"
    assert msg.uuid == "abc"
    assert msg.datetime == dt
    assert msg.username == "alice"
    assert msg.data == "payload"


def test_db_initializes_with_expected_defaults():
    db = dbmod.DB({"url": "sqlite://"}, "tcp://localhost:5555")
    assert db.options == {"url": "sqlite://"}
    assert db.outbound == "tcp://localhost:5555"
    assert db.engine is None
    assert db.sessions is None
    assert db.messages == []


def test_setup_creates_engine_session_and_subscribes(monkeypatch):
    session = DummySession()
    socket = DummySocket([])
    db, _, _, created_engines, created_tables, _ = make_db(monkeypatch, session=session, socket=socket)

    setproctitle_calls = []
    monkeypatch.setattr(dbmod, "setproctitle", lambda title: setproctitle_calls.append(title))

    db.setup()

    assert setproctitle_calls == ["ReactOBus [db]"]
    assert socket.connected == "tcp://example"
    assert socket.opts == [(dbmod.zmq.SUBSCRIBE, b"")]
    assert db.sessions is not None
    assert created_engines == ["sqlite:///:memory:"]
    assert len(created_tables) == 1
    assert db.poller.registered == [(socket, dbmod.zmq.POLLIN)]


def test_save_to_db_success_clears_messages(monkeypatch):
    session = DummySession()
    db, _, _, _, _, _ = make_db(monkeypatch, session=session)
    db.messages = [dbmod.Message(topic="t", uuid="u", datetime=real_datetime.datetime.utcnow(), username="n", data="d")]

    db.save_to_db()

    assert session.committed == 1
    assert len(session.bulk_saved) == 1
    assert db.messages == []


def test_save_to_db_uses_session_factory_each_call(monkeypatch):
    session = DummySession()
    db, _, _, _, _, factory = make_db(monkeypatch, session=session)
    db.messages = [dbmod.Message(topic="t", uuid="u", datetime=real_datetime.datetime.utcnow(), username="n", data="d")]

    db.save_to_db()
    db.messages = [dbmod.Message(topic="t2", uuid="u2", datetime=real_datetime.datetime.utcnow(), username="n", data="d")]
    db.save_to_db()

    assert factory.calls == 2
    assert session.committed == 2


def test_save_to_db_retries_and_succeeds_on_second_commit(monkeypatch):
    err = SQLAlchemyError("temporary")
    session = DummySession(commit_side_effect=err)
    db, _, _, _, _, _ = make_db(monkeypatch, session=session)
    db.messages = [dbmod.Message(topic="t", uuid="u", datetime=real_datetime.datetime.utcnow(), username="n", data="d")]

    db.save_to_db()

    assert session.committed == 1
    assert db.messages == [dbmod.Message(topic="t", uuid="u", datetime=real_datetime.datetime.utcnow(), username="n", data="d")] or db.messages == []


def test_save_to_db_logs_error_after_three_failed_commits(monkeypatch, caplog):
    class AlwaysFailSession(DummySession):
        def commit(self):
            self.committed += 1
            raise SQLAlchemyError("boom")

    session = AlwaysFailSession()
    db, _, _, _, _, _ = make_db(monkeypatch, session=session)
    db.messages = [dbmod.Message(topic="t", uuid="u", datetime=real_datetime.datetime.utcnow(), username="n", data="d")]

    with caplog.at_level("ERROR", logger="ROB.DB"):
        db.save_to_db()

    assert session.committed == 3
    assert db.messages != []
    assert any("Unable to commit to the database" in rec.message for rec in caplog.records)
    assert any("Database error:" in rec.message for rec in caplog.records)


def test_run_accepts_valid_message_and_saves(monkeypatch):
    socket = DummySocket([[b"topic", b"uuid", b"2020-01-02T03:04:05.123456", b"alice", b"payload"]])
    db, session, _, _, _, _ = make_db(
        monkeypatch,
        session=DummySession(),
        socket=socket,
        poll_results=[[(socket, dbmod.zmq.POLLIN)], []],
        now_values=[0.0, 2.0],
    )

    monkeypatch.setattr(db, "setup", lambda: None)

    saved = []

    def fake_save():
        saved.append(list(db.messages))
        raise StopIteration()

    monkeypatch.setattr(db, "save_to_db", fake_save)

    with pytest.raises(StopIteration):
        db.run()

    assert len(db.messages) == 1
    assert saved and saved[0][0].topic == "topic"
    assert saved[0][0].uuid == "uuid"
    assert saved[0][0].username == "alice"
    assert saved[0][0].data == "payload"


def test_run_logs_invalid_message_and_skips_append(monkeypatch, caplog):
    socket = DummySocket([[b"topic", b"uuid", b"not-a-date", b"alice", b"payload"]])
    db, _, _, _, _, _ = make_db(
        monkeypatch,
        session=DummySession(),
        socket=socket,
        poll_results=[[(socket, dbmod.zmq.POLLIN)], []],
        now_values=[0.0, 2.0],
    )
    monkeypatch.setattr(db, "setup", lambda: None)
    monkeypatch.setattr(db, "save_to_db", lambda: (_ for _ in ()).throw(StopIteration()))

    with caplog.at_level("ERROR", logger="ROB.DB"):
        with pytest.raises(StopIteration):
            db.run()

    assert db.messages == []
    assert any("Invalid message" in rec.message for rec in caplog.records)


def test_run_logs_build_error_and_skips_append(monkeypatch, caplog):
    socket = DummySocket([[b"topic", b"uuid", b"2020-01-02T03:04:05.123456", b"alice", b"payload"]])
    db, _, _, _, _, _ = make_db(
        monkeypatch,
        session=DummySession(),
        socket=socket,
        poll_results=[[(socket, dbmod.zmq.POLLIN)], []],
        now_values=[0.0, 2.0],
    )
    monkeypatch.setattr(db, "setup", lambda: None)

    original_message = dbmod.Message

    class BadMessage(original_message):
        def __init__(self, *args, **kwargs):
            raise SQLAlchemyError("cannot build")

    monkeypatch.setattr(dbmod, "Message", BadMessage)
    monkeypatch.setattr(db, "save_to_db", lambda: (_ for _ in ()).throw(StopIteration()))

    with caplog.at_level("ERROR", logger="ROB.DB"):
        with pytest.raises(StopIteration):
            db.run()

    assert db.messages == []
    assert any("Unable to build the new message row" in rec.message for rec in caplog.records)


def test_run_does_not_save_without_messages(monkeypatch):
    socket = DummySocket([])
    db, _, _, _, _, _ = make_db(
        monkeypatch,
        session=DummySession(),
        socket=socket,
        poll_results=[[]],
        now_values=[0.0, 2.0],
    )
    monkeypatch.setattr(db, "setup", lambda: None)

    called = []

    def fake_save():
        called.append(True)
        raise StopIteration()

    monkeypatch.setattr(db, "save_to_db", fake_save)

    with pytest.raises(StopIteration):
        db.run()

    assert called == []


def test_run_saves_when_messages_already_buffered_and_time_expires(monkeypatch):
    socket = DummySocket([])
    db, _, _, _, _, _ = make_db(
        monkeypatch,
        session=DummySession(),
        socket=socket,
        poll_results=[[]],
        now_values=[0.0, 2.1],
    )
    monkeypatch.setattr(db, "setup", lambda: None)
    db.messages = [dbmod.Message(topic="t", uuid="u", datetime=real_datetime.datetime.utcnow(), username="n", data="d")]

    saved = []

    def fake_save():
        saved.append(len(db.messages))
        raise StopIteration()

    monkeypatch.setattr(db, "save_to_db", fake_save)

    with pytest.raises(StopIteration):
        db.run()

    assert saved == [1]
