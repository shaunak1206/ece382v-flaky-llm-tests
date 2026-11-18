import datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import reactobus.db as dbmod


class BreakLoop(Exception):
    pass


def make_message(**kwargs):
    defaults = dict(
        topic="topic",
        uuid="uuid-1",
        datetime=datetime.datetime(2020, 1, 2, 3, 4, 5, 123456),
        username="alice",
        data="payload",
    )
    defaults.update(kwargs)
    return dbmod.Message(**defaults)


def test_message_model_has_expected_table_name_and_columns():
    assert dbmod.Message.__tablename__ == "messages"
    cols = set(dbmod.Message.__table__.columns.keys())
    assert cols == {"id", "topic", "uuid", "datetime", "username", "data"}


def test_message_instance_stores_values():
    msg = make_message()
    assert msg.topic == "topic"
    assert msg.uuid == "uuid-1"
    assert msg.datetime == datetime.datetime(2020, 1, 2, 3, 4, 5, 123456)
    assert msg.username == "alice"
    assert msg.data == "payload"


def test_db_init_sets_expected_defaults():
    db = dbmod.DB({"url": "sqlite:///:memory:"}, "tcp://127.0.0.1:5555")
    assert db.options == {"url": "sqlite:///:memory:"}
    assert db.outbound == "tcp://127.0.0.1:5555"
    assert db.engine is None
    assert db.sessions is None
    assert db.messages == []


def test_setup_initializes_engine_sessions_and_poller(monkeypatch):
    created = {}

    class FakeSocket:
        def __init__(self):
            self.setsockopt_calls = []
            self.connected = []

        def setsockopt(self, opt, value):
            self.setsockopt_calls.append((opt, value))

        def connect(self, addr):
            self.connected.append(addr)

    class FakeContext:
        def socket(self, socktype):
            created["socktype"] = socktype
            created["socket"] = FakeSocket()
            return created["socket"]

    class FakePoller:
        def __init__(self):
            created["poller"] = self
            self.register_calls = []

        def register(self, sock, flag):
            self.register_calls.append((sock, flag))

    def fake_instance():
        created["context_called"] = True
        return FakeContext()

    monkeypatch.setattr(dbmod.zmq.Context, "instance", staticmethod(fake_instance))
    monkeypatch.setattr(dbmod.zmq, "Poller", FakePoller)

    created["title"] = []

    def fake_setproctitle(title):
        created["title"].append(title)

    monkeypatch.setattr(dbmod, "setproctitle", fake_setproctitle)

    created["create_engine_args"] = []

    def fake_create_engine(url):
        created["create_engine_args"].append(url)
        engine = object()
        created["engine"] = engine
        return engine

    monkeypatch.setattr(dbmod, "create_engine", fake_create_engine)

    created["create_all_called"] = []

    class FakeMeta:
        def create_all(self, engine):
            created["create_all_called"].append(engine)

    monkeypatch.setattr(dbmod.Base, "metadata", FakeMeta(), raising=False)

    created["sessionmaker_kwargs"] = []

    def fake_sessionmaker(*args, **kwargs):
        created["sessionmaker_kwargs"].append(kwargs)

        def factory():
            return "session"

        return factory

    monkeypatch.setattr(dbmod, "sessionmaker", fake_sessionmaker)

    db = dbmod.DB({"url": "sqlite:///test.db"}, "tcp://example:5555")
    db.setup()

    assert created["title"] == ["ReactOBus [db]"]
    assert created["context_called"] is True
    assert created["socktype"] == dbmod.zmq.SUB
    assert created["socket"].setsockopt_calls == [(dbmod.zmq.SUBSCRIBE, b"")]
    assert created["socket"].connected == ["tcp://example:5555"]
    assert created["poller"].register_calls == [(created["socket"], dbmod.zmq.POLLIN)]
    assert created["create_engine_args"] == ["sqlite:///test.db"]
    assert created["create_all_called"] == [created["engine"]]
    assert db.engine is created["engine"]
    assert callable(db.sessions)


def test_save_to_db_commits_and_clears_messages(monkeypatch):
    db = dbmod.DB({"url": "sqlite:///:memory:"}, "tcp://example")
    db.messages = [make_message(), make_message(uuid="uuid-2")]

    calls = {"bulk": 0, "commit": 0}

    class FakeSession:
        def bulk_save_objects(self, objects):
            calls["bulk"] += 1
            self.objects = list(objects)

        def commit(self):
            calls["commit"] += 1

    db.sessions = lambda: FakeSession()
    db.save_to_db()

    assert calls["bulk"] == 1
    assert calls["commit"] == 1
    assert db.messages == []


def test_save_to_db_retries_until_success(monkeypatch):
    db = dbmod.DB({"url": "sqlite:///:memory:"}, "tcp://example")
    db.messages = [make_message()]

    attempts = {"bulk": 0, "commit": 0}

    class FakeSession:
        def bulk_save_objects(self, objects):
            attempts["bulk"] += 1
            if attempts["bulk"] < 3:
                raise dbmod.SQLAlchemyError("boom")

        def commit(self):
            attempts["commit"] += 1

    db.sessions = lambda: FakeSession()
    db.save_to_db()

    assert attempts["bulk"] == 3
    assert attempts["commit"] == 1
    assert db.messages == []


def test_save_to_db_logs_error_after_exhausting_retries(monkeypatch):
    db = dbmod.DB({"url": "sqlite:///:memory:"}, "tcp://example")
    db.messages = [make_message()]

    class FakeSession:
        def bulk_save_objects(self, objects):
            raise dbmod.SQLAlchemyError("always bad")

        def commit(self):
            pytest.fail("commit should not be called")

    db.sessions = lambda: FakeSession()

    errors = []

    monkeypatch.setattr(dbmod.LOG, "error", lambda msg, *args: errors.append(msg % args if args else msg))

    db.save_to_db()

    assert db.messages == [pytest.approx(db.messages[0], rel=0)] or len(db.messages) == 1
    assert errors[0] == "Unable to commit to the database, retrying later on"
    assert "Database error: always bad" in errors[1]


def test_run_builds_message_and_saves_after_one_second(monkeypatch):
    db = dbmod.DB({"url": "sqlite:///:memory:"}, "tcp://example")

    class FakeSub:
        def recv_multipart(self):
            return [
                b"topic-a",
                b"uuid-a",
                b"2020-01-02T03:04:05.123456",
                b"bob",
                b"hello",
            ]

    class FakePoller:
        def __init__(self):
            self.calls = 0

        def poll(self, timeout):
            self.calls += 1
            if self.calls == 1:
                return {db.sub: dbmod.zmq.POLLIN}
            raise BreakLoop()

    monkeypatch.setattr(db, "setup", lambda: None)
    db.sub = FakeSub()
    db.poller = FakePoller()

    times = iter([100.0, 102.0])
    monkeypatch.setattr(dbmod.time, "time", lambda: next(times))

    saved = []

    def fake_save():
        saved.append([m.topic for m in db.messages])
        raise BreakLoop()

    monkeypatch.setattr(db, "save_to_db", fake_save)

    with pytest.raises(BreakLoop):
        db.run()

    assert len(db.messages) == 1
    msg = db.messages[0]
    assert msg.topic == "topic-a"
    assert msg.uuid == "uuid-a"
    assert msg.username == "bob"
    assert msg.data == "hello"
    assert msg.datetime == datetime.datetime(2020, 1, 2, 3, 4, 5, 123456)
    assert saved == [["topic-a"]]


def test_run_logs_invalid_message_and_continues(monkeypatch):
    db = dbmod.DB({"url": "sqlite:///:memory:"}, "tcp://example")

    class FakeSub:
        def recv_multipart(self):
            return [b"bad", b"message"]

    class FakePoller:
        def __init__(self):
            self.calls = 0

        def poll(self, timeout):
            self.calls += 1
            if self.calls == 1:
                return {db.sub: dbmod.zmq.POLLIN}
            raise BreakLoop()

    monkeypatch.setattr(db, "setup", lambda: None)
    db.sub = FakeSub()
    db.poller = FakePoller()
    monkeypatch.setattr(dbmod.time, "time", lambda: 100.0)

    errors = []
    monkeypatch.setattr(dbmod.LOG, "error", lambda msg, *args: errors.append(msg % args if args else msg))

    monkeypatch.setattr(db, "save_to_db", lambda: (_ for _ in ()).throw(BreakLoop()))

    with pytest.raises(BreakLoop):
        db.run()

    assert errors and errors[0].startswith("Invalid message:")
    assert db.messages == []


def test_run_skips_messages_when_datetime_is_invalid(monkeypatch):
    db = dbmod.DB({"url": "sqlite:///:memory:"}, "tcp://example")

    class FakeSub:
        def recv_multipart(self):
            return [
                b"topic-a",
                b"uuid-a",
                b"not-a-datetime",
                b"bob",
                b"hello",
            ]

    class FakePoller:
        def __init__(self):
            self.calls = 0

        def poll(self, timeout):
            self.calls += 1
            if self.calls == 1:
                return {db.sub: dbmod.zmq.POLLIN}
            raise BreakLoop()

    monkeypatch.setattr(db, "setup", lambda: None)
    db.sub = FakeSub()
    db.poller = FakePoller()
    monkeypatch.setattr(dbmod.time, "time", lambda: 100.0)

    errors = []
    monkeypatch.setattr(dbmod.LOG, "error", lambda msg, *args: errors.append(msg % args if args else msg))
    monkeypatch.setattr(db, "save_to_db", lambda: (_ for _ in ()).throw(BreakLoop()))

    with pytest.raises(BreakLoop):
        db.run()

    assert errors and errors[0].startswith("Invalid message:")
    assert db.messages == []


def test_run_waits_forever_when_no_messages(monkeypatch):
    db = dbmod.DB({"url": "sqlite:///:memory:"}, "tcp://example")

    poll_timeouts = []

    class FakePoller:
        def poll(self, timeout):
            poll_timeouts.append(timeout)
            raise BreakLoop()

    monkeypatch.setattr(db, "setup", lambda: None)
    db.poller = FakePoller()
    db.messages = []

    with pytest.raises(BreakLoop):
        db.run()

    assert poll_timeouts == [-1]


def test_run_uses_timeout_when_messages_are_buffered(monkeypatch):
    db = dbmod.DB({"url": "sqlite:///:memory:"}, "tcp://example")

    poll_timeouts = []

    class FakePoller:
        def poll(self, timeout):
            poll_timeouts.append(timeout)
            raise BreakLoop()

    monkeypatch.setattr(db, "setup", lambda: None)
    db.poller = FakePoller()
    db.messages = [make_message()]

    with pytest.raises(BreakLoop):
        db.run()

    assert poll_timeouts == [1000]


def test_run_does_not_save_when_messages_list_is_empty_identity_check(monkeypatch):
    db = dbmod.DB({"url": "sqlite:///:memory:"}, "tcp://example")

    class EmptyButDifferentList(list):
        pass

    class FakePoller:
        def __init__(self):
            self.calls = 0

        def poll(self, timeout):
            self.calls += 1
            if self.calls == 1:
                return {}
            raise BreakLoop()

    monkeypatch.setattr(db, "setup", lambda: None)
    db.poller = FakePoller()
    db.messages = EmptyButDifferentList()

    saved = []
    monkeypatch.setattr(db, "save_to_db", lambda: saved.append(True))
    monkeypatch.setattr(dbmod.time, "time", lambda: 100.0)

    with pytest.raises(BreakLoop):
        db.run()

    assert saved == []


def test_setup_creates_sqlite_tables_in_real_database(tmp_path):
    db_file = tmp_path / "reactobus.sqlite"
    db = dbmod.DB({"url": f"sqlite:///{db_file}"}, "tcp://example")
    db.setup()

    engine = create_engine(f"sqlite:///{db_file}")
    insp = dbmod.inspect(engine)
    assert "messages" in insp.get_table_names()
    cols = {c["name"] for c in insp.get_columns("messages")}
    assert {"id", "topic", "uuid", "datetime", "username", "data"} <= cols
