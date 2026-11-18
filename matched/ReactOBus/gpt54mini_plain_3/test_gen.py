import datetime
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import SQLAlchemyError

import reactobus.db as dbmod


class DummySession:
    def __init__(self, save_side_effect=None):
        self.saved = []
        self.committed = 0
        self.save_side_effect = save_side_effect

    def bulk_save_objects(self, objects):
        self.saved.append(list(objects))
        if callable(self.save_side_effect):
            self.save_side_effect(objects)
        elif self.save_side_effect is not None:
            raise self.save_side_effect

    def commit(self):
        self.committed += 1


class DummySessionsFactory:
    def __init__(self, session):
        self.session = session
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.session


class DummySocket:
    def __init__(self):
        self.opts = []
        self.connected_to = None

    def setsockopt(self, opt, value):
        self.opts.append((opt, value))

    def connect(self, outbound):
        self.connected_to = outbound


class DummyPoller:
    def __init__(self):
        self.registered = []
        self.results = []

    def register(self, sock, event):
        self.registered.append((sock, event))

    def poll(self, timeout):
        if not self.results:
            return []
        item = self.results.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def test_message_model_table_name():
    assert dbmod.Message.__tablename__ == "messages"


def test_message_model_has_expected_columns():
    cols = dbmod.Message.__table__.columns
    assert set(cols.keys()) == {"id", "topic", "uuid", "datetime", "username", "data"}
    assert cols["id"].primary_key is True
    assert cols["topic"].index is True
    assert cols["uuid"].index is True
    assert cols["datetime"].index is True
    assert cols["username"].index is True


def test_message_instance_can_store_values():
    dt = datetime.datetime(2020, 1, 2, 3, 4, 5)
    msg = dbmod.Message(
        topic="topic",
        uuid="uuid",
        datetime=dt,
        username="alice",
        data="payload",
    )
    assert msg.topic == "topic"
    assert msg.uuid == "uuid"
    assert msg.datetime == dt
    assert msg.username == "alice"
    assert msg.data == "payload"


def test_db_initial_state():
    db = dbmod.DB({"url": "sqlite://"}, "tcp://outbound")
    assert db.options == {"url": "sqlite://"}
    assert db.outbound == "tcp://outbound"
    assert db.engine is None
    assert db.sessions is None
    assert db.messages == []


def test_setup_creates_engine_metadata_and_session_factory(monkeypatch):
    created = {}
    metadata_calls = []

    class DummyEngine:
        pass

    def fake_create_engine(url):
        created["url"] = url
        return DummyEngine()

    def fake_create_all(engine):
        metadata_calls.append(engine)

    fake_socket = DummySocket()
    fake_poller = DummyPoller()

    monkeypatch.setattr(dbmod, "create_engine", fake_create_engine)
    monkeypatch.setattr(dbmod.Base.metadata, "create_all", fake_create_all)
    monkeypatch.setattr(dbmod.zmq.Context, "instance", staticmethod(lambda: SimpleNamespace(socket=lambda kind: fake_socket)))
    monkeypatch.setattr(dbmod.zmq, "Poller", lambda: fake_poller)
    monkeypatch.setattr(dbmod, "setproctitle", lambda title: created.setdefault("title", title))

    db = dbmod.DB({"url": "sqlite:///test.db"}, "tcp://127.0.0.1:5555")
    db.setup()

    assert created["url"] == "sqlite:///test.db"
    assert created["title"] == "ReactOBus [db]"
    assert db.engine.__class__ is DummyEngine
    assert db.sessions is not None
    assert metadata_calls == [db.engine]
    assert fake_socket.connected_to == "tcp://127.0.0.1:5555"
    assert fake_socket.opts == [(dbmod.zmq.SUBSCRIBE, b"")]
    assert fake_poller.registered == [(fake_socket, dbmod.zmq.POLLIN)]


def test_save_to_db_success_clears_messages(monkeypatch):
    session = DummySession()
    db = dbmod.DB({"url": "sqlite://"}, "tcp://outbound")
    db.sessions = DummySessionsFactory(session)
    db.messages = [dbmod.Message(topic="t", uuid="u", datetime=datetime.datetime.now(), username="n", data="d")]

    db.save_to_db()

    assert db.messages == []
    assert session.committed == 1
    assert len(session.saved) == 1
    assert session.saved[0][0].topic == "t"


def test_save_to_db_retries_until_success(monkeypatch):
    attempts = {"n": 0}

    def side_effect(_objects):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise SQLAlchemyError("boom")

    session = DummySession(save_side_effect=side_effect)
    db = dbmod.DB({"url": "sqlite://"}, "tcp://outbound")
    db.sessions = DummySessionsFactory(session)
    db.messages = [dbmod.Message(topic="t", uuid="u", datetime=datetime.datetime.now(), username="n", data="d")]

    db.save_to_db()

    assert attempts["n"] == 3
    assert session.committed == 1
    assert db.messages == []


def test_save_to_db_keeps_messages_on_failure(monkeypatch):
    session = DummySession(save_side_effect=SQLAlchemyError("always fails"))
    db = dbmod.DB({"url": "sqlite://"}, "tcp://outbound")
    db.sessions = DummySessionsFactory(session)
    db.messages = [dbmod.Message(topic="t", uuid="u", datetime=datetime.datetime.now(), username="n", data="d")]

    db.save_to_db()

    assert db.messages
    assert session.committed == 0


def test_save_to_db_uses_new_session_each_call():
    session = DummySession()
    factory = DummySessionsFactory(session)
    db = dbmod.DB({"url": "sqlite://"}, "tcp://outbound")
    db.sessions = factory
    db.messages = [dbmod.Message(topic="t", uuid="u", datetime=datetime.datetime.now(), username="n", data="d")]

    db.save_to_db()
    db.messages = [dbmod.Message(topic="t2", uuid="u2", datetime=datetime.datetime.now(), username="n2", data="d2")]
    db.save_to_db()

    assert factory.calls == 2
    assert session.committed == 2


def test_run_appends_valid_message_and_saves_when_time_expires(monkeypatch):
    fake_socket = DummySocket()
    fake_poller = DummyPoller()
    fake_poller.results = [
        [(fake_socket, dbmod.zmq.POLLIN)],
        [],
    ]

    payload = [
        b"topic1",
        b"uuid1",
        b"2020-01-02T03:04:05.000000",
        b"alice",
        b"hello",
    ]

    db = dbmod.DB({"url": "sqlite://"}, "tcp://outbound")
    db.setup = lambda: None
    db.sub = fake_socket
    db.poller = fake_poller
    db.messages = []
    saved = {"called": 0}

    monkeypatch.setattr(dbmod.time, "time", lambda: 0 if saved["called"] == 0 else 2)

    def fake_save():
        saved["called"] += 1
        raise StopIteration

    db.save_to_db = fake_save
    fake_socket.recv_multipart = lambda: payload

    with pytest.raises(StopIteration):
        db.run()

    assert len(db.messages) == 1
    assert db.messages[0].topic == "topic1"
    assert db.messages[0].uuid == "uuid1"
    assert db.messages[0].username == "alice"
    assert db.messages[0].data == "hello"
    assert saved["called"] == 1


def test_run_ignores_invalid_message_format(monkeypatch, caplog):
    fake_socket = DummySocket()
    fake_poller = DummyPoller()
    fake_poller.results = [[(fake_socket, dbmod.zmq.POLLIN)]]

    db = dbmod.DB({"url": "sqlite://"}, "tcp://outbound")
    db.setup = lambda: None
    db.sub = fake_socket
    db.poller = fake_poller
    db.messages = []
    db.save_to_db = lambda: (_ for _ in ()).throw(AssertionError("should not save"))

    fake_socket.recv_multipart = lambda: [b"topic", b"uuid", b"bad-datetime", b"user", b"data"]

    with pytest.raises(IndexError):
        db.run()

    assert db.messages == []


def test_run_does_not_append_on_sqlalchemy_error(monkeypatch):
    fake_socket = DummySocket()
    fake_poller = DummyPoller()
    fake_poller.results = [[(fake_socket, dbmod.zmq.POLLIN)]]

    db = dbmod.DB({"url": "sqlite://"}, "tcp://outbound")
    db.setup = lambda: None
    db.sub = fake_socket
    db.poller = fake_poller
    db.messages = []
    db.save_to_db = lambda: None

    class BadMessage:
        def __iter__(self):
            raise SQLAlchemyError("build fail")

    fake_socket.recv_multipart = lambda: BadMessage()

    with pytest.raises(SQLAlchemyError):
        db.run()

    assert db.messages == []


def test_run_uses_timeout_minus_one_when_no_pending_messages(monkeypatch):
    fake_socket = DummySocket()
    poll_args = []

    class RecordingPoller(DummyPoller):
        def poll(self, timeout):
            poll_args.append(timeout)
            raise StopIteration

    db = dbmod.DB({"url": "sqlite://"}, "tcp://outbound")
    db.setup = lambda: None
    db.sub = fake_socket
    db.poller = RecordingPoller()
    db.messages = []
    db.save_to_db = lambda: None

    with pytest.raises(StopIteration):
        db.run()

    assert poll_args == [-1]


def test_run_uses_timeout_1000_when_messages_pending(monkeypatch):
    fake_socket = DummySocket()
    poll_args = []

    class RecordingPoller(DummyPoller):
        def poll(self, timeout):
            poll_args.append(timeout)
            raise StopIteration

    db = dbmod.DB({"url": "sqlite://"}, "tcp://outbound")
    db.setup = lambda: None
    db.sub = fake_socket
    db.poller = RecordingPoller()
    db.messages = [dbmod.Message(topic="t", uuid="u", datetime=datetime.datetime.now(), username="n", data="d")]
    db.save_to_db = lambda: None

    with pytest.raises(StopIteration):
        db.run()

    assert poll_args == [1000]


def test_run_does_not_save_when_no_messages_pending_and_no_input(monkeypatch):
    fake_socket = DummySocket()
    fake_poller = DummyPoller()
    fake_poller.results = [[]]

    db = dbmod.DB({"url": "sqlite://"}, "tcp://outbound")
    db.setup = lambda: None
    db.sub = fake_socket
    db.poller = fake_poller
    db.messages = []
    called = {"n": 0}
    db.save_to_db = lambda: called.__setitem__("n", called["n"] + 1)

    monkeypatch.setattr(dbmod.time, "time", lambda: 0)

    with pytest.raises(IndexError):
        db.run()

    assert called["n"] == 0
    assert db.messages == []
