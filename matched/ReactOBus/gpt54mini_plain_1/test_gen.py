import datetime
import logging

import pytest
from sqlalchemy.exc import SQLAlchemyError

import reactobus.db as db


class FakeSubSocket:
    def __init__(self):
        self.setsockopt_calls = []
        self.connected_to = None

    def setsockopt(self, option, value):
        self.setsockopt_calls.append((option, value))

    def connect(self, outbound):
        self.connected_to = outbound


class FakePoller:
    def __init__(self):
        self.registered = []
        self.poll_calls = []
        self.responses = []

    def register(self, sock, flag):
        self.registered.append((sock, flag))

    def poll(self, timeout):
        self.poll_calls.append(timeout)
        if not self.responses:
            raise SystemExit("stop")
        return self.responses.pop(0)


class FakeSession:
    def __init__(self, fail_times=0):
        self.fail_times = fail_times
        self.bulk_saved = []
        self.commits = 0

    def bulk_save_objects(self, objects):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise SQLAlchemyError("boom")
        self.bulk_saved.append(list(objects))

    def commit(self):
        self.commits += 1


class FakeSessionMaker:
    def __init__(self, session):
        self.session = session
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.session


def test_message_table_name_and_columns():
    assert db.Message.__tablename__ == "messages"
    cols = db.Message.__table__.columns.keys()
    assert cols == ["id", "topic", "uuid", "datetime", "username", "data"]


def test_db_init_sets_expected_attributes():
    options = {"url": "sqlite:///:memory:"}
    obj = db.DB(options, "tcp://127.0.0.1:5555")
    assert obj.options == options
    assert obj.outbound == "tcp://127.0.0.1:5555"
    assert obj.engine is None
    assert obj.sessions is None
    assert obj.messages == []


def test_setup_configures_zmq_and_sqlalchemy(monkeypatch):
    fake_sub = FakeSubSocket()
    fake_poller = FakePoller()
    created = {}
    created_all = []

    class FakeContext:
        def socket(self, kind):
            created["socket_kind"] = kind
            return fake_sub

        @staticmethod
        def instance():
            return FakeContext()

    class FakeEngine:
        pass

    def fake_create_engine(url):
        created["url"] = url
        return FakeEngine()

    def fake_create_all(engine):
        created_all.append(engine)

    class FakeBase:
        metadata = type("Meta", (), {"create_all": staticmethod(fake_create_all)})()

    class FakeSessionMaker:
        def __init__(self, bind):
            created["session_bind"] = bind

    monkeypatch.setattr(db.zmq, "Context", FakeContext)
    monkeypatch.setattr(db.zmq, "Poller", lambda: fake_poller)
    monkeypatch.setattr(db.zmq, "SUB", 1)
    monkeypatch.setattr(db.zmq, "SUBSCRIBE", 2)
    monkeypatch.setattr(db, "create_engine", fake_create_engine)
    monkeypatch.setattr(db, "sessionmaker", FakeSessionMaker)
    monkeypatch.setattr(db, "Base", FakeBase)
    monkeypatch.setattr(db, "setproctitle", lambda title: created.setdefault("title", title))

    obj = db.DB({"url": "sqlite:///example.db"}, "tcp://outbound")
    obj.setup()

    assert created["title"] == "ReactOBus [db]"
    assert created["socket_kind"] == 1
    assert fake_sub.setsockopt_calls == [(2, b"")]
    assert fake_sub.connected_to == "tcp://outbound"
    assert fake_poller.registered == [(fake_sub, db.zmq.POLLIN)]
    assert created["url"] == "sqlite:///example.db"
    assert created_all == [obj.engine]
    assert created["session_bind"] is obj.engine
    assert obj.poller is fake_poller


def test_save_to_db_success_clears_messages(monkeypatch):
    session = FakeSession()
    obj = db.DB({"url": "sqlite://"}, "outbound")
    obj.sessions = FakeSessionMaker(session)
    obj.messages = [db.Message(topic="t", uuid="u", datetime=datetime.datetime.utcnow(), username="n", data="d")]

    obj.save_to_db()

    assert session.commits == 1
    assert len(session.bulk_saved) == 1
    assert obj.messages == []


def test_save_to_db_retries_until_success(monkeypatch):
    session = FakeSession(fail_times=2)
    obj = db.DB({"url": "sqlite://"}, "outbound")
    obj.sessions = FakeSessionMaker(session)
    obj.messages = [db.Message(topic="t", uuid="u", datetime=datetime.datetime.utcnow(), username="n", data="d")]

    obj.save_to_db()

    assert session.commits == 1
    assert session.fail_times == 0
    assert obj.messages == []


def test_save_to_db_logs_error_after_three_failures(caplog):
    session = FakeSession(fail_times=3)
    obj = db.DB({"url": "sqlite://"}, "outbound")
    obj.sessions = FakeSessionMaker(session)
    obj.messages = [db.Message(topic="t", uuid="u", datetime=datetime.datetime.utcnow(), username="n", data="d")]

    with caplog.at_level(logging.ERROR, logger="ROB.DB"):
        obj.save_to_db()

    assert obj.messages  # still pending
    assert "Unable to commit to the database" in caplog.text
    assert "Database error: boom" in caplog.text


def test_run_appends_message_from_valid_payload(monkeypatch):
    fake_sub = FakeSubSocket()
    fake_poller = FakePoller()
    dt_str = "2020-01-02T03:04:05.123456"
    fake_poller.responses = [{fake_sub: db.zmq.POLLIN}]

    class DummyDB(db.DB):
        def setup(self):
            self.sub = fake_sub
            self.poller = fake_poller
            self.messages = []

    obj = DummyDB({"url": "sqlite://"}, "outbound")

    def fake_recv():
        return [b"topic", b"uuid", dt_str.encode(), b"user", b"payload"]

    fake_sub.recv_multipart = fake_recv

    with pytest.raises(SystemExit):
        obj.run()

    assert len(obj.messages) == 1
    msg = obj.messages[0]
    assert msg.topic == "topic"
    assert msg.uuid == "uuid"
    assert msg.datetime == datetime.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%f")
    assert msg.username == "user"
    assert msg.data == "payload"


def test_run_logs_invalid_message_and_skips_it(monkeypatch, caplog):
    fake_sub = FakeSubSocket()
    fake_poller = FakePoller()
    fake_poller.responses = [{fake_sub: db.zmq.POLLIN}]

    class DummyDB(db.DB):
        def setup(self):
            self.sub = fake_sub
            self.poller = fake_poller
            self.messages = []

    obj = DummyDB({"url": "sqlite://"}, "outbound")
    fake_sub.recv_multipart = lambda: [b"topic", b"uuid", b"bad-date", b"user", b"payload"]

    with caplog.at_level(logging.ERROR, logger="ROB.DB"):
        with pytest.raises(SystemExit):
            obj.run()

    assert obj.messages == []
    assert "Invalid message" in caplog.text


def test_run_logs_sqlalchemy_error_when_building_message(monkeypatch, caplog):
    fake_sub = FakeSubSocket()
    fake_poller = FakePoller()
    fake_poller.responses = [{fake_sub: db.zmq.POLLIN}]

    class DummyDB(db.DB):
        def setup(self):
            self.sub = fake_sub
            self.poller = fake_poller
            self.messages = []

    obj = DummyDB({"url": "sqlite://"}, "outbound")
    fake_sub.recv_multipart = lambda: [b"topic", b"uuid", b"2020-01-02T03:04:05.123456", b"user", b"payload"]

    class ExplodingMessage:
        def __init__(self, **kwargs):
            raise SQLAlchemyError("cannot build row")

    monkeypatch.setattr(db, "Message", ExplodingMessage)

    with caplog.at_level(logging.ERROR, logger="ROB.DB"):
        with pytest.raises(SystemExit):
            obj.run()

    assert obj.messages == []
    assert "Unable to build the new message row" in caplog.text


def test_run_saves_when_messages_reach_threshold(monkeypatch):
    fake_sub = FakeSubSocket()
    fake_poller = FakePoller()
    fake_poller.responses = [{fake_sub: db.zmq.POLLIN}, {}]

    class DummyDB(db.DB):
        def setup(self):
            self.sub = fake_sub
            self.poller = fake_poller
            self.messages = []

    obj = DummyDB({"url": "sqlite://"}, "outbound")
    fake_sub.recv_multipart = lambda: [b"topic", b"uuid", b"2020-01-02T03:04:05.123456", b"user", b"payload"]

    saved = []

    def fake_save():
        saved.append(len(obj.messages))
        raise SystemExit("stop after save")

    obj.save_to_db = fake_save

    with pytest.raises(SystemExit):
        obj.run()

    assert saved == [1]


def test_run_saves_when_time_threshold_exceeded(monkeypatch):
    fake_sub = FakeSubSocket()
    fake_poller = FakePoller()
    fake_poller.responses = [{fake_sub: db.zmq.POLLIN}, {}]

    class DummyDB(db.DB):
        def setup(self):
            self.sub = fake_sub
            self.poller = fake_poller
            self.messages = []

    obj = DummyDB({"url": "sqlite://"}, "outbound")
    fake_sub.recv_multipart = lambda: [b"topic", b"uuid", b"2020-01-02T03:04:05.123456", b"user", b"payload"]

    times = iter([10.0, 12.1])

    monkeypatch.setattr(db.time, "time", lambda: next(times))

    saved = []

    def fake_save():
        saved.append(True)
        raise SystemExit("stop after save")

    obj.save_to_db = fake_save

    with pytest.raises(SystemExit):
        obj.run()

    assert saved == [True]


def test_run_uses_long_poll_when_no_pending_messages(monkeypatch):
    fake_sub = FakeSubSocket()
    fake_poller = FakePoller()
    fake_poller.responses = []

    class DummyDB(db.DB):
        def setup(self):
            self.sub = fake_sub
            self.poller = fake_poller
            self.messages = []

    obj = DummyDB({"url": "sqlite://"}, "outbound")

    with pytest.raises(SystemExit):
        obj.run()

    assert fake_poller.poll_calls == [-1]


def test_run_uses_short_poll_when_messages_pending(monkeypatch):
    fake_sub = FakeSubSocket()
    fake_poller = FakePoller()
    fake_poller.responses = []

    class DummyDB(db.DB):
        def setup(self):
            self.sub = fake_sub
            self.poller = fake_poller
            self.messages = [db.Message(topic="t", uuid="u", datetime=datetime.datetime.utcnow(), username="n", data="d")]

    obj = DummyDB({"url": "sqlite://"}, "outbound")

    with pytest.raises(SystemExit):
        obj.run()

    assert fake_poller.poll_calls == [1000]
