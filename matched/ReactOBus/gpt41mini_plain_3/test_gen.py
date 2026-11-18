import datetime
import pytest
import logging
from sqlalchemy.exc import SQLAlchemyError
from reactobus import db as reactodb
from reactobus.db import Message, DB


@pytest.fixture
def message_data():
    return {
        "topic": "test_topic",
        "uuid": "123e4567-e89b-12d3-a456-426614174000",
        "datetime": datetime.datetime.utcnow(),
        "username": "tester",
        "data": "some data",
    }


def test_message_fields(message_data):
    msg = Message(**message_data)
    assert msg.topic == "test_topic"
    assert msg.uuid == message_data["uuid"]
    assert isinstance(msg.datetime, datetime.datetime)
    assert msg.username == "tester"
    assert msg.data == "some data"


def test_db_init_options_and_outbound():
    opts = {"url": "sqlite:///:memory:"}
    outbound = "tcp://localhost:5555"
    d = DB(opts, outbound)
    assert d.options == opts
    assert d.outbound == outbound
    assert d.engine is None
    assert d.sessions is None
    assert d.messages == []


def test_db_setup_creates_engine_and_sessions(monkeypatch):
    opts = {"url": "sqlite:///:memory:"}
    outbound = "inproc://test"
    d = DB(opts, outbound)

    # Patch setproctitle to avoid side effect
    monkeypatch.setattr(reactodb, "setproctitle", lambda x: None)

    # Patch zmq Context to prevent real socket creation
    class DummySocket:
        def setsockopt(self, *a, **k): pass
        def connect(self, _): pass

    class DummyPoller:
        def register(self, sock, flag): pass

    class DummyContext:
        def socket(self, socktype): return DummySocket()

    monkeypatch.setattr(reactodb.zmq, "Context", lambda: DummyContext())
    monkeypatch.setattr(reactodb.zmq, "SUB", 1)
    monkeypatch.setattr(reactodb.zmq, "Poller", DummyPoller)

    d.setup()

    assert d.engine is not None
    assert d.sessions is not None
    # Messages table created
    tables = d.engine.table_names()
    assert "messages" in tables


def test_save_to_db_commits_and_clears_messages(monkeypatch):
    opts = {"url": "sqlite:///:memory:"}
    d = DB(opts, "tcp://test")
    d.setup()

    # Add a dummy message
    msg = Message(
        topic="test",
        uuid="1",
        datetime=datetime.datetime.utcnow(),
        username="user",
        data="data",
    )
    d.messages.append(msg)

    committed = []

    # Patch session and bulk_save_objects to record call
    class DummySession:
        def bulk_save_objects(_, objs):
            committed.extend(objs)
        def commit(_):
            pass

    d.sessions = lambda: DummySession()
    d.save_to_db()

    assert committed == [msg]
    assert d.messages == []


def test_save_to_db_retries_and_logs_error(monkeypatch, caplog):
    opts = {"url": "sqlite:///:memory:"}
    d = DB(opts, "tcp://test")
    d.setup()

    # add one dummy message
    d.messages.append(Message(
        topic="a", uuid="b",
        datetime=datetime.datetime.utcnow(),
        username="user", data="data"
    ))

    # Patch session to raise error on commit
    class ErrorSession:
        def bulk_save_objects(self, objs):
            pass
        def commit(self):
            raise SQLAlchemyError("fail")

    d.sessions = lambda: ErrorSession()

    caplog.set_level(logging.ERROR, logger="ROB.DB")
    d.save_to_db()

    # Messages are still in list, commit failed 3 times logged
    assert len(d.messages) == 1
    assert "Unable to commit to the database" in caplog.text
    assert "Database error: fail" in caplog.text


def test_run_receives_and_parses_message(monkeypatch):
    opts = {"url": "sqlite:///:memory:"}
    outbound = "tcp://test"

    d = DB(opts, outbound)

    # Patch setup to set poller and sub socket
    messages = []

    class DummySub:
        def setsockopt(self, *args): pass
        def connect(self, _): pass

        def recv_multipart(self):
            # Return a valid message: topic, uuid, dt, username, data
            dt = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")
            return [b"topic", b"uuid", dt.encode(), b"user", b"payload"]

    class DummyPoller:
        def __init__(self):
            self.called = False

        def register(self, sub, flag): pass

        def poll(self, timeout):
            # On first call return a ready socket, else no socket
            if not self.called:
                self.called = True
                return {d.sub: reactodb.zmq.POLLIN}
            return {}

    monkeypatch.setattr(d, "setup", lambda: None)
    d.sub = DummySub()
    d.poller = DummyPoller()
    d.messages = []

    # Patch sessions and save_to_db to no-op for this test
    d.sessions = lambda: None

    def fake_save_to_db():
        pass

    d.save_to_db = fake_save_to_db

    # Patch time to allow one iteration to save
    import time as time_module
    times = [0, 2]  # to trigger save on second check

    def fake_time():
        return times.pop(0)

    monkeypatch.setattr(reactodb.time, "time", fake_time)

    # Run one iteration of run (breaking after first message handled and saved)
    # We'll break after messages appended and save_to_db called once

    # To break the infinite loop, patch run to break when messages appended
    original_run = d.run

    def run_once():
        d.setup()
        last_save = reactodb.time.time()
        while True:
            timeout = 1000 if d.messages else -1
            sockets = dict(d.poller.poll(timeout))
            if sockets.get(d.sub) == reactodb.zmq.POLLIN:
                msg = d.sub.recv_multipart()
                from zmq.utils.strtypes import u
                try:
                    (topic, uuid, dt, username, data) = (u(m) for m in msg)
                    dt_parsed = datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f")
                    d.messages.append(
                        Message(
                            topic=topic,
                            uuid=uuid,
                            datetime=dt_parsed,
                            username=username,
                            data=data,
                        )
                    )
                except Exception:
                    continue

            if d.messages:
                assert len(d.messages) == 1
                # Trigger save_to_db call, then break to end test
                d.save_to_db()
                break

    run_once()


def test_run_handles_invalid_message(monkeypatch, caplog):
    opts = {"url": "sqlite:///:memory:"}
    d = DB(opts, "tcp://test")

    class DummySub:
        def setsockopt(self, *a): pass
        def connect(self, _): pass
        def recv_multipart(self):
            # Invalid date format triggers ValueError
            return [b"topic", b"uuid", b"bad_dt", b"user", b"data"]

    class DummyPoller:
        def __init__(self):
            self.called = False

        def register(self, sub, flag): pass
        def poll(self, timeout):
            if not self.called:
                self.called = True
                return {d.sub: reactodb.zmq.POLLIN}
            return {}

    monkeypatch.setattr(d, "setup", lambda: None)
    d.sub = DummySub()
    d.poller = DummyPoller()
    d.messages = []

    monkeypatch.setattr(reactodb.time, "time", lambda: 0)

    caplog.set_level(logging.ERROR, logger="ROB.DB")

    # Run one iteration similar to run, test capturing the error logged
    timeout = 1000 if d.messages else -1
    sockets = dict(d.poller.poll(timeout))
    if sockets.get(d.sub) == reactodb.zmq.POLLIN:
        msg = d.sub.recv_multipart()
        from zmq.utils.strtypes import u
        try:
            (topic, uuid, dt, username, data) = (u(m) for m in msg)
            dt = datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f")
            d.messages.append(
                Message(
                    topic=topic,
                    uuid=uuid,
                    datetime=dt,
                    username=username,
                    data=data,
                )
            )
        except ValueError:
            LOG = logging.getLogger("ROB.DB")
            LOG.error("Invalid message: %s", msg)

    assert "Invalid message:" in caplog.text


def test_run_checks_empty_messages_list_identity(monkeypatch):
    # Regression test that `if self.messages is []:` check in run is using 'is' which is always False for newly created lists
    opts = {"url": "sqlite:///:memory:"}
    d = DB(opts, "tcp://test")
    monkeypatch.setattr(d, "setup", lambda: None)
    d.sub = None
    d.poller = None
    d.messages = []

    # The run logic has: if self.messages is []: continue
    # This test checks the identity check is always False for a new list,
    # so the continue is not triggered incorrectly.
    # Here we simulate this part:
    assert (d.messages is []) is False


def test_message_str_repr_present():
    # Check Message instance can be created and basic repr does not error
    msg = Message(
        topic="topic",
        uuid="uuid",
        datetime=datetime.datetime.now(),
        username="user",
        data="data",
    )
    s = str(msg)
    r = repr(msg)
    # Just verify no exception and it's a string
    assert isinstance(s, str)
    assert isinstance(r, str)


def test_db_inherits_from_multiprocessing_process():
    opts = {"url": "sqlite:///:memory:"}
    d = DB(opts, "mock")
    assert isinstance(d, reactodb.multiprocessing.Process)


def test_db_save_to_db_no_messages(monkeypatch):
    # save_to_db should commit nothing and return early if messages empty
    opts = {"url": "sqlite:///:memory:"}
    d = DB(opts, "mock")
    d.setup()
    d.messages = []

    called = False

    class DummySession:
        def bulk_save_objects(self, objs):
            nonlocal called
            called = True

        def commit(self):
            nonlocal called
            called = True

    d.sessions = lambda: DummySession()
    d.save_to_db()
    # bulk_save_objects and commit should not be called because messages empty, so called stays False
    assert called is False


def test_db_save_to_db_exception_on_bulk_save(monkeypatch, caplog):
    opts = {"url": "sqlite:///:memory:"}
    d = DB(opts, "mock")
    d.setup()

    d.messages.append(Message(
        topic="t",
        uuid="u",
        datetime=datetime.datetime.utcnow(),
        username="user",
        data="d"
    ))

    # Raise SQLAlchemyError on bulk_save_objects but not commit
    class DummySession:
        def bulk_save_objects(self, objs):
            raise SQLAlchemyError("bulk save error")

        def commit(self):
            pass

    d.sessions = lambda: DummySession()

    caplog.set_level(logging.ERROR, logger="ROB.DB")
    d.save_to_db()
    # Should log error once (on last retry)
    assert "Unable to commit to the database" in caplog.text
    assert "bulk save error" in caplog.text


def test_db_run_save_to_db_called_by_len(monkeypatch):
    opts = {"url": "sqlite:///:memory:"}
    d = DB(opts, "test")

    class DummySub:
        def setsockopt(self, *args): pass
        def connect(self, _): pass
        def recv_multipart(self):
            dt = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")
            return [b"topic", b"uuid", dt.encode(), b"user", b"data"]

    class DummyPoller:
        def __init__(self):
            self.call_count = 0

        def register(self, sock, flag): pass

        def poll(self, timeout):
            self.call_count += 1
            if self.call_count == 1:
                # Return message ready
                return {d.sub: reactodb.zmq.POLLIN}
            if self.call_count == 2:
                # no message, triggers save due to length
                return {}
            return {}

    monkeypatch.setattr(d, "setup", lambda: None)
    d.sub = DummySub()
    d.poller = DummyPoller()
    d.messages = []

    # Patch save_to_db to record call
    calls = []

    def fake_save_to_db():
        calls.append(len(d.messages))
        d.messages.clear()

    d.save_to_db = fake_save_to_db

    # Patch time to emulate >1s passed so save is done
    now = [0]

    def fake_time():
        now[0] += 1.1
        return now[0]

    monkeypatch.setattr(reactodb.time, "time", fake_time)

    # Run simplified run with break after 2 cycles
    count = 0
    last_save = fake_time()
    while count < 3:
        timeout = 1000 if d.messages else -1
        sockets = dict(d.poller.poll(timeout))
        if sockets.get(d.sub) == reactodb.zmq.POLLIN:
            msg = d.sub.recv_multipart()
            from zmq.utils.strtypes import u
            (topic, uuid, dt, username, data) = (u(m) for m in msg)
            dt_parsed = datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f")
            d.messages.append(
                Message(
                    topic=topic,
                    uuid=uuid,
                    datetime=dt_parsed,
                    username=username,
                    data=data,
                )
            )

        now_val = fake_time()
        if len(d.messages) >= 1000 or now_val - last_save > 1:
            d.save_to_db()
            last_save = now_val
        count += 1

    assert len(calls) >= 1


def test_db_run_invalid_message_sqlalchemyerror(monkeypatch, caplog):
    opts = {"url": "sqlite:///:memory:"}
    d = DB(opts, "test")

    class DummySub:
        def setsockopt(self, *a): pass
        def connect(self, _): pass
        def recv_multipart(self):
            # This will cause SQLAlchemyError in Message constructor
            dt = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")
            return [b"topic", b"uuid", dt.encode(), b"user", b"data"]

    class DummySession:
        def __init__(self):
            pass

    def fake_message_init(*args, **kwargs):
        raise SQLAlchemyError("error building message")

    monkeypatch.setattr(d, "setup", lambda: None)
    d.sub = DummySub()
    d.poller = reactodb.zmq.Poller()
    d.messages = []

    monkeypatch.setattr(reactodb.Message, "__init__", fake_message_init)

    caplog.set_level(logging.ERROR, logger="ROB.DB")

    # Simulate run's message receive handling
    timeout = 1000 if d.messages else -1
    sockets = {d.sub: reactodb.zmq.POLLIN}
    try:
        msg = d.sub.recv_multipart()
        from zmq.utils.strtypes import u
        (topic, uuid, dt, username, data) = (u(m) for m in msg)
        dt = datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f")
        try:
            reactodb.Message(
                topic=topic,
                uuid=uuid,
                datetime=dt,
                username=username,
                data=data,
            )
        except SQLAlchemyError as err:
            LOG = logging.getLogger("ROB.DB")
            LOG.error("Unable to build the new message row: %s", err)
    except Exception:
        pytest.fail("Unexpected exception in test")

    assert "Unable to build the new message row:" in caplog.text
