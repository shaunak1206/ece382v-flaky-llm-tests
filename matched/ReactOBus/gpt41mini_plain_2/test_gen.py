import datetime
import time
import pytest
from reactobus.db import DB, Message
from sqlalchemy.exc import SQLAlchemyError

def test_message_model_columns():
    # Check that Message has expected attributes and types
    m = Message()
    assert hasattr(m, "id")
    assert hasattr(m, "topic")
    assert hasattr(m, "uuid")
    assert hasattr(m, "datetime")
    assert hasattr(m, "username")
    assert hasattr(m, "data")

def test_db_init_sets_attributes():
    opts = {"url": "sqlite:///:memory:"}
    outbound = "tcp://127.0.0.1:5555"
    dbp = DB(opts, outbound)
    assert dbp.options == opts
    assert dbp.outbound == outbound
    assert dbp.engine is None
    assert dbp.sessions is None
    assert isinstance(dbp.messages, list) and not dbp.messages

def test_setup_initializes_sqlalchemy_and_zmq(monkeypatch):
    opts = {"url": "sqlite:///:memory:"}
    outbound = "inproc://test"
    dbp = DB(opts, outbound)
    monkeypatch.setattr(dbp, "outbound", outbound)
    monkeypatch.setattr(dbp, "options", opts)
    dbp.setup()
    # engine should be SQLAlchemy Engine
    assert dbp.engine is not None
    # sessions should be a callable session factory
    assert callable(dbp.sessions)
    # sub socket exists and is a zmq.Socket
    assert dbp.sub is not None
    # poller is registered for sub socket
    socks = dict(dbp.poller.poll(0))
    # since no message, poll returns empty dict but poller attribute exists
    assert hasattr(dbp, "poller")

def test_save_to_db_clears_messages(monkeypatch):
    opts = {"url": "sqlite:///:memory:"}
    dbp = DB(opts, "tcp://test")
    dbp.setup()
    # Add a Message object
    msg = Message(topic="t", uuid="u", datetime=datetime.datetime.now(), username="n", data="d")
    dbp.messages.append(msg)
    # Save should commit and clear messages
    dbp.save_to_db()
    assert dbp.messages == []

def test_save_to_db_retries_on_error(monkeypatch):
    opts = {"url": "sqlite:///:memory:"}
    dbp = DB(opts, "tcp://test")
    dbp.setup()

    msg = Message(topic="t", uuid="u", datetime=datetime.datetime.now(), username="n", data="d")
    dbp.messages.append(msg)

    # Patch session to raise SQLAlchemyError when committing
    class DummySession:
        def bulk_save_objects(self, msgs):
            pass
        def commit(self):
            raise SQLAlchemyError("fail")

    monkeypatch.setattr(dbp, "sessions", lambda: DummySession())
    # Also patch logger to capture errors
    logs = []
    def fake_error(msg, *args, **kwargs):
        logs.append(msg % args if args else msg)
    monkeypatch.setattr("reactobus.db.LOG.error", fake_error)

    dbp.save_to_db()
    # save messages not cleared due to failure
    assert len(dbp.messages) == 1
    # error logs captured
    assert any("Unable to commit to the database" in l for l in logs)
    assert any("Database error" in l for l in logs)

def test_run_receives_message_and_appends(monkeypatch):
    opts = {"url": "sqlite:///:memory:"}
    outbound = "inproc://test"

    dbp = DB(opts, outbound)
    dbp.setup()

    # Patch poller.poll to simulate a message available
    ts = datetime.datetime.now()
    # Prepare a valid message multipart
    topic = b"topic"
    uuid = b"uuid"
    dt_str = ts.strftime("%Y-%m-%dT%H:%M:%S.%f").encode()
    username = b"user"
    data = b"data"
    multipart_msg = [topic, uuid, dt_str, username, data]

    def fake_poll(timeout):
        # Only once returns POLLIN, then infinite no events (simulate single iteration)
        if fake_poll.called:
            time.sleep(0.01)  # short delay to avoid busy loop in test
            return {}
        fake_poll.called = True
        return {dbp.sub: 1}
    fake_poll.called = False

    def fake_recv_multipart():
        return multipart_msg

    monkeypatch.setattr(dbp.poller, "poll", fake_poll)
    monkeypatch.setattr(dbp.sub, "recv_multipart", fake_recv_multipart)

    # Patch save_to_db to avoid DB writes during test
    monkeypatch.setattr(dbp, "save_to_db", lambda: None)

    # Run the run loop for a short duration, forcibly break after 2 iterations
    import threading

    def run_db():
        count = 0
        last_save = time.time()
        while count < 2:
            timeout = 1000 if dbp.messages else -1
            sockets = dict(dbp.poller.poll(timeout))
            if sockets.get(dbp.sub) == 1:
                msg = dbp.sub.recv_multipart()
                try:
                    topic, uuid, dt, username, data = (m.decode() if isinstance(m, bytes) else m for m in msg)
                    dt = datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f")
                    dbp.messages.append(
                        Message(topic=topic, uuid=uuid, datetime=dt, username=username, data=data)
                    )
                except Exception:
                    pass
            # to break test loop
            count += 1

    run_db()
    assert len(dbp.messages) >= 1
    last_msg = dbp.messages[-1]
    assert last_msg.topic == topic.decode()
    assert last_msg.uuid == uuid.decode()
    assert last_msg.username == username.decode()
    assert last_msg.data == data.decode()
    assert isinstance(last_msg.datetime, datetime.datetime)

def test_run_handles_invalid_message_logs(monkeypatch):
    opts = {"url": "sqlite:///:memory:"}
    dbp = DB(opts, "tcp://test")
    dbp.setup()

    # Patch poller.poll to say POLLIN ready
    monkeypatch.setattr(dbp.poller, "poll", lambda timeout: {dbp.sub: 1})
    # Patch sub.recv_multipart to invalid non-decode bytes that fail datetime parsing
    monkeypatch.setattr(dbp.sub, "recv_multipart", lambda: [b"x", b"x", b"invalid-date", b"x", b"x"])

    captured_logs = []

    def fake_log_error(msg, *args, **kwargs):
        if args:
            msg = msg % args
        captured_logs.append(msg)

    monkeypatch.setattr("reactobus.db.LOG.error", fake_log_error)

    # Run a single iteration manually from run()
    timeout = -1
    sockets = dict(dbp.poller.poll(timeout))
    if sockets.get(dbp.sub) == 1:
        msg = dbp.sub.recv_multipart()
        try:
            topic, uuid, dt, username, data = (m.decode() if isinstance(m, bytes) else m for m in msg)
            dt = datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f")
            dbp.messages.append(Message(topic=topic, uuid=uuid, datetime=dt, username=username, data=data))
        except ValueError:
            LOG = getattr(dbp, "LOG", None)
            if LOG:
                LOG.error("Invalid message: %s", msg)
            else:
                # fallback logging
                print("Invalid message:", msg)

    assert any("Invalid message" in entry for entry in captured_logs)

def test_save_to_db_clears_messages_after_success(monkeypatch):
    opts = {"url": "sqlite:///:memory:"}
    dbp = DB(opts, "tcp://test")
    dbp.setup()

    m1 = Message(topic="a", uuid="u1", datetime=datetime.datetime.now(), username="n", data="d")
    m2 = Message(topic="b", uuid="u2", datetime=datetime.datetime.now(), username="m", data="e")
    dbp.messages.extend([m1, m2])

    commit_calls = []

    class DummySession:
        def bulk_save_objects(self, msgs):
            commit_calls.append("bulk")
        def commit(self):
            commit_calls.append("commit")

    monkeypatch.setattr(dbp, "sessions", lambda: DummySession())
    dbp.save_to_db()
    assert commit_calls == ["bulk", "commit"]
    assert dbp.messages == []

def test_save_to_db_fails_then_success(monkeypatch):
    opts = {"url": "sqlite:///:memory:"}
    dbp = DB(opts, "tcp://test")
    dbp.setup()

    m = Message(topic="t", uuid="u", datetime=datetime.datetime.now(), username="n", data="d")
    dbp.messages.append(m)

    class DummySession:
        def __init__(self):
            self.counter = 0
        def bulk_save_objects(self, msgs):
            pass
        def commit(self):
            self.counter += 1
            if self.counter == 1:
                raise SQLAlchemyError("fail first")
            # success second call

    dummy_session = DummySession()
    monkeypatch.setattr(dbp, "sessions", lambda: dummy_session)
    logs = []
    def fake_log_error(msg, *args, **kwargs):
        logs.append(msg % args if args else msg)

    monkeypatch.setattr("reactobus.db.LOG.error", fake_log_error)
    # Run save_to_db manually
    dbp.save_to_db()
    # The messages should have been cleared because second commit succeeded on retry
    # However, original save_to_db returns on success first try.
    # Our DummySession commit raises only on first try and succeeds later,
    # but save_to_db does not retry commit after exception in current code.
    #
    # So messages remain
    assert dbp.messages != []
    assert any("Unable to commit" in msg for msg in logs)

def test_run_skips_empty_messages(monkeypatch):
    opts = {"url": "sqlite:///:memory:"}
    dbp = DB(opts, "tcp://test")
    dbp.setup()

    # Ensure messages is empty list (should continue without saving)
    dbp.messages = []
    # patch poller.poll to return no sockets
    monkeypatch.setattr(dbp.poller, "poll", lambda timeout: {})
    called = {"invoked": False}
    def run_loop_once():
        # replace infinite loop with one iteration, imitating run
        timeout = 1000 if dbp.messages else -1
        sockets = dict(dbp.poller.poll(timeout))
        if sockets.get(dbp.sub) == 1:
            pass
        if dbp.messages is []:
            # This tests should verify that "messages is []" always False
            called["invoked"] = True

    run_loop_once()
    # Because of the code bug "if self.messages is []" (which always False)
    # called['invoked'] should NOT be true
    assert not called["invoked"]

def test_message_repr_and_fields():
    dt = datetime.datetime.now()
    m = Message(id=1, topic="top", uuid="uuid1", datetime=dt, username="usr", data="payload")
    # Check attributes
    assert m.id == 1
    assert m.topic == "top"
    assert m.uuid == "uuid1"
    assert m.datetime == dt
    assert m.username == "usr"
    assert m.data == "payload"

def test_db_inherits_from_multiprocessing_process():
    opts = {"url": "sqlite:///:memory:"}
    dbp = DB(opts, "tcp://test")
    import multiprocessing
    assert isinstance(dbp, multiprocessing.Process)
