# -*- coding: utf-8 -*-
# vim: set ts=4

# Copyright 2016 Rémi Duraffort
# This file is part of ReactOBus.
#
# ReactOBus is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ReactOBus is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ReactOBus.  If not, see <http://www.gnu.org/licenses/>

import datetime
import logging
import multiprocessing
from unittest import mock

import pytest
import sqlalchemy
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from reactobus import db as db_module
from reactobus.db import DB, Message


OUTBOUND = "tcp://127.0.0.1:5556"


def make_message(index=0):
    return Message(
        topic="topic.%d" % index,
        uuid="uuid-%d" % index,
        datetime=datetime.datetime(2016, 1, 1, 12, 0, index % 60, 123456),
        username="user%d" % index,
        data='{"index": %d}' % index,
    )


@pytest.fixture
def fake_zmq(monkeypatch):
    fake = mock.MagicMock()
    monkeypatch.setattr(db_module, "zmq", fake)
    return fake


@pytest.fixture
def fake_setproctitle(monkeypatch):
    fake = mock.MagicMock()
    monkeypatch.setattr(db_module, "setproctitle", fake)
    return fake


@pytest.fixture
def db_url(tmp_path):
    return "sqlite:///%s" % (tmp_path / "reactobus.db")


@pytest.fixture
def db_obj(db_url, fake_zmq, fake_setproctitle):
    obj = DB({"url": db_url}, OUTBOUND)
    obj.setup()
    yield obj
    if obj.engine is not None:
        obj.engine.dispose()


# ---------------------------------------------------------------------------
# Message model
# ---------------------------------------------------------------------------

def test_message_tablename():
    assert Message.__tablename__ == "messages"


def test_message_has_expected_columns():
    columns = set(Message.__table__.c.keys())
    assert columns == {"id", "topic", "uuid", "datetime", "username", "data"}


def test_message_id_is_primary_key():
    assert Message.__table__.c.id.primary_key is True


def test_message_indexed_columns():
    indexed = {c.name for idx in Message.__table__.indexes for c in idx.columns}
    assert indexed == {"topic", "uuid", "datetime", "username"}


def test_message_datetime_is_timezone_aware_column():
    assert Message.__table__.c.datetime.type.timezone is True


def test_message_init_sets_attributes():
    dt = datetime.datetime(2016, 5, 4, 3, 2, 1, 500)
    msg = Message(
        topic="a.b",
        uuid="1234",
        datetime=dt,
        username="bob",
        data="payload",
    )
    assert msg.topic == "a.b"
    assert msg.uuid == "1234"
    assert msg.datetime == dt
    assert msg.username == "bob"
    assert msg.data == "payload"


def test_message_id_is_none_before_persist():
    assert make_message().id is None


def test_message_roundtrip_through_database(db_url, fake_zmq, fake_setproctitle):
    db_obj = DB({"url": db_url}, OUTBOUND)
    db_obj.setup()
    try:
        session = db_obj.sessions()
        session.add(make_message(7))
        session.commit()
        session.close()

        session = db_obj.sessions()
        row = session.query(Message).one()
        assert row.id is not None
        assert row.topic == "topic.7"
        assert row.uuid == "uuid-7"
        assert row.username == "user7"
        assert row.data == '{"index": 7}'
        session.close()
    finally:
        db_obj.engine.dispose()


# ---------------------------------------------------------------------------
# DB construction
# ---------------------------------------------------------------------------

def test_db_is_multiprocessing_process():
    assert issubclass(DB, multiprocessing.Process)


def test_db_init_stores_options_and_outbound():
    options = {"url": "sqlite://"}
    obj = DB(options, OUTBOUND)
    assert obj.options is options
    assert obj.outbound == OUTBOUND


def test_db_init_defaults():
    obj = DB({"url": "sqlite://"}, OUTBOUND)
    assert obj.engine is None
    assert obj.sessions is None
    assert obj.messages == []


def test_db_init_messages_not_shared_between_instances():
    first = DB({"url": "sqlite://"}, OUTBOUND)
    second = DB({"url": "sqlite://"}, OUTBOUND)
    first.messages.append(make_message())
    assert second.messages == []


# ---------------------------------------------------------------------------
# DB.setup
# ---------------------------------------------------------------------------

def test_setup_sets_process_title(db_url, fake_zmq, fake_setproctitle):
    obj = DB({"url": db_url}, OUTBOUND)
    obj.setup()
    fake_setproctitle.assert_called_once_with("ReactOBus [db]")
    obj.engine.dispose()


def test_setup_subscribes_and_connects_to_outbound(db_url, fake_zmq, fake_setproctitle):
    obj = DB({"url": db_url}, OUTBOUND)
    obj.setup()
    fake_zmq.Context.instance.assert_called_once_with()
    socket = fake_zmq.Context.instance.return_value.socket.return_value
    socket.setsockopt.assert_called_once_with(fake_zmq.SUBSCRIBE, b"")
    socket.connect.assert_called_once_with(OUTBOUND)
    obj.engine.dispose()


def test_setup_registers_poller(db_url, fake_zmq, fake_setproctitle):
    obj = DB({"url": db_url}, OUTBOUND)
    obj.setup()
    poller = fake_zmq.Poller.return_value
    poller.register.assert_called_once_with(obj.sub, fake_zmq.POLLIN)
    obj.engine.dispose()


def test_setup_creates_engine_and_sessionmaker(db_url, fake_zmq, fake_setproctitle):
    obj = DB({"url": db_url}, OUTBOUND)
    obj.setup()
    assert obj.engine is not None
    assert str(obj.engine.url) == db_url
    assert obj.sessions is not None
    obj.engine.dispose()


def test_setup_creates_messages_table(db_url, fake_zmq, fake_setproctitle):
    obj = DB({"url": db_url}, OUTBOUND)
    obj.setup()
    assert "messages" in sqlalchemy.inspect(obj.engine).get_table_names()
    obj.engine.dispose()


# ---------------------------------------------------------------------------
# DB.save_to_db
# ---------------------------------------------------------------------------

def test_save_to_db_persists_messages(db_obj):
    db_obj.messages = [make_message(i) for i in range(3)]
    db_obj.save_to_db()

    session = db_obj.sessions()
    assert session.query(Message).count() == 3
    session.close()


def test_save_to_db_clears_pending_messages_on_success(db_obj):
    db_obj.messages = [make_message(i) for i in range(5)]
    db_obj.save_to_db()
    assert db_obj.messages == []


def test_save_to_db_bulk_saves_many_messages(db_obj):
    db_obj.messages = [make_message(i) for i in range(250)]
    db_obj.save_to_db()

    session = db_obj.sessions()
    assert session.query(Message).count() == 250
    session.close()


def test_save_to_db_with_no_messages_writes_nothing(db_obj):
    db_obj.messages = []
    db_obj.save_to_db()

    session = db_obj.sessions()
    assert session.query(Message).count() == 0
    session.close()
    assert db_obj.messages == []


def test_save_to_db_retries_then_succeeds(db_obj):
    session = mock.MagicMock()
    session.commit.side_effect = [SQLAlchemyError("temporary"), None]
    db_obj.sessions = mock.MagicMock(return_value=session)
    db_obj.messages = [make_message(1)]

    db_obj.save_to_db()

    assert session.commit.call_count == 2
    assert db_obj.messages == []


def test_save_to_db_gives_up_after_three_attempts(db_obj, caplog):
    caplog.set_level(logging.ERROR, logger="ROB.DB")
    session = mock.MagicMock()
    session.commit.side_effect = SQLAlchemyError("boom")
    db_obj.sessions = mock.MagicMock(return_value=session)
    pending = [make_message(1), make_message(2)]
    db_obj.messages = pending

    db_obj.save_to_db()

    assert session.commit.call_count == 3
    assert db_obj.messages is pending
    assert len(db_obj.messages) == 2


def test_save_to_db_logs_errors_when_commit_fails(db_obj, caplog):
    caplog.set_level(logging.ERROR, logger="ROB.DB")
    session = mock.MagicMock()
    session.commit.side_effect = SQLAlchemyError("boom")
    db_obj.sessions = mock.MagicMock(return_value=session)
    db_obj.messages = [make_message(1)]

    db_obj.save_to_db()

    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any("Unable to commit" in m for m in messages)
    assert any("boom" in m for m in messages)
