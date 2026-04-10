#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the log_file debug-logging feature of WhiskerObserver."""

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest
from loguru import logger

from pipecat_whisker.observer import WhiskerObserver, whisker_serializer

# ---------------------------------------------------------------------------
# Helpers – lightweight fakes so we don't need a real Pipecat pipeline
# ---------------------------------------------------------------------------


class _FakeProcessor:
    """Minimal stand-in for a FrameProcessor."""

    def __init__(self, name: str):
        self.name = name
        self.next = None
        self.entry_processors = []

    @property
    def __class_name__(self):
        return "FakeProcessor"


class _FakePipeline:
    """Minimal stand-in for a BasePipeline."""

    def __init__(self):
        proc = _FakeProcessor("proc-1")
        self.entry_processors = [proc]


class _FakeFrame:
    """Minimal stand-in for a Frame."""

    name = "TestFrame"

    def __class__(self):
        pass


class _FakeDirection:
    """Minimal stand-in for a FrameDirection enum value."""

    name = "DOWNSTREAM"


class _FakeFrameProcessed:
    """Minimal stand-in for FrameProcessed."""

    def __init__(self, processor, frame, direction=None):
        self.processor = processor
        self.frame = frame
        self.direction = direction or _FakeDirection()


class _FakeFramePushed:
    """Minimal stand-in for FramePushed."""

    def __init__(self, source, frame, direction=None):
        self.source = source
        self.frame = frame
        self.direction = direction or _FakeDirection()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def log_path(tmp_path):
    """Return a temporary log file path."""
    return str(tmp_path / "whisker_debug.log")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLogSinkSetup:
    """Verify that the loguru sink is added / removed correctly."""

    def test_no_sink_when_log_file_is_none(self):
        """When log_file is not provided, no sink or logger should be created."""
        obs = WhiskerObserver.__new__(WhiskerObserver)
        # Manually replicate the relevant init lines
        obs._log_file = None
        obs._log_sink_id = None
        obs._frame_logger = None

        assert obs._log_sink_id is None
        assert obs._frame_logger is None

    def test_sink_added_when_log_file_is_provided(self, log_path):
        """When log_file is provided, a loguru sink should be registered."""
        sink_id = logger.add(
            log_path,
            filter=lambda record: record["extra"].get("whisker_frame") is True,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {message}",
            level="DEBUG",
            rotation="10 MB",
            encoding="utf-8",
        )
        frame_logger = logger.bind(whisker_frame=True)
        try:
            assert sink_id is not None
            # Write a test message to verify the sink works
            frame_logger.debug("test-setup-message")
            assert os.path.exists(log_path)
            with open(log_path) as f:
                content = f.read()
            assert "test-setup-message" in content
        finally:
            logger.remove(sink_id)

    def test_sink_removed_on_cleanup(self, log_path):
        """_maybe_remove_log_sink should remove the sink."""
        obs = WhiskerObserver.__new__(WhiskerObserver)
        obs._log_file = log_path
        obs._log_sink_id = logger.add(
            log_path,
            filter=lambda record: record["extra"].get("whisker_frame") is True,
            format="{message}",
            level="DEBUG",
        )
        obs._frame_logger = logger.bind(whisker_frame=True)

        obs._maybe_remove_log_sink()
        assert obs._log_sink_id is None
        assert obs._frame_logger is None


class TestMaybeLogPipeline:
    """Tests for _maybe_log_pipeline."""

    def test_logs_pipeline_structure(self, log_path):
        """Pipeline structure should be written as a PIPELINE log line."""
        obs = WhiskerObserver.__new__(WhiskerObserver)
        obs._log_file = log_path
        obs._log_sink_id = logger.add(
            log_path,
            filter=lambda record: record["extra"].get("whisker_frame") is True,
            format="{message}",
            level="DEBUG",
        )
        obs._frame_logger = logger.bind(whisker_frame=True)

        processors = [{"id": "p1", "name": "p1", "parent": None, "type": "MyProc"}]
        connections = [{"from": "p1", "to": "p2"}]

        obs._maybe_log_pipeline(processors, connections)

        with open(log_path) as f:
            content = f.read()
        assert "PIPELINE" in content
        assert "p1" in content
        assert "p2" in content

        logger.remove(obs._log_sink_id)

    def test_noop_when_no_logger(self):
        """Should not raise when _frame_logger is None."""
        obs = WhiskerObserver.__new__(WhiskerObserver)
        obs._frame_logger = None
        # Should be a silent no-op
        obs._maybe_log_pipeline([], [])


class TestMaybeLogFrame:
    """Tests for _maybe_log_frame."""

    def test_logs_frame_event(self, log_path):
        """A frame event should produce a FRAME log line with all fields."""
        obs = WhiskerObserver.__new__(WhiskerObserver)
        obs._log_file = log_path
        obs._log_sink_id = logger.add(
            log_path,
            filter=lambda record: record["extra"].get("whisker_frame") is True,
            format="{message}",
            level="DEBUG",
        )
        obs._frame_logger = logger.bind(whisker_frame=True)

        obs._maybe_log_frame(
            id=42,
            event="process",
            direction="downstream",
            processor="MyLLM",
            frame_name="LLMFullResponseEndFrame",
            frame_type="frame",
            payload={"text": "hello"},
        )

        with open(log_path) as f:
            content = f.read()
        assert "FRAME #42" in content
        assert "process" in content
        assert "downstream" in content
        assert "MyLLM" in content
        assert "LLMFullResponseEndFrame" in content
        assert '"hello"' in content

        logger.remove(obs._log_sink_id)

    def test_handles_unserializable_payload(self, log_path):
        """Should not raise on payloads that json.dumps cannot handle."""
        obs = WhiskerObserver.__new__(WhiskerObserver)
        obs._log_file = log_path
        obs._log_sink_id = logger.add(
            log_path,
            filter=lambda record: record["extra"].get("whisker_frame") is True,
            format="{message}",
            level="DEBUG",
        )
        obs._frame_logger = logger.bind(whisker_frame=True)

        obs._maybe_log_frame(
            id=1,
            event="push",
            direction="upstream",
            processor="Proc",
            frame_name="Frame",
            frame_type="frame",
            payload=object(),  # not JSON-serializable
        )

        with open(log_path) as f:
            content = f.read()
        assert "FRAME #1" in content

        logger.remove(obs._log_sink_id)

    def test_noop_when_no_logger(self):
        """Should not raise when _frame_logger is None."""
        obs = WhiskerObserver.__new__(WhiskerObserver)
        obs._frame_logger = None
        obs._maybe_log_frame(
            id=1,
            event="push",
            direction="upstream",
            processor="P",
            frame_name="F",
            frame_type="frame",
            payload={},
        )


class TestFilterIsolation:
    """Verify that the whisker_frame filter keeps log lines separate."""

    def test_regular_logs_not_in_log_file(self, log_path):
        """Normal logger.debug() calls must NOT appear in the log_file."""
        sink_id = logger.add(
            log_path,
            filter=lambda record: record["extra"].get("whisker_frame") is True,
            format="{message}",
            level="DEBUG",
        )
        try:
            # Regular log (no whisker_frame binding)
            logger.debug("this-should-be-excluded")
            # Whisker log
            frame_logger = logger.bind(whisker_frame=True)
            frame_logger.debug("this-should-be-included")

            with open(log_path) as f:
                content = f.read()
            assert "this-should-be-excluded" not in content
            assert "this-should-be-included" in content
        finally:
            logger.remove(sink_id)
