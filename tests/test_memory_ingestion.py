from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from graphiti_core.nodes import EpisodeType

from diffbot_memory.server import (
    GRAPHITI_EXTRACTION_INSTRUCTIONS,
    MEMORY_CANDIDATES_TYPE,
    _ingest_queued_episode,
    _prepare_queue_item,
)


class FakeGraphiti:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def add_episode(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class FakeExtractor:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def generate_response(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.response


def _structured_episode(**overrides: Any) -> dict[str, Any]:
    episode = {
        "started_at": "2026-07-04T10:00:00+00:00",
        "completed_at": "2026-07-04T10:00:01+00:00",
        "command": "remember the dock",
        "completion_status": "completed",
        "tool_events": [],
    }
    episode.update(overrides)
    return episode


def test_source_text_prepares_text_episode() -> None:
    item, status = _prepare_queue_item(
        name="manual",
        episode_body="the dock is by the window",
        group_id="diffbot",
        source="text",
        reference_time="2026-07-04T10:00:00Z",
    )

    assert status == {"status": "queued", "group_id": "diffbot", "source": "text"}
    assert item is not None
    assert item["source"] == "text"
    assert item["episode_body"] == "the dock is by the window"
    assert item["group_id"] == "diffbot"
    assert isinstance(item["reference_time"], datetime)


def test_source_json_accepts_only_known_structured_episode() -> None:
    structured = json.dumps(
        _structured_episode(
            command="remember the dock",
            tool_events=[
                {
                    "tool": "semantic.find",
                    "category": "semantic",
                    "output": '{"matches":[{"label":"dock"}]}',
                }
            ],
        )
    )

    item, status = _prepare_queue_item(
        name="automatic",
        episode_body=structured,
        group_id="diffbot",
        source="json",
        reference_time="2026-07-04T10:00:00Z",
    )

    assert status == {"status": "queued", "group_id": "diffbot", "source": "json"}
    assert item is not None
    assert item["source"] == "json"
    assert item["structured_episode"]["command"] == "remember the dock"

    dropped_item, dropped_status = _prepare_queue_item(
        name="automatic",
        episode_body='{"tool":"raw-result"}',
        group_id="diffbot",
        source="json",
        reference_time="2026-07-04T10:00:00Z",
    )

    assert dropped_item is None
    assert dropped_status == {
        "status": "dropped",
        "reason": "unknown_json_shape",
        "group_id": "diffbot",
    }


def test_text_ingestion_uses_episode_type_text() -> None:
    async def run() -> None:
        graphiti = FakeGraphiti()
        extractor = FakeExtractor({})
        item, _ = _prepare_queue_item(
            name="manual",
            episode_body="the dock is by the window",
            group_id="diffbot",
            source="text",
            reference_time="2026-07-04T10:00:00Z",
        )
        assert item is not None

        await _ingest_queued_episode(graphiti, extractor, item)

        assert extractor.calls == []
        assert graphiti.calls[0]["source"] == EpisodeType.text
        assert graphiti.calls[0]["episode_body"] == "the dock is by the window"

    asyncio.run(run())


def test_empty_candidate_extraction_skips_graphiti_ingestion() -> None:
    async def run() -> None:
        graphiti = FakeGraphiti()
        extractor = FakeExtractor(
            {
                "type": MEMORY_CANDIDATES_TYPE,
                "user_preferences": [],
                "spatial_facts": [],
                "observations": [],
            }
        )
        item, _ = _prepare_queue_item(
            name="automatic",
            episode_body=json.dumps(_structured_episode(command="go to the kitchen")),
            group_id="diffbot",
            source="json",
            reference_time="2026-07-04T10:00:00Z",
        )
        assert item is not None

        await _ingest_queued_episode(graphiti, extractor, item)

        assert len(extractor.calls) == 1
        assert graphiti.calls == []

    asyncio.run(run())


def test_candidate_extraction_success_uses_json_episode_and_instructions() -> None:
    async def run() -> None:
        graphiti = FakeGraphiti()
        extractor = FakeExtractor(
            {
                "type": MEMORY_CANDIDATES_TYPE,
                "user_preferences": ["User prefers short spoken status updates."],
                "spatial_facts": ["The charging dock is by the window."],
                "observations": [],
            }
        )
        item, _ = _prepare_queue_item(
            name="automatic",
            episode_body=json.dumps(
                _structured_episode(
                    command="remember that I prefer short spoken status updates",
                    tool_events=[
                        {
                            "tool": "speak.say",
                            "category": "speech",
                            "arguments": {"text": "I will remember that."},
                        }
                    ],
                )
            ),
            group_id="diffbot",
            source="json",
            reference_time="2026-07-04T10:00:00Z",
        )
        assert item is not None

        await _ingest_queued_episode(graphiti, extractor, item)

        assert len(graphiti.calls) == 1
        call = graphiti.calls[0]
        body = json.loads(call["episode_body"])
        assert call["source"] == EpisodeType.json
        assert call["custom_extraction_instructions"] == GRAPHITI_EXTRACTION_INSTRUCTIONS
        assert body["type"] == MEMORY_CANDIDATES_TYPE
        assert body["user_preferences"] == ["User prefers short spoken status updates."]

    asyncio.run(run())
