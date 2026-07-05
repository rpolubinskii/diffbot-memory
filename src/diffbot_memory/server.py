"""diffbot-memory: a thin Graphiti MCP service we fully control.

Replaces the prebuilt zepai/knowledge-graph-mcp image, whose LLM provider was
hardwired to the OpenAI cloud Responses API. Here the extraction LLM and embedder
use graphiti-core's OpenAI-compatible (chat-completions) clients pointed at local
Ollama, and FastMCP's DNS-rebinding host check is disabled so LAN clients (diffbot-mcp)
connect directly — no reverse proxy. Tools mirror what diffbot-mcp's MemoryGateway
calls: add_memory / search_memory_facts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Literal

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.client import Message
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EpisodeType
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("diffbot-memory")

OLLAMA_URL = os.getenv("OPENAI_API_URL", "http://host.docker.internal:11434/v1")
OLLAMA_KEY = os.getenv("OPENAI_API_KEY", "ollama")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma4:26b")
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "bge-m3")
EMBEDDER_DIM = int(os.getenv("EMBEDDER_DIM", "1024"))
GROUP_ID = os.getenv("GRAPHITI_GROUP_ID", "diffbot")
FALKOR_HOST = os.getenv("FALKORDB_HOST", "falkordb")
FALKOR_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
FALKOR_PASSWORD = os.getenv("FALKORDB_PASSWORD", "")
FALKOR_DB = os.getenv("FALKORDB_DATABASE", "default_db")
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8100"))
SEMAPHORE_LIMIT = int(os.getenv("SEMAPHORE_LIMIT", "2"))
MEMORY_CANDIDATES_TYPE = "diffbot.memory_candidates.v1"
GRAPHITI_EXTRACTION_INSTRUCTIONS = """
Extract only durable user preferences, spatial facts, and observations from the JSON candidate buckets.
Ignore task outcomes, success/failure bookkeeping, tool names, JSON field names, request/result mechanics,
call IDs, timestamps except episode time, transient sensor readings, and speech/logging artifacts.
Preserve temporal and spatial qualifiers when they change the meaning of the learned fact.
""".strip()
MEMORY_CANDIDATE_EXTRACTION_PROMPT = """
You extract durable robot memory candidates from a structured DiffBot command episode.
Judge the command, final assistant text, and sanitized tool events yourself.
Return only strict JSON with this shape:
{"type":"diffbot.memory_candidates.v1","user_preferences":[],"spatial_facts":[],"observations":[]}

Rules:
- user_preferences: stable operator preferences or standing instructions.
- spatial_facts: durable locations, object placements, named places, or map relationships.
- observations: stable facts observed through vision or semantic-map tools.
- Do not include task history, success or failure status, tool mechanics, tool names, call IDs, timestamps,
  raw coordinates unless needed for the fact, speech output, logs, or transient sensor readings.
- If nothing durable was learned, return the same object with all three arrays empty.
""".strip()


class MemoryCandidates(BaseModel):
    type: Literal["diffbot.memory_candidates.v1"] = MEMORY_CANDIDATES_TYPE
    user_preferences: list[str] = Field(default_factory=list)
    spatial_facts: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)


class NoopReranker(CrossEncoderClient):
    """No-op cross-encoder: hybrid (vector + BM25 + graph) search still works; we
    skip the rerank pass to avoid both cloud logprobs (OpenAIRerankerClient) and a
    heavy local model (BGERerankerClient)."""

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        del query
        return [(passage, 1.0) for passage in passages]


def _build_llm() -> OpenAIGenericClient:
    return OpenAIGenericClient(
        config=LLMConfig(
            api_key=OLLAMA_KEY,
            base_url=OLLAMA_URL,
            model=MODEL_NAME,
            small_model=MODEL_NAME,
        )
    )


def _build_graphiti(llm: OpenAIGenericClient | None = None) -> Graphiti:
    graph_llm = llm or _build_llm()
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=OLLAMA_KEY,
            base_url=OLLAMA_URL,
            embedding_model=EMBEDDER_MODEL,
            embedding_dim=EMBEDDER_DIM,
        )
    )
    driver = FalkorDriver(
        host=FALKOR_HOST,
        port=FALKOR_PORT,
        password=FALKOR_PASSWORD or None,
        database=FALKOR_DB,
    )
    return Graphiti(
        graph_driver=driver,
        llm_client=graph_llm,
        embedder=embedder,
        cross_encoder=NoopReranker(),
        max_coroutines=SEMAPHORE_LIMIT,
    )


class Memory:
    """Owns the Graphiti client and a background ingestion queue so add_memory
    returns immediately instead of blocking the caller on LLM extraction.

    Initialized lazily on the first tool call (FastMCP's lifespan isn't reliably
    wired to app startup for the streamable-HTTP transport), so init happens on the
    server's running event loop — which the background worker and async clients need.
    """

    def __init__(self) -> None:
        self.graphiti: Graphiti | None = None
        self.extractor_llm: OpenAIGenericClient | None = None
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._started = False

    async def ensure_started(self) -> None:
        if self._started:
            return
        async with self._lock:
            if self._started:
                return
            self.extractor_llm = _build_llm()
            self.graphiti = _build_graphiti(self.extractor_llm)
            await self.graphiti.build_indices_and_constraints()
            self._worker = asyncio.create_task(self._run_worker())
            self._started = True
            log.info(
                "diffbot-memory ready (llm=%s embedder=%s dim=%d group=%s falkor=%s:%d)",
                MODEL_NAME, EMBEDDER_MODEL, EMBEDDER_DIM, GROUP_ID, FALKOR_HOST, FALKOR_PORT,
            )

    async def _run_worker(self) -> None:
        assert self.graphiti is not None
        assert self.extractor_llm is not None
        while True:
            episode = await self.queue.get()
            try:
                await _ingest_queued_episode(self.graphiti, self.extractor_llm, episode)
            except Exception:
                log.exception("episode ingestion failed")
            finally:
                self.queue.task_done()


memory = Memory()


mcp = FastMCP(
    "diffbot-memory",
    host=MCP_HOST,
    port=MCP_PORT,
    # Stateless JSON-over-HTTP. Our tools carry no per-session state (add_memory queues,
    # search_memory_facts queries), so MCP session continuity buys us nothing — and
    # requiring it broke the Spring AI client after a transitive starlette bump (it stopped
    # echoing the session id on follow-ups -> 400). Stateless mode drops the session
    # requirement; json_response returns plain JSON instead of SSE for simple clients.
    stateless_http=True,
    json_response=True,
    # Disable DNS-rebinding host validation: this is a trusted-LAN service bound to
    # 0.0.0.0, and the default localhost-only allow-list would reject diffbot-mcp.
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@mcp.tool()
async def add_memory(
    name: str,
    episode_body: str,
    group_id: str = "",
    source: str = "text",
    reference_time: str | None = None,
) -> dict[str, Any]:
    """Queue an episode for ingestion into the temporal knowledge graph."""
    queue_item, status = _prepare_queue_item(
        name=name,
        episode_body=episode_body,
        group_id=group_id,
        source=source,
        reference_time=reference_time,
    )
    if queue_item is None:
        return status

    await memory.ensure_started()
    await memory.queue.put(queue_item)
    return status


def _prepare_queue_item(
    *,
    name: str,
    episode_body: str,
    group_id: str,
    source: str,
    reference_time: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    effective_group_id = group_id or GROUP_ID
    effective_source = _normalize_source(source)
    parsed_time = _parse_time(reference_time)

    if effective_source == "json":
        structured_episode = _parse_structured_memory_episode(episode_body)
        if structured_episode is None:
            return None, {
                "status": "dropped",
                "reason": "unknown_json_shape",
                "group_id": effective_group_id,
            }
        return (
            {
                "name": name or "diffbot structured episode",
                "source": "json",
                "structured_episode": structured_episode,
                "group_id": effective_group_id,
                "reference_time": parsed_time,
            },
            {"status": "queued", "group_id": effective_group_id, "source": "json"},
        )

    return (
        {
            "name": name or "episode",
            "source": "text",
            "episode_body": episode_body,
            "group_id": effective_group_id,
            "reference_time": parsed_time,
        },
        {"status": "queued", "group_id": effective_group_id, "source": "text"},
    )


async def _ingest_queued_episode(
    graphiti: Graphiti,
    extractor_llm: OpenAIGenericClient,
    episode: dict[str, Any],
) -> None:
    if episode["source"] == "text":
        await graphiti.add_episode(
            name=episode["name"],
            episode_body=episode["episode_body"],
            source_description="diffbot",
            reference_time=episode["reference_time"],
            source=EpisodeType.text,
            group_id=episode["group_id"],
        )
        return

    try:
        candidates = await _extract_memory_candidates(
            extractor_llm,
            episode["structured_episode"],
            group_id=episode["group_id"],
        )
    except Exception:
        log.warning(
            "memory candidate extraction failed",
            extra={"source": "json", "group_id": episode["group_id"]},
            exc_info=True,
        )
        return

    candidate_body = _candidate_episode_body(candidates)
    if candidate_body is None:
        log.info(
            "memory episode dropped with no durable candidates",
            extra={"source": "json", "group_id": episode["group_id"]},
        )
        return

    await graphiti.add_episode(
        name=episode["name"],
        episode_body=candidate_body,
        source_description="diffbot",
        reference_time=episode["reference_time"],
        source=EpisodeType.json,
        group_id=episode["group_id"],
        custom_extraction_instructions=GRAPHITI_EXTRACTION_INSTRUCTIONS,
    )


async def _extract_memory_candidates(
    llm: OpenAIGenericClient,
    structured_episode: dict[str, Any],
    *,
    group_id: str,
) -> MemoryCandidates:
    response = await llm.generate_response(
        [
            Message(role="system", content=MEMORY_CANDIDATE_EXTRACTION_PROMPT),
            Message(
                role="user",
                content=json.dumps(structured_episode, ensure_ascii=False, sort_keys=True),
            ),
        ],
        response_model=MemoryCandidates,
        max_tokens=1200,
        group_id=group_id,
        prompt_name="diffbot.memory_candidates",
    )
    if isinstance(response, MemoryCandidates):
        return response
    return MemoryCandidates.model_validate(response)


def _candidate_episode_body(candidates: MemoryCandidates) -> str | None:
    data = candidates.model_dump()
    data["user_preferences"] = _clean_candidate_list(data["user_preferences"])
    data["spatial_facts"] = _clean_candidate_list(data["spatial_facts"])
    data["observations"] = _clean_candidate_list(data["observations"])
    if not any(data[bucket] for bucket in ("user_preferences", "spatial_facts", "observations")):
        return None
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _clean_candidate_list(values: list[str]) -> list[str]:
    return [value.strip() for value in values if isinstance(value, str) and value.strip()]


def _parse_structured_memory_episode(episode_body: str) -> dict[str, Any] | None:
    try:
        value = json.loads(episode_body)
    except json.JSONDecodeError:
        return None
    if not _is_structured_memory_episode(value):
        return None
    return value


def _is_structured_memory_episode(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("command"), str)
        and isinstance(value.get("started_at"), str)
        and isinstance(value.get("completed_at"), str)
        and isinstance(value.get("completion_status"), str)
        and (
            "tool_events" not in value
            or isinstance(value.get("tool_events"), list)
        )
    )


def _normalize_source(source: str) -> Literal["text", "json"]:
    return "json" if source.lower() == "json" else "text"


@mcp.tool()
async def search_memory_facts(
    query: str,
    group_ids: list[str] | None = None,
    max_facts: int = 10,
) -> dict[str, Any]:
    """Search the graph for facts relevant to a natural-language query."""
    await memory.ensure_started()
    if memory.graphiti is None:
        return {"facts": []}
    edges = await memory.graphiti.search(
        query,
        group_ids=group_ids or [GROUP_ID],
        num_results=max_facts,
    )
    facts = [
        {"fact": edge.fact, "valid_at": _iso(edge.valid_at), "invalid_at": _iso(edge.invalid_at)}
        for edge in edges
    ]
    return {"facts": facts}


def _parse_time(value: str | None) -> datetime:
    if value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
