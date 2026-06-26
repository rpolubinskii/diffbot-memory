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
import logging
import os
from datetime import datetime, timezone
from typing import Any

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EpisodeType
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

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


class NoopReranker(CrossEncoderClient):
    """No-op cross-encoder: hybrid (vector + BM25 + graph) search still works; we
    skip the rerank pass to avoid both cloud logprobs (OpenAIRerankerClient) and a
    heavy local model (BGERerankerClient)."""

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        del query
        return [(passage, 1.0) for passage in passages]


def _build_graphiti() -> Graphiti:
    llm = OpenAIGenericClient(
        config=LLMConfig(
            api_key=OLLAMA_KEY,
            base_url=OLLAMA_URL,
            model=MODEL_NAME,
            small_model=MODEL_NAME,
        )
    )
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
        llm_client=llm,
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
            self.graphiti = _build_graphiti()
            await self.graphiti.build_indices_and_constraints()
            self._worker = asyncio.create_task(self._run_worker())
            self._started = True
            log.info(
                "diffbot-memory ready (llm=%s embedder=%s dim=%d group=%s falkor=%s:%d)",
                MODEL_NAME, EMBEDDER_MODEL, EMBEDDER_DIM, GROUP_ID, FALKOR_HOST, FALKOR_PORT,
            )

    async def _run_worker(self) -> None:
        assert self.graphiti is not None
        while True:
            episode = await self.queue.get()
            try:
                await self.graphiti.add_episode(
                    name=episode["name"],
                    episode_body=episode["episode_body"],
                    source_description="diffbot",
                    reference_time=episode["reference_time"],
                    source=EpisodeType.text,
                    group_id=episode["group_id"],
                )
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
    del source  # all episodes ingested as text
    await memory.ensure_started()
    await memory.queue.put(
        {
            "name": name or "episode",
            "episode_body": episode_body,
            "group_id": group_id or GROUP_ID,
            "reference_time": _parse_time(reference_time),
        }
    )
    return {"status": "queued", "group_id": group_id or GROUP_ID}


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
