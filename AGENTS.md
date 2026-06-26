# Agent Instructions

## Overview

`diffbot-memory` is the agent's persistent memory service: a thin Python MCP server
(`src/diffbot_memory/server.py`) built on **graphiti-core** (bi-temporal knowledge graph
over FalkorDB), with the extraction LLM + embeddings on local Ollama. Runs on the RTX 3090.
`diffbot-mcp` is the only client — it wraps this and exposes curated `memory.remember` /
`memory.recall` tools to the agent (see `diffbot-mcp` `MemoryGateway`).

We run graphiti-core directly instead of the upstream `zepai/knowledge-graph-mcp` image:
that image's LLM is hardwired to the OpenAI cloud Responses API (ignores base_url, and Ollama
doesn't implement `/v1/responses`). This service uses `OpenAIGenericClient` (chat-completions)
on Ollama and owns its FastMCP settings (DNS-rebinding disabled → no reverse proxy).

## Build / Run

- `docker compose up -d --build` — builds the service + starts FalkorDB.
- `docker compose logs -f diffbot-memory` — ready line + ingestion errors. A 401 to
  `api.openai.com` means the LLM isn't on Ollama; an `/embeddings` connection error means
  Ollama isn't reachable.
- MCP at `:8100/mcp/`; FalkorDB UI at `:3100`.

## Key facts

- Models (Ollama, **0.0.0.0:11434**): LLM `MODEL_NAME` (Ollama-native gemma4 — NOT an HF gemma4
  GGUF, which fails with "unknown architecture: gemma4"), embedder `bge-m3` (`EMBEDDER_DIM=1024`).
- Tools `add_memory` / `search_memory_facts` match diffbot-mcp's MemoryGateway — keep names/params
  in sync if either changes. Writes are queued (background); search is hybrid + a no-op reranker.
- `GRAPHITI_GROUP_ID=diffbot` must match diffbot-mcp's group.
- Pinned to graphiti-core 0.28.2 / mcp 1.26.0 / falkordb 1.6.0 (verified versions).
- Self-hosted/offline.
