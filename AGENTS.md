# Agent Instructions

## Overview

`diffbot-memory` is the agent's persistent memory service: a Graphiti bi-temporal knowledge graph
over FalkorDB, served as an MCP server, with extraction LLM + embeddings on local Ollama. It runs on
the RTX 3090 host. `diffbot-mcp` is the only client — it wraps this and exposes curated
`memory.remember` / `memory.recall` tools to the agent (see `diffbot-mcp` `MemoryGateway`).

This service is **mostly configuration**, not code: the upstream `zepai/knowledge-graph-mcp` image
is configured via `config.yaml` (`${VAR}` expansion from `.env`) and `docker-compose.yml`. Don't
fork the Graphiti server unless a curated change is unavoidable; prefer config.

## Build / Run

- `docker compose up -d` — start FalkorDB + the Graphiti MCP server.
- `docker compose logs -f diffbot-memory` — watch ingestion; malformed-JSON extraction errors mean
  the local LLM quant is too lossy.
- MCP at `:8100/mcp/`; FalkorDB UI at `:3100`.

## Key facts

- Models (Ollama): LLM `hf.co/unsloth/gemma-4-31B-it-GGUF:IQ4_XS`, embedder `bge-m3`
  (`dimensions: 1024` — must match the model).
- `GRAPHITI_GROUP_ID=diffbot` must match `diffbot-mcp`'s configured group.
- Keep `SEMAPHORE_LIMIT` low (one local model). Run Ollama with flash-attention + `q8_0` KV cache +
  capped `num_ctx` so the LLM and embedder coexist in 24 GB.
- Self-hosted/offline: no external calls when pointed at local Ollama.
