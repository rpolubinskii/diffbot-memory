# diffbot-memory

Persistent, temporal memory for the DiffBot agent — a [Graphiti](https://github.com/getzep/graphiti)
bi-temporal knowledge graph (facts carry `valid_at`/`invalid_at`) served over MCP, backed by
FalkorDB, with extraction + embeddings run locally on Ollama. Consumed by `diffbot-mcp`, which
re-exposes the curated `memory.remember` / `memory.recall` tools to the agent.

```
diffbot-agent ──MCP──> diffbot-mcp ──MCP client──> diffbot-memory (this)
                                                        ├── FalkorDB (graph)
                                                        └── Ollama (LLM + embedder)
```

## Prerequisites (run on the RTX 3090 host)

1. Ollama with the models pulled:
   ```bash
   ollama pull hf.co/unsloth/gemma-4-31B-it-GGUF:IQ4_XS   # extraction LLM
   ollama pull bge-m3                                      # embedder (1024-dim)
   ```
2. To fit IQ4_XS + bge-m3 in 24 GB, run Ollama with KV-cache quantization and a capped context:
   ```bash
   OLLAMA_FLASH_ATTENTION=1
   OLLAMA_KV_CACHE_TYPE=q8_0
   # cap the gemma context (e.g. num_ctx 16384 via a Modelfile or OLLAMA_CONTEXT_LENGTH)
   ```

## Run

```bash
cp .env.example .env      # adjust OPENAI_API_URL if Ollama isn't on host.docker.internal
docker compose up -d
```

- MCP endpoint: `http://<3090-host>:8100/mcp/`
- FalkorDB web UI: `http://<3090-host>:3100`

## How diffbot-mcp connects

`diffbot-mcp` adds an MCP client connection to this endpoint and proxies `memory.remember` →
Graphiti `add_memory` and `memory.recall` → `search_memory_facts`. The `group_id` here
(`GRAPHITI_GROUP_ID=diffbot`) must match `diffbot-mcp`'s configured group.

## Verify

- List tools / call `add_memory` then `search_memory_facts` with an MCP client against `:8100/mcp/`;
  the fact should come back. Inspect the graph in the FalkorDB UI (`:3100`).
- **Watch the logs for malformed-JSON / schema ingestion errors.** Graphiti depends on reliable
  structured output; if IQ4_XS produces frequent extraction failures, raise the quant (QAT-q4 or q5)
  — that's the signal the model is too lossy for extraction.

## Notes

- `embedder.dimensions` is pinned to **1024** for bge-m3 (not the upstream 1536 default); changing
  the embedder means rebuilding the graph's vector index.
- One `group_id` graph holds both automatic episodes (written by the agent after each command) and
  deliberate facts (the model's `memory.remember`). Graphiti merges them temporally.
- Self-hosted and offline-capable: no external API calls when pointed at local Ollama.
