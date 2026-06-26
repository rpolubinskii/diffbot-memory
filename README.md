# diffbot-memory

Persistent, temporal memory for the DiffBot agent — a thin MCP service (this repo)
built on [graphiti-core](https://github.com/getzep/graphiti): a bi-temporal knowledge
graph (facts carry `valid_at`/`invalid_at`) over FalkorDB, with extraction + embeddings
run locally on Ollama. Consumed by `diffbot-mcp`, which re-exposes curated
`memory.remember` / `memory.recall` tools to the agent.

```
diffbot-agent ──MCP──> diffbot-mcp ──MCP client──> diffbot-memory (this)
                                                        ├── FalkorDB (graph)
                                                        └── Ollama (LLM + embedder)
```

We run graphiti-core directly (`src/diffbot_memory/server.py`) rather than the upstream
`zepai/knowledge-graph-mcp` image, because that image's LLM provider is hardwired to the
OpenAI **cloud Responses API** (it ignores a custom base_url and `/v1/responses` isn't
something Ollama implements). Here the LLM uses graphiti-core's `OpenAIGenericClient`
(chat-completions) pointed at Ollama, and we own the FastMCP settings — so DNS-rebinding
host validation is disabled in-app and diffbot-mcp connects straight to `:8100` (no proxy).

## Prerequisites (on the RTX 3090 host)

1. Ollama listening on all interfaces so the container can reach it:
   ```bash
   sudo systemctl edit ollama      # [Service] / Environment="OLLAMA_HOST=0.0.0.0:11434"
   sudo systemctl restart ollama
   ```
2. Models pulled (use Ollama-native models — an imported HF gemma4 GGUF won't load:
   llama.cpp lacks the `gemma4` arch):
   ```bash
   ollama pull gemma4:26b      # extraction LLM (chat-completions, tools/structured output)
   ollama pull bge-m3          # embedder (1024-dim)
   ```

## Run

```bash
cp .env.example .env       # adjust MODEL_NAME / OPENAI_API_URL if needed
docker compose up -d --build
docker compose logs -f diffbot-memory   # expect: "diffbot-memory ready (...)"
```

- MCP endpoint: `http://<3090-host>:8100/mcp/`
- FalkorDB web UI: `http://<3090-host>:3100`

## Tools

- `add_memory(name, episode_body, group_id, source, reference_time)` — queues an episode for
  background ingestion (extraction never blocks the caller).
- `search_memory_facts(query, group_ids, max_facts)` — hybrid search; returns
  `{"facts": [{fact, valid_at, invalid_at}, …]}`.

These names/params match `diffbot-mcp`'s `MemoryGateway`, so diffbot-mcp needs no change.
`GRAPHITI_GROUP_ID=diffbot` must match diffbot-mcp's configured group.

## Verify

- `docker compose logs diffbot-memory` shows the ready line and no extraction/embedding errors.
  A 401 to `api.openai.com` would mean the LLM isn't pointed at Ollama; an `/embeddings` connection
  error means Ollama isn't reachable (check `OLLAMA_HOST=0.0.0.0`).
- From the agent, establish a fact, then recall it next turn; confirm an episode lands in the
  FalkorDB UI (`:3100`).

## Notes

- Reranking uses a **no-op cross-encoder** (hybrid vector+BM25+graph search still works) to avoid
  cloud logprobs (`OpenAIRerankerClient`) and a heavy local model (`BGERerankerClient`). Swap in
  `BGERerankerClient` later if recall quality needs it.
- Self-hosted/offline: no external calls when pointed at local Ollama.
