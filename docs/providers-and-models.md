# Providers and models

The harness is vendor-agnostic: a provider is nothing more than a **base URL**,
an **API key** and a **model id** for an OpenAI-compatible endpoint. Everything
else (Groq's free-tier limits, Ollama's local context) is optional metadata.

## Built-in presets

| Preset | Default endpoint | Default model |
|---|---|---|
| `groq` | `https://api.groq.com/openai/v1` | `openai/gpt-oss-120b` |
| `nvidia` | `https://integrate.api.nvidia.com/v1` | `deepseek-ai/deepseek-v4.1-flash` |
| `openai` | `https://api.openai.com/v1` | `gpt-4o-mini` |
| `openrouter` | `https://openrouter.ai/api/v1` | `openrouter/auto` |
| `together` | `https://api.together.xyz/v1` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| `deepseek` | `https://api.deepseek.com/v1` | `deepseek-chat` |
| `mistral` | `https://api.mistral.ai/v1` | `mistral-large-latest` |
| `cerebras` | `https://api.cerebras.ai/v1` | `llama3.3-70b` |
| `fireworks` | `https://api.fireworks.ai/inference/v1` | `accounts/fireworks/models/llama-v3p3-70b-instruct` |
| `perplexity` | `https://api.perplexity.ai` | `sonar` |
| `ollama` | `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`) | `OLLAMA_MODEL` |
| `custom` | `AI_BASE_URL` / `CUSTOM_BASE_URL` | `AI_MODEL` / `CUSTOM_MODEL` |

Shorthand aliases exist for convenience (`nim`/`nvidia-nim` → `nvidia`,
`open-router` → `openrouter`, `local` → `ollama`, `openai-compatible` →
`custom`, …). Any other name is accepted as a custom provider; it simply has no
preset defaults.

## Custom providers

A provider name never has to be a preset:

```dotenv
LLM_PROVIDER=myhost
MYHOST_BASE_URL=http://192.168.1.20:8000/v1
MYHOST_API_KEY=not-needed
MYHOST_MODEL=my-model
```

An unknown name resolves from its own prefixed variables (upper-cased, with
non-alphanumerics replaced by `_`). Endpoints on `localhost`, `127.0.0.1`, `::1`,
`0.0.0.0` or a `*.local` host are treated as local: they are allowed to run
without a real API key, and they use the larger local input budget
(`OLLAMA_MAX_INPUT_TOKENS`).

## Resolution precedence

For a provider called `NAME`, each of the three values comes from the first
source that has one:

1. `NAME_BASE_URL` / `NAME_API_KEY` / `NAME_MODEL` (provider-specific)
2. `AI_BASE_URL` / `AI_API_KEY` / `AI_MODEL` (the active provider's values;
   legacy `LLM_BASE_URL`, `LLM_API_KEY`, `OPENAI_API_KEY`, `LLM_MODEL` spellings
   are also read)
3. the preset in `PRESETS` (known providers only)

The generic `AI_*` block belongs to the provider named by `LLM_PROVIDER`, and it
is also the fallback when the configured provider is local or the name is
custom. That prevents switching from NVIDIA to Groq from sending the NVIDIA key
and model to Groq: instead you get a clear "not configured" error.

Notes:

- Ollama deliberately never inherits a remote `AI_API_KEY` just because one is
  present; it always uses the placeholder `"local"`.
- A legacy `OPENAI_API_KEY=gsk_…` value is still accepted as a Groq key, and a
  Groq key mistakenly pointed at the `openai` preset produces an explicit error
  instead of a 401 from the wrong host.
- Switching providers with `--provider` or `/provider` keeps the conversation;
  the next turn simply uses the new endpoint. Each provider remembers its own
  active model in `.agent/models.json`.

## Model selection and discovery

- `/model` shows the active model, its budget and today's usage.
- `/model list` (or `--list-models`) calls the provider's OpenAI-compatible
  `GET /models` endpoint. Discovery works for any provider without a bundled
  table. The list is fetched on demand (bounded by `MODEL_DISCOVERY_TIMEOUT`,
  default 20 s) and remembered for the session, so `/model 7` works without
  retyping a long vendor id.
- `/model <id|alias|#>` switches the model. Aliases exist for the bundled Groq
  catalog (e.g. `/model 20b`, `/model qwen`).

Caveat: a catalogue can advertise models the key cannot actually call;
`/model <id>` is the source of truth.

## Request budgets

A request is bounded by four separate values, resolved per provider **and**
model (`src/models.py`):

- **context window** — the model's total input + output capacity;
- **maximum input** — what the harness allows the request body to carry;
- **maximum output** — the generation cap, reserved inside the window;
- **rate limits** — the account's TPM/RPM/RPD/TPD buckets.

Precedence for each value:

```
MODEL_<TAG>_*  >  NAME_* (provider prefix)  >  bundled metadata  >  globals
```

`<TAG>` is the model id upper-cased with non-alphanumerics replaced by `_`
(`minimaxai/minimax-m3.1-flash` → `MINIMAXAI_MINIMAX_M3_1_FLASH`). The
provider prefix follows the same rule as the connection variables
(`NVIDIA_MAX_OUTPUT_TOKENS`, `GROQ_TPM`, …). A plain `MODEL_CONTEXT_WINDOW`
global is also read.

The generic `MAX_OUTPUT_TOKENS` is a request, not a guarantee: it is clamped
against the context window and the TPM bucket (which holds input *and* output),
so a lower model-specific or provider-specific maximum always wins and an 8K
per-minute bucket is never asked for a large input plus a large output. The
input ceiling is similarly clamped by the remaining room, never below
`MIN_REQUEST_INPUT_TOKENS` (400), with `RATE_LIMIT_HEADROOM` (200) reserved as a
safety margin. Local providers use `OLLAMA_MAX_INPUT_TOKENS` (default 32,000)
because Ollama has no per-minute ceiling, only a context window.

## Bundled model limits

Only Groq's published free-tier numbers are bundled (`GROQ_MODELS` in
`src/models.py`). Groq applies its limits **per model, per organization** —
every coding model gets its own bucket, so switching models is the only way to
get more throughput; a smaller model does not give a bigger budget.

| Model | TPM | RPM | RPD | TPD | Context |
|---|---|---|---|---|---|
| `openai/gpt-oss-120b` | 8,000 | 30 | 1,000 | 200,000 | 131,072 |
| `openai/gpt-oss-20b` | 8,000 | 30 | 1,000 | 200,000 | 131,072 |
| `qwen/qwen3.8-27b` | 8,000 | 30 | 1,000 | 200,000 | (unknown) |
| `openai/gpt-oss-safeguard-20b` | 2,000 | 3 | 1,000 | 200,000 | 131,072 |

Any other provider/model has no bundled limits and falls back to the global
`MAX_INPUT_TOKENS` with no daily guard; the agent still reacts to the
provider's own 429 responses and rate-limit headers.

## Request-format compatibility

- The request loop uses OpenAI's `max_tokens` by default and automatically
  retries with `max_completion_tokens` when a newer reasoning model rejects the
  old name (driven by the provider's own error text).
- The OpenAI SDK is built with `max_retries=0`, so retries are made visibly in
  the agent loop instead of inside the SDK (see
  [token budget](token-budget-and-compaction.md)).
- Some providers return a tool call as ordinary text instead of the native
  `tool_calls` field. Two text protocols are recognised: a bare JSON object
  (`{"name", "arguments"}` or `{"tool"/"function", "parameters"}`), and the
  MiniMax XML-ish serialization degraded into literal text with its token-noise
  string. Prose replies are never rewritten; a malformed or truncated text call
  fails the turn instead of executing half a call.
- Providers differ in how they report usage (`prompt_tokens` vs
  `input_tokens`, nested details objects); `src/usage_report.py` normalises all
  of them into one value. Missing fields stay `None` and are never invented.

## Limitations

- Model discovery lists what a provider advertises, not what the key can call.
- Providers without bundled limits or `NAME_*` / `MODEL_<TAG>_*` overrides have
  no local daily guard.
- A model that hangs without sending response bytes cannot be distinguished
  from a slow one; the harness waits up to `LLM_TIMEOUT` (default 120 s).
- Tool calls written as prose rather than JSON or a recognised text protocol
  are not executed.

For the complete variable list see
[configuration.md](configuration.md).
