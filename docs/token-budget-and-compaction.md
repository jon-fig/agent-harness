# Token budget and compaction

Every agent iteration resends the whole conversation, so the cost of a turn
grows with the square of the iteration count. The harness bounds that growth at
the points that matter: the tool surface, the system prompt, tool output,
history, and the request itself.

## Three different things called "tokens"

The harness deliberately keeps these apart:

1. **Estimate** — made locally before a request is sent. Always carries
   provenance (exact tokenizer vs. approximate fallback).
2. **Reported usage** — what the provider returned in `response.usage`. This is
   what history accounting prefers.
3. **Budget** — what the request is checked against (context window, input
   ceiling, rate-limit buckets).

An estimate is never presented as provider-reported usage, and a local budget
never silently overrides a provider limit.

## Context window vs. input/output limits

- The **context window** is the model's total capacity (input + reserved
  output together).
- `MAX_INPUT_TOKENS` (default 6,500) is the harness's ceiling on the estimated
  request body.
- `MAX_OUTPUT_TOKENS` (default 4,000) is the generation cap *requested* from the
  model; `src/models.py` clamps it per model/provider, so a lower
  model-specific or provider-specific maximum always wins.

## Token estimation and tokenizer selection

`src/tokenization.py` resolves the best tokenizer for the active model, in this
order:

1. An explicit per-model variable: `TOKENIZER_<SANITIZED_MODEL_ID>` (e.g.
   `TOKENIZER_MINIMAXAI_MINIMAX_M3_1_FLASH`).
2. `TOKENIZER_MAP` — a JSON object of `fnmatch` pattern → spec, e.g.
   `{"qwen/*": "tiktoken:cl100k_base"}`.
3. `TOKENIZER_DEFAULT`.
4. A known OpenAI model family, matched by regex on the model id: `gpt-oss` →
   `o200k_harmony`, `gpt-4o`/`gpt-4.1`/`gpt-5`/`o1…` → `o200k_base`,
   `gpt-4`/`gpt-3.5`/embeddings → `cl100k_base`, codex models → `p50k_base`.
5. The approximate fallback: `characters / 4`, rounded up.

A spec can be `tiktoken:<encoding>`, `approx[:chars_per_token]`, `hf:<repo>` or
`file:<tokenizer.json>`. The `hf:`/`file:` forms need the optional `tokenizers`
package; without it they degrade to the approximate tokenizer instead of
failing. Only a *string* counted with an exact tokenizer is reported as exact;
any structured payload (chat messages) is always labelled approximate because
chat framing is not represented by a raw tokenizer count.

### Calibration

Models without a compatible tokenizer start from chars/4 and **self-calibrate**:
after each response the observed characters-per-reported-input-token ratio is
clamped to [1.5, 8.0], smoothed (`TOKENIZER_CALIBRATION` smoothing factor 0.5)
and persisted per provider/model in `.agent/tokenizer.json`. One odd response
cannot skew later estimates, and a later process starts from the learned value
instead of 4.0. Only the approximate tokenizer is re-based; an exact one
already agrees with the provider. `TOKENIZER_CALIBRATION=0` keeps the fixed
ratio.

## Tool-output truncation

`src/mcp_client.py` is the single point where tool output enters history. Output
longer than `TOOL_OUTPUT_MAX_CHARS` (default 4,000 **characters**) is
middle-truncated: the head (column names, top of output) and the tail
(traceback, last rows) are kept, the middle is dropped, and the notice names how
many characters were omitted. A truncated JSON document is explicitly marked as
no longer valid JSON, so the model cannot mistake the fragment for a complete
result.

Characters and tokens are different units: 4,000 characters is roughly 1,000
tokens of English prose, fewer for code/JSON, more for non-Latin scripts. The
character cap is applied where output is produced; the token cost is only known
as part of the request estimate.

## Conversation history management

`compact_history` in `src/budget.py` collapses every tool exchange (an assistant
message with `tool_calls` plus its tool replies) except the newest
`HISTORY_KEEP_EXCHANGES` (default 2):

- long string arguments keep their head (160 chars) with an explicit
  `[N chars collapsed]` marker;
- assistant prose is truncated to 240 chars;
- each old tool result becomes one line: the call happened, the original size in
  KB, and — at the `summary` detail level — the result's first line (a header
  row, a row count, a saved path). Errors are preserved (truncated to 240
  chars), never summarised away, because the model may still need to correct
  them.

Already-collapsed messages are never rewritten again (repeated compaction is
idempotent), user messages and assistant prose are never rewritten, and the full
text stays on disk where the agent should re-read it anyway.

## Automatic compaction

`maybe_auto_compact` prices the **complete** request (system prompt + tool
schemas + every message) first, then decides:

- Eligibility: `AUTO_COMPACT` enabled (default) **and** at least
  `AUTO_COMPACT_MIN_TURNS` user turns (default 2) **and** the measured request
  exceeds `AUTO_COMPACT_THRESHOLD` (default `0.6` = 60% of the input budget; a
  value above 1 is an absolute token count).
- Target: collapse older tool exchanges until `AUTO_COMPACT_TARGET` (default
  `0.5`) is reached or no meaningful reduction remains.
- Progression: cheap first (`summary` keeps each result's first line), then
  aggressive (`terse` drops it). The newest exchange is never collapsed during
  automatic compaction — only hard fitting may go to zero as a last resort.
- A compaction that saves fewer than 4 tokens is neither applied nor retried.
- The floor: the smallest request the policy can produce is measured too. When
  most of the request is protected text (system prompt, tool schemas, user
  messages, assistant prose, newest exchange), the log names that floor instead
  of pretending the target is reachable.

When the request still does not fit after everything, the turn is refused with a
clear `BudgetExceeded` message instead of sending something the provider will
reject. `/compact` runs the same collapse on demand; `AUTO_COMPACT=0` disables
the automatic path.

## Request sizing and pre-flight checks

`build_payload` tries progressively stronger compactions
(`HISTORY_KEEP_EXCHANGES` → 1 → 0 exchanges kept verbatim) until the estimated
input fits the budget. If nothing fits, the smallest attempt is returned with
`fits=False`. As a further fallback the request is priced and sent **without**
the tool block, which is the cheapest way to still get an answer when the
conversation alone is close to the limit.

Before the request is sent:

- **Size check** — estimated input must be within the model's input budget;
  otherwise `BudgetExceeded`.
- **Daily guard** — the projected request (estimated input + reserved output)
  is added to today's totals and checked against bundled RPD/TPD limits. The
  check is projective so the last request that fits cannot push usage over the
  limit. Cached prompt tokens are excluded from the daily token check, matching
  how the provider counts.
- **Pacing** — the provider's `x-ratelimit-remaining-tokens` /
  `x-ratelimit-reset-tokens` headers are tracked per provider/model. When the
  remaining window cannot cover the next request, the harness waits out the
  window (capped at 60 s) instead of provoking a 429.

## HTTP responses

| Response | Handling |
|---|---|
| **413** (request too large) | Compacts the entire history once (nothing kept verbatim) and retries; if it still does not fit, the turn fails. |
| **429** (rate limited) | Waits for the provider's own hint — `retry-after` or `x-ratelimit-reset-tokens`, which accepts durations like `7.66s` or `1m2.4s` — falling back to an exponential delay (2 s doubling, capped at 60 s). |
| **5xx** | Bounded backoff (starts at 2 s, doubles up to 60 s, plus jitter). |
| **400** rejecting `max_tokens` | Switches to `max_completion_tokens` and retries once. |

Retries are bounded by `LLM_MAX_RETRIES` (default 2) *per response class* in the
loop; the OpenAI SDK itself is built with `max_retries=0` so these decisions are
made visibly in the agent loop, not inside the SDK.

## Output and daily limits

- `finish_reason=length` on a final answer: the answer is returned with a
  visible `[Answer truncated at the model's N-token output limit…]` marker.
- `finish_reason=length` while composing a tool call: the call is **not**
  executed (half a `write_file` body would replace a good file) and the turn
  reports the problem.
- `MAX_ITERATIONS` (default 8) caps model calls per user turn;
  `MAX_TOOL_CALLS_PER_TURN` (default 20) separately caps tool executions, since
  one model call can request many.
- The final iteration runs without tools and gets an appended note telling the
  model to summarise what it completed and what is left.

## Daily usage accounting

`src/usage.py` counts requests and tokens per provider/model per day in
`.agent/usage.json`, keyed by UTC date (matching providers that reset at
midnight UTC). Two kinds of tokens are stored **separately**:

- **reported** — accumulated from `response.usage`;
- **estimated** — the local estimate, used only when the provider returned no
  usable total, never double-counted.

The guard compares *billable* usage (total minus cached tokens) plus the
projection against the RPD/TPD limits and refuses a projected violation with the
reset time (midnight UTC). The counters are local to this machine: they are a
guard rail, not a billing system, and a second machine using the same key is
invisible to them.

`/model` reports today's usage with the same reported/estimated distinction
(`usage.describe`).
