# Client Migration Guide

This release replaces the old `GET /qa/stream?input_message=…&token=jessica` endpoint with an authenticated `POST /qa/stream`. Legacy clients will receive `401` (missing/wrong `X-API-Key`) or `405` (method not allowed) until they migrate.

---

## What changed

| Aspect       | Old                                                                   | New                                                                |
| ------------ | --------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Method       | `GET`                                                                 | `POST`                                                             |
| Path         | `/qa/stream`                                                          | `/qa/stream` (unchanged)                                           |
| Auth         | `?token=jessica` query param (static, in URL)                         | `X-API-Key: <secret>` header                                       |
| Input        | `?input_message=…` query param                                        | JSON body `{ "input_message": "...", "session_id": "..." }`        |
| Input cap    | None (unbounded)                                                      | `MAX_INPUT_CHARS` (default 4000) → `413` on overflow               |
| Errors mid-stream | Broken/empty SSE                                                 | Typed `event: error` SSE frame, then `stream_end`                  |
| Cancellation | Server kept generating after client disconnect                        | Server cancels the run on the next chunk boundary                  |
| Readiness    | None — request would block on cold init                               | `GET /healthz` reports `{"ready": true/false}`; `503` until ready  |

The SSE event names emitted on the success path (`message_chunk`, `tool_message`, `end_of_ai_response`, `stream_end`) are **unchanged** — only the request shape and the new `event: error` frame are new.

---

## Action items per client

1. **Rotate the credential.** The string `jessica` was published in the old README and must be considered compromised. Issue a new `API_KEY` (≥ 16 chars), give it to the client, and store it as a secret — never bake it into a URL.
2. **Move the secret out of the URL** and into the `X-API-Key` request header.
3. **Switch to `POST` with a JSON body.** Native `EventSource` is GET-only and cannot send headers — see the browser snippet below for the replacement.
4. **Handle the new `413`, `503`, and `event: error` cases.** They were not reachable before and your client probably crashes on them today.
5. **Cap the user input at `MAX_INPUT_CHARS`** in your UI to avoid round-trip rejections.

---

## Drop-in replacements

### cURL

**Before**

```bash
curl -N "http://localhost:8000/qa/stream?input_message=มาตรฐาน&token=jessica"
```

**After**

```bash
curl -N -X POST http://localhost:8000/qa/stream \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input_message":"มาตรฐาน"}'
```

### Browser

**Before** (`EventSource`)

```javascript
const es = new EventSource(
  `http://localhost:8000/qa/stream?input_message=${encodeURIComponent(q)}&token=jessica`
);
es.addEventListener("message_chunk", (e) => {
  const { content } = JSON.parse(e.data);
  appendToken(content);
});
es.addEventListener("stream_end", () => es.close());
```

**After** (`fetch` + `ReadableStream`)

```javascript
const res = await fetch("http://localhost:8000/qa/stream", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": import.meta.env.VITE_API_KEY,
  },
  body: JSON.stringify({ input_message: q }),
  signal: abortController.signal, // cancel mid-stream when the user navigates away
});

if (res.status === 401) throw new Error("bad API key");
if (res.status === 413) throw new Error("question too long");
if (res.status === 503) throw new Error("server not ready");
if (!res.ok || !res.body) throw new Error(`unexpected ${res.status}`);

const reader = res.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });

  // SSE frames are separated by a blank line.
  let sep;
  while ((sep = buffer.indexOf("\n\n")) !== -1) {
    const frame = buffer.slice(0, sep);
    buffer = buffer.slice(sep + 2);

    let event = "message";
    let data = "";
    for (const line of frame.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    }

    if (event === "message_chunk") {
      appendToken(JSON.parse(data).content);
    } else if (event === "tool_message") {
      // optional: surface retrieval hits
    } else if (event === "error") {
      showError(JSON.parse(data).message);
    } else if (event === "stream_end") {
      return;
    }
  }
}
```

> If swapping out `EventSource` for `fetch` is too invasive, an SSE polyfill that supports POST + headers (e.g. [`@microsoft/fetch-event-source`](https://github.com/Azure/fetch-event-source)) is a smaller change.

---

## Status-code reference

| Status | Meaning                                                                 | Client action                                  |
| ------ | ----------------------------------------------------------------------- | ---------------------------------------------- |
| `200`  | Stream opened — read SSE frames                                         | Process events as above                        |
| `401`  | Missing or wrong `X-API-Key`                                            | Stop; surface auth error; do not retry         |
| `413`  | `input_message` exceeds `MAX_INPUT_CHARS`                               | Trim input, ask user to shorten, retry         |
| `422`  | Body failed Pydantic validation (e.g. empty `input_message`)            | Fix payload, do not retry as-is                |
| `503`  | `qa_service` not yet wired (vector index missing or app still starting) | Poll `GET /healthz` until `ready: true`, retry |

---

## Verifying the migration

1. `curl -fsS http://localhost:8000/healthz` → `{"status":"ok","ready":true}`.
2. The new `POST` example above returns `200` and streams Thai-language tokens.
3. Sending the legacy `GET /qa/stream?token=jessica&...` returns `401` (the global `X-API-Key` dependency rejects it) — confirms the old shape is fully retired.
