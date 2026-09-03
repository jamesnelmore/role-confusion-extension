"""Minimal OpenRouter chat client (stdlib only) with tool-calling support.

The API key is read from, in order:
  1. $OPENROUTER_API_KEY
  2. ./.env               (this project)
  3. ~/.config/openrouter/key
Never printed, never logged.
"""

import json
import os
import pathlib
import ssl
import time
import urllib.error
import urllib.request

URL = "https://openrouter.ai/api/v1/chat/completions"

try:
    import certifi
    _CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _CTX = ssl.create_default_context()


def _from_dotenv(name, path):
    p = pathlib.Path(path).expanduser()
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == name:
            return v.strip().strip("'\"")
    return None


def load_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    for path in (".env", "~/.config/openrouter/key"):
        key = _from_dotenv("OPENROUTER_API_KEY", path)
        if key:
            return key
    raise SystemExit(
        "OPENROUTER_API_KEY not found. Put it in ./.env as:\n"
        "  OPENROUTER_API_KEY=sk-or-...")


class OpenRouter:
    def __init__(self, key=None):
        self.key = key or load_key()

    def chat(self, model, messages, tools=None, tool_choice=None,
             temperature=0.0, max_tokens=1024, timeout=180, extra=None):
        body = {"model": model, "messages": messages,
                "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
        if extra:
            body.update(extra)
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            URL, data=data,
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json",
                     "HTTP-Referer": "https://role-confusion-eval.local",
                     "X-Title": "role-confusion-eval"})
        last = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (402, 429, 500, 502, 503, 520, 524) and attempt < 4:
                    time.sleep(2 ** attempt * 3)
                    continue
                detail = e.read().decode("utf-8", "replace")[:400]
                raise RuntimeError(f"OpenRouter HTTP {e.code} for {model!r}: {detail}")
            except (urllib.error.URLError, TimeoutError) as e:
                last = e
                if attempt < 4:
                    time.sleep(2 ** attempt * 3)
                    continue
                raise RuntimeError(f"OpenRouter network error for {model!r}: {e}")
        raise RuntimeError(f"OpenRouter failed for {model!r}: {last}")


if __name__ == "__main__":
    import sys
    m = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-4o-mini"
    r = OpenRouter().chat(m, [{"role": "user", "content": "Reply with exactly: pong"}],
                          max_tokens=10)
    print("key loaded OK; model reply:",
          r["choices"][0]["message"]["content"].strip())
