"""LLM back-ends. Every call is appended to a JSONL log (Section 8.4.4:
"the full set of prompts and replies was maintained for every run").

Providers
  anthropic : Anthropic Messages API (needs ANTHROPIC_API_KEY)
  openai    : any OpenAI-compatible /chat/completions endpoint (OpenAI, vLLM,
              Ollama, LM Studio...). Set base_url and OPENAI_API_KEY if needed.
  mock      : no network. Returns the first parent with jittered PARAMS. Only
              for testing the pipeline.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request


class NeedReply(Exception):
    """Raised by the manual provider when a prompt is waiting for a reply file."""


class LLM:
    def __init__(self, cfg: dict, log_path: str):
        self.provider = cfg.get("provider", "anthropic")
        self.model = cfg.get("model")
        self.temperature = cfg.get("temperature", 1.0)
        self.max_tokens = cfg.get("max_tokens", 8000)
        self.base_url = cfg.get("base_url", "https://api.openai.com/v1")
        self.retries = cfg.get("retries", 3)
        self.log_path = log_path
        self._rng_state = 0
        self.reply_dir = cfg.get("reply_dir") or os.path.join(os.path.dirname(log_path), "manual")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # ------------------------------------------------------------------
    def complete(self, system: str, user: str, meta: dict) -> str:
        last = None
        for attempt in range(self.retries):
            try:
                t0 = time.time()
                reply = getattr(self, f"_{self.provider}")(system, user, meta)
                self._log(system, user, reply, meta, time.time() - t0, None)
                return reply
            except NeedReply:
                raise
            except Exception as e:  # network / rate limit
                last = e
                self._log(system, user, None, meta, 0, repr(e))
                time.sleep(min(60, 5 * 2 ** attempt))
        raise RuntimeError(f"LLM failed after {self.retries} attempts: {last!r}")

    def _log(self, system, user, reply, meta, secs, err):
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "provider": self.provider,
                                "model": self.model, "temperature": self.temperature, "meta": meta,
                                "system": system, "user": user, "reply": reply,
                                "seconds": secs, "error": err}) + "\n")

    # ------------------------------------------------------------------
    def _anthropic(self, system, user, meta):
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                     temperature=self.temperature, system=system,
                                     messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    def _openai(self, system, user, meta):
        body = json.dumps({"model": self.model, "temperature": self.temperature,
                           "max_tokens": self.max_tokens,
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}]}).encode()
        req = urllib.request.Request(self.base_url.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    def _manual(self, system, user, meta):
        """Operator mode: the prompt is written to <reply_dir>/gGG_kKK.prompt.md and the
        reply is read from gGG_kKK.reply.md once someone (or some model) has written it."""
        os.makedirs(self.reply_dir, exist_ok=True)
        stem = os.path.join(self.reply_dir, f"g{meta['gen']:02d}_k{meta['k']:02d}")
        if os.path.exists(stem + ".reply.md"):
            return open(stem + ".reply.md").read()
        with open(stem + ".prompt.md", "w") as f:
            f.write(f"<!-- op={meta['op']} parents={meta['parents']} -->\n# SYSTEM\n{system}\n\n# USER\n{user}\n")
        raise NeedReply(stem)

    def _mock(self, system, user, meta):
        import numpy as np
        self._rng_state += 1
        rng = np.random.default_rng(self._rng_state)
        if meta.get("op") == "E1" and os.path.exists("designs/example_non_mrg.py"):
            return "```python\n" + open("designs/example_non_mrg.py").read() + "```"
        code = meta.get("fallback_code") or extract_code(user) or ""
        def jitter(m):
            v = float(m.group(2))
            return f'{m.group(1)}{v * float(rng.uniform(0.6, 1.6)):.4g}'
        code = re.sub(r'("t_intensity":\s*)([0-9.eE+-]+)', jitter, code)
        code = re.sub(r'("t_orientation":\s*)([0-9.eE+-]+)', jitter, code)
        return f"```python\n{code}\n```"


def extract_code(text: str) -> str | None:
    if not text:
        return None
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    if "def segment" in text:
        return text.strip() + "\n"
    return None
