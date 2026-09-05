"""Local gpt-oss Inspect provider with optional role-direction projection.

Inspect's HF provider cannot parse Harmony tool calls, so we render the chat
template and parse output ourselves. Projection removes a probe-derived
direction (userness, or userness−toolness) from Harmony tool-call / tool-result
tokens at chosen layers. Feature: pre-MLP (`post_attention_layernorm`).

Probes: vendor `02-train-role-probes.ipynb` pickle (or data/role_probes.pkl).
"""

from __future__ import annotations

import contextlib
import gc
import json
import pickle
import re
import sys

import numpy as np
import torch
from inspect_ai.model import (
    ChatCompletionChoice, ChatMessageAssistant, ContentReasoning, ContentText,
    GenerateConfig, ModelOutput, ModelUsage,
)
from inspect_ai.model._model import ModelAPI
from inspect_ai.model._registry import modelapi
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo
from transformers import AutoModelForCausalLM, AutoTokenizer

_PREFILL_CHUNK = 512
_ROLES = ["system", "user", "cot", "assistant", "tool"]
_TOOL_RESULT_RE = re.compile(
    r"<\|start\|>(?:functions\.[^\s<]+|tool)\b.*?<\|message\|>.*?<\|end\|>", re.DOTALL,
)
_TOOL_CALL_SPAN_RE = re.compile(
    r"<\|start\|>assistant\b[^<]*to=functions\.[^\s<]+.*?<\|message\|>.*?(?:<\|call\|>|<\|end\|>)",
    re.DOTALL,
)
# From vendor experiments/agent-injections/01-run-user-injections-gpt-oss.ipynb
_PARSE_CALL_RE = re.compile(
    r"(?:^|<\|start\|>\s*assistant\s*)"
    r"(?:(?:to=(?P<to1>functions\.[^\s<]+)\s*<\|channel\|>\s*(?P<chan1>commentary|analysis))"
    r"|(?:<\|channel\|>\s*(?P<chan2>commentary|analysis)\s*to=(?P<to2>functions\.[^\s<]+)))"
    r"\s*(?:<\|constrain\|>\s*)?[A-Za-z0-9_.+-]*\s*"
    r"<\|message\|>(?P<args>.*?)"
    r"(?:<\|call\|>|(?=<\|end\|>|<\|start\|>\s*assistant|$))",
    re.DOTALL | re.IGNORECASE,
)
_FINAL_RE = re.compile(
    r"(?:^|<\|start\|>\s*assistant\s*)<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)",
    re.DOTALL,
)
_ANALYSIS_RE = re.compile(
    r"(?:^|<\|start\|>\s*assistant\s*)<\|channel\|>\s*analysis\s*<\|message\|>(.*?)(?:<\|end\|>|$)",
    re.DOTALL,
)


def _direction(coef: np.ndarray, kind: str) -> torch.Tensor:
    u, t = _ROLES.index("user"), _ROLES.index("tool")
    v = coef[u] - (coef.mean(0) if kind == "userness" else coef[t])
    return torch.tensor(v / np.linalg.norm(v), dtype=torch.float32)


def load_vectors(probes_path: str, kind: str, layers: list[int]) -> dict[int, torch.Tensor]:
    with open(probes_path, "rb") as f:
        probes = pickle.load(f)
    out: dict[int, torch.Tensor] = {}
    for L in layers:
        p = next(x for x in probes if x["layer_ix"] == L and list(x["role_space"]) == _ROLES)
        out[L] = _direction(p["probe"].coef_, kind)
    return out


def tool_token_indices(tokenizer: AutoTokenizer, text: str) -> set[int]:
    spans = [(m.start(), m.end()) for m in _TOOL_RESULT_RE.finditer(text)]
    spans += [(m.start(), m.end()) for m in _TOOL_CALL_SPAN_RE.finditer(text)]
    if not spans:
        return set()
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt")
    hits: set[int] = set()
    for i, (a, b) in enumerate(enc["offset_mapping"][0].tolist()):
        if a == b == 0:
            continue
        if any(a < end and b > start for start, end in spans):
            hits.add(i)
    return hits


@contextlib.contextmanager
def project_tool_tokens(model, vectors: dict[int, torch.Tensor], layers: list[int], positions: set[int]):
    """Project `vectors[L]` out of `positions` at pre-MLP of each L. Decode (T=1) untouched."""
    blocks = model.model.layers
    param = next(model.parameters())
    handles = []
    pos_cell: dict[int, torch.Tensor | None] = {}

    def stash(layer_ix: int):
        def pre(_m, _a, kwargs):
            pos_cell[layer_ix] = kwargs.get("position_ids") if kwargs else None
        return pre

    def make_hook(vec: torch.Tensor, layer_ix: int):
        u = (vec.to(dtype=param.dtype, device=param.device))
        u = u / (u.norm() + 1e-8)

        def hook(_m, _inp, kwargs, out):
            hs = out[0] if isinstance(out, tuple) else out
            if hs.shape[1] == 1:
                return out
            pid = (kwargs.get("position_ids") if kwargs else None) or pos_cell.get(layer_ix)
            if pid is not None:
                idx = [i for i, g in enumerate(pid[0].tolist()) if g in positions]
            else:
                idx = [i for i in positions if i < hs.shape[1]]
            if not idx:
                return out
            t = torch.as_tensor(idx, device=hs.device)
            hs = hs.clone()
            sl = hs[:, t, :]
            hs[:, t, :] = sl - (sl * u).sum(-1, keepdim=True) * u
            return (hs,) + out[1:] if isinstance(out, tuple) else hs
        return hook

    try:
        for L in layers:
            handles.append(blocks[L].register_forward_pre_hook(stash(L), with_kwargs=True))
            handles.append(blocks[L].post_attention_layernorm.register_forward_hook(
                make_hook(vectors[L], L), with_kwargs=True,
            ))
        yield
    finally:
        for h in handles:
            h.remove()


def _parse_args(raw: str) -> dict:
    raw = raw.strip()
    try:
        payload = json.loads(raw)
        if isinstance(payload, str):
            payload = json.loads(payload)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    return {"command": raw.strip('"')}


def parse_harmony(text: str) -> tuple[str, list[str], list[tuple[str, dict]]]:
    calls = []
    for m in _PARSE_CALL_RE.finditer(text):
        fqn = m.group("to1") or m.group("to2")
        calls.append((fqn.split(".", 1)[-1], _parse_args(m.group("args"))))
    fm = _FINAL_RE.search(text)
    analysis = [m.group(1).strip() for m in _ANALYSIS_RE.finditer(text)]
    return (fm.group(1).strip() if fm else ""), analysis, calls


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(getattr(c, "text", "") or "" for c in content)
    return str(content) if content is not None else ""


def _to_chat_messages(messages: list) -> list[dict]:
    out: list[dict] = []
    for msg in messages:
        if msg.role in ("system", "user"):
            out.append({"role": msg.role, "content": _text_of(msg.content)})
        elif msg.role == "assistant":
            m: dict = {"role": "assistant", "content": _text_of(msg.content)}
            if getattr(msg, "tool_calls", None):
                m["tool_calls"] = [{
                    "type": "function", "id": tc.id,
                    "function": {"name": tc.function, "arguments": json.dumps(tc.arguments)},
                } for tc in msg.tool_calls]
            out.append(m)
        elif msg.role == "tool":
            out.append({
                "role": "tool",
                "name": getattr(msg, "function", None) or "bash",
                "content": _text_of(msg.content),
            })
    return out


class SteeredHuggingFaceAPI(ModelAPI):
    def __init__(self, model_name, base_url=None, api_key=None, config=GenerateConfig(), **args):
        super().__init__(model_name, base_url, api_key, [], config)
        self.project: str = args.get("project", "off")
        self.max_new_tokens = int(args.get("max_new_tokens", 2048))
        hf_id = args.get("hf_model", "openai/gpt-oss-20b")
        self.layers = [int(x) for x in str(args.get("layers", "")).split(",") if x.strip()]
        self.vectors: dict[int, torch.Tensor] = {}
        if self.project != "off":
            probes = args.get("probes") or "data/role_probes.pkl"
            self.vectors = load_vectors(probes, self.project, self.layers)
        print(f"[local] project={self.project} layers={self.layers} model={hf_id}", file=sys.stderr)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(hf_id, dtype="auto", device_map="cuda").eval()
        self._id_call = self.tokenizer.convert_tokens_to_ids("<|call|>")
        self._id_return = self.tokenizer.convert_tokens_to_ids("<|return|>")

    def max_connections(self) -> int:
        return 1

    @torch.no_grad()
    def _generate_text(self, prompt: str, config: GenerateConfig) -> tuple[str, int, int]:
        ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        n_in = ids["input_ids"].shape[1]
        temp = config.temperature if config.temperature is not None else 1.0
        kwargs: dict = dict(
            max_new_tokens=config.max_tokens or self.max_new_tokens,
            eos_token_id=[self._id_call, self._id_return],
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            use_cache=True, prefill_chunk_size=_PREFILL_CHUNK,
            do_sample=bool(temp), **({"temperature": temp} if temp else {}),
        )
        ctx = contextlib.nullcontext()
        if self.project != "off":
            pos = tool_token_indices(self.tokenizer, prompt)
            # progress.py watches scored samples; this is too noisy for tail -f
            ctx = project_tool_tokens(self.model, self.vectors, self.layers, pos)
        try:
            with ctx:
                out = self.model.generate(**ids, **kwargs)
            new = out[0][n_in:]
            if len(new) and new[-1].item() in (self._id_call, self._id_return):
                new = new[:-1]
            text = self.tokenizer.decode(new, skip_special_tokens=False)
            return text, n_in, len(new)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def generate(self, input, tools: list[ToolInfo], tool_choice: ToolChoice, config: GenerateConfig) -> ModelOutput:
        schemas = [{
            "type": "function",
            "function": {
                "name": t.name, "description": t.description,
                "parameters": t.parameters.model_dump(exclude_none=True) if hasattr(t.parameters, "model_dump") else t.parameters,
            },
        } for t in tools] if tools else None
        prompt = self.tokenizer.apply_chat_template(
            _to_chat_messages(input), tools=schemas, add_generation_prompt=True, tokenize=False,
        )
        gen, n_in, n_out = self._generate_text(prompt, config)
        final, analysis, calls = parse_harmony(gen)
        parts = [ContentReasoning(reasoning=a) for a in analysis]
        if final:
            parts.append(ContentText(text=final))
        tcs = [ToolCall(id=f"call_{i}", function=n, arguments=a, type="function") for i, (n, a) in enumerate(calls)]
        msg = ChatMessageAssistant(
            content=parts or final or "", tool_calls=tcs or None,
            model=self.model_name, source="generate", metadata={"harmony_raw": gen},
        )
        return ModelOutput(
            model=self.model_name,
            choices=[ChatCompletionChoice(message=msg, stop_reason="tool_calls" if tcs else "stop")],
            usage=ModelUsage(input_tokens=n_in, output_tokens=n_out, total_tokens=n_in + n_out),
        )


@modelapi(name="steered-hf")
def steered_hf() -> type[ModelAPI]:
    return SteeredHuggingFaceAPI
