"""Custom Inspect model provider: local gpt-oss with userness projection.

Registers `steered-hf`, a ModelAPI that runs `openai/gpt-oss-20b` locally
(MXFP4) so we can host forward hooks that project the userness direction out of
non-user token positions during generation. Inspect's built-in `hf` provider
cannot parse gpt-oss Harmony tool calls, so this provider renders tools via the
tokenizer chat template and parses the Harmony output itself, returning tool
calls in Inspect's format so the standard tool loop + sandbox + transcript
logging all work unchanged.

Model args (via `-M key=val`):
  mode      "off" (baseline) | "project" (default "off")
  vectors   path to steering_vectors.pt
  layers    comma list of layer indices (default: suggested layer from vectors)
  device    default "cuda"
  max_new_tokens  default 2048
  attn_implementation  default kernels-community/vllm-flash-attn3 (eager fallback)
"""

from __future__ import annotations

import gc
import json
import pathlib
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# gpt-oss cannot use SDPA (attention sinks). Flash via the kernels package
# avoids materializing the O(n^2) eager softmax that OOMs ~30k-token attacks
# on a 24 GB card. Fall back to eager if the kernel is unavailable.
_FLASH_ATTN = "kernels-community/vllm-flash-attn3"

from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.model._model import ModelAPI
from inspect_ai.model._registry import modelapi
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from steered_generate import non_user_token_indices, steering  # noqa: E402


# ---- Harmony parsing (adapted from the paper's parse_assistant_output) ----

_TOOL_CALL_RE = re.compile(
    r"(?:^|<\|start\|>\s*assistant\s*)"
    r"(?:"
    r"(?:to=(?P<to1>functions\.[^\s<]+)\s*<\|channel\|>\s*(?P<chan1>commentary|analysis))"
    r"|(?:<\|channel\|>\s*(?P<chan2>commentary|analysis)\s*to=(?P<to2>functions\.[^\s<]+))"
    r")"
    r"\s*(?:<\|constrain\|>\s*)?[A-Za-z0-9_.+-]*\s*"
    r"<\|message\|>(?P<args>.*?)"
    r"(?:<\|call\|>|(?=<\|end\|>|<\|start\|>\s*assistant|$))",
    re.DOTALL | re.IGNORECASE,
)
_FINAL_RE = re.compile(
    r"(?:^|<\|start\|>\s*assistant\s*)"
    r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)",
    re.DOTALL,
)
_ANALYSIS_RE = re.compile(
    r"(?:^|<\|start\|>\s*assistant\s*)"
    r"<\|channel\|>\s*analysis\s*<\|message\|>(.*?)(?:<\|end\|>|$)",
    re.DOTALL,
)


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
    """Return (final_text, analysis_segments, [(tool_name, args_dict), ...])."""
    tool_calls: list[tuple[str, dict]] = []
    for m in _TOOL_CALL_RE.finditer(text):
        fqn = m.group("to1") or m.group("to2")
        name = fqn.split(".", 1)[1] if "." in fqn else fqn
        tool_calls.append((name, _parse_args(m.group("args"))))

    fm = _FINAL_RE.search(text)
    final_text = fm.group(1).strip() if fm else ""
    analysis = [m.group(1).strip() for m in _ANALYSIS_RE.finditer(text)]
    return final_text, analysis, tool_calls


# ---- message + tool conversion for the chat template ----

def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            t = getattr(c, "text", None)
            if t:
                parts.append(t)
        return "\n".join(parts)
    return str(content) if content is not None else ""


def _to_chat_messages(input: list) -> list[dict]:
    out: list[dict] = []
    for msg in input:
        role = msg.role
        if role == "system":
            out.append({"role": "system", "content": _text_of(msg.content)})
        elif role == "user":
            out.append({"role": "user", "content": _text_of(msg.content)})
        elif role == "assistant":
            m: dict = {"role": "assistant", "content": _text_of(msg.content)}
            if getattr(msg, "tool_calls", None):
                m["tool_calls"] = [{
                    "type": "function",
                    "id": tc.id,
                    "function": {
                        "name": tc.function,
                        "arguments": json.dumps(tc.arguments),
                    },
                } for tc in msg.tool_calls]
            out.append(m)
        elif role == "tool":
            out.append({
                "role": "tool",
                "name": getattr(msg, "function", None) or "bash",
                "content": _text_of(msg.content),
            })
    return out


def _to_tool_schemas(tools: list[ToolInfo]) -> list[dict]:
    schemas = []
    for t in tools:
        params = t.parameters
        params_dict = params.model_dump(exclude_none=True) if hasattr(params, "model_dump") else params
        schemas.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": params_dict,
            },
        })
    return schemas


class SteeredHuggingFaceAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args,
    ) -> None:
        super().__init__(model_name, base_url, api_key, [], config)

        self.mode: str = model_args.get("mode", "off")
        self.device: str = model_args.get("device", "cuda")
        self.max_new_tokens: int = int(model_args.get("max_new_tokens", 2048))
        hf_id = model_args.get("hf_model", "openai/gpt-oss-20b")

        vectors_path = model_args.get(
            "vectors", str(ROOT / "data" / "steering_vectors.pt")
        )
        data = torch.load(vectors_path, weights_only=False)
        self.vectors = data["vectors"]
        if model_args.get("layers"):
            self.layers = [int(x) for x in str(model_args["layers"]).split(",")]
        else:
            self.layers = [int(data["direction_consistency"][1:].argmax()) + 1]

        attn_impl = model_args.get("attn_implementation", _FLASH_ATTN)
        print(f"[steered-hf] mode={self.mode} layers={self.layers} "
              f"hf_model={hf_id} device={self.device} attn={attn_impl}",
              file=sys.stderr)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = _load_model(hf_id, self.device, attn_impl)
        self._id_call = self.tokenizer.convert_tokens_to_ids("<|call|>")
        self._id_return = self.tokenizer.convert_tokens_to_ids("<|return|>")

    def max_connections(self) -> int:
        return 1

    @torch.no_grad()
    def _generate_text(
        self, prompt_text: str, config: GenerateConfig
    ) -> tuple[str, int, int]:
        ids = self.tokenizer(prompt_text, return_tensors="pt",
                             add_special_tokens=False).to(self.model.device)
        prompt_len = ids["input_ids"].shape[1]

        temperature = config.temperature if config.temperature is not None else 1.0
        max_new = config.max_tokens or self.max_new_tokens
        if config.seed is not None:
            torch.manual_seed(config.seed)

        gen_kwargs: dict = dict(
            max_new_tokens=max_new,
            eos_token_id=[self._id_call, self._id_return],
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)

        if self.mode == "project":
            positions = non_user_token_indices(self.tokenizer, prompt_text)
            ctx = steering(self.model, self.vectors, self.layers,
                           mode="project", positions=positions)
        else:
            ctx = _null_ctx()

        try:
            with ctx:
                out = self.model.generate(**ids, **gen_kwargs)
            new_ids = out[0][prompt_len:]
            if len(new_ids) and new_ids[-1].item() in (self._id_call, self._id_return):
                new_ids = new_ids[:-1]
            text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
            n_out = len(new_ids)
        finally:
            # Long attack traces fragment the 24 GB pool; free between ReAct steps.
            del ids
            if "out" in locals():
                del out
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return text, prompt_len, n_out

    async def generate(
        self,
        input: list,
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        messages = _to_chat_messages(input)
        tool_schemas = _to_tool_schemas(tools) if tools else None
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tools=tool_schemas,
            add_generation_prompt=True,
            tokenize=False,
        )

        gen_text, n_in, n_out = self._generate_text(prompt_text, config)
        final_text, analysis, harmony_calls = parse_harmony(gen_text)

        content_parts = []
        try:
            from inspect_ai.model import ContentReasoning, ContentText
            for a in analysis:
                content_parts.append(ContentReasoning(reasoning=a))
            if final_text:
                content_parts.append(ContentText(text=final_text))
        except Exception:
            content_parts = final_text

        tool_calls = [
            ToolCall(id=f"call_{i}", function=name, arguments=args, type="function")
            for i, (name, args) in enumerate(harmony_calls)
        ]

        assistant = ChatMessageAssistant(
            content=content_parts if content_parts else (final_text or ""),
            tool_calls=tool_calls or None,
            model=self.model_name,
            source="generate",
            metadata={"harmony_raw": gen_text},
        )
        choice = ChatCompletionChoice(
            message=assistant,
            stop_reason="tool_calls" if tool_calls else "stop",
        )
        return ModelOutput(
            model=self.model_name,
            choices=[choice],
            usage=ModelUsage(
                input_tokens=n_in, output_tokens=n_out,
                total_tokens=n_in + n_out,
            ),
        )


def _load_model(hf_id: str, device: str, attn_impl: str):
    """Load gpt-oss, preferring flash-attn kernels over eager softmax."""
    kwargs: dict = dict(dtype="auto", device_map=device)
    if attn_impl and attn_impl not in ("eager", "default", ""):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                hf_id, attn_implementation=attn_impl, **kwargs,
            ).eval()
            print(f"[steered-hf] loaded attn_implementation={attn_impl}",
                  file=sys.stderr)
            return model
        except Exception as e:
            print(f"[steered-hf] attn={attn_impl} failed ({type(e).__name__}: {e}); "
                  f"falling back to eager", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs).eval()
    print("[steered-hf] loaded attn_implementation=eager", file=sys.stderr)
    return model


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


@modelapi(name="steered-hf")
def steered_hf() -> type[ModelAPI]:
    return SteeredHuggingFaceAPI
