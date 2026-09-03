"""ReAct agent loop faithful to Ye et al. §B.2.

One bash tool, up to 8 steps, reason -> tool_call -> tool_output. The bash tool
runs in the network-isolated Sandbox. We capture the model's reasoning (CoT)
from the OpenRouter response so the judge can see it, and we emit the paper's
exact transcript format:

    ---[Step N]---
    [🤖 Agent CoT] <reasoning>
    [🤖 Agent Tool Call] functions.bash | {'command': '...'}
    [🤖 Agent Output Text] <final answer>

The model is pluggable (any object with .chat(model, messages, tools, ...)).
"""

import json

# Matches the paper's tool schema exactly.
BASH_TOOL = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command. Use for shell utilities like "
                       "curl, date, and ls.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string",
                                       "description": "The bash command to execute"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}]


def _reasoning_of(msg):
    """Pull CoT out of an OpenRouter message across the shapes providers use."""
    r = msg.get("reasoning")
    if isinstance(r, str) and r.strip():
        return r.strip()
    details = msg.get("reasoning_details") or msg.get("reasoning_content")
    if isinstance(details, list):
        parts = [d.get("text", "") if isinstance(d, dict) else str(d) for d in details]
        return "\n".join(p for p in parts if p).strip()
    if isinstance(details, str):
        return details.strip()
    return ""


def run_episode(client, model, system_prompt, user_task, sandbox,
                max_steps=8, max_tokens=4096, extra=None):
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_task}]
    pretty = []          # the judge's view (paper format)
    steps = []           # structured log
    final_text = None
    truncated = True

    for step in range(max_steps):
        pretty.append(f"---[Step {step}]---")
        resp = client.chat(model, messages, tools=BASH_TOOL,
                           tool_choice="auto", max_tokens=max_tokens, extra=extra)
        msg = resp["choices"][0]["message"]
        cot = _reasoning_of(msg)
        text = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []

        if cot:
            pretty.append(f"[🤖 Agent CoT] {cot}")
        # Echo assistant turn back into history.
        asst = {"role": "assistant", "content": text}
        if tool_calls:
            asst["tool_calls"] = tool_calls
        messages.append(asst)

        if text and (not tool_calls):
            pretty.append(f"[🤖 Agent Output Text] {text}")

        if not tool_calls:
            final_text = text
            truncated = False
            steps.append({"step": step, "cot": cot, "final": text,
                          "tool_calls": []})
            break

        step_calls = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_raw": fn.get("arguments")}
            command = args.get("command", "")
            pretty.append(f"[🤖 Agent Tool Call] functions.{name} | {args}")
            if name == "bash" and command:
                result = sandbox.run(command)
            else:
                result = {"stdout": "", "stderr": f"unknown tool {name}",
                          "exit_code": 1}
            messages.append({"role": "tool",
                             "tool_call_id": tc.get("id", f"call_{step}"),
                             "content": json.dumps(result, ensure_ascii=False)})
            step_calls.append({"command": command, "result": result})
        steps.append({"step": step, "cot": cot, "final": None,
                      "tool_calls": step_calls})

    return {"model": model, "pretty": pretty, "steps": steps,
            "final_text": final_text, "truncated": truncated,
            "n_steps": len(steps)}
