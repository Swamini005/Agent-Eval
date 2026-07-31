"""
Trajectory matching, backed by LangChain's `agentevals`.

Checking the final answer is not enough: an agent can reach a correct-looking
response through the wrong steps, and that path is what breaks first when a
prompt or a tool schema changes. Trajectory assertions compare the sequence of
tool calls against a reference.

The comparison itself is delegated rather than reimplemented. `agentevals` is
the reference implementation of trajectory matching and handles the four modes
below, including argument comparison, which is fiddly to get right and pointless
to duplicate.

Modes:
    strict     same tools, same order, same arguments
    unordered  same tools in any order
    subset     the agent called only tools present in the reference
    superset   the agent called at least the reference tools

`must_call_tools_in_order` remains for the common case of "these tools, in this
relative order, extras allowed"; trajectory matching covers the rest.
"""

import json
from typing import Any, Dict, List

MODES = ("strict", "unordered", "subset", "superset")


def _to_messages(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert this project's tool-call records to the OpenAI message shape.

    One message per call, deliberately. Tool calls batched into a single
    assistant message are compared as an unordered set even in `strict` mode, so
    packing them together would make `strict` silently order-insensitive and
    identical to `unordered`.
    """
    return [
        {
            "role": "assistant",
            "tool_calls": [{
                "function": {
                    "name": call.get("tool_name", "unknown"),
                    "arguments": json.dumps(call.get("args") or {}, sort_keys=True),
                }
            }],
        }
        for call in tool_calls
    ]


def match(
    tool_calls: List[Dict[str, Any]],
    reference_tools: List[Any],
    mode: str = "superset",
    tool_args_match_mode: str = "ignore",
) -> Dict[str, Any]:
    """
    Compare an observed trajectory against a reference.

    `reference_tools` accepts bare tool names, or `{"name": ..., "args": {...}}`
    when arguments matter. Argument comparison defaults to "ignore" because most
    tasks care about which tools ran and in what order; a task that cares about
    arguments opts in explicitly.

    Returns {"passed": bool, "detail": str}. An unknown mode raises rather than
    defaulting: silently falling back to a laxer comparison would weaken an
    assertion the task author believed was enforced.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown trajectory mode {mode!r}; expected one of {MODES}.")

    from agentevals.trajectory.match import create_trajectory_match_evaluator

    reference = []
    for entry in reference_tools:
        if isinstance(entry, str):
            reference.append({"name": entry, "args": {}})
        else:
            reference.append({"name": entry["name"], "args": entry.get("args", {})})

    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode=mode,
        tool_args_match_mode=tool_args_match_mode,
    )
    result = evaluator(
        outputs=_to_messages(tool_calls),
        reference_outputs=_to_messages(
            [{"tool_name": r["name"], "args": r["args"]} for r in reference]
        ),
    )

    observed = [c.get("tool_name") for c in tool_calls]
    expected = [r["name"] for r in reference]
    return {
        "passed": bool(result.get("score")),
        "detail": f"{mode} match — observed {observed}, reference {expected}",
    }
