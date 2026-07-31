"""
Universal adapter: evaluate an agent this project has never seen.

The suite is meant to run against any agent from any framework. That claim only
holds if plugging one in is trivial, so this wraps an arbitrary callable or
object resolved from a dotted path -- no subclass to write, no file to add:

    python demo_pipeline.py --target=external --agent=their_module:their_agent

Supported shapes, tried in order:

    f(prompt) -> str                      plain function
    f(prompt) -> dict                     dict with a response/output/text key
    obj.run(prompt) / .invoke(prompt)     LangChain, CrewAI, most frameworks
    obj(prompt)                           anything callable
    obj.invoke({"input": prompt})         LangChain AgentExecutor / LCEL
    async def f(prompt)                   awaited, not returned unawaited

The last two are not niceties. An async agent whose coroutine is returned rather
than awaited never runs at all, and str() of a coroutine is a perfectly plausible
looking response -- so the suite would grade an agent that never executed. A
LangChain AgentExecutor rejects a bare string, which the runner would record as
the agent failing every task. Both look like a bad agent and are actually a bad
adapter, which is the worst failure this file can have.

Tool calls are read from the result when the agent reports them, under any of
the common key names. An agent that reports none is not assumed to have made
none: `capabilities()` declares tool visibility as unknown, so tool-dependent
assertions report themselves unmeasured rather than failing an agent for a fact
this adapter could not observe.
"""

import asyncio
import importlib
import inspect
from dataclasses import asdict, is_dataclass
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Generator, List, Optional

from app.adapters.base import BaseAgentAdapter
from app.adapters.registry import AdapterRegistry

logger = logging.getLogger(__name__)

# Result keys checked for the agent's answer, most specific first.
RESPONSE_KEYS = ("response", "output", "result", "answer", "text", "content", "final_output")

# Result keys checked for a list of tool calls.
TOOL_CALL_KEYS = ("tool_calls", "tools_called", "intermediate_steps", "actions", "steps")

# Per-call keys for the tool's name and arguments.
NAME_KEYS = ("tool_name", "name", "tool", "function")
ARG_KEYS = ("args", "arguments", "input", "inputs", "tool_input", "parameters")


def resolve(path: str) -> Any:
    """
    Import a dotted path and return the object.

    Accepts "module:attribute" or "module.attribute". Raises with the path
    included, because a typo here is the most likely failure and the message is
    the only thing the operator sees.
    """
    if not path:
        raise ValueError(
            "No agent specified. Pass --agent=module:callable, or set AGENT_PATH."
        )

    module_path, _, attribute = path.partition(":")
    if not attribute:
        module_path, _, attribute = path.rpartition(".")

    if not module_path or not attribute:
        raise ValueError(
            f"Could not parse agent path {path!r}. "
            f"Expected 'module:callable' or 'module.callable'."
        )

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(f"Could not import {module_path!r} from agent path {path!r}: {e}") from e

    try:
        return getattr(module, attribute)
    except AttributeError as e:
        raise ValueError(f"{module_path!r} has no attribute {attribute!r}") from e


# Input shapes tried when an agent rejects a bare string. `input` is the
# LangChain AgentExecutor convention; `messages` covers chat-style graphs.
DICT_INPUTS = (
    lambda p: {"input": p},
    lambda p: {"messages": [{"role": "user", "content": p}]},
)


def _run(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # A loop is already running on this thread; give the coroutine its own.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _joined(chunks: List[Any]) -> Any:
    """Collapse a drained stream into a single result."""
    if chunks and all(isinstance(c, str) for c in chunks):
        return "".join(chunks)
    # Streamed state updates rather than text: the last chunk is the final state.
    return chunks[-1] if chunks else ""


def _await(result: Any) -> Any:
    """
    Resolve whatever an async or streaming agent handed back.

    Returning a coroutine or generator unresolved means the agent never ran,
    while str() of either still produces a response-shaped string -- so the suite
    would score an agent that did nothing, and report it as the agent's failure.
    """
    if inspect.isasyncgen(result):
        async def drain():
            return [chunk async for chunk in result]
        return _joined(_run(drain()))
    if inspect.isawaitable(result):
        return _await(_run(result))
    if inspect.isgenerator(result):
        return _joined(list(result))
    return result


def instantiate(obj: Any) -> Any:
    """
    An agent handed over as a class is the normal case, so construct it.

    Without this, getattr(cls, "invoke") is an unbound function and the prompt
    binds to `self`, which fails in a way that reads like the agent's fault.
    """
    if not isinstance(obj, type):
        return obj
    try:
        return obj()
    except TypeError as e:
        raise ValueError(
            f"{obj.__name__} needs constructor arguments, so it cannot be built "
            f"automatically. Pass an already-constructed instance instead: "
            f"--agent=your_module:agent_instance ({e})"
        ) from e


def call_agent(agent: Any, prompt: str) -> Any:
    """
    Invoke an agent of unknown shape, trying the common conventions.

    A TypeError from the call is treated as a signature mismatch and the dict
    input shapes are tried. That can invoke an agent twice if it raised TypeError
    from inside its own body -- unavoidable without inspecting a foreign
    signature, and far cheaper than reporting a working agent as wholly broken.
    The shape that succeeded is logged so the operator can see what was used.
    """
    for method in ("run", "invoke", "__call__"):
        fn = getattr(agent, method, None)
        if not callable(fn):
            continue

        try:
            return _await(fn(prompt))
        except TypeError as string_error:
            for build in DICT_INPUTS:
                payload = build(prompt)
                try:
                    result = _await(fn(payload))
                except TypeError:
                    continue
                logger.info("Agent accepted %s(%s)", method, ", ".join(payload))
                return result
            raise TypeError(
                f"{type(agent).__name__}.{method}() accepted neither a string nor "
                f"{{'input': ...}} nor {{'messages': ...}}: {string_error}"
            ) from string_error

    raise TypeError(
        f"{type(agent).__name__} is not callable and has no run() or invoke() method."
    )


def _content_of(message: Any) -> str:
    """Text of one chat message, dict-shaped or object-shaped."""
    from app.agent.llm import text_of

    if isinstance(message, dict):
        return text_of(message.get("content", ""))
    return text_of(message)


def _as_mapping(result: Any) -> Optional[Dict[str, Any]]:
    """
    A dict view of a result object, for agents that return neither.

    Frameworks routinely return a dataclass or a model instance. Falling through
    to str() on one yields "AgentResult(final_output='...')" -- a string that
    looks like an answer, scores like an answer, and is not one.
    """
    if is_dataclass(result) and not isinstance(result, type):
        return asdict(result)

    known = RESPONSE_KEYS + TOOL_CALL_KEYS + ("messages",)
    found = {k: getattr(result, k) for k in known if hasattr(result, k)}
    return found or None


def extract_response(result: Any) -> str:
    """The agent's answer, whatever container it came back in."""
    from app.agent.llm import text_of

    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        for key in RESPONSE_KEYS:
            value = result.get(key)
            if value is None:
                continue
            return value if isinstance(value, str) else text_of(value)
        # LangGraph and LCEL return the conversation, not an answer field.
        messages = result.get("messages")
        if isinstance(messages, (list, tuple)) and messages:
            return _content_of(messages[-1])
        return str(result)

    if getattr(result, "content", None) is not None:
        return text_of(result)

    mapping = _as_mapping(result)
    if mapping is not None:
        return extract_response(mapping)

    return str(result)


def extract_tool_calls(result: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Tool calls reported by the agent, normalised, or None when it reported none.

    None and [] are different and the distinction matters: None means this
    adapter could not observe tool use, [] means the agent stated it used no
    tools. Collapsing them would let a task assert "tool X was called" against an
    agent whose tool use is simply invisible here.
    """
    if not isinstance(result, dict):
        mapping = _as_mapping(result)
        if mapping is None:
            return None
        result = mapping

    raw = next((result[k] for k in TOOL_CALL_KEYS if result.get(k) is not None), None)
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        return None

    calls = []
    for entry in raw:
        if isinstance(entry, str):
            calls.append({"tool_name": entry, "args": {}, "result": ""})
            continue
        if isinstance(entry, (list, tuple)) and entry:
            # LangChain intermediate_steps: (AgentAction, observation)
            action = entry[0]
            calls.append({
                "tool_name": str(getattr(action, "tool", "unknown")),
                "args": getattr(action, "tool_input", {}) or {},
                "result": str(entry[1]) if len(entry) > 1 else "",
            })
            continue
        if isinstance(entry, dict):
            name = next((entry[k] for k in NAME_KEYS if entry.get(k)), "unknown")
            if isinstance(name, dict):
                name = name.get("name", "unknown")
            args = next((entry[k] for k in ARG_KEYS if entry.get(k) is not None), {})
            calls.append({
                "tool_name": str(name),
                "args": args if isinstance(args, dict) else {"input": args},
                "result": str(entry.get("result") or entry.get("output") or ""),
            })
    return calls


@AdapterRegistry.register("external")
class ExternalAgentAdapter(BaseAgentAdapter):
    """Wraps a third-party agent so the suite can evaluate it unchanged."""

    def __init__(self, agent_path: Optional[str] = None):
        self.agent_path = agent_path or os.getenv("AGENT_PATH", "")
        self.agent: Any = None
        self.tool_calls: Optional[List[Dict[str, Any]]] = None
        self.response = ""
        self.metrics: Dict[str, Any] = {}
        self.raw_result: Any = None

    def initialize(self, config: Dict[str, Any]) -> None:
        self.agent_path = config.get("agent_path") or self.agent_path
        if self.agent is None:
            self.agent = instantiate(resolve(self.agent_path))
            logger.info("External agent loaded from %s", self.agent_path)
        self.tool_calls = None
        self.response = ""
        self.metrics = {}

    def run(self, task: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        start = time.time()
        # Deliberately not wrapped: a third-party agent that raises has failed,
        # and the runner records that as a task failure with the error attached.
        self.raw_result = call_agent(self.agent, task)

        self.response = extract_response(self.raw_result)
        self.tool_calls = extract_tool_calls(self.raw_result)
        self.metrics = {
            "execution_time_seconds": round(time.time() - start, 3),
            "tool_calls_count": len(self.tool_calls) if self.tool_calls else 0,
        }
        plan = self.raw_result.get("plan", []) if isinstance(self.raw_result, dict) else []
        return {"response": self.response, "plan": plan, "intent": {}}

    def stream(self, task: str) -> Generator[Dict[str, Any], None, None]:
        yield self.run(task)

    def get_trace(self) -> List[Dict[str, Any]]:
        return [{"type": "ExternalAgent", "content": self.response}] if self.response else []

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        return self.tool_calls or []

    def get_execution_graph(self) -> Dict[str, Any]:
        return {"nodes": [{"id": "external", "name": self.agent_path or "external agent"}],
                "edges": []}

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics

    def cleanup(self) -> None:
        self.tool_calls = None
        self.response = ""
        self.raw_result = None

    def capabilities(self):
        """
        Unknown until proven otherwise.

        A foreign agent's tool surface cannot be introspected, so every task is
        attempted rather than skipped. Under-running a supplied agent would
        produce a fast, flattering, meaningless result.
        """
        from app.benchmarks.selection import AgentCapabilities

        observed = {c["tool_name"] for c in (self.tool_calls or [])}
        return AgentCapabilities(tools=observed or None, plans=True, retrieves=True)

    @property
    def reports_tool_calls(self) -> bool:
        """False when the agent gave no tool information at all."""
        return self.tool_calls is not None
