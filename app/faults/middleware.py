import threading
import time
from importlib import import_module
from typing import Dict, Any, List, Generator, Optional
from unittest.mock import patch
from app.adapters.base import BaseAgentAdapter
from app.faults.engine import FaultInjectionEngine
from app.faults.registry import FaultRegistry
from app.packs import PackRegistry

# Guards every window in which tool patches are installed. unittest.mock.patch
# rebinds module attributes, which are process-global and not thread-safe, so
# without this two concurrent tasks can restore each other's saved originals and
# leave a MagicMock permanently in place.
_TOOL_PATCH_LOCK = threading.RLock()

class FaultInjectionMiddleware(BaseAgentAdapter):
    """
    Middleware that wraps any BaseAgentAdapter and dynamically injects configured 
    failures (Reasoning, Tool, Memory, Retrieval, Prompt, Model, Performance, Config).
    Uses dynamic patching to remain independent of any concrete agent framework.
    """
    
    def __init__(
        self,
        target_adapter: BaseAgentAdapter,
        engine: FaultInjectionEngine,
        domain: str = ""
    ):
        """
        Args:
            target_adapter: The agent being faulted.
            engine: This task's private fault engine.
            domain: Selects the domain pack that declares where tool-layer faults
                attach. With no pack, tool faults are skipped rather than being
                pointed at tools that may not exist.
        """
        self.target = target_adapter
        self.engine = engine
        self.domain = domain
        # Set by run(): seconds spent inside the agent, excluding time queued on
        # the tool patch lock. None until the first run.
        self._agent_seconds: Optional[float] = None

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Intercepts initialization configuration parameters.

        The runner calls this once per attempt, so it is also the point at which
        per-attempt fault state is cleared. Without the reset a retried task
        reports its faults once per attempt.
        """
        self.engine.reset_run_state()
        self.engine.current_task_id = config.get("task_id")
        mutated_config = dict(config)
        
        # 1. Broken API / Configuration Fault
        fault = self.engine.trigger_fault("broken_api", "configuration")
        if fault:
            handler = FaultRegistry.get_handler("broken_api")
            mutated_config = handler(mutated_config, fault)
            self.engine.log_trigger(fault, "API endpoint corruption", "Initialization with broken/invalid credentials")
            
        # 2. Temperature Increase / Model Fault
        fault = self.engine.trigger_fault("temperature_increase", "model")
        if fault:
            handler = FaultRegistry.get_handler("temperature_increase")
            mutated_config = handler(mutated_config, fault)
            self.engine.log_trigger(fault, "Elevate model temperature to 2.0", "Passed mutated settings during init")

        # 3. Max Token Reduction / Model Fault
        fault = self.engine.trigger_fault("max_token_reduction", "model")
        if fault:
            handler = FaultRegistry.get_handler("max_token_reduction")
            mutated_config = handler(mutated_config, fault)
            self.engine.log_trigger(fault, "Limit model output to 5 tokens", "Passed token limits during init")
            
        self.target.initialize(mutated_config)

    def run(self, task: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Runs the wrapped agent, intercepting prompt inputs, tool calls, and outputs.
        """
        self.engine.increment_steps()
        mutated_task = task
        
        # 1. Prompt Corruption
        fault = self.engine.trigger_fault("prompt_corruption", "prompt")
        if fault:
            handler = FaultRegistry.get_handler("prompt_corruption")
            mutated_task = handler(mutated_task, fault)
            self.engine.log_trigger(fault, "Corrupt prompt spelling", f"Task text mutated to: {mutated_task}")

        # 2. Prompt Injection
        fault = self.engine.trigger_fault("prompt_injection", "prompt")
        if fault:
            handler = FaultRegistry.get_handler("prompt_injection")
            mutated_task = handler(mutated_task, fault)
            self.engine.log_trigger(fault, "Inject override prompt block", f"Task text mutated to: {mutated_task}")

        # 3. Instruction Deletion
        fault = self.engine.trigger_fault("instruction_deletion", "prompt")
        if fault:
            handler = FaultRegistry.get_handler("instruction_deletion")
            mutated_task = handler(mutated_task, fault)
            self.engine.log_trigger(fault, "Delete verb commands", f"Task text mutated to: {mutated_task}")
            
        # 4. Performance Latency
        fault = self.engine.trigger_fault("artificial_latency", "performance")
        if fault:
            delay = fault.parameters.get("delay_seconds", 1.0)
            time.sleep(delay)
            self.engine.log_trigger(fault, f"Introduce {delay}s latency", f"Delayed task execution by {delay}s")

        # Tool faults replace module-level attributes, which every thread in the
        # runner's pool shares. Serialise the patched window so concurrent tasks
        # cannot interleave start/stop and leave a mock installed after the task
        # that owned it has finished -- that corrupts unrelated tasks, and even
        # later experiment arms in the same process.
        tool_patches = self._setup_tool_patches()

        # Nothing is patched, so nothing is shared, so nothing needs serialising.
        # The lock used to be taken unconditionally and held across the whole of
        # target.run(), so every task in the pool ran one at a time regardless of
        # the configured concurrency.
        if not tool_patches:
            started = time.perf_counter()
            result = self.target.run(mutated_task, config)
            self._agent_seconds = time.perf_counter() - started
            return self._apply_reasoning_faults(result)

        with _TOOL_PATCH_LOCK:
            for p in tool_patches:
                p.start()
            try:
                # Timed inside the lock, deliberately. The runner times run()
                # from the outside, which includes however long this task queued
                # behind other tasks' patch windows -- so p95_latency was
                # reporting lock contention as agent latency, and grew with the
                # configured concurrency rather than shrinking. Measured here,
                # the number describes the agent, which is what the gate's
                # threshold is about.
                started = time.perf_counter()
                result = self.target.run(mutated_task, config)
                self._agent_seconds = time.perf_counter() - started

                # Check for reasoning bypass / shortcut injection in returned state/plan
                return self._apply_reasoning_faults(result)
            finally:
                for p in reversed(tool_patches):
                    p.stop()

    def stream(self, task: str) -> Generator[Dict[str, Any], None, None]:
        """
        Streams wrapped steps, intercepting yields.
        """
        self.engine.increment_steps()
        mutated_task = task
        
        # Apply prompt mutations
        fault = self.engine.trigger_fault("prompt_corruption", "prompt")
        if fault:
            handler = FaultRegistry.get_handler("prompt_corruption")
            mutated_task = handler(mutated_task, fault)
            self.engine.log_trigger(fault, "Prompt corruption", "Mutated stream prompt")
            
        with _TOOL_PATCH_LOCK:
            tool_patches = self._setup_tool_patches()
            for p in tool_patches:
                p.start()
            try:
                for chunk in self.target.stream(mutated_task):
                    yield chunk
            finally:
                for p in reversed(tool_patches):
                    p.stop()

    def get_trace(self) -> List[Dict[str, Any]]:
        trace = self.target.get_trace()
        
        # Memory/Context Truncation Fault
        fault = self.engine.trigger_fault("context_truncation", "memory")
        if fault:
            trace = trace[-1:] if trace else []
            self.engine.log_trigger(fault, "Limit context size to last message", "Truncated trace data")
            
        # Conversation Reset Fault
        fault = self.engine.trigger_fault("conversation_reset", "memory")
        if fault:
            trace = []
            self.engine.log_trigger(fault, "Reset state history", "Cleared trace history")
            
        return trace

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        return self.target.get_tool_calls()

    def get_execution_graph(self) -> Dict[str, Any]:
        return self.target.get_execution_graph()

    def get_metrics(self) -> Dict[str, Any]:
        metrics = dict(self.target.get_metrics())
        # Time actually spent running the agent, excluding any wait for the tool
        # patch lock. The runner prefers this over its own wall-clock timing so
        # latency describes the agent rather than the harness's queueing.
        if self._agent_seconds is not None:
            metrics["agent_seconds"] = self._agent_seconds
        return metrics

    def cleanup(self) -> None:
        self.target.cleanup()

    def get_injected_faults(self) -> List[Dict[str, Any]]:
        """Faults this middleware injected, plus any injected further down the chain."""
        return [log.model_dump() for log in self.engine.logs] + self.target.get_injected_faults()

    def get_retrieval_documents(self) -> List[Dict[str, Any]]:
        """
        Delegates to the wrapped agent. Retrieval faults act on the tool layer, so
        the documents reported here are the ones the agent genuinely received --
        corrupted content included.
        """
        return self.target.get_retrieval_documents()

    # --- Internals ---
    
    def _apply_reasoning_faults(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Mutates plan/reasoning parameters if scheduled."""
        mutated = dict(result)
        
        # Planner Bypass
        fault = self.engine.trigger_fault("planner_bypass", "reasoning")
        if fault:
            handler = FaultRegistry.get_handler("planner_bypass")
            mutated["plan"] = handler(mutated.get("plan", []), fault)
            self.engine.log_trigger(fault, "Clear execution planner", "Plan wiped from response payload")
            
        # Planner Shortcut
        fault = self.engine.trigger_fault("planner_shortcut", "reasoning")
        if fault:
            handler = FaultRegistry.get_handler("planner_shortcut")
            mutated["plan"] = handler(mutated.get("plan", []), fault)
            self.engine.log_trigger(fault, "Truncate plan sequence", "Plan sequence limited to 1 item")

        # Planner Bypass Confirmation
        fault = self.engine.trigger_fault("planner_bypass_confirmation", "reasoning")
        if fault:
            handler = FaultRegistry.get_handler("planner_bypass_confirmation")
            mutated["plan"] = handler(mutated.get("plan", []), fault)
            self.engine.log_trigger(fault, "Forced booking without confirmation", "Plan forced to execute booking tools")
            
        return mutated

    def _setup_tool_patches(self) -> List[Any]:
        """
        Build the tool-layer patches for whichever faults fire this run.

        Patch targets come from the domain pack rather than being hardcoded, so
        the middleware carries no knowledge of any particular agent's tools. A
        fault type with no target declared for this domain simply does not fire.
        """
        pack = PackRegistry.get(self.domain)
        if pack is None:
            return []

        patches = []
        for fault_type, target in pack.fault_targets.items():
            fault = self.engine.trigger_fault(fault_type, "tool")
            if not fault:
                continue

            if target.returns is None:
                # No replacement value declared: wrap the real function so the
                # fault changes timing without changing output. Replacing the
                # return value here would confound a latency fault with a
                # correctness fault and make the resulting score unattributable.
                delay = fault.parameters.get("delay_seconds", 0.5)
                patches.append(patch(target.target, side_effect=self._delayed(target.target, delay)))
                self.engine.log_trigger(
                    fault,
                    f"Delay {target.target} by {delay}s",
                    f"Wrapped {target.target} with a {delay}s sleep, output unchanged"
                )
            else:
                replacement = target.returns
                patches.append(patch(target.target, side_effect=lambda *a, _v=replacement, **k: _v))
                self.engine.log_trigger(
                    fault,
                    f"Replace {target.target} output",
                    f"Patched {target.target} to return {replacement!r}"
                )

        return patches

    @staticmethod
    def _delayed(import_path: str, delay: float):
        """Wrap the real function at `import_path` so it sleeps before returning."""
        module_path, _, attr = import_path.rpartition(".")
        original = getattr(import_module(module_path), attr)

        def wrapped(*args, **kwargs):
            time.sleep(delay)
            return original(*args, **kwargs)

        return wrapped
