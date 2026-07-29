import time
import random
from typing import Dict, Any, List, Generator, Optional
from unittest.mock import patch
from app.adapters.base import BaseAgentAdapter
from app.faults.engine import FaultInjectionEngine
from app.faults.registry import FaultRegistry
from app.faults.models import FaultConfig

class FaultInjectionMiddleware(BaseAgentAdapter):
    """
    Middleware that wraps any BaseAgentAdapter and dynamically injects configured 
    failures (Reasoning, Tool, Memory, Retrieval, Prompt, Model, Performance, Config).
    Uses dynamic patching to remain independent of any concrete agent framework.
    """
    
    def __init__(self, target_adapter: BaseAgentAdapter, engine: FaultInjectionEngine):
        self.target = target_adapter
        self.engine = engine

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Intercepts initialization configuration parameters.
        """
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

        # Intercept tool executions during run
        tool_patches = self._setup_tool_patches()
        
        # Activate all mock patches
        for p in tool_patches:
            p.start()
            
        try:
            result = self.target.run(mutated_task, config)
            
            # Check for reasoning bypass / shortcut injection in returned state/plan
            result = self._apply_reasoning_faults(result)
            return result
        finally:
            for p in tool_patches:
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
            
        tool_patches = self._setup_tool_patches()
        for p in tool_patches:
            p.start()
            
        try:
            for chunk in self.target.stream(mutated_task):
                yield chunk
        finally:
            for p in tool_patches:
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
        return self.target.get_metrics()

    def cleanup(self) -> None:
        self.target.cleanup()

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
        Prepares standard mock service overrides (patching app.services.mocks methods)
        to trigger simulated Tool execution failures.
        """
        patches = []
        
        # 1. Tool Latency
        fault = self.engine.trigger_fault("tool_latency", "tool")
        if fault:
            delay = fault.parameters.get("delay_seconds", 0.5)
            def delay_decorator(orig_fn):
                def wrapped(*args, **kwargs):
                    time.sleep(delay)
                    return orig_fn(*args, **kwargs)
                return wrapped
            
            # Patch weather service as a demonstration
            patches.append(patch("app.services.mocks.mock_weather_forecast", side_effect=delay_decorator(
                lambda *args, **kwargs: {"city": "Paris", "date": "2026-08-01", "temperature_celsius": 18, "condition": "Overcast", "humidity_percentage": 60, "wind_speed_kph": 5}
            )))
            self.engine.log_trigger(fault, f"Inject tool latency of {delay}s", "Weather mock service patched with sleep delay")

        # 2. Random Tool Failure
        fault = self.engine.trigger_fault("random_tool_failure", "tool")
        if fault:
            patches.append(patch("app.services.mocks.mock_flight_search", side_effect=lambda *args, **kwargs: []))
            self.engine.log_trigger(fault, "Force empty flight returns", "Flight mock service patched to return empty lists")
            
        # 3. Malformed Tool Output
        fault = self.engine.trigger_fault("malformed_tool_output", "tool")
        if fault:
            patches.append(patch(
                "app.services.mocks.mock_restaurant_search", 
                side_effect=lambda *args, **kwargs: [{"name": "Error Response", "cuisine": "XML", "rating": 0.0, "average_price_usd": 0.0, "popular_dish": "<xml>error</xml>"}]
            ))
            self.engine.log_trigger(fault, "Force malformed XML dishes", "Restaurant mock service patched to return XML tags")

        # 4. Context Corruption
        fault = self.engine.trigger_fault("context_corruption", "tool")
        if fault:
            patches.append(patch("app.services.mocks.mock_policy_document", side_effect=lambda *args, **kwargs: "[CORRUPTED POLICY CONTENT]"))
            self.engine.log_trigger(fault, "Corrupt policy context", "Policy mock service patched to return corrupted placeholder")
            
        return patches
