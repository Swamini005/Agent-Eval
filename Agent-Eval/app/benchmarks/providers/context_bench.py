from typing import List, Dict, Any
from app.benchmarks.base import BaseBenchmarkProvider
from app.benchmarks.registry import BenchmarkRegistry
from app.benchmarks.models import UnifiedBenchmarkTask

@BenchmarkRegistry.register("contextbench")
class ContextBenchProvider(BaseBenchmarkProvider):
    """
    Normalizes ContextBench dataset format into UnifiedBenchmarkTask.
    Expects input fields like id, query, context_type, task_domain, etc.
    """
    
    def load_tasks(self, raw_data: List[Dict[str, Any]]) -> List[UnifiedBenchmarkTask]:
        tasks = []
        for index, item in enumerate(raw_data):
            task_id = str(item.get("id", f"cb_{index}"))
            tasks.append(UnifiedBenchmarkTask(
                id=task_id,
                benchmark="contextbench",
                category=item.get("context_type", "general"),
                domain=item.get("task_domain", "contextual"),
                difficulty=item.get("complexity", "medium"),
                prompt=item.get("query", item.get("prompt", "")),
                expected_answer=item.get("expected_output", item.get("expected_answer")),
                expected_tools=item.get("tools_to_call", []),
                ground_truth=item.get("ground_truth_data", {}),
                metadata={
                    "original_item": item,
                    "context_length": item.get("context_length_tokens", 0)
                }
            ))
        return tasks
