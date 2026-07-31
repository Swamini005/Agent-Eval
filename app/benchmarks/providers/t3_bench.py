from typing import List, Dict, Any
from app.benchmarks.base import BaseBenchmarkProvider
from app.benchmarks.registry import BenchmarkRegistry
from app.benchmarks.models import UnifiedBenchmarkTask

@BenchmarkRegistry.register("t3bench")
class T3BenchProvider(BaseBenchmarkProvider):
    """
    Normalizes T3Bench dataset format into UnifiedBenchmarkTask.
    Expects input fields like t3_id, instruction, scenario, etc.
    """
    
    def load_tasks(self, raw_data: List[Dict[str, Any]]) -> List[UnifiedBenchmarkTask]:
        tasks = []
        for index, item in enumerate(raw_data):
            task_id = str(item.get("t3_id", f"t3_{index}"))
            tasks.append(UnifiedBenchmarkTask(
                id=task_id,
                benchmark="t3bench",
                category=item.get("scenario", "general"),
                domain=item.get("application", "api"),
                difficulty=item.get("level", "medium"),
                prompt=item.get("instruction", item.get("prompt", "")),
                expected_answer=item.get("answer_keys", item.get("expected_answer")),
                expected_tools=item.get("tool_sequence", []),
                ground_truth=item.get("metrics_ground_truth", {}),
                metadata={
                    "original_item": item,
                    "agent_type": item.get("agent_type", "generic")
                }
            ))
        return tasks
