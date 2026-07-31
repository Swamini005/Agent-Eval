from typing import List, Dict, Any
from app.benchmarks.base import BaseBenchmarkProvider
from app.benchmarks.registry import BenchmarkRegistry
from app.benchmarks.models import UnifiedBenchmarkTask

@BenchmarkRegistry.register("harbor")
class HarborIndexProvider(BaseBenchmarkProvider):
    """
    Normalizes Harbor Index dataset format into UnifiedBenchmarkTask.
    Expects input fields like task_id, task_prompt, target_response, etc.
    """
    
    def load_tasks(self, raw_data: List[Dict[str, Any]]) -> List[UnifiedBenchmarkTask]:
        tasks = []
        for index, item in enumerate(raw_data):
            task_id = str(item.get("task_id", f"harbor_{index}"))
            tasks.append(UnifiedBenchmarkTask(
                id=task_id,
                benchmark="harbor",
                category=item.get("category", "general"),
                domain=item.get("domain", "web"),
                difficulty=item.get("difficulty_level", "medium"),
                prompt=item.get("task_prompt", item.get("prompt", "")),
                expected_answer=item.get("target_response", item.get("expected_answer")),
                expected_tools=item.get("required_tools", []),
                ground_truth=item.get("validation_rules", {}),
                metadata={
                    "original_item": item,
                    "evaluation_strategy": item.get("eval_strategy", "exact_match")
                }
            ))
        return tasks
