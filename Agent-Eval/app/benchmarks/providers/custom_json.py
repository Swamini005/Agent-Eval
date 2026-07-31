from typing import List, Dict, Any
from app.benchmarks.base import BaseBenchmarkProvider
from app.benchmarks.registry import BenchmarkRegistry
from app.benchmarks.models import UnifiedBenchmarkTask

@BenchmarkRegistry.register("custom_json")
class CustomJSONProvider(BaseBenchmarkProvider):
    """
    Parses general custom JSON datasets.
    Handles variations of common task formats gracefully.
    """
    
    def load_tasks(self, raw_data: List[Dict[str, Any]]) -> List[UnifiedBenchmarkTask]:
        tasks = []
        for index, item in enumerate(raw_data):
            task_id = str(
                item.get("id") or 
                item.get("task_id") or 
                item.get("uuid") or 
                f"custom_{index}"
            )
            
            prompt = str(
                item.get("prompt") or 
                item.get("query") or 
                item.get("instruction") or 
                item.get("input") or 
                ""
            )
            
            expected_answer = (
                item.get("expected_answer") or 
                item.get("expected_output") or 
                item.get("target_response") or 
                item.get("answer")
            )
            
            expected_tools = (
                item.get("expected_tools") or 
                item.get("required_tools") or 
                item.get("tools") or 
                []
            )
            
            tasks.append(UnifiedBenchmarkTask(
                id=task_id,
                benchmark=item.get("benchmark", "custom"),
                category=item.get("category", "general"),
                domain=item.get("domain", "custom"),
                difficulty=item.get("difficulty", "medium"),
                prompt=prompt,
                expected_answer=str(expected_answer) if expected_answer is not None else None,
                expected_tools=expected_tools if isinstance(expected_tools, list) else [expected_tools],
                ground_truth=item.get("ground_truth") or item.get("validation_rules") or {},
                metadata=item.get("metadata") or {"original_item": item}
            ))
        return tasks
