import time
from typing import List, Dict, Any, Optional
from app.benchmarks.models import UnifiedBenchmarkTask
from app.benchmarks.filters import BenchmarkFilters
from app.adapters.base import BaseAgentAdapter

class BenchmarkDispatcher:
    """
    Dispatcher that controls loading, filtering, and executing benchmark tasks 
    against a unified Agent Adapter.
    """
    
    def __init__(self, agent_adapter: BaseAgentAdapter):
        self.agent_adapter = agent_adapter

    def dispatch(
        self,
        tasks: List[UnifiedBenchmarkTask],
        filter_benchmarks: Optional[List[str]] = None,
        shuffle_tasks: bool = False,
        shuffle_seed: Optional[int] = None,
        sample_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Dispatches filtered tasks to the agent adapter and calculates evaluation analytics.
        """
        processed_tasks = list(tasks)
        
        # 1. Apply filter by benchmarks
        if filter_benchmarks:
            processed_tasks = BenchmarkFilters.filter_by_benchmarks(processed_tasks, filter_benchmarks)
            
        # 2. Apply shuffle
        if shuffle_tasks:
            processed_tasks = BenchmarkFilters.shuffle(processed_tasks, seed=shuffle_seed)
            
        # 3. Apply sample N tasks
        if sample_n is not None:
            processed_tasks = BenchmarkFilters.sample(processed_tasks, sample_n)

        # 4. Execute runs
        results = []
        successful_runs = 0
        total_execution_time = 0.0
        
        for task in processed_tasks:
            # Re-initialize adapter for each run to avoid state bleeding
            self.agent_adapter.initialize({"session_id": f"dispatch_{task.id}", "task_id": task.id})
            
            start_time = time.time()
            try:
                run_result = self.agent_adapter.run(task.prompt)
                duration = time.time() - start_time
                
                tool_calls = self.agent_adapter.get_tool_calls()
                metrics = self.agent_adapter.get_metrics()
                
                # Simple evaluation checking: Did it run expected tools?
                executed_tool_names = [t.get("tool_name") for t in tool_calls]
                tool_coverage = 0.0
                if task.expected_tools:
                    matched_tools = set(executed_tool_names).intersection(set(task.expected_tools))
                    tool_coverage = len(matched_tools) / len(task.expected_tools)
                else:
                    tool_coverage = 1.0 # No tools expected
                
                results.append({
                    "task_id": task.id,
                    "benchmark": task.benchmark,
                    "prompt": task.prompt,
                    "response": run_result.get("response", ""),
                    "plan": run_result.get("plan", []),
                    "tool_calls": tool_calls,
                    "metrics": metrics,
                    "tool_coverage": tool_coverage,
                    "success": True
                })
                successful_runs += 1
                total_execution_time += duration
            except Exception as e:
                results.append({
                    "task_id": task.id,
                    "benchmark": task.benchmark,
                    "success": False,
                    "error": str(e)
                })
            finally:
                self.agent_adapter.cleanup()
                
        # 5. Summarize evaluation metadata
        success_rate = (successful_runs / len(processed_tasks)) if processed_tasks else 0.0
        
        return {
            "summary": {
                "total_tasks_attempted": len(processed_tasks),
                "successful_executions": successful_runs,
                "success_rate": success_rate,
                "total_execution_time_seconds": round(total_execution_time, 3),
                "average_execution_time_seconds": round(total_execution_time / successful_runs, 3) if successful_runs else 0.0
            },
            "results": results
        }
