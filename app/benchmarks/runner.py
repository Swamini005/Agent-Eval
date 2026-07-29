import json
import time
import os
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.benchmarks.models import UnifiedBenchmarkTask
from app.adapters.base import BaseAgentAdapter
from app.faults.middleware import FaultInjectionMiddleware
from app.telemetry import telemetry_tracker

class BenchmarkRunner:
    """
    Harbor-compatible Benchmark Runner.
    Runs normalized benchmark tasks in parallel against an Agent Adapter
    with fault injection, retry logic, and complete telemetry capture.
    """
    
    def __init__(
        self,
        agent_adapter: BaseAgentAdapter,
        concurrency: int = 4,
        max_retries: int = 2
    ):
        self.agent_adapter = agent_adapter
        self.concurrency = concurrency
        self.max_retries = max_retries

    def run_benchmark(self, tasks: List[UnifiedBenchmarkTask], output_dir: str = ".") -> Dict[str, Any]:
        """
        Runs tasks in parallel using a ThreadPoolExecutor.
        Writes all telemetry reports to execution.json.
        """
        results = []
        
        # Determine parallel workers
        workers = min(self.concurrency, len(tasks)) if tasks else 1
        
        print(f"Starting parallel execution of {len(tasks)} tasks using {workers} workers (retries={self.max_retries})")
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._execute_task_with_retries, task): task 
                for task in tasks
            }
            
            for future in as_completed(futures):
                task = futures[future]
                try:
                    task_result = future.result()
                    results.append(task_result)
                except Exception as e:
                    results.append({
                        "task_id": task.id,
                        "benchmark": task.benchmark,
                        "success": False,
                        "errors": [f"ThreadPool Execution crashed: {str(e)}"]
                    })
                    
        # Output reports
        output_payload = {
            "summary": {
                "total_tasks": len(tasks),
                "successful_runs": sum(1 for r in results if r.get("success", False)),
                "failed_runs": sum(1 for r in results if not r.get("success", False)),
            },
            "tasks": results
        }
        
        output_file = os.path.join(output_dir, "execution.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)
            
        print(f"Benchmark execution telemetry successfully saved to {output_file}")
        return output_payload

    def _execute_task_with_retries(self, task: UnifiedBenchmarkTask) -> Dict[str, Any]:
        """
        Executes a single task with retry limits.
        Captures complete prompt, response, tool, memory, and latency details.
        """
        attempt = 0
        errors = []
        start_time = 0.0
        duration = 0.0
        
        # Create a Langfuse trace for this task execution
        trace_payload = telemetry_tracker.create_trace(
            task_id=task.id,
            benchmark_name=task.benchmark,
            category=task.category,
            difficulty=task.difficulty,
            prompt=task.prompt,
            metadata=task.metadata
        )
        trace_id = trace_payload["trace_id"]
        deep_link = trace_payload["deep_link"]
        
        session_id = f"thread_run_{task.id}"
        
        while attempt <= self.max_retries:
            try:
                # Re-initialize state before each retry
                self.agent_adapter.initialize({"session_id": session_id, "task_id": task.id})
                
                # Fetch Langfuse callback handler if enabled
                cb_handler = telemetry_tracker.get_callback_handler(trace_id)
                run_config = {"callbacks": [cb_handler]} if cb_handler else None
                
                start_time = time.time()
                
                # Run execution passing callbacks
                run_result = self.agent_adapter.run(task.prompt, config=run_config)
                duration = time.time() - start_time
                
                # Fetch details from adapter
                trace = self.agent_adapter.get_trace()
                tool_calls = self.agent_adapter.get_tool_calls()
                graph_schema = self.agent_adapter.get_execution_graph()
                metrics = self.agent_adapter.get_metrics()
                
                # Token estimation if metrics are absent
                prompt_tokens = metrics.get("prompt_tokens", len(task.prompt) // 4)
                completion_tokens = metrics.get("completion_tokens", len(run_result.get("response", "")) // 4)
                total_tokens = prompt_tokens + completion_tokens
                cost = round(prompt_tokens * 0.000005 + completion_tokens * 0.000015, 6)
                
                # Capture reasoning nodes from plan & observations
                reasoning_nodes = []
                for idx, step in enumerate(run_result.get("plan", [])):
                    reasoning_nodes.append({
                        "node_id": f"plan_step_{idx}",
                        "description": step,
                        "status": "completed" if idx < metrics.get("tool_calls_count", 0) else "planned"
                    })
                
                # Retrieval documents mock placeholder
                retrieval_documents = [
                    {
                        "source": f"doc_{i}",
                        "content": f"Information relevant to {task.category}"
                    } for i in range(2)
                ]
                
                # Evaluate tool coverage
                executed_tool_names = [t.get("tool_name") for t in tool_calls]
                tool_coverage = 0.0
                if task.expected_tools:
                    matched_tools = set(executed_tool_names).intersection(set(task.expected_tools))
                    tool_coverage = len(matched_tools) / len(task.expected_tools)
                else:
                    tool_coverage = 1.0
                
                # Detect injected faults from middleware
                injected_faults = []
                if isinstance(self.agent_adapter, FaultInjectionMiddleware):
                    injected_faults = [log.model_dump() for log in self.agent_adapter.engine.logs]
                
                # Log evaluation and reasoning spans to Langfuse
                telemetry_tracker.log_evaluation(
                    trace_id=trace_id,
                    score=tool_coverage,
                    tool_coverage=tool_coverage,
                    errors=errors,
                    injected_faults=injected_faults
                )
                
                telemetry_tracker.log_span(
                    trace_id=trace_id,
                    span_name="execute_agent",
                    span_input=task.prompt,
                    span_output=run_result.get("response", ""),
                    latency=duration
                )
                
                return {
                    "task_id": task.id,
                    "benchmark": task.benchmark,
                    "category": task.category,
                    "prompt": task.prompt,
                    "response": run_result.get("response", ""),
                    "tool_calls": tool_calls,
                    "execution_graph": graph_schema,
                    "latency_seconds": round(duration, 3),
                    "cost_usd": cost,
                    "tokens": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    },
                    "errors": [],
                    "memory_state": trace,
                    "retrieval_documents": retrieval_documents,
                    "reasoning_nodes": reasoning_nodes,
                    "success": True,
                    "attempts": attempt + 1,
                    "langfuse_trace_id": trace_id,
                    "langfuse_deep_link": deep_link
                }
            except Exception as e:
                errors.append(str(e))
                attempt += 1
                print(f"Execution failed for task {task.id} (attempt {attempt}/{self.max_retries + 1}): {str(e)}")
            finally:
                self.agent_adapter.cleanup()
                
        # If all attempts fail
        return {
            "task_id": task.id,
            "benchmark": task.benchmark,
            "category": task.category,
            "prompt": task.prompt,
            "response": "",
            "tool_calls": [],
            "execution_graph": {"nodes": [], "edges": []},
            "latency_seconds": round(time.time() - start_time, 3) if start_time > 0 else 0.0,
            "cost_usd": 0.0,
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "errors": errors,
            "memory_state": [],
            "retrieval_documents": [],
            "reasoning_nodes": [],
            "success": False,
            "attempts": attempt,
            "langfuse_trace_id": trace_id,
            "langfuse_deep_link": deep_link
        }
