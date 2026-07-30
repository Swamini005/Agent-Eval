import uuid
import tempfile
import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage
from app.agent.graph import travel_agent_graph

# Telemetry, adapters, and benchmark imports
from app.adapters.factory import AgentFactory
from app.faults.models import FaultConfig
from app.faults.engine import FaultInjectionEngine
from app.faults.middleware import FaultInjectionMiddleware
from app.benchmarks.models import UnifiedBenchmarkTask
from app.benchmarks.runner import BenchmarkRunner
from app.evaluation.models import EvaluationTaskInput, EvaluationExecutionInput
from app.evaluation.engine import EvaluationEngine


app = FastAPI(
    title="AI Travel Assistant API",
    description="Production-grade travel planning assistant state machine endpoints",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to the assistant")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Conversation session ID for memory retention")

class ChatResponse(BaseModel):
    session_id: str
    response: str
    plan: List[str]
    intent: Dict[str, Any]
    reasoning: str

class EvalItem(BaseModel):
    id: Optional[str] = None
    input: str

class EvalResponseItem(BaseModel):
    id: Optional[str] = None
    input: str
    response: str
    plan: List[str]
    intent: Dict[str, Any]

class EvaluationRequest(BaseModel):
    items: List[EvalItem]

class EvaluationResponse(BaseModel):
    results: List[EvalResponseItem]

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Primary endpoint for chatting with the AI Travel Assistant.
    Maintains memory using session_id as thread_id.
    """
    config = {"configurable": {"thread_id": request.session_id}}
    
    try:
        # Run state machine
        state = await travel_agent_graph.ainvoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        )
        
        return ChatResponse(
            session_id=request.session_id,
            response=state.get("response", "Sorry, I encountered an issue processing your request."),
            plan=state.get("plan", []),
            intent=state.get("intent", {}),
            reasoning=state.get("reasoning", "")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")

@app.post("/api/evaluate", response_model=EvaluationResponse)
async def evaluate_endpoint(request: EvaluationRequest):
    """
    Evaluation pipeline integration endpoint.
    Runs multiple queries in isolated environments (unique session IDs)
    and returns predictions/intents/plans for easy comparison with ground truth.
    """
    results = []
    for item in request.items:
        # Unique session for each evaluation item to prevent memory bleed
        session_id = f"eval_{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            state = await travel_agent_graph.ainvoke(
                {"messages": [HumanMessage(content=item.input)]},
                config=config
            )
            results.append(EvalResponseItem(
                id=item.id,
                input=item.input,
                response=state.get("response", ""),
                plan=state.get("plan", []),
                intent=state.get("intent", {})
            ))
        except Exception as e:
            # Return partial results / failed flag rather than crashing the whole batch
            results.append(EvalResponseItem(
                id=item.id,
                input=item.input,
                response=f"Error executing graph: {str(e)}",
                plan=[],
                intent={}
            ))
            
    return EvaluationResponse(results=results)


class TaskPayload(BaseModel):
    id: str
    benchmark: str
    category: str
    domain: str
    difficulty: str
    prompt: str
    expected_answer: Optional[str] = None
    expected_tools: List[str] = Field(default_factory=list)
    ground_truth: Optional[Dict[str, Any]] = Field(
        None,
        description="Declarative machine-checkable assertions; see AssertionMetric."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

class FaultPayload(BaseModel):
    id: str
    type: str
    component: str
    severity: str = "warning"
    probability: float = 1.0
    scheduling: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)

class BenchmarkRunRequest(BaseModel):
    tasks: List[TaskPayload]
    faults: List[FaultPayload] = Field(default_factory=list)
    concurrency: int = 4
    max_retries: int = 1
    seed: int = Field(
        0,
        description="Seed for probabilistic fault injection. Fixed by default so a "
                    "given request yields the same injections on every run."
    )

class BenchmarkRunResponse(BaseModel):
    summary: Dict[str, Any]
    results: List[Dict[str, Any]]

@app.post("/api/benchmark/run", response_model=BenchmarkRunResponse)
def benchmark_run_endpoint(request: BenchmarkRunRequest):
    """
    Executes a complete evaluation benchmark run over HTTP.
    Accepts task lists, applies fault injection middleware, executes agent concurrently,
    runs the evaluation metrics suite, and returns comprehensive reports.
    """
    # 1. Configure Fault Injection
    fault_configs = [
        FaultConfig(
            id=f.id,
            type=f.type,
            component=f.component,
            severity=f.severity,
            probability=f.probability,
            scheduling=f.scheduling,
            parameters=f.parameters
        ) for f in request.faults
    ]
    fault_engine = FaultInjectionEngine(fault_configs, seed=request.seed)

    # Each task declares its own domain, so a single run may span several packs.
    domains_by_task = {t.id: t.domain for t in request.tasks}

    # 2. Fail fast if the agent cannot be constructed at all, while we are still
    #    on the request thread and can return a meaningful status code.
    try:
        AgentFactory.create_agent("langgraph")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create agent adapter: {e}")

    # 3. Build a private adapter and fault engine per task. Adapters and engines
    #    both carry per-run mutable state, so sharing either across the runner's
    #    thread pool would interleave telemetry and misattribute fault logs.
    #    This runs on worker threads, where HTTPException would not reach the
    #    client, so construction failures surface as per-task errors instead.
    def build_adapter(task_id: str):
        return FaultInjectionMiddleware(
            AgentFactory.create_agent("langgraph"),
            fault_engine.fork(task_id),
            domain=domains_by_task.get(task_id, "")
        )

    # 3. Normalize Tasks
    unified_tasks = [
        UnifiedBenchmarkTask(
            id=t.id,
            benchmark=t.benchmark,
            category=t.category,
            domain=t.domain,
            difficulty=t.difficulty,
            prompt=t.prompt,
            expected_answer=t.expected_answer,
            expected_tools=t.expected_tools,
            ground_truth=t.ground_truth,
            metadata=t.metadata
        ) for t in request.tasks
    ]
    
    # 4. Execute Benchmark Runner in parallel
    runner = BenchmarkRunner(build_adapter, concurrency=request.concurrency, max_retries=request.max_retries)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Run execution
        runner.run_benchmark(unified_tasks, output_dir=temp_dir)
        fault_engine.save_reports(workspace_path=temp_dir)
        
        # Load reports
        execution_file = os.path.join(temp_dir, "execution.json")
        fault_report_file = os.path.join(temp_dir, "fault_report.json")
        
        with open(execution_file, "r", encoding="utf-8") as f:
            execution_data = json.load(f)
        with open(fault_report_file, "r", encoding="utf-8") as f:
            fault_report = json.load(f)
            
        # 5. Evaluate Execution Outputs
        eval_tasks = [
            EvaluationTaskInput(
                task_id=t.id,
                benchmark=t.benchmark,
                category=t.category,
                domain=t.domain,
                prompt=t.prompt,
                expected_answer=t.expected_answer,
                expected_tools=t.expected_tools,
                ground_truth=t.ground_truth
            ) for t in unified_tasks
        ]
        
        eval_executions = [
            EvaluationExecutionInput(
                task_id=e["task_id"],
                category=e.get("category", "general"),
                response=e["response"],
                latency_seconds=e["latency_seconds"],
                cost_usd=e["cost_usd"],
                tool_calls=e["tool_calls"],
                tokens=e["tokens"],
                memory_state=e["memory_state"],
                retrieval_documents=e["retrieval_documents"],
                reasoning_nodes=e["reasoning_nodes"],
                errors=e["errors"]
            ) for e in execution_data["tasks"]
        ]
        
        eval_engine = EvaluationEngine()
        reports = eval_engine.evaluate_run(
            tasks=eval_tasks,
            executions=eval_executions,
            fault_report=fault_report,
            output_dir=temp_dir
        )
        
        # Format API Response
        return BenchmarkRunResponse(
            summary={
                "global_average_score": reports["summary"]["global_average_score"],
                "total_tasks_evaluated": reports["summary"]["total_tasks_evaluated"],
                "summary_metrics": reports["summary"]["summary_metrics"],
                "execution_summary": execution_data["summary"],
                "fault_summary": fault_report["summary"]
            },
            results=reports["results"]
        )

