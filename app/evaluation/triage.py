"""
Triage: choose which cases a commit should be evaluated against.

Running all thirty tasks on every commit is slow and mostly wasted. Triage picks
the subset that matters, in two layers:

    rules      deterministic, from the diff and the agent's capabilities.
               This is the *floor* -- it can never be reduced.
    advisor    an LLM reading the diff and the project's failure history.
               It may only ADD to the floor. It cannot remove anything.

That asymmetry is the whole design. An LLM that could *shrink* the set would be
able to produce a fast green build that proves nothing -- the exact failure this
project exists to catch -- and its reasoning is not reproducible, so a reviewer
could never tell it had happened. Restricted to adding, the worst it can do is
waste time, and the floor keeps the gate defensible.

Everything the advisor returns is validated against the real task and regression
registries. A hallucinated id is dropped and recorded, never silently ignored.
If the advisor errors, times out, or is not configured, triage returns the rules
floor and says so.

The advisor is not deterministic. It is called at temperature 0, but that is not
a guarantee any provider makes: three runs of one unchanged diff against Gemini
returned three different proposals. The floor was identical every time, so the
gate stays defensible -- but a re-run of the same commit would otherwise show a
different test set, which is not something you can explain to a reviewer. Passing
`cache_dir` pins the decision: the proposal is stored against the inputs that
define it, so re-running a commit replays it instead of asking again.
"""

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, Set

from app.benchmarks.models import UnifiedBenchmarkTask

logger = logging.getLogger(__name__)

# Advisor output is capped so a confused model cannot expand a targeted run back
# into the full suite and quietly undo the point of triage.
# Deliberately tight. Given a loose brief the model suggests everything plausibly
# related -- a trial run added the full 10 and turned a 7-task selection into 17,
# which is barely different from running all 30. The advisor earns its place by
# catching the one or two non-obvious cases, not by padding.
MAX_SUGGESTED_TASKS = 3
MAX_SUGGESTED_REGRESSIONS = 2

PROMPT = """You are triaging an automated evaluation suite for an AI agent.

A deterministic rule set has ALREADY selected every task that directly exercises
the changed code. Those will run regardless. Your job is narrow: name at most
{max_tasks} additional tasks that are genuinely at risk and that the rules would
MISS.

Rules for your answer:
- Suggesting NOTHING is the correct answer most of the time. Prefer it.
- Do not suggest a task merely because it is in the same category, or because the
  change "could affect" it. The rules already cover direct exposure.
- Only suggest a task if you can name the SPECIFIC mechanism by which this change
  breaks it, and explain why the already-selected tasks would not catch it.
- Every suggestion costs time and money. A vague reason means do not suggest it.
- If several tasks share one reason, that reason is too vague. Suggest none of them.

CHANGED FILES:
{changed_files}

TASKS ALREADY SELECTED BY RULES:
{mandatory}

TASKS AVAILABLE TO ADD (id: category -- prompt):
{catalogue}

REGRESSIONS AVAILABLE TO ADD:
{regressions}

HISTORICAL FAILURE COUNTS BY CATEGORY:
{history}

Reply with JSON only, no prose, in exactly this shape:
{{"tasks": [{{"id": "...", "reason": "..."}}],
  "regressions": [{{"label": "...", "reason": "..."}}]}}
"""


class TriageDecision:
    """What to run, and why each item is in the set."""

    def __init__(self):
        self.mandatory: Set[str] = set()
        self.suggested: Set[str] = set()
        self.regressions: Set[str] = set()
        self.provenance: Dict[str, str] = {}
        self.reasons: Dict[str, str] = {}
        self.advisor_status = "not run"
        self.rejected: List[str] = []
        self.rule_reason = ""
        self.replayed = False

    @property
    def task_ids(self) -> Set[str]:
        return self.mandatory | self.suggested

    def summary(self) -> Dict[str, Any]:
        return {
            "task_ids": sorted(self.task_ids),
            "mandatory": sorted(self.mandatory),
            "suggested": sorted(self.suggested),
            "regressions": sorted(self.regressions),
            "provenance": self.provenance,
            "reasons": self.reasons,
            "advisor_status": self.advisor_status,
            "rejected_suggestions": self.rejected,
            "rule_reason": self.rule_reason,
            "advisor_replayed": self.replayed,
        }

    def describe(self, total: int) -> str:
        return (f"{len(self.task_ids)} of {total} tasks "
                f"({len(self.mandatory)} required by rules, "
                f"{len(self.suggested)} added by advisor); "
                f"advisor {self.advisor_status}")


def _catalogue(tasks: List[UnifiedBenchmarkTask], exclude: Set[str]) -> str:
    lines = [
        f"{t.id}: {t.category} -- {t.prompt[:70]}"
        for t in tasks if t.id not in exclude
    ]
    return "\n".join(lines) or "(none)"


def ask_advisor(
    tasks: List[UnifiedBenchmarkTask],
    mandatory: Set[str],
    changed_files: List[str],
    regressions: List[str],
    history: Dict[str, int],
) -> Dict[str, Any]:
    """
    Ask the model which additional cases are at risk.

    Raises on any failure. The caller treats that as "advisor unavailable" and
    proceeds with the rules floor, so a model outage can never block a build or
    silently narrow what runs.
    """
    from app.agent.llm import get_llm, text_of
    from langchain_core.messages import HumanMessage

    prompt = PROMPT.format(
        max_tasks=MAX_SUGGESTED_TASKS,
        changed_files="\n".join(changed_files) or "(unknown)",
        mandatory=", ".join(sorted(mandatory)) or "(none)",
        catalogue=_catalogue(tasks, mandatory),
        regressions=", ".join(regressions) or "(none)",
        history=json.dumps(history) if history else "(none recorded)",
    )

    # temperature=0: the same diff must select the same tests. At the agent's
    # configured temperature two runs of one commit proposed different
    # regressions, which would make a triaged run unreproducible.
    raw = text_of(get_llm(temperature=0.0).invoke([HumanMessage(content=prompt)])).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].removeprefix("json").strip()
    return json.loads(raw)


def _proposal_key(changed_files: List[str], mandatory: Set[str], suite_sha: str) -> str:
    """
    Identity of an advisor question.

    Provider and model are included for the same reason the experiment cache
    includes them: a proposal from one model must never be replayed as though
    another had produced it. The prompt and caps are hashed in too, so editing
    the brief invalidates decisions made under the old one.
    """
    from app.config import settings

    material = json.dumps({
        "changed_files": sorted(changed_files),
        "mandatory": sorted(mandatory),
        "suite_sha": suite_sha,
        "provider": settings.LLM_PROVIDER,
        "model": settings.MODEL_NAME,
        "prompt": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()[:16],
        "caps": [MAX_SUGGESTED_TASKS, MAX_SUGGESTED_REGRESSIONS],
    }, sort_keys=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def triage(
    tasks: List[UnifiedBenchmarkTask],
    mandatory_ids: Set[str],
    rule_reason: str = "",
    changed_files: Optional[List[str]] = None,
    available_regressions: Optional[List[str]] = None,
    failure_history: Optional[Dict[str, int]] = None,
    use_advisor: bool = True,
    cache_dir: Optional[str] = None,
    suite_sha: str = "",
) -> TriageDecision:
    """
    Decide the run set. The rules floor is always included.

    `mandatory_ids` comes from impact analysis and capability selection. The
    advisor sees what was already chosen and proposes additions only.

    `cache_dir` pins the advisor's answer so re-running a commit selects the same
    tests. What is stored is the raw proposal, not the finished decision, so
    validation against the live registries runs again on replay -- a task deleted
    since the entry was written is rejected rather than resurrected from cache.
    """
    known = {t.id for t in tasks}
    decision = TriageDecision()
    decision.mandatory = set(mandatory_ids) & known
    decision.rule_reason = rule_reason
    for task_id in decision.mandatory:
        decision.provenance[task_id] = "rule"

    if not use_advisor:
        decision.advisor_status = "disabled"
        return decision

    # Same rule the experiment cache follows: without a suite hash, an edited
    # task set would silently reuse a decision made against the old one.
    if cache_dir and not suite_sha:
        raise ValueError("suite_sha is required when cache_dir is set.")

    cache_path = None
    if cache_dir:
        key = _proposal_key(changed_files or [], decision.mandatory, suite_sha)
        cache_path = os.path.join(cache_dir, f"{key}.json")

    proposal = None
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                proposal = json.load(f)
            decision.replayed = True
        except (OSError, json.JSONDecodeError):
            proposal = None

    if proposal is None:
        try:
            proposal = ask_advisor(
                tasks, decision.mandatory, changed_files or [],
                available_regressions or [], failure_history or {},
            )
        except Exception as e:
            # Includes DeterministicMode when no model is configured, which is the
            # normal case in CI. The floor stands either way.
            decision.advisor_status = f"unavailable ({type(e).__name__})"
            logger.info("Triage advisor unavailable: %s", e)
            return decision

        if cache_path:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(proposal, f, indent=2)

    valid_regressions = set(available_regressions or [])

    for entry in (proposal.get("tasks") or [])[:MAX_SUGGESTED_TASKS]:
        task_id = (entry or {}).get("id", "")
        if task_id in known:
            if task_id not in decision.mandatory:
                decision.suggested.add(task_id)
                decision.provenance[task_id] = "advisor"
                decision.reasons[task_id] = (entry.get("reason") or "")[:200]
        elif task_id:
            # Recorded rather than dropped quietly: a model inventing task ids is
            # a signal about the advisor, and hiding it would waste the signal.
            decision.rejected.append(f"task {task_id!r} does not exist")

    for entry in (proposal.get("regressions") or [])[:MAX_SUGGESTED_REGRESSIONS]:
        label = (entry or {}).get("label", "")
        if label in valid_regressions:
            decision.regressions.add(label)
            decision.reasons[label] = (entry.get("reason") or "")[:200]
        elif label:
            decision.rejected.append(f"regression {label!r} does not exist")

    decision.advisor_status = "replayed" if decision.replayed else "ok"
    return decision


def selected_tasks(tasks: List[UnifiedBenchmarkTask],
                   decision: TriageDecision) -> List[UnifiedBenchmarkTask]:
    """Tasks to run, in the suite's own order."""
    chosen = decision.task_ids
    return [t for t in tasks if t.id in chosen]
