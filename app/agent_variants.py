"""
Agent variants as configuration, so an improvement is a diffable row.

An agent whose behaviour lives only in code cannot be ablated: two versions
differ by a commit, and attributing a score change to one specific change means
reading the diff and hoping. Declaring the behaviour that varies means each
improvement is one file, one row in a table, and one hash in a report.

Each variant is hashed (`agent_cfg_sha`) and stamped into its results, so a
reported gain is tied to the exact configuration that produced it.
"""

import hashlib
import json
import os
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

VARIANTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agents"
)


class AgentVariant(BaseModel):
    """One configuration of the agent under test."""

    name: str
    description: str = ""

    extra_tool_triggers: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Additional keyword triggers per tool, merged over the "
                    "adapter's defaults. Fixes tools the agent fails to reach.",
    )
    policy_guard: bool = Field(
        False,
        description="When a retrieval tool returns a policy document, summarise "
                    "rather than echo it verbatim. Quoting the document back "
                    "leaks the exact figures a corrupted-context task forbids.",
    )
    retry_on_tool_error: bool = Field(
        False,
        description="Retry a tool once when it raises, feeding the error back.",
    )
    max_tool_calls: Optional[int] = Field(
        None, description="Hard ceiling on tool calls per task."
    )

    @property
    def sha(self) -> str:
        """Content hash of the behavioural fields, excluding name/description."""
        payload = self.model_dump(exclude={"name", "description"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def available_variants() -> List[str]:
    if not os.path.isdir(VARIANTS_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(VARIANTS_DIR) if f.endswith(".yaml"))


def load_variant(name: str) -> AgentVariant:
    path = os.path.join(VARIANTS_DIR, f"{name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No agent variant '{name}' at {path}. Available: {available_variants()}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return AgentVariant(**yaml.safe_load(f))
