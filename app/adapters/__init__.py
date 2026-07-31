"""
Agent adapters, discovered automatically.

Every module in this package is imported on first use so its
``@AdapterRegistry.register`` decorator runs. Dropping a new adapter file into
this directory is therefore enough to make it evaluable -- no edit here, no edit
to the pipeline.

That matters for the intended workflow: commit an adapter on a branch, and CI
evaluates it against the same task suite as every other agent. With the previous
explicit import list, a newly committed adapter silently failed to register and
the suite reported nothing about it, which looks identical to an agent that has
no problems.
"""

import importlib
import logging
import pkgutil

from app.adapters.base import BaseAgentAdapter
from app.adapters.registry import AdapterRegistry
from app.adapters.factory import AgentFactory

logger = logging.getLogger(__name__)

# Modules that define the contract rather than an implementation.
_INFRASTRUCTURE = {"base", "registry", "factory"}


def discover() -> list:
    """
    Import every adapter module in this package and return the registered names.

    An adapter that fails to import is logged and skipped rather than taking the
    whole run down: one broken agent must not prevent the others being evaluated.
    The failure is visible in the log and the agent is absent from the registry,
    so nothing reports a score for it.
    """
    for module in pkgutil.iter_modules(__path__):
        if module.name.startswith("_") or module.name in _INFRASTRUCTURE:
            continue
        try:
            importlib.import_module(f"{__name__}.{module.name}")
        except Exception as e:
            logger.error("Adapter module %r failed to import: %s", module.name, e)
    return AdapterRegistry.available()


discover()

__all__ = [
    "BaseAgentAdapter",
    "AdapterRegistry",
    "AgentFactory",
    "discover",
]
