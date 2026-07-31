"""
Decide which agents a commit should be evaluated against.

    python -m scripts.select_targets [--base=origin/main] [--default=langgraph]

Prints one adapter name per line, for CI to loop over.

Committing an adapter under app/adapters/ should evaluate *that* adapter, which
is the whole point of the workflow: push an agent on a branch and see the suite
run against it. Without this, CI always evaluates the default target and a newly
committed agent is never measured -- indistinguishable, in the report, from an
agent with no problems.

When nothing agent-specific changed, the default target still runs so every
commit produces a comparable baseline result.
"""

import argparse
import subprocess
import sys

import app.adapters  # noqa: F401  (auto-discovers every adapter)
from app.adapters.registry import AdapterRegistry

# Modules in app/adapters/ that define the contract, not an agent.
INFRASTRUCTURE = {"base", "registry", "factory", "__init__", "external"}


def changed_paths(base: str) -> list:
    for args in (["diff", "--name-only", f"{base}...HEAD"],
                 ["diff", "--name-only", base]):
        try:
            out = subprocess.check_output(["git", *args], text=True,
                                          stderr=subprocess.DEVNULL, timeout=30)
        except Exception:
            continue
        paths = [p.strip().replace("\\", "/") for p in out.splitlines() if p.strip()]
        if paths:
            return paths
    return []


def targets_from_changes(paths: list, default: str) -> list:
    """
    Adapter names implied by the changed files.

    Only adapters that actually registered are returned: a file that failed to
    import is absent from the registry, and evaluating a name that cannot be
    constructed would fail the job for a reason unrelated to agent quality.
    """
    registered = set(AdapterRegistry.available())
    selected = []

    for path in paths:
        if not path.startswith("app/adapters/") or not path.endswith(".py"):
            continue
        name = path.rsplit("/", 1)[-1][:-3]
        if name in INFRASTRUCTURE:
            # A change to the contract affects every agent, so fall back to
            # evaluating all of them rather than guessing.
            #
            # `or [default]` because an adapter that fails to import is absent
            # from the registry, so a broken dependency can empty this set. CI
            # loops over the printed list, and an empty list runs the body zero
            # times -- evaluating nothing while every step reports success.
            return sorted(registered - {"external"}) or [default]
        if name in registered and name not in selected:
            selected.append(name)

    return selected or [default]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--default", default="langgraph")
    args = parser.parse_args()

    paths = changed_paths(args.base)
    targets = targets_from_changes(paths, args.default)

    # Anything that is not a target name goes to stderr, so the caller can read
    # stdout directly as the list.
    print(f"changed files: {len(paths)}", file=sys.stderr)
    print(f"targets: {targets}", file=sys.stderr)

    for target in targets:
        print(target)


if __name__ == "__main__":
    main()
