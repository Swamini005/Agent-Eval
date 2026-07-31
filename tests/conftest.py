"""
Shared test configuration.

Environment is set before any application or DeepEval import so the settings
below actually take effect -- both libraries read their configuration at import
time, and setting these inside a fixture would be too late.
"""

import os

# Force the agent's deterministic rule-based path, so the suite needs no key, no
# network, and returns the same numbers every run.
#
# Assigned, not setdefault(). The deepeval pytest plugin loads any .env into
# os.environ before conftest runs, so setdefault() found LLM_PROVIDER already
# present and did nothing: adding a .env with a real provider silently turned the
# unit tests into live model calls. They went from 0.6s to 32s, spent tokens on
# every run, and failed outright when the key was absent or expired -- failures
# that point at the code and are nothing of the kind.
#
# Set EVAL_ALLOW_LIVE_TESTS=1 to opt a run into whatever .env configures.
if os.environ.get("EVAL_ALLOW_LIVE_TESTS") != "1":
    os.environ["LLM_PROVIDER"] = "mock"

# DeepEval reports usage to a third party by default. An evaluation harness
# should not transmit anything about the runs it is measuring.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("ERROR_REPORTING", "NO")
