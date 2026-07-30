"""
Shared test configuration.

Environment is set before any application or DeepEval import so the settings
below actually take effect -- both libraries read their configuration at import
time, and setting these inside a fixture would be too late.
"""

import os

# Force the agent's deterministic rule-based path. Without this the default
# provider is "google" (app/config.py) and every node attempts a network call
# that will fail anyway, making runs slow and their latency measurements
# meaningless. Tests previously required this to be exported by hand.
os.environ.setdefault("LLM_PROVIDER", "mock")

# DeepEval reports usage to a third party by default. An evaluation harness
# should not transmit anything about the runs it is measuring.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("ERROR_REPORTING", "NO")
