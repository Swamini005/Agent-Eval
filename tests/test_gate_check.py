import subprocess
import sys
import os

def test_gate_check_baseline_success():
    # Run the demo pipeline to generate current baseline results
    res_pipeline = subprocess.run(
        [sys.executable, "demo_pipeline.py", "--mode=ci"],
        capture_output=True,
        text=True
    )
    assert res_pipeline.returncode == 0
    
    # Run the gate check, it should pass (returncode 0)
    res_gate = subprocess.run(
        [sys.executable, "app/evaluation/gate_check.py"],
        capture_output=True,
        text=True
    )
    assert res_gate.returncode == 0
    assert "CI GATE PASSED" in res_gate.stdout
