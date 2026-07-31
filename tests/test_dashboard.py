import pytest
import pandas as pd
from dashboard import get_git_info, generate_pdf_report

def test_dashboard_git_info():
    git_info = get_git_info()
    assert "branch" in git_info
    assert "commit" in git_info
    assert "ci_status" in git_info

def test_pdf_generation_report():
    # Make a dummy DataFrame
    df = pd.DataFrame([
        {"task_id": "t1", "benchmark": "harbor", "prompt": "Search flight JFK to LHR", "overall_score": 0.85}
    ])
    summary = {
        "global_average_score": 0.85,
        "total_tasks_evaluated": 1
    }
    
    pdf_bytes = generate_pdf_report(df, summary)
    assert len(pdf_bytes) > 0
    # PDF files start with standard %PDF magic bytes
    assert pdf_bytes.startswith(b"%PDF")
