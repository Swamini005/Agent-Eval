import os
import yaml
import pytest

def test_github_actions_workflow_yaml():
    yaml_path = ".github/workflows/ci.yml"
    assert os.path.exists(yaml_path)
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
        
    assert content is not None
    assert "name" in content
    on_key = True if True in content else "on"
    assert on_key in content
    
    triggers = content[on_key]
    assert "push" in triggers
    assert "pull_request" in triggers
    assert "schedule" in triggers
    assert "workflow_dispatch" in triggers
    
    assert "jobs" in content
    assert "evaluate-agent" in content["jobs"]
    assert content["jobs"]["evaluate-agent"]["runs-on"] == "ubuntu-latest"
