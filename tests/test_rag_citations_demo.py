import os
import subprocess
import sys
from pathlib import Path


def test_rag_citations_demo_requires_openai_api_key():
    root = Path(__file__).parents[1]
    env = os.environ | {"OPENAI_API_KEY": ""}
    result = subprocess.run(
        [sys.executable, root / "scripts" / "rag_citations_demo.py"],
        cwd=root,
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 1
    assert "OPENAI_API_KEY" in result.stderr
