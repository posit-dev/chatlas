import subprocess
from pathlib import Path

scripts = [
    "_generate_anthropic_types.py",
    "_generate_google_types.py",
    "_generate_openai_types.py",
]

absolute_scripts = [Path(__file__).parent / script for script in scripts]

for script in absolute_scripts:
    subprocess.run(["python", script], check=False)
