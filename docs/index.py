from pathlib import Path

docs_dir = Path(__file__).parent
readme = docs_dir.parent / "README.md"

with open(readme, "r") as f:
    readme_src = f.read()

index_src = f"""
---
pagetitle: "chatlas"
---

{readme_src}
"""

# On Github, asset links are relative to the root directory,
# but on the Quarto site, they are relative to the docs directory.
# So, we need to adjust the asset links in the README.
index_src = index_src.replace("docs/images/", "images/")

index = docs_dir / "index.qmd"

with open(index, "w") as f:
    f.write(index_src)
