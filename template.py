"""
template.py - Crée la structure du projet
"""
from pathlib import Path

dirs = ["src", "tests", "data/raw", "data/vector_db"]
for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)

for f in ["src/__init__.py", "tests/__init__.py"]:
    Path(f).touch()

print("✓ Structure créée")