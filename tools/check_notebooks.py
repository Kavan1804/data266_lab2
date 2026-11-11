"""Syntax checks every notebook cell, and checks the first and last cells."""
import ast
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
bad = 0
for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    nb = nbformat.read(path, as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    for i, cell in enumerate(code):
        source = re.sub(r"^\s*[%!].*$", "", cell.source, flags=re.M)
        try:
            ast.parse(source)
        except SyntaxError as err:
            bad += 1
            print(path.name, "cell", i, "syntax error:", err)
    first_ok = nb.cells[0].cell_type == "code" and "rev-parse" in nb.cells[0].source and "SEED = 266" in nb.cells[0].source
    last_ok = nb.cells[-1].cell_type == "code" and "ZipFile" in nb.cells[-1].source
    executed = sum(1 for c in code if c.get("outputs"))
    print(f"{path.name}: {len(code)} code cells | first cell ok: {first_ok} | last cell ok: {last_ok} | cells with outputs: {executed}")
    bad += (not first_ok) + (not last_ok)
sys.exit(1 if bad else 0)
