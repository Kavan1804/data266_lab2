"""Repo checks before tagging. It looks in what git would commit."""
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True).stdout.split("\n")
files = [ROOT / f for f in files if f]
problems = []

questions = []
source = next((p for p in (ROOT / "DATA266_Lab2_Part1/Questions.csv", ROOT / "data/part1/Questions.csv") if p.exists()), None)
if source:
    with open(source, newline="", encoding="utf-8") as f:
        questions = [row["question"].strip() for row in csv.DictReader(f)]

secret = re.compile(r"(hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|KGAT_[A-Za-z0-9]{20,})")
question_id = re.compile(r"\bQ0\d\d\b")
for path in files:
    if not path.exists():
        continue
    if path.stat().st_size > 20e6:
        problems.append(f"large file: {path.name} ({path.stat().st_size / 1e6:.0f} MB)")
    if path.suffix.lower() in {".pdf", ".safetensors", ".ckpt", ".pt", ".bin", ".zip"}:
        problems.append(f"should not be committed: {path.name}")
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".zip", ".pdf"}:
        continue
    text = path.read_text(errors="ignore")
    if secret.search(text):
        problems.append(f"possible secret in {path.name}")
    if path.name != "check_repo.py" and question_id.search(text):
        problems.append(f"question ID in {path.name}: {question_id.search(text).group(0)}")
    for q in questions:
        if len(q) > 15 and q in text:
            problems.append(f"question text in {path.name}: {q[:50]}")

print(f"checked {len(files)} files, compared with {len(questions)} questions")
print("\n".join(problems) if problems else "no problems found")
sys.exit(1 if problems else 0)
