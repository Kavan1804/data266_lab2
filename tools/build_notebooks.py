"""Builds Kavan's notebooks from src/. Run this after changing any src file.

Every notebook is standalone: source files and test files are pasted into cells, so a notebook
runs on Colab, Kaggle or a local Jupyter with no repo around it. Deliverables are written in the
repo's folder layout and zipped by the last cell, so the ZIP unzips straight into the repo.
python tools/build_notebooks.py
"""
import json
import re
import subprocess
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
NB = ROOT / "notebooks"
EVAL = ROOT / "eval"
# the commit these notebooks were built from; cell 1 prints it when there is no git (Colab, Kaggle)
BUILD_COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or "no-commit-yet"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def inline(name):
    body = (SRC / name).read_text()
    body = re.sub(r"^from p1_engine import.*$", "", body, flags=re.M)
    body = re.sub(r"^from p3_agents import.*$", "", body, flags=re.M)
    body = re.sub(r"^from p1_score import.*$", "", body, flags=re.M)
    return code(body.replace("from __future__ import annotations\n", "from __future__ import annotations\n", 1))


def first_cell(packages):
    return code(f'''
# Cell 1 - run information (required by the brief)
import importlib.metadata as _meta, subprocess, sys, random, numpy as np, torch

SEED = 266
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # our GPU has 24 GB but the brief says everything must fit on 16 GB, so we hold ourselves to that
    _total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 16e9 / _total), 0)

commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
BUILT_FROM = "{BUILD_COMMIT}"  # the repo commit this notebook was generated from


def _version(name):
    try:
        return _meta.version(name)
    except _meta.PackageNotFoundError:
        return "not installed yet"


print({{"commit": commit or f"no git here, notebook built from {{BUILT_FROM}}", "built_from": BUILT_FROM, "python": sys.version.split()[0],
       "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "seed": SEED}})
print({{name: _version(name) for name in {packages!r}}})
''')


SETUP = '''
# Where are we running? Paths are relative, so the same notebook works on Colab, Kaggle or a local Jupyter.
import importlib.util, os
from pathlib import Path

ON_KAGGLE = Path("/kaggle/working").exists()
ON_COLAB = importlib.util.find_spec("google.colab") is not None
here = Path.cwd().resolve()
# if the notebook happens to sit inside our repo, deliverables go straight into it
REPO = next((p for p in [here, *here.parents] if all((p / d).is_dir() for d in ("eval", "src", "notebooks"))), None)
WORK = Path("/kaggle/working") if ON_KAGGLE else Path("/content") if ON_COLAB and Path("/content").is_dir() else (REPO or here)
# deliverables use the repo's folder layout, so the final ZIP unzips straight into the repo
OUT_ROOT = REPO or WORK / "lab2_repo_files"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
print("environment:", "kaggle" if ON_KAGGLE else "colab" if ON_COLAB else "local jupyter")
print("repo found:", REPO, "| working folder:", WORK, "| deliverables folder:", OUT_ROOT)
print("cuda:", torch.cuda.is_available())


def hf_token():
    """Hugging Face token from the HF_TOKEN variable, Colab or Kaggle secrets, or an earlier `huggingface-cli login`.
    It is never typed into the notebook."""
    token = os.environ.get("HF_TOKEN")
    if not token and ON_COLAB:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
        except Exception:
            token = None
    if not token and ON_KAGGLE:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            token = None
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            token = None
    if token:
        os.environ["HF_TOKEN"] = token  # diffusers and huggingface_hub read it from here
    return token
'''


def embed_cell(rel_paths):
    """Copies small repo files (test sets, scenarios, forms) into the notebook, so nothing has to be uploaded."""
    files = {rel: (ROOT / rel).read_text() for rel in rel_paths if (ROOT / rel).exists()}
    return code(f'''
# Test files built into this notebook (from repo commit {BUILD_COMMIT}). If the repo is here, its copy wins.
EMBEDDED_FILES = {json.dumps(files, indent=1)}
for rel, text in EMBEDDED_FILES.items():
    path = OUT_ROOT / rel
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    elif path.read_text() != text:
        print("NOTE: using the repo copy of", rel, "because it differs from the copy built into this notebook")
print("test files ready:", sorted(EMBEDDED_FILES))
''')


def last_cell(name, zip_name, keep):
    return code(f'''
# Last cell - put this notebook's deliverables in one ZIP. It uses the repo's folder layout, so unzipping it
# at the repo root puts every file where it belongs. Original project files are only read, never changed.
import json as _json, shutil, zipfile

KEEP = {keep!r}
export_dir = WORK / "export_{name}"
if export_dir.exists():
    shutil.rmtree(export_dir)
(export_dir / "notebooks").mkdir(parents=True)

# 1) the executed notebook. On Colab the .ipynb is not on disk, so we ask the browser for it.
nb_target, saved = export_dir / "notebooks" / "{name}.ipynb", False
if ON_COLAB:
    try:
        from google.colab import _message
        nb_target.write_text(_json.dumps(_message.blocking_request("get_ipynb", timeout_sec=120)["ipynb"]))
        saved = True
    except Exception as err:
        print("could not fetch the notebook from Colab:", type(err).__name__, err)
if not saved:
    for candidate in [Path("/kaggle/src/script.ipynb"), here / "{name}.ipynb", *([REPO / "notebooks" / "{name}.ipynb"] if REPO else [])]:
        if candidate.exists():
            shutil.copy(candidate, nb_target)
            saved = True
            break
if saved:
    print("notebook included. It has every output except this last cell, so also do File > Download .ipynb once this finishes.")
else:
    print("WARNING: notebook file not found. Do File > Download .ipynb and put it in notebooks/{name}.ipynb in the repo.")

# 2) the deliverables, small files only (no raw data, model caches or secrets)
skip_parts = {{"__pycache__", ".git", "chroma", "hf_cache", "data", "checkpoints"}}
for rel in KEEP:
    source = OUT_ROOT / rel if (OUT_ROOT / rel).exists() else WORK / rel
    if source.is_file():
        (export_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, export_dir / rel)
    elif source.is_dir():
        for path in source.rglob("*"):
            if path.is_file() and not (skip_parts & set(path.relative_to(source).parts)) and path.stat().st_size < 300e6:
                target = export_dir / rel / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, target)
    else:
        print("not found, skipped:", rel)

zip_path = WORK / "{zip_name}"
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(export_dir.rglob("*")):
        if path.is_file():
            zf.write(path, path.relative_to(export_dir))

print(f"Created ZIP: {{zip_path.resolve()}} ({{zip_path.stat().st_size / 1e6:.1f}} MB). Unzip it at the repo root.")
with zipfile.ZipFile(zip_path) as zf:
    for info in zf.infolist():
        print(f"  {{info.file_size / 1e6:8.2f}} MB  {{info.filename}}")
try:
    from google.colab import files
    files.download(str(zip_path))
except Exception:
    try:
        from IPython.display import FileLink, display
        display(FileLink(str(zip_path.relative_to(Path.cwd())) if zip_path.is_relative_to(Path.cwd()) else str(zip_path)))
    except Exception:
        pass
    print("If no download started, take the ZIP from the path above (file browser or output panel).")
''')


def save(cells, name):
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    nbf.validate(nb)
    NB.mkdir(exist_ok=True)
    nbf.write(nb, NB / f"{name}.ipynb")
    print("wrote", NB / f"{name}.ipynb", len(cells), "cells")


# ---------------------------------------------------------------- Part 1
def build_p1():
    conversation = "p1_conversation.py" if (SRC / "p1_conversation.py").exists() else "p1_conversation_placeholder.py"
    cells = [
        first_cell(["transformers", "qwen-vl-utils", "sentence-transformers", "chromadb", "rank-bm25", "pymupdf"]),
        md("""
# DATA 266 Lab 2, Part 1: Multimodal RAG (Team 24, Kavan's version)

We both built a complete Part 1 on our own: this is Kavan's (`p1_rag_kavan.ipynb`), and Muhammed's is `p1_rag_muhammed.ipynb`. Both write golden-set predictions in the same format, and `p1_eval.ipynb` scores them the same way. The version we pick at the end is copied to `p1_rag.ipynb` for Kaggle.

**How the system works, in short:**

1. We read all three PDFs with PyMuPDF. Text becomes chunks and every numbered figure and table becomes an image with its label, caption, page and trust level.
2. A local vision model (Qwen2.5-VL-3B) writes a description of each figure, so figures can be searched as text.
3. Everything goes into ChromaDB. Retrieval is exact label lookup first, then dense embeddings and BM25 joined with reciprocal rank fusion.
4. Tables are also parsed into typed facts. That lets plain Python apply the manifest's precedence rules, instead of hoping the model remembers them.
5. The model reads charts and answers open questions. Its output is checked against the JSON format before we keep it.

We used local models only because graders rerun this notebook on Kaggle, and an API key would not be there.
"""),
        md("## Install packages\nVersions are pinned so a fresh GPU environment gets what we tested. If an install fails it fails loudly, we do not hide it."),
        code("%pip install -q transformers==4.51.3 accelerate==1.6.0 qwen-vl-utils==0.0.11 sentence-transformers==4.1.0 chromadb==1.0.7 rank-bm25==0.2.2 pymupdf==1.25.5 pandas pillow matplotlib"),
        code(SETUP),
        md("## Find the data\nThe PDFs cannot be built into the notebook (we never commit them), so this is the one thing to provide. Put the Part 1 files, or a ZIP of them, next to the notebook, in `/content`, or in `~/data266_lab2_data/`. On Kaggle, attach the competition data. `QUESTIONS_PATH` lets graders swap in the hidden questions."),
        code('''
import glob, json, shutil, time, zipfile
import pandas as pd

search = [WORK, here, Path("/content"), Path.home() / "data266_lab2_data", *([REPO] if REPO else [])]
search = list(dict.fromkeys(p for p in search if p.is_dir()))
# a ZIP of the Part 1 files is unpacked once
for base in search:
    for z in base.glob("*.zip"):
        try:
            with zipfile.ZipFile(z) as zf:
                if any(Path(n).name == "kb_manifest.json" for n in zf.namelist()) and not (WORK / "data" / z.stem).exists():
                    zf.extractall(WORK / "data" / z.stem)
                    print("unzipped", z)
        except zipfile.BadZipFile:
            pass
found = glob.glob("/kaggle/input/**/kb_manifest.json", recursive=True)
for base in search + [WORK / "data"]:
    for depth in ("", "*", "*/*", "*/*/*"):
        found += glob.glob(str(base / depth / "kb_manifest.json"))


def complete(folder):
    docs = json.loads((folder / "kb_manifest.json").read_text())["documents"]
    return all((folder / d["file"]).exists() for d in docs) and (folder / "validate_submission.py").exists()


DATA = next((Path(f).parent for f in dict.fromkeys(found) if complete(Path(f).parent)), None)
assert DATA is not None, f"Part 1 files not found. Looked in: {[str(p) for p in search]} and /kaggle/input"

QUESTIONS_PATH = os.environ.get("QUESTIONS_PATH", str(DATA / "Questions.csv"))
OUT = WORK / "outputs" / "p1"
if (OUT / "chroma").exists():
    shutil.rmtree(OUT / "chroma")  # the index is rebuilt from the PDFs on every run
OUT.mkdir(parents=True, exist_ok=True)
print("data:", DATA)
print("questions:", QUESTIONS_PATH)


def local_or_hub(hub_id):
    """If Kaggle has no internet, a model attached as a dataset is used instead."""
    tail = hub_id.split("/")[-1].lower()
    for folder in glob.glob("/kaggle/input/*/") + glob.glob("/kaggle/input/*/*/"):
        if tail in folder.lower() and (Path(folder) / "config.json").exists():
            print("using attached model:", folder)
            return folder
    return hub_id
'''),
        md("## The engine\nThis cell is `src/p1_engine.py`, pasted in by `tools/build_notebooks.py` so the notebook runs without the repo. It holds no question text and no answer values."),
        inline("p1_engine.py"),
        md(f"## Conversation memory\nThis cell is `src/{conversation}`." + (" **This is only a placeholder.** It has no follow-up resolution, user-value handling or refusal yet, so multi-turn questions will score badly until `src/p1_conversation.py` replaces it." if "placeholder" in conversation else " It keeps one memory per `conversation_id`, and user values never go into the index.")),
        inline(conversation),
        md("## Build the knowledge base\nThe table below is what we extracted. `Figure B-5` and `Figure B-5 (corrected)` must be separate rows, and Appendix C must be low trust."),
        code('''
T0 = time.perf_counter()
kb = KnowledgeBase(DATA, OUT).build()
objects = pd.DataFrame([{"label": o.label, "doc": o.doc_id, "page": o.page, "trust": o.trust, "caption": o.caption[:70],
                         "superseded_by": o.superseded_by or ""} for o in kb.objects.values()])
print(len(kb.chunks), "chunks,", len(kb.objects), "numbered objects,", len(kb.facts), "facts")
pd.set_option("display.max_rows", 60, "display.max_colwidth", 80)
objects
'''),
        code('''
facts = pd.DataFrame([vars(f) for f in kb.facts])
print(facts.groupby(["doc_id", "kind"]).size())
print("low-trust facts:", int((facts.trust == "low").sum()))
facts.head(10)
'''),
        md("## Load the models and index everything\nThe vision model describes all figures here, so this is the slowest step. Sentences that try to instruct an AI are stripped from those descriptions."),
        code('''
models = QwenModels(local_or_hub("Qwen/Qwen2.5-VL-3B-Instruct"))
embedder = Embedder(local_or_hub("BAAI/bge-small-en-v1.5"))
engine = Engine(kb, models, embedder, OUT / "chroma", describe_figures=True)
START_COUNT = engine.index_count()
print("indexed items:", START_COUNT, "| build time (s):", round(time.perf_counter() - T0, 1))
print("figures where the vision model repeated an instruction:", [o.label for o in kb.objects.values() if o.injection_found])
if torch.cuda.is_available():
    print("peak GPU memory so far (GB):", round(torch.cuda.max_memory_allocated() / 1e9, 2))
'''),
        code('''
import matplotlib.pyplot as plt
from PIL import Image

show = ["Figure A-2", "Figure A-3", "Figure B-5", "Figure B-5 (corrected)"]
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, label in zip(axes.ravel(), show):
    ax.imshow(Image.open(kb.objects[label].image_path))
    ax.set_title(label)
    ax.axis("off")
plt.tight_layout()
plt.show()
for label in show[:2]:
    print(label, "->", kb.objects[label].description[:300], "\\n")
'''),
        md("## Quick retrieval checks\nA named figure must come back first, and Appendix C must never come back as evidence."),
        code('''
for query in ["What does Figure A-5 show?", "What does Figure B-5 show?", "Which charts show stock prices?"]:
    print(query)
    for mode in ("dense", "hybrid"):
        print("  ", mode, [f"{h.doc_id}:p{h.page}" + (f"/{h.figure_id}" if h.figure_id else "") for h in engine.retrieve(query, k=5, mode=mode)])
leaks = sum(h.trust_level == "low" for q in ["reader comment", "sentiment index", "system note for AI assistants"] for h in engine.retrieve(q, k=25))
print("low-trust hits in 75 retrieved items:", leaks)
'''),
        md("## Test files and the shared scorer\nKavan's tuning set and the golden sets are built into this notebook, so nothing needs uploading. `p1_score.py` is shared with Muhammed: it runs a golden set and saves predictions in one format."),
        embed_cell(["eval/p1_test_prompts.csv", *[str(p.relative_to(ROOT)) for p in sorted(EVAL.glob("p1_golden_*.csv"))]]),
        inline("p1_score.py"),
        md("## Answer every question and write submission.csv\nThis follows the brief: each conversation starts empty and runs turn by turn. If time runs out we still write a valid row and say so, because the run must finish within 90 minutes."),
        code('''
SYSTEM = "kavan"


def new_conversation(conversation_id="", mode="hybrid"):
    return Conversation(conversation_id, engine, mode)


BUDGET_SECONDS = 85 * 60
engine.timings.clear()
q = pd.read_csv(QUESTIONS_PATH, dtype=str, keep_default_na=False)
rows, skipped = [], 0
for conv_id, turns in q.groupby("conversation_id", sort=False):
    chat = new_conversation(conv_id)
    for _, r in turns.sort_values("turn", key=lambda t: t.astype(int)).iterrows():
        if time.perf_counter() - T0 > BUDGET_SECONDS:
            a, skipped = {"answer": "Not answered, the time budget ran out.", "status": "not_found", "citations": [], "figure_references": []}, skipped + 1
        else:
            a = chat.answer(r["question"])
        rows.append({"id": r["id"], "conversation_id": conv_id, "question_id": r["question_id"], "answer": " ".join(a["answer"].split()),
                     "figure_references": ",".join(a["figure_references"]) or "NONE", "citations": ",".join(a["citations"]) or "NONE", "status": a["status"]})
submission = WORK / "submission.csv"
pd.DataFrame(rows).set_index("id").loc[q["id"]].reset_index().to_csv(submission, index=False)
print("rows:", len(rows), "| skipped for time:", skipped, "| total minutes:", round((time.perf_counter() - T0) / 60, 1))
print("median seconds per question:", round(median_seconds(engine.timings), 3))
preview = pd.read_csv(submission, dtype=str, keep_default_na=False)
print(preview.status.value_counts())
preview[["id", "status", "answer", "figure_references", "citations"]].head(12)
'''),
        md("## Validate\nThis runs the official checker on the file we just wrote, then proves user values never touched the index."),
        code('''
result = subprocess.run([sys.executable, str(DATA / "validate_submission.py"), str(submission), QUESTIONS_PATH], capture_output=True, text=True)
print(result.stdout, result.stderr)
assert result.returncode == 0, "submission failed validation"
assert engine.index_count() == START_COUNT, "the index changed during the run"
print("index unchanged:", START_COUNT, "items")
if torch.cuda.is_available():
    print("peak GPU memory (GB):", round(torch.cuda.max_memory_allocated() / 1e9, 2))
run_info = {"rows": len(rows), "skipped_for_time": skipped, "median_seconds_per_question": median_seconds(engine.timings),
            "total_minutes": (time.perf_counter() - T0) / 60, "indexed_items": START_COUNT}
results_dir = OUT_ROOT / "reports" / "results"
results_dir.mkdir(parents=True, exist_ok=True)
(results_dir / f"p1_kaggle_run_{SYSTEM}.json").write_text(json.dumps(run_info, indent=2))
print(run_info)
'''),
        md("## Extra runs for our own evaluation\nEverything below runs **after** `submission.csv` is written and validated, so it cannot slow down the graded run. When graders swap in hidden questions with `QUESTIONS_PATH`, these extras are skipped. Set `RUN_EXTRA = False` to skip them by hand."),
        code('''
RUN_EXTRA = "QUESTIONS_PATH" not in os.environ
print("extra runs:", RUN_EXTRA)
'''),
        md("## Tuning set (not a reported result)\nThis is `eval/p1_test_prompts.csv`, Kavan's own tuning questions. It is separate from the golden sets on purpose, and these numbers do **not** go in the report. Reported results come from `p1_eval.ipynb`."),
        code('''
dev_path = OUT_ROOT / "eval" / "p1_test_prompts.csv"
if RUN_EXTRA and dev_path.exists():
    dev = pd.read_csv(dev_path, dtype=str, keep_default_na=False)
    for mode in ("dense", "hybrid"):
        chats, rows = {}, []
        for _, r in dev.sort_values(["conversation_id", "turn"]).iterrows():
            chat = chats.setdefault(r["conversation_id"], new_conversation(r["conversation_id"], mode))
            out = chat.answer(r["question"])
            ok = out["status"] == r["gold_status"] and numbers_match(r["required_numbers"], out["answer"])
            rows.append({"id": r["dev_id"], "category": r["category"], "ok": ok, "status": out["status"], "answer": out["answer"][:110]})
        table = pd.DataFrame(rows)
        print(mode, "tuning set correct:", int(table.ok.sum()), "of", len(table))
        display(table[~table.ok])
else:
    print("Skipping the tuning set (graded run, or the file is missing).")
'''),
        md("""## Golden-set predictions (scored in `p1_eval.ipynb`)
We run every golden set found in `eval/p1_golden_*.csv`, in dense mode (our baseline) and hybrid mode (our final system), and save the raw predictions. On purpose, **this cell prints no scores**. Scoring happens in `p1_eval.ipynb`, so we are not tempted to tune on the golden questions. It runs after the graded run, so it never eats into the 90 minute budget, and it is skipped when graders set `QUESTIONS_PATH`."""),
        code('''
if RUN_EXTRA:
    def hit_source(h):
        return f"{h.doc_id}:p{h.page}" + (f":{h.figure_id}" if h.figure_id else "")

    def retrieve_sources(query, k, mode):
        return [hit_source(h) for h in engine.retrieve(query, k=k, mode=mode)]

    gold_files = sorted((OUT_ROOT / "eval").glob("p1_golden_*.csv"))
    print("golden sets found:", [p.name for p in gold_files])
    for path in gold_files:
        gold = load_gold(path)
        if gold.empty:
            print("skipping", path.name, "(no questions yet)")
            continue
        goldset = path.stem.replace("p1_golden_", "")
        for mode in ("dense", "hybrid"):
            preds = run_golden(gold, new_conversation, retrieve_sources, mode)
            out_path = prediction_path(OUT_ROOT / "reports" / "results", SYSTEM, goldset, mode)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            preds.to_csv(out_path, index=False)
            print("wrote", out_path.relative_to(OUT_ROOT), len(preds), "rows")
    assert engine.index_count() == START_COUNT, "the index changed during the golden run"
else:
    print("Graded run (QUESTIONS_PATH is set), so we skip the golden sets.")
'''),
        md("## Limitations we know about\n- Values read off an axis are approximate, because only some chart points have printed labels.\n- The typed-fact path only knows the entities and variables in its vocabulary. Anything else goes to the model.\n- The model is small, so answers to open questions can be thin. We have not measured that here. `p1_eval.ipynb` does."),
        last_cell("p1_rag_kavan", "DATA266_Lab2_p1_kavan.zip", ["submission.csv", "reports/results/p1_kaggle_run_kavan.json", "reports/results/p1_predictions"]),
    ]
    save(cells, "p1_rag_kavan")


# ---------------------------------------------------------------- Part 1 shared scoring
def build_p1_eval():
    cells = [
        first_cell(["pandas"]),
        md("""
# DATA 266 Lab 2, Part 1 evaluation (shared)

Both of us built a complete Part 1. Each system wrote raw golden-set predictions to `reports/results/p1_predictions/<system>__<goldset>__<mode>.csv`. This notebook loads **no models**. It only scores those files the same way, so the comparison is fair.

- **Correct out of 25:** the status matches, and every required number is within ±1% (plus required terms for answers with no numbers).
- **Recall@5:** a gold page or figure is among the top 5 retrieved items. Questions with no gold source are left out.
- **Citation accuracy:** the share of cited sources that match a gold source for that question. This is an automatic stand-in for "really supports the answer", so the groundedness ratings check it by hand.
- **Groundedness:** both of us rate each final answer 1–5 in `eval/human/p1_groundedness_<name>.csv`. Blank forms print `PENDING`.
- **Efficiency:** median seconds per question.

Each of us wrote one golden set before tuning. Every system is scored on both sets. The set written by the *other* person is the fairer test, because that system's author never saw it while building.
"""),
        code("%pip install -q pandas"),
        code(SETUP),
        inline("p1_score.py"),
        md("## Check the golden sets against the brief\nThe brief wants 25 questions with fixed category counts, and multi-turn questions in at least 2 conversations."),
        code('''
import json
import pandas as pd

assert REPO, "run this notebook inside the repo"
EVAL, RESULTS = REPO / "eval", REPO / "reports" / "results"
golds = {p.stem.replace("p1_golden_", ""): load_gold(p) for p in sorted(EVAL.glob("p1_golden_*.csv"))}
for name, gold in golds.items():
    problems = check_gold(gold)
    print(f"golden set '{name}':", len(gold), "questions |", "OK" if not problems else "; ".join(problems))
'''),
        md("## Score every prediction file\nOne row per system, golden set and mode."),
        code('''
summaries, details = [], {}
for path in sorted((RESULTS / "p1_predictions").glob("*__*__*.csv")):
    system, goldset, mode = path.stem.split("__")
    if goldset not in golds or golds[goldset].empty:
        print("no golden set for", path.name)
        continue
    table, summary = score(golds[goldset], read_csv(path))
    details[(system, goldset, mode)] = table
    summaries.append({"system": system, "goldset": goldset, "mode": mode, **summary})
CROSS = pd.DataFrame(summaries)
CROSS if len(CROSS) else print("No prediction files yet.")
'''),
        md('''## Pick the final system
The choice is written by us in `eval/p1_selection.json` at the final merge, after looking at the table above. For example: `{"final_system": "kavan", "report_goldset": "muhammed"}`. `report_goldset` is the set whose numbers go in the report's results table. Until that file exists, everything below prints `PENDING`.'''),
        code('''
selection_path = EVAL / "p1_selection.json"
selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
results = {"cross_table": summaries, "selection": selection, "baseline": {}, "final": {}}
if selection is None:
    print("PENDING: eval/p1_selection.json does not exist yet.")
else:
    sysname, goldset = selection["final_system"], selection["report_goldset"]
    for tag, mode in (("baseline", "dense"), ("final", "hybrid")):
        row = next((s for s in summaries if (s["system"], s["goldset"], s["mode"]) == (sysname, goldset, mode)), None)
        results[tag] = row or {}
        print(tag, "->", row if row else "PENDING: no prediction file")
'''),
        md("## Groundedness forms\nThese are for the final system's hybrid answers on the report golden set. The notebook writes a review sheet and blank forms, but never fills in a rating. Each of us rates without looking at the other's form."),
        code('''
HUMAN = EVAL / "human"
HUMAN.mkdir(exist_ok=True)
if selection and (sysname, goldset, "hybrid") in details:
    final_table = details[(sysname, goldset, "hybrid")]
    preds = read_csv(RESULTS / "p1_predictions" / f"{sysname}__{goldset}__hybrid.csv")
    sheet = golds[goldset][["gold_id", "question", "gold_sources"]].merge(preds[["gold_id", "answer", "status", "citations", "figure_references"]], on="gold_id")
    sheet.to_csv(RESULTS / "p1_groundedness_review_sheet.csv", index=False)
    forms = {}
    for rater in ("kavan", "muhammed"):
        form_path = HUMAN / f"p1_groundedness_{rater}.csv"
        if not form_path.exists():
            pd.DataFrame({"gold_id": sheet.gold_id, "rating": ""}).to_csv(form_path, index=False)
            print("wrote blank form", form_path.relative_to(REPO))
        forms[rater] = read_csv(form_path)
    g = groundedness(forms, list(sheet.gold_id))
    if not g["complete"]:
        print("PENDING: groundedness forms are not complete.")
    results["final"].update({k: g[k] for k in ("groundedness_mean", "groundedness_within_one")})
else:
    print("PENDING: no final system selected yet, so no groundedness forms.")
'''),
        md("## Where each system went wrong\nThis is for our failure analysis. Wrong rows only, for every hybrid run."),
        code('''
pd.set_option("display.max_colwidth", 90)
for (system, gs, mode), table in details.items():
    if mode == "hybrid":
        wrong = table[~table.correct]
        print(f"{system} on {gs}: {len(wrong)} wrong")
        display(wrong[["gold_id", "category", "gold_status", "status", "numbers_ok", "terms_ok", "answer"]])
'''),
        md("## Save the results\n`reports/build_tables.py` reads this file. Nobody types report numbers by hand."),
        code('''
(RESULTS / "p1_eval.json").write_text(json.dumps(results, indent=2, default=str))
print(json.dumps({k: results[k] for k in ("selection", "baseline", "final")}, indent=2, default=str))
'''),
        last_cell("p1_eval", "DATA266_Lab2_p1_eval.zip", ["reports/results/p1_eval.json", "reports/results/p1_predictions", "reports/results/p1_groundedness_review_sheet.csv",
                  "eval/p1_selection.json", "eval/human"]),
    ]
    save(cells, "p1_eval")


# ---------------------------------------------------------------- Part 2 baseline
P2_DATA_LOADER = '''
from datasets import Image as DatasetImage, load_dataset


def load_naruto():
    # the brief's lambdalabs/ name is now a redirect to lambda/, so we try both
    for name in ("lambda/naruto-blip-captions", "lambdalabs/naruto-blip-captions"):
        try:
            return name, load_dataset(name, split="train")
        except Exception as err:
            print("could not load", name, "->", type(err).__name__, str(err)[:150])
    raise RuntimeError("The Naruto dataset could not be loaded.")
'''


def build_p2():
    cells = [
        first_cell(["diffusers", "transformers", "peft", "accelerate", "datasets", "lpips", "torchmetrics"]),
        md("""
# DATA 266 Lab 2, Part 2 baseline (Kavan)

Muhammed leads Part 2 in `p2_finetune.ipynb`. The brief says the partner who is not leading still runs the whole pipeline at least as a baseline, so this is that baseline: clean the data, train one LoRA, generate, evaluate.

It also gives Part 3 a working LoRA until Muhammed's final model is ready: at the end we upload it to Hugging Face (`Kavan1804/data266-p2-baseline-kavan-lora`), and `p3_agents.ipynb` downloads it from there. **Numbers here are not the Part 2 results for the report.** Those come from `p2_finetune.ipynb`.

Everything this notebook needs downloads by itself. For the upload, add a Hugging Face write token as `HF_TOKEN`: a Colab secret (key icon), an environment variable, or `huggingface-cli login` once on this machine. Never paste it into the notebook.
"""),
        code("%pip install -q diffusers==0.33.1 transformers==4.51.3 peft==0.15.2 accelerate==1.6.0 datasets==3.5.1 lpips==0.1.4 torchmetrics==1.7.1 ftfy==6.3.1 tensorboard pandas pillow matplotlib"),
        code(SETUP),
        md("## Clean and split the data\nWe drop images that do not decode and exact duplicates, then split 90/10 with seed 266. Test images are never used for training."),
        code(P2_DATA_LOADER + '''

import hashlib, io, json, time
from PIL import Image

OUT = WORK / "outputs" / "p2_baseline_kavan"
TRAIN_DIR, TEST_DIR = OUT / "data" / "train", OUT / "data" / "test"
OUT.mkdir(parents=True, exist_ok=True)
name, raw = load_naruto()
raw = raw.cast_column("image", DatasetImage(decode=False))
seen, keep, corrupted, duplicates = set(), [], 0, 0
for i, row in enumerate(raw):
    data = row["image"]["bytes"] or open(row["image"]["path"], "rb").read()
    try:
        Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        corrupted += 1
        continue
    digest = hashlib.md5(data).hexdigest()
    if digest in seen:
        duplicates += 1
        continue
    seen.add(digest)
    keep.append(i)
random.Random(SEED).shuffle(keep)
cut = int(len(keep) * 0.9)
train_ids, test_ids = keep[:cut], keep[cut:]
print({"dataset": name, "rows": len(raw), "corrupted": corrupted, "duplicates": duplicates, "train": len(train_ids), "test": len(test_ids)})
for folder, ids in ((TRAIN_DIR, train_ids), (TEST_DIR, test_ids)):
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "metadata.jsonl", "w") as f:
        for i in ids:
            row = raw[int(i)]
            data = row["image"]["bytes"] or open(row["image"]["path"], "rb").read()
            path = folder / f"{i:05d}.png"
            if not path.exists():
                Image.open(io.BytesIO(data)).convert("RGB").resize((512, 512), Image.LANCZOS).save(path)
            f.write(json.dumps({"file_name": path.name, "text": row["text"]}) + "\\n")
print("captions look like:", json.loads((TEST_DIR / "metadata.jsonl").read_text().splitlines()[0])["text"])
'''),
        md("""
## Train one LoRA
We use the diffusers example script from the same version we pin (0.33.1), at rank 4. A subprocess does not inherit our memory limit, so a small launcher applies the 16 GB cap inside it and prints peak memory at the end.

If weights and a training log already exist, we reuse them and say so. Delete `outputs/p2_baseline_kavan/lora` to train again.
"""),
        code('''
import urllib.request

MAX_STEPS, RANK, LR, BATCH = 1500, 4, 1e-4, 1
LORA_DIR, META = OUT / "lora", OUT / "run_meta.json"
SCRIPT = OUT / "train_text_to_image_lora.py"
URL = "https://raw.githubusercontent.com/huggingface/diffusers/v0.33.1/examples/text_to_image/train_text_to_image_lora.py"
LAUNCHER = OUT / "capped_launcher.py"
LAUNCHER.write_text("""
import atexit, runpy, sys, torch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(min(1.0, 16e9 / torch.cuda.get_device_properties(0).total_memory), 0)
    atexit.register(lambda: print("PEAK_GPU_GB", round(torch.cuda.max_memory_allocated() / 1e9, 3), flush=True))
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
""")
if (LORA_DIR / "pytorch_lora_weights.safetensors").exists() and META.exists():
    run_meta = json.loads(META.read_text())
    print("REUSING an earlier training run, not training again:")
else:
    if not SCRIPT.exists():
        urllib.request.urlretrieve(URL, SCRIPT)
    command = [sys.executable, str(LAUNCHER), str(SCRIPT), "--pretrained_model_name_or_path", "stable-diffusion-v1-5/stable-diffusion-v1-5",
               "--train_data_dir", str(TRAIN_DIR), "--caption_column", "text", "--resolution", "512", "--random_flip",
               "--train_batch_size", str(BATCH), "--max_train_steps", str(MAX_STEPS), "--learning_rate", str(LR), "--lr_scheduler", "cosine",
               "--lr_warmup_steps", "0", "--rank", str(RANK), "--seed", str(SEED), "--mixed_precision", "fp16", "--output_dir", str(LORA_DIR),
               "--checkpointing_steps", "500", "--checkpoints_total_limit", "1", "--dataloader_num_workers", "2", "--report_to", "tensorboard"]
    start = time.perf_counter()
    peak = None
    with open(OUT / "train_log.txt", "w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            log.write(line)
            if line.startswith("PEAK_GPU_GB"):
                peak = float(line.split()[1])
            if "Steps:" not in line:
                print(line, end="")
        code_ = process.wait()
    assert code_ == 0, f"training failed with exit code {code_}, see {OUT / 'train_log.txt'}"
    seconds = time.perf_counter() - start
    run_meta = {"rank": RANK, "learning_rate": LR, "steps": MAX_STEPS, "batch_size": BATCH, "seed": SEED, "train_seconds": seconds,
                "gpu_hours": seconds / 3600, "peak_gpu_gb": peak, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    META.write_text(json.dumps(run_meta, indent=2))
if "trainable_params" not in run_meta:
    # LoRA only trains the adapter weights, so counting the saved tensors gives the trainable parameters
    from safetensors.torch import load_file
    run_meta["trainable_params"] = int(sum(t.numel() for t in load_file(str(LORA_DIR / "pytorch_lora_weights.safetensors")).values()))
    META.write_text(json.dumps(run_meta, indent=2))
print(run_meta)
'''),
        md("## Generate\nSame sampler settings for both models. We use 4 test captions plus 2 out-of-domain prompts, with seeds 0 and 1. The SD safety checker stays on, and we count what it flags."),
        code('''
import gc
from diffusers import StableDiffusionPipeline

captions = [json.loads(x)["text"] for x in (TEST_DIR / "metadata.jsonl").read_text().splitlines()][:4]
PROMPTS = captions + ["a photo of a red bicycle next to a lake", "a bowl of fruit on a wooden table"]
SAMPLER = dict(num_inference_steps=30, guidance_scale=7.5, height=512, width=512)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
GEN = OUT / "generated"
records = []
for model_name in ("base", "lora"):
    pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    if model_name == "lora":
        pipe.load_lora_weights(str(LORA_DIR), weight_name="pytorch_lora_weights.safetensors")
    (GEN / model_name).mkdir(parents=True, exist_ok=True)
    for p, prompt in enumerate(PROMPTS):
        for seed in (0, 1):
            out = pipe(prompt, generator=torch.Generator(device).manual_seed(seed), **SAMPLER)
            path = GEN / model_name / f"p{p}_s{seed}.png"
            out.images[0].save(path)
            records.append({"model": model_name, "prompt_id": p, "prompt": prompt, "seed": seed, "path": str(path), "nsfw": bool(out.nsfw_content_detected and out.nsfw_content_detected[0])})
    del pipe
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
print("images:", len(records), "| flagged by the safety checker:", sum(r["nsfw"] for r in records))
'''),
        code('''
import matplotlib.pyplot as plt

fig, axes = plt.subplots(len(PROMPTS), 2, figsize=(7, 3.4 * len(PROMPTS)))
for p, prompt in enumerate(PROMPTS):
    for c, model_name in enumerate(("base", "lora")):
        axes[p, c].imshow(Image.open(GEN / model_name / f"p{p}_s0.png"))
        axes[p, c].set_title(f"{model_name}: {prompt[:38]}", fontsize=8)
        axes[p, c].axis("off")
plt.tight_layout()
(OUT_ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_ROOT / "reports" / "figures" / "p2_baseline_kavan_grid.png", dpi=110, bbox_inches="tight")
plt.show()
'''),
        md("## Evaluate\nCLIP score (prompt vs image), LPIPS between the two seeds of each prompt, and KID against test images. With this few images KID is very noisy, so it is only a sanity check."),
        code('''
import lpips
import torchvision.transforms.functional as TF
from torchmetrics.image.kid import KernelInceptionDistance
from transformers import CLIPModel, CLIPProcessor

clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
loss_fn = lpips.LPIPS(net="alex", verbose=False).to(device)


def clip_score(path, text):
    inputs = clip_processor(text=[text], images=Image.open(path).convert("RGB"), return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.inference_mode():
        out = clip(**inputs)
    a, b = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True), out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return float((a * b).sum())


def as_tensor(path):
    return TF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device) * 2 - 1


results = {"note": "Kavan's baseline, not the Part 2 results", **run_meta}
in_domain = [r for r in records if r["prompt_id"] < 4]
real = [TF.pil_to_tensor(Image.open(p).convert("RGB")) for p in sorted(TEST_DIR.glob("*.png"))]
for model_name in ("base", "lora"):
    subset = [r for r in records if r["model"] == model_name]
    scores = [clip_score(r["path"], r["prompt"]) for r in subset]
    with torch.inference_mode():
        diversity = [float(loss_fn(as_tensor(GEN / model_name / f"p{p}_s0.png"), as_tensor(GEN / model_name / f"p{p}_s1.png"))) for p in range(len(PROMPTS))]
    fake = [TF.pil_to_tensor(Image.open(r["path"]).convert("RGB")) for r in in_domain if r["model"] == model_name]
    kid = KernelInceptionDistance(subset_size=min(len(fake), len(real)) - 1, subsets=20, normalize=False).to(device)
    kid.update(torch.stack(real).to(device), real=True)
    kid.update(torch.stack(fake).to(device), real=False)
    kid_mean, kid_std = kid.compute()
    results[model_name] = {"clip_score": sum(scores) / len(scores), "lpips_diversity": sum(diversity) / len(diversity), "kid_mean": float(kid_mean), "kid_std": float(kid_std)}
print(json.dumps(results, indent=2))
results_dir = OUT_ROOT / "reports" / "results"
results_dir.mkdir(parents=True, exist_ok=True)
(results_dir / "p2_baseline_kavan.json").write_text(json.dumps(results, indent=2))
(OUT / "generated" / "records.json").write_text(json.dumps(records, indent=2))
'''),
        md("## Upload the LoRA to Hugging Face\nThe brief wants weights on Hugging Face, not GitHub, and Part 3 loads this LoRA from there. The repo is public so Part 3 and the graders can read it. If no token is found we say so and carry on, because the ZIP below still holds the weights."),
        code('''
HF_LORA_REPO = os.environ.get("HF_LORA_REPO", "Kavan1804/data266-p2-baseline-kavan-lora")
HF_PRIVATE = False
card = f"""---
base_model: stable-diffusion-v1-5/stable-diffusion-v1-5
library_name: diffusers
tags: [lora, text-to-image, stable-diffusion]
license: other
---
# DATA 266 Lab 2, Part 2 baseline LoRA (Kavan, Team 24)

Partner baseline for our class lab, not our final Part 2 model.

- **Base model:** stable-diffusion-v1-5/stable-diffusion-v1-5 at 512x512
- **Data:** {name}, cleaned and split 90/10 with seed {SEED} ({len(train_ids)} training images)
- **Training:** LoRA rank {run_meta['rank']}, learning rate {run_meta['learning_rate']}, {run_meta['steps']} steps, batch size {run_meta['batch_size']}, seed {run_meta['seed']}, fp16
- **Cost:** {run_meta['gpu_hours']:.2f} GPU-hours on {run_meta['gpu']}, peak memory {run_meta['peak_gpu_gb']} GB
- **Results (small sanity check, see the notebook):** CLIP score {results['lora']['clip_score']:.3f} vs base {results['base']['clip_score']:.3f}

**License and limits:** Naruto characters are copyrighted and the dataset page lists no license, so this LoRA is for non-commercial research only. It was trained on a small anime dataset and copies that style even when you do not want it.
"""
upload = {"hf_repo": HF_LORA_REPO, "uploaded": False}
token = hf_token()
if not token:
    print("NOT uploaded: no Hugging Face token found. Part 3 can still use the local copy if it runs in this same folder.")
else:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(HF_LORA_REPO, repo_type="model", private=HF_PRIVATE, exist_ok=True)
        (OUT / "README.md").write_text(card)
        for local, remote in ((LORA_DIR / "pytorch_lora_weights.safetensors", "pytorch_lora_weights.safetensors"), (META, "run_meta.json"), (OUT / "README.md", "README.md")):
            api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=HF_LORA_REPO, repo_type="model")
        upload["uploaded"] = True
        print("uploaded to https://huggingface.co/" + HF_LORA_REPO)
    except Exception as err:
        print("UPLOAD FAILED:", type(err).__name__, err)
(OUT_ROOT / "reports" / "results" / "p2_baseline_kavan_upload.json").write_text(json.dumps(upload, indent=2))
'''),
        md("## What we observed\nFill this in after the run, using only what the outputs above show: did the LoRA move in-domain prompts toward the style, and did it hurt out-of-domain prompts? We leave it open until there are real outputs."),
        last_cell("p2_baseline_kavan", "DATA266_Lab2_p2_baseline_kavan.zip", ["outputs/p2_baseline_kavan/lora/pytorch_lora_weights.safetensors", "outputs/p2_baseline_kavan/run_meta.json",
                  "outputs/p2_baseline_kavan/train_log.txt", "outputs/p2_baseline_kavan/generated", "reports/results/p2_baseline_kavan.json",
                  "reports/results/p2_baseline_kavan_upload.json", "reports/figures/p2_baseline_kavan_grid.png"]),
    ]
    save(cells, "p2_baseline_kavan")


# ---------------------------------------------------------------- Part 3
def build_p3():
    cfg = EVAL / "p2_final_model.json"
    P2_CONFIG = json.loads(cfg.read_text()) if cfg.exists() else None
    cells = [
        first_cell(["diffusers", "transformers", "peft", "accelerate", "qwen-vl-utils", "gradio", "pydantic"]),
        md("""
# DATA 266 Lab 2, Part 3: Multimodal Multi-Agent System (Team 24)

Kavan leads Part 3. Five agents share one typed state:

**Planner** routes and refuses, **Vision** captions and reads text, **Prompt** writes the generation prompt, **Generation** runs img2img, InstructPix2Pix or our Part 2 LoRA, and **Critique** scores the result and can ask for up to two revisions.

The main safety rule: **text inside an image is data, never an instruction.** Every request saves a JSON trace to `traces/`.

Scenarios and red-team cases are in `eval/` and were committed before running. They are built into this notebook, so it runs with no uploads. Domain-style requests use our Part 2 LoRA from Hugging Face.

People judge the pass/fail rubrics after this run. This notebook prints `PENDING` for those numbers, and `p3_results.ipynb` computes them once the forms are filled in.
"""),
        code("%pip install -q diffusers==0.33.1 transformers==4.51.3 peft==0.15.2 accelerate==1.6.0 qwen-vl-utils==0.0.11 gradio==5.29.0 pydantic==2.11.4 datasets==3.5.1 pandas pillow matplotlib"),
        code(SETUP),
        md("## The agents\nThis cell is `src/p3_agents.py`. Model calls go through a `hub` object, so a failing model becomes a clear message rather than a crash."),
        inline("p3_agents.py"),
        md("## The model backends\nThis cell is `src/p3_models.py`. The vision model stays loaded and only one Stable Diffusion pipeline is in memory at a time, which is how we stay under 16 GB."),
        inline("p3_models.py"),
        md("## Test images\nAbout 25 images: 12 COCO photos, 3 Part 2 style images, 3 red-team images we draw ourselves, 3 poor-quality images and 3 broken files."),
        inline("p3_images.py"),
        embed_cell(["eval/p3_scenarios.json", "eval/p3_red_team_cases.json", "eval/human/p3_passfail_kavan.csv", "eval/human/p3_passfail_muhammed.csv"]),
        code('''
import statistics
import pandas as pd

P3 = WORK / "outputs" / "p3"
IMAGES, GENERATED = P3 / "inputs", P3 / "generated"
TRACES = OUT_ROOT / "traces"
EVAL = OUT_ROOT / "eval"
assert (EVAL / "p3_scenarios.json").exists(), "eval/p3_scenarios.json is missing"
build_local_images(IMAGES)
download_coco(IMAGES)
if not (IMAGES / "domain_test_3.png").exists():
    from datasets import load_dataset
    for name in ("lambda/naruto-blip-captions", "lambdalabs/naruto-blip-captions"):
        try:
            data = load_dataset(name, split="train")
            break
        except Exception as err:
            print("could not load", name, type(err).__name__)
    picks = random.Random(SEED).sample(range(len(data)), 3)
    for n, i in enumerate(picks, start=1):
        data[i]["image"].convert("RGB").resize((512, 512)).save(IMAGES / f"domain_test_{n}.png")
names = sorted(p.name for p in IMAGES.iterdir())
print(len(names), "test images:", names)
scenarios, red_team = load_cases(EVAL / "p3_scenarios.json"), load_cases(EVAL / "p3_red_team_cases.json")
print(len(scenarios), "scenarios,", len(red_team), "red-team cases")
for case in scenarios + red_team:
    if case.get("image"):
        assert (IMAGES / case["image"]).exists(), case["image"]
'''),
        md("## Load the models\n`P2_LORA` says which Part 2 LoRA to use. Before the final merge it is Kavan's baseline. After the merge it is Muhammed's selected run, built in from `eval/p2_final_model.json`. A copy on this machine from an earlier Part 2 run is used if it exists, otherwise the LoRA downloads from Hugging Face. If neither works, domain requests return a clear error instead of crashing. The output below shows which LoRA we used."),
        code(f'''
P2_LORA = {P2_CONFIG!r}  # built in from eval/p2_final_model.json; None means Kavan's baseline (DEFAULT_LORA)
if os.environ.get("P2_LORA_REPO"):  # handy at the demo: point at another LoRA without editing code
    P2_LORA = {{**(P2_LORA or DEFAULT_LORA), "hf_repo": os.environ["P2_LORA_REPO"], "local_path": None}}
hf_token()  # only needed if the LoRA repo is private
LORA = find_lora([WORK, OUT_ROOT], P2_LORA)
hub = ModelHub(LORA)
print("Part 2 LoRA:", LORA)
print("GPU memory after loading (GB):", round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else "cpu")


def run_case(case, use_critique, tag):
    image = str(IMAGES / case["image"]) if case.get("image") else None
    state = RequestState(request_id=f"{{tag}}_{{case['id']}}", user_text=case["request"], image_path=image, questions=case.get("questions", []), use_critique=use_critique, seed=SEED)
    start = time.perf_counter()
    state = run_request(state, hub, GENERATED / tag, TRACES / tag)
    return state, time.perf_counter() - start
'''),
        md("## One request, end to end\nWe look at one full trace first, because that is the easiest way to see what each agent did."),
        code('''
state, seconds = run_case(scenarios[3], True, "first_look")
print("route:", state.route, "| status:", state.final_status, "| revisions:", state.revisions, "| seconds:", round(seconds, 1))
print("prompt:", state.prompt)
for event in state.trace:
    print(f"{event.agent:<11} {event.seconds:7.2f}s  {str(event.output)[:150]}")
if state.final_image:
    display(Image.open(state.final_image).resize((384, 384)))
'''),
        md("## Run the 15 scenarios with and without critique\n`final` has the critique loop. `baseline` is the same system without it (the output safety filter stays on in both). Both use the same seeds."),
        code('''
RUNS = {}
for tag, use_critique in (("baseline", False), ("final", True)):
    rows = []
    for case in scenarios:
        state, seconds = run_case(case, use_critique, tag)
        checks = auto_check(case, state)
        rows.append({"id": case["id"], "category": case["category"], "route": state.route, "status": state.final_status, "revisions": state.revisions, "seconds": seconds,
                     "auto_checks_ok": all(checks.values()), "failed_checks": [k for k, v in checks.items() if not v], "image": state.final_image,
                     "critic_decisions": [c.decision for c in state.critiques], "clip_scores": [round(c.clip_score, 3) for c in state.critiques], "message": state.message})
    RUNS[tag] = pd.DataFrame(rows)
    print(tag, "| auto checks ok:", int(RUNS[tag].auto_checks_ok.sum()), "of 15 | median seconds:", round(statistics.median(RUNS[tag].seconds), 2))
    display(RUNS[tag].drop(columns=["image", "message"]))
'''),
        md("## Images for the human review\nThese are the outputs Kavan and Muhammed judge against the rubric, independently. Nothing below is scored by a machine."),
        code('''
import matplotlib.pyplot as plt

made = [c for c in scenarios if RUNS["final"].set_index("id").loc[c["id"], "image"]]
fig, axes = plt.subplots(max(len(made), 1), 2, figsize=(7, 3.6 * max(len(made), 1)), squeeze=False)
for row, case in enumerate(made):
    for col, tag in enumerate(("baseline", "final")):
        path = RUNS[tag].set_index("id").loc[case["id"], "image"]
        if path:
            axes[row, col].imshow(Image.open(path))
        axes[row, col].set_title(f"{case['id']} {tag}: {case['request'][:36]}", fontsize=8)
        axes[row, col].axis("off")
plt.tight_layout()
plt.show()
for case in scenarios:
    if case["category"] == "caption_question" or not RUNS["final"].set_index("id").loc[case["id"], "image"]:
        print(case["id"], "|", case["request"], "->", RUNS["final"].set_index("id").loc[case["id"], "message"])
review = pd.DataFrame([{"scenario_id": c["id"], "request": c["request"], "rubric": c["rubric"], **{f"{t}_image": RUNS[t].set_index("id").loc[c["id"], "image"] for t in RUNS}} for c in scenarios])
review.to_csv(P3 / "review_sheet.csv", index=False)
print("review sheet:", P3 / "review_sheet.csv")
for c in scenarios[:3]:
    trace = json.loads((TRACES / "final" / f"final_{c['id']}.json").read_text())
    print(c["id"], "vision:", trace["vision"])
'''),
        md("## Red-team cases\nAn attack counts as a success if the system produced what it should have refused, repeated a forbidden claim, or crashed. The code decides this from the trace, and we read the output below to double check it."),
        code('''
rows = []
for case in red_team:
    state, seconds = run_case(case, True, "red_team")
    rows.append({"id": case["id"], "category": case["category"], "route": state.route, "status": state.final_status, "attack_succeeded": attack_succeeded(case, state), "seconds": seconds, "message": state.message})
RED = pd.DataFrame(rows)
print("attack successes:", int(RED.attack_succeeded.sum()), "of", len(RED))
pd.set_option("display.max_colwidth", 110)
RED
'''),
        md("## The Gradio demo\nThis shows each agent's output for a request. For the demo, set `KEEP_APP_OPEN = True`, or `LAB2_KEEP_GRADIO=1` in the environment, and rerun this cell. By default we start it, test one request and close it, so Run All finishes."),
        inline("p3_app.py"),
        code('''
KEEP_APP_OPEN = os.environ.get("LAB2_KEEP_GRADIO") == "1"
demo = build_app(hub, GENERATED / "app", TRACES / "app")
# a fresh test through the same function the buttons call
outputs = handle_request(hub, GENERATED / "app", TRACES / "app", str(IMAGES / "coco_bear.jpg"), "Describe this image.", True)
print("app test status:", outputs[0])
outputs = handle_request(hub, GENERATED / "app", TRACES / "app", str(IMAGES / "broken_empty.jpg"), "Describe this image.", True)
print("app test on a broken file:", outputs[0])
demo.launch(share=ON_KAGGLE or ON_COLAB, prevent_thread_lock=not KEEP_APP_OPEN)
if not KEEP_APP_OPEN:
    demo.close()
    print("app started and closed fine")
'''),
        md("## Results table\nWe save the raw runs to `reports/results/p3_runs.json`, then build the table. The human numbers come from the forms in `eval/human/`. Right after this run they print as `PENDING` and are stored as null. Once both forms are filled in, `p3_results.ipynb` recomputes the table from the saved runs on a laptop, with no GPU. Nothing here is estimated."),
        inline("p3_results.py"),
        code('''
runs_path = OUT_ROOT / "reports" / "results" / "p3_runs.json"
runs_path.parent.mkdir(parents=True, exist_ok=True)
# to_json handles numpy types and keeps lists, so p3_results.ipynb reads back exactly what we saw
runs_path.write_text(json.dumps({"baseline": json.loads(RUNS["baseline"].to_json(orient="records")),
                                 "final": json.loads(RUNS["final"].to_json(orient="records")),
                                 "red_team": json.loads(RED.to_json(orient="records"))}, indent=1))
results = compute_results(RUNS, RED, EVAL / "human")
(OUT_ROOT / "reports" / "results" / "p3.json").write_text(json.dumps(results, indent=2))
print(json.dumps({k: results[k] for k in ("red_team_attack_success", "red_team_cases")}))
results_table(results)
'''),
        md("## Upload the test images to Hugging Face\nThe brief wants the Part 3 inputs on Hugging Face, and `links.txt` points there. If no token is found we say so and carry on."),
        code('''
HF_INPUTS_REPO = os.environ.get("HF_P3_INPUTS_REPO", "Kavan1804/data266-p3-inputs")
upload = {"hf_repo": HF_INPUTS_REPO, "uploaded": False}
token = hf_token()
if not token:
    print("NOT uploaded: no Hugging Face token found.")
else:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(HF_INPUTS_REPO, repo_type="dataset", private=False, exist_ok=True)
        api.upload_folder(repo_id=HF_INPUTS_REPO, repo_type="dataset", folder_path=str(IMAGES))
        upload["uploaded"] = True
        print("uploaded to https://huggingface.co/datasets/" + HF_INPUTS_REPO)
    except Exception as err:
        print("UPLOAD FAILED:", type(err).__name__, err)
(OUT_ROOT / "reports" / "results" / "p3_inputs_upload.json").write_text(json.dumps(upload, indent=2))
'''),
        last_cell("p3_agents", "DATA266_Lab2_p3_submission.zip", ["traces", "outputs/p3/generated", "outputs/p3/review_sheet.csv", "reports/results/p3.json", "reports/results/p3_runs.json",
                  "reports/results/p3_inputs_upload.json"]),
    ]
    # imports used across cells and repeated in the inlined source
    save(cells, "p3_agents")


# ---------------------------------------------------------------- Part 3 human results (laptop, no GPU)
def build_p3_results():
    cells = [
        first_cell(["pandas"]),
        md("""
# DATA 266 Lab 2, Part 3 results with the human forms (laptop, no GPU)

`p3_agents.ipynb` runs on the GPU and saves every scenario and red-team run to `reports/results/p3_runs.json`. The human pass/fail judgments happen after that run, so this small notebook recomputes the Part 3 results table from the saved runs plus `eval/human/p3_passfail_kavan.csv` and `p3_passfail_muhammed.csv`. It loads no models. Run it in the repo once both forms are filled in. Disagreements go in `eval/human/p3_resolved.csv`.
"""),
        code("%pip install -q pandas"),
        code(SETUP),
        inline("p3_results.py"),
        code('''
import json
import pandas as pd

assert REPO, "run this notebook inside the repo, after unzipping the p3_agents ZIP into it"
saved = json.loads((REPO / "reports" / "results" / "p3_runs.json").read_text())
RUNS = {tag: pd.DataFrame(saved[tag]) for tag in ("baseline", "final")}
RED = pd.DataFrame(saved["red_team"])
results = compute_results(RUNS, RED, REPO / "eval" / "human")
(REPO / "reports" / "results" / "p3.json").write_text(json.dumps(results, indent=2))
print(json.dumps({k: results[k] for k in ("red_team_attack_success", "red_team_cases")}))
results_table(results)
'''),
        last_cell("p3_results", "DATA266_Lab2_p3_results.zip", ["reports/results/p3.json", "eval/human"]),
    ]
    save(cells, "p3_results")


if __name__ == "__main__":
    build_p1()
    build_p1_eval()
    build_p2()
    build_p3()
    build_p3_results()
