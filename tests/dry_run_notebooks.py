"""Runs notebook cells with FAKE models in a throwaway copy of the repo.

This finds bugs like missing imports or broken tables before GPU time is spent.
Nothing it prints or writes is a result. It never touches the real repo folders,
and it never saves outputs into the real notebooks.

python tests/dry_run_notebooks.py p1 p3
"""
import builtins
import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))


class FakeP1Models:
    def __init__(self, *a, **k):
        pass

    def text(self, prompt, system="", max_new_tokens=0):
        return json.dumps({"answer": "fake answer", "status": "answered", "sources": ["S1"]}) if "JSON:" in prompt else "fake answer"

    def vision(self, prompt, path, system="", max_new_tokens=0):
        return "fake chart description"


def run_cells(nb_path, tmp, patches, stop_after=None):
    nb = nbformat.read(nb_path, as_version=4)
    space = {"__name__": "__main__", "display": lambda *a, **k: None}
    os.chdir(tmp / "notebooks")
    import matplotlib
    matplotlib.use("Agg")
    for i, cell in enumerate(c for c in nb.cells if c.cell_type == "code"):
        source = re.sub(r"^\s*[%!].*$", "", cell.source, flags=re.M)
        for name, value in patches.get("before", {}).items():
            if name in space or i == 0:
                space[name] = value
        try:
            exec(compile(source, f"{nb_path.name}[cell {i}]", "exec"), space)
        except Exception:
            print(f"FAILED in {nb_path.name} code cell {i}:\n{source[:300]}\n")
            raise
        for name, value in patches.get("after", {}).items():
            if name in space:
                space[name] = value
    return space


def fresh_copy(tmp):
    for name in ("notebooks", "src", "eval"):
        shutil.copytree(ROOT / name, tmp / name)
    data = next(p for p in (ROOT / "DATA266_Lab2_Part1", ROOT / "data/part1") if p.exists())
    shutil.copytree(data, tmp / "DATA266_Lab2_Part1")


def dry_p1(tmp):
    from p1_engine import Embedder

    class CpuEmbedder(Embedder):
        def __init__(self, model_id):
            super().__init__(model_id, device="cpu")

    space = run_cells(tmp / "notebooks/p1_rag_kavan.ipynb", tmp, {"after": {"QwenModels": FakeP1Models, "Embedder": CpuEmbedder, "hf_token": lambda: None}})
    assert (tmp / "submission.csv").exists()
    assert (tmp / "DATA266_Lab2_p1_kavan.zip").exists()
    assert space["engine"].index_count() == space["START_COUNT"]
    print("p1 dry run ok")


def dry_p3(tmp):
    from test_offline import FakeHub
    from PIL import Image

    class Hub(FakeHub):
        def __init__(self, *a, **k):
            super().__init__()
            self.lora = {"source": "fake"}

    def fake_download(folder):
        import p3_images
        for name in p3_images.COCO:
            Image.effect_noise((256, 256), 90).convert("RGB").save(folder / name)
        for n in (1, 2, 3):
            Image.effect_noise((256, 256), 90).convert("RGB").save(folder / f"domain_test_{n}.png")

    space = run_cells(tmp / "notebooks/p3_agents.ipynb", tmp, {"after": {"ModelHub": Hub, "download_coco": fake_download, "hf_token": lambda: None}})
    results = json.loads((tmp / "reports/results/p3.json").read_text())
    assert results["final"]["scenarios_passed"] is None, "blank forms must stay pending"
    assert results["red_team_attack_success"] == 0
    assert len(list((tmp / "traces/final").glob("*.json"))) == 15 and len(list((tmp / "traces/red_team").glob("*.json"))) == 10
    assert (tmp / "DATA266_Lab2_p3_submission.zip").exists()
    # now fill the forms with made-up test votes and check that the table really computes from them
    import pandas as pd
    for who, flip in (("kavan", False), ("muhammed", True)):
        path = tmp / "eval/human" / f"p3_passfail_{who}.csv"
        form = pd.read_csv(path, dtype=str, keep_default_na=False)
        form["pass"] = ["1"] * len(form)
        form["critic_agree"] = ["1"] * len(form)
        if flip:
            form.loc[0, "pass"] = "0"  # one disagreement on S01 in the baseline
        form.to_csv(path, index=False)
    nb = nbformat.read(nb_path := tmp / "notebooks/p3_agents.ipynb", as_version=4)
    cell = next(c.source for c in nb.cells if c.cell_type == "code" and "compute_results(RUNS" in c.source)
    exec(compile(cell, "results cell", "exec"), space)
    results = json.loads((tmp / "reports/results/p3.json").read_text())
    assert results["baseline"]["scenarios_passed"] is None, "a disagreement must wait for the resolved file"
    assert results["final"]["scenarios_passed"] == 15 and results["final"]["human_agreement"] == 1.0
    assert abs(results["baseline"]["human_agreement"] - 14 / 15) < 1e-9 and results["final"]["critic_agreement"] == 1.0
    (tmp / "eval/human/p3_resolved.csv").write_text("scenario_id,mode,pass\nS01,baseline,1\n")
    exec(compile(cell, "results cell", "exec"), space)
    assert json.loads((tmp / "reports/results/p3.json").read_text())["baseline"]["scenarios_passed"] == 15
    print("p3 dry run ok")


if __name__ == "__main__":
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    for part in sys.argv[1:] or ["p1", "p3"]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp).resolve()
            fresh_copy(tmp)
            {"p1": dry_p1, "p3": dry_p3}[part](tmp)
            os.chdir(ROOT)
