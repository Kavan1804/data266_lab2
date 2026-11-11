"""Writes the report's results tables from reports/results/*.json.

Report numbers must come from notebook output, so nobody types them in by hand.
Missing values print as PENDING.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS, TABLES = HERE / "results", HERE / "tables"
TABLES.mkdir(exist_ok=True)


def load(name):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else {}


def show(value, fmt="{:.2f}"):
    if value is None:
        return r"\textit{PENDING}"
    return fmt.format(value) if isinstance(value, float) else str(value)


def table(name, caption, label, header, rows):
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    spec = "l" + "c" * (len(header) - 1)
    (TABLES / f"{name}.tex").write_text(
        f"\\begin{{table}}[!t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}{{{spec}}}\n\\hline\n"
        + " & ".join(header) + " \\\\\n\\hline\n" + body + " \\\\\n\\hline\n\\end{tabular}\n\\end{table}\n")


def p1():
    data = load("p1_eval.json")
    base, final = data.get("baseline", {}), data.get("final", {})
    measures = [("Correct out of 25", "correct_of_25"), ("Recall@5", "recall_at_5"), ("Citation accuracy", "citation_accuracy"),
                ("Groundedness mean (1--5)", "groundedness_mean"), ("Agreement within one point", "groundedness_within_one"),
                ("Median seconds per question", "median_seconds_per_question")]
    rows = [[label, show(base.get(key)), show(final.get(key))] for label, key in measures]
    table("p1", "Part 1 results on our 25 golden questions", "tab:p1", ["Measure", "Dense baseline", "Final hybrid"], rows)


def p2():
    data = load("p2.json")
    models = [data.get(k, {}) for k in ("base", "run1", "run2")]
    measures = [("KID (mean)", "kid_mean", "{:.4f}"), ("KID (std)", "kid_std", "{:.4f}"), ("CLIP score", "clip_score", "{:.3f}"), ("LPIPS diversity", "lpips_diversity", "{:.3f}"),
                ("Trainable parameters", "trainable_params", "{}"), ("Learning rate", "learning_rate", "{}"), ("Steps", "steps", "{}"), ("Batch size", "batch_size", "{}"),
                ("GPU-hours", "gpu_hours", "{:.2f}"), ("Peak GPU memory (GB)", "peak_gpu_gb", "{:.2f}"), ("Win rate, style", "win_rate_style", "{:.2f}"), ("Win rate, prompt", "win_rate_prompt", "{:.2f}")]
    rows = [[label] + [show(m.get(key), fmt) for m in models] for label, key, fmt in measures]
    table("p2", "Part 2 results", "tab:p2", ["Measure", "Base", "Run 1", "Run 2"], rows)


def p3():
    data = load("p3.json")
    base, final = data.get("baseline", {}), data.get("final", {})
    rows = [["Scenarios passed (of 15)", show(base.get("scenarios_passed")), show(final.get("scenarios_passed"))],
            ["Human pass/fail agreement", show(base.get("human_agreement")), show(final.get("human_agreement"))],
            ["Critic agreement", "n/a", show(final.get("critic_agreement"))],
            ["Median seconds per request", show(base.get("median_seconds_per_request")), show(final.get("median_seconds_per_request"))],
            ["Red-team attacks that succeeded", "n/a", f"{show(data.get('red_team_attack_success'))} of {show(data.get('red_team_cases'))}"]]
    table("p3", "Part 3 results", "tab:p3", ["Measure", "No critique", "With critique"], rows)


if __name__ == "__main__":
    p1(), p2(), p3()
    print("wrote tables to", TABLES)
