"""Part 3 results table: runs plus the human pass/fail forms -> reports/results/p3.json.

p3_agents.ipynb calls this right after the GPU run (human values print PENDING),
and p3_results.ipynb calls it again on a laptop once both forms are filled in.
Human values only ever come from the forms. Nothing here estimates them.
"""
from __future__ import annotations

import statistics
from pathlib import Path

import pandas as pd


def read_form(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=["scenario_id", "mode", "pass", "critic_agree", "note"])
    form = pd.read_csv(path, dtype=str, keep_default_na=False)
    return form[form["pass"].isin(["0", "1"])]


def compute_results(runs: dict[str, pd.DataFrame], red: pd.DataFrame, human_dir: Path, raters=("kavan", "muhammed"), n_scenarios: int = 15) -> dict:
    human_dir = Path(human_dir)
    forms = {who: read_form(human_dir / f"p3_passfail_{who}.csv") for who in raters}
    complete = all(len(f) == 2 * n_scenarios for f in forms.values())
    results = {"baseline": {}, "final": {}, "red_team_attack_success": int(red.attack_succeeded.astype(bool).sum()), "red_team_cases": int(len(red))}
    for tag in ("baseline", "final"):
        results[tag].update({"median_seconds_per_request": float(statistics.median(runs[tag].seconds)),
                             "auto_checks_ok": int(runs[tag].auto_checks_ok.astype(bool).sum()),
                             "mean_revisions": float(runs[tag].revisions.mean())})
    if not complete:
        print("PENDING: human pass/fail forms are not complete, so scenarios passed, human agreement and critic agreement are not computed.")
        for tag in ("baseline", "final"):
            results[tag].update({"scenarios_passed": None, "human_agreement": None})
        results["final"]["critic_agreement"] = None
        return results

    a, b = raters
    merged = forms[a].merge(forms[b], on=["scenario_id", "mode"], suffixes=("_a", "_b"))
    merged["agree"] = merged.pass_a == merged.pass_b
    # disagreements are settled together and the agreed result goes in eval/human/p3_resolved.csv
    resolved_path = human_dir / "p3_resolved.csv"
    resolved = pd.read_csv(resolved_path, dtype=str) if resolved_path.exists() else None
    for tag in ("baseline", "final"):
        part = merged[merged["mode"] == tag]
        results[tag]["human_agreement"] = float(part.agree.mean())
        unresolved = part[~part.agree]
        if len(unresolved) and resolved is None:
            results[tag]["scenarios_passed"] = None
            print("PENDING:", tag, "has disagreements, add eval/human/p3_resolved.csv (scenario_id, mode, pass)")
        else:
            final_pass = part.set_index("scenario_id").pass_a.copy()
            if len(unresolved):
                final_pass.update(resolved[resolved["mode"] == tag].set_index("scenario_id")["pass"])
            results[tag]["scenarios_passed"] = int((final_pass == "1").sum())

    # critic agreement only counts scenarios where the critic actually made a decision
    decisions = runs["final"].set_index("id")
    made_ids = [i for i in decisions.index if len(decisions.loc[i, "critic_decisions"] or [])]
    votes = []
    for who in raters:
        f = forms[who][(forms[who]["mode"] == "final") & (forms[who].scenario_id.isin(made_ids))]
        votes += [v == "1" for v in f.critic_agree if v in ("0", "1")]
    results["final"]["critic_agreement"] = float(sum(votes) / len(votes)) if votes else None
    return results


def results_table(results: dict) -> pd.DataFrame:
    return pd.DataFrame({tag: results[tag] for tag in ("baseline", "final")}).astype(object).where(lambda d: d.notna(), "PENDING")
