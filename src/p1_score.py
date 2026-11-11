"""Part 1 golden-set runner and scorer, shared by both members. No models, only pandas.

Each member's Part 1 notebook calls run_golden() and writes one prediction file per
(system, golden set, mode) to reports/results/p1_predictions/<system>__<goldset>__<mode>.csv.
notebooks/p1_eval.ipynb then scores every file the same way, so both systems are
compared on equal terms. Nothing in here knows any question or answer.
"""
from __future__ import annotations

import re
import statistics
import time
from pathlib import Path

import pandas as pd

GOLD_COLUMNS = ["gold_id", "conversation_id", "turn", "category", "question", "gold_answer", "required_numbers",
                "required_terms", "gold_status", "gold_sources", "gold_figure_references", "notes"]
PRED_COLUMNS = ["gold_id", "conversation_id", "turn", "answer", "status", "citations", "figure_references", "top5_sources", "seconds"]
STATUSES = {"answered", "not_found", "conflict", "false_premise", "session_override"}
# the brief's six golden-set categories and how many questions each needs
CATEGORY_QUOTA = {"fact_table": 5, "figure": 6, "two_source": 3, "multi_turn": 5, "conflict_revision": 3, "unanswerable_false_premise_injection": 3}


# ---------------------------------------------------------------- reading files
def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_gold(path: str | Path) -> pd.DataFrame:
    gold = read_csv(path)
    for col in GOLD_COLUMNS:
        if col not in gold.columns:
            gold[col] = ""  # older files may not have required_terms
    return gold[GOLD_COLUMNS]


def check_gold(gold: pd.DataFrame) -> list[str]:
    """Problems with a golden set, as plain sentences. An empty list means it follows the brief."""
    problems = []
    if len(gold) != 25:
        problems.append(f"has {len(gold)} questions, the brief needs 25")
    if gold.gold_id.duplicated().any():
        problems.append("gold_id values are not unique")
    counts = gold.category.value_counts().to_dict()
    for cat, need in CATEGORY_QUOTA.items():
        if counts.get(cat, 0) < need:
            problems.append(f"category {cat}: {counts.get(cat, 0)} of {need}")
    unknown = set(counts) - set(CATEGORY_QUOTA)
    if unknown:
        problems.append(f"unknown categories: {sorted(unknown)}")
    multi = gold[gold.category == "multi_turn"].conversation_id.nunique()
    if multi < 2:
        problems.append(f"multi_turn questions use {multi} conversation(s), the brief needs at least 2")
    bad = set(gold.gold_status) - STATUSES
    if bad:
        problems.append(f"invalid gold_status values: {sorted(bad)}")
    return problems


# ---------------------------------------------------------------- running a system
def ordered_turns(gold: pd.DataFrame):
    """Conversations in file order, turns in numeric order, like the Kaggle loop."""
    for conv_id, turns in gold.groupby("conversation_id", sort=False):
        yield conv_id, turns.sort_values("turn", key=lambda t: t.astype(int))


def run_golden(gold: pd.DataFrame, new_conversation, retrieve_sources, mode: str) -> pd.DataFrame:
    """Runs one system on one golden set.

    new_conversation(conversation_id, mode) -> object with .answer(question) -> dict (the required JSON)
    retrieve_sources(query, k, mode) -> list of source strings like "DOC1:p3" or "DOC1:p3:Figure 1-2"
    If the conversation object has .last_standalone (the follow-up rewritten as a full question),
    Recall@5 uses that as the query, otherwise the raw question.
    """
    rows = []
    for conv_id, turns in ordered_turns(gold):
        chat = new_conversation(conv_id, mode)
        for _, r in turns.iterrows():
            start = time.perf_counter()
            out = chat.answer(r["question"])
            seconds = time.perf_counter() - start
            query = getattr(chat, "last_standalone", None) or r["question"]
            top5 = retrieve_sources(query, 5, mode)
            rows.append({"gold_id": r["gold_id"], "conversation_id": conv_id, "turn": r["turn"],
                         "answer": " ".join(str(out.get("answer", "")).split()), "status": out.get("status", ""),
                         "citations": ",".join(out.get("citations", [])) or "NONE",
                         "figure_references": ",".join(out.get("figure_references", [])) or "NONE",
                         "top5_sources": ";".join(top5) or "NONE", "seconds": round(seconds, 4)})
    return pd.DataFrame(rows, columns=PRED_COLUMNS)


def prediction_path(results_dir: str | Path, system: str, goldset: str, mode: str) -> Path:
    return Path(results_dir) / "p1_predictions" / f"{system}__{goldset}__{mode}.csv"


# ---------------------------------------------------------------- matching
def split_list(cell: str, sep: str = ",") -> list[str]:
    cell = (cell or "").strip()
    return [] if cell in ("", "NONE") else [x.strip() for x in cell.split(sep) if x.strip()]


def numbers_in(text: str) -> list[float]:
    text = text.replace("−", "-").replace("–", "-")
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+", text)]


def numbers_match(required: str, answer: str) -> bool:
    """Every required number appears in the answer within +-1% (relative), as the brief says."""
    found = numbers_in(answer)
    wanted = [float(x.replace(",", "")) for x in split_list(required, ";")]  # semicolon separated, so 1,200 stays one number
    return all(any(abs(f - w) <= abs(w) * 0.01 + 1e-9 for f in found) for w in wanted)


def terms_match(required: str, answer: str) -> bool:
    """For answers without numbers: every required term (semicolon separated) appears, case-insensitive."""
    low = answer.lower()
    return all(t.lower() in low for t in split_list(required, ";"))


def parse_source(item: str) -> tuple[str, int | None, str | None]:
    """'DOC1:p3' -> (DOC1, 3, None); 'DOC1:Figure 1-2' -> (DOC1, None, 'Figure 1-2');
    'DOC1:p3:Figure 1-2' -> (DOC1, 3, 'Figure 1-2')."""
    parts = [p.strip() for p in item.split(":")]
    doc, page, label = parts[0].upper(), None, None
    for p in parts[1:]:
        m = re.fullmatch(r"p(\d+)", p, re.I)
        if m:
            page = int(m.group(1))
        elif p:
            label = p
    return doc, page, label


def source_matches(item: str, gold_item: str) -> bool:
    d1, p1, l1 = parse_source(item)
    d2, p2, l2 = parse_source(gold_item)
    if d1 != d2:
        return False
    return (p2 is not None and p1 == p2) or (l2 is not None and l1 is not None and l1.lower() == l2.lower())


# ---------------------------------------------------------------- scoring
def score(gold: pd.DataFrame, pred: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-question table plus the summary numbers for the results table."""
    merged = gold.merge(pred, on="gold_id", how="left", suffixes=("", "_pred"))
    rows = []
    for _, r in merged.iterrows():
        answer, status = str(r.get("answer") or ""), str(r.get("status") or "")
        gold_sources = split_list(r["gold_sources"])
        number_ok, term_ok = numbers_match(r["required_numbers"], answer), terms_match(r["required_terms"], answer)
        status_ok = status == r["gold_status"]
        top5 = split_list(str(r.get("top5_sources") or ""), ";")
        cites = split_list(str(r.get("citations") or ""))
        rows.append({
            "gold_id": r["gold_id"], "category": r["category"], "gold_status": r["gold_status"], "status": status,
            "status_ok": status_ok, "numbers_ok": number_ok, "terms_ok": term_ok,
            "correct": bool(status_ok and number_ok and term_ok and answer),
            # questions with no gold source (e.g. not_found) are left out of the two retrieval measures
            "recall_at_5": (any(source_matches(t, g) for t in top5 for g in gold_sources) if gold_sources else None),
            "citations_total": len(cites) if gold_sources else 0,
            "citations_supported": sum(any(source_matches(c, g) for g in gold_sources) for c in cites) if gold_sources else 0,
            "seconds": float(r["seconds"]) if str(r.get("seconds") or "") not in ("", "nan") else None,
            "answer": answer,
        })
    table = pd.DataFrame(rows)
    recall = table.recall_at_5.dropna()
    seconds = table.seconds.dropna()
    summary = {
        "n_questions": int(len(table)),
        "correct_of_25": int(table.correct.sum()),
        "status_correct": int(table.status_ok.sum()),
        "recall_at_5": float(recall.astype(bool).mean()) if len(recall) else None,
        "recall_questions": int(len(recall)),
        "citation_accuracy": float(table.citations_supported.sum() / table.citations_total.sum()) if table.citations_total.sum() else None,
        "citations_counted": int(table.citations_total.sum()),
        "median_seconds_per_question": float(statistics.median(seconds)) if len(seconds) else None,
        "missing_predictions": int(merged["status"].isna().sum() if "status" in merged else len(merged)),
    }
    return table, summary


def groundedness(forms: dict[str, pd.DataFrame], gold_ids: list[str]) -> dict:
    """Mean 1-5 rating over both raters and the share of questions where they agree within one point.
    Returns None values while any form is incomplete. Ratings only ever come from people."""
    ratings = {}
    for rater, form in forms.items():
        form = form[form.gold_id.isin(gold_ids) & form.rating.isin(["1", "2", "3", "4", "5"])]
        if len(form) != len(gold_ids):
            return {"groundedness_mean": None, "groundedness_within_one": None, "complete": False}
        ratings[rater] = form.set_index("gold_id").rating.astype(int)
    frame = pd.DataFrame(ratings)
    within = (frame.max(axis=1) - frame.min(axis=1) <= 1).mean() if frame.shape[1] >= 2 else None
    return {"groundedness_mean": float(frame.values.mean()), "groundedness_within_one": None if within is None else float(within), "complete": True}
