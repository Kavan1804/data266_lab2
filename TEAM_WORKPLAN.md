# Team 24 Work Plan, DATA 266 Lab 2

**Pair_Programming_Team_24:** Kavan Siddesh Kumar and Muhammed Faiq Salman
**Written:** Mon Nov 9, 2026 | **Due:** Tue Nov 17, 2026, 11:59 PM PT | **Demo:** Nov 18–24

This is our only planning document. We want to work fully on our own from today until the final merge, without passing half-finished work back and forth. So everything one of us needs from the other is fixed here up front: file names, folders, formats and dates. If this file and the brief (`DATA266_Lab2_Requirements.pdf`) ever disagree, the brief wins.

This replaces the older `DATA266_Lab2_Handoff_Muhammed.pdf` from Nov 5. That PDF split Part 1 into halves. We changed the plan, so each of us now builds a complete Part 1.

**How we run things:** every notebook is **standalone**. It does not need the repo around it, so you can open just the one `.ipynb` in Colab (we use Colab on an RTX 4090) or a local Jupyter and press Run all. Anything a notebook needs is either built into it (code, test sets, scenarios) or downloaded by it (models, datasets). The only exception is the Part 1 PDFs, which we never commit. The last cell downloads a ZIP laid out like the repo, so you unzip it at the repo root and push. Part 1 must also run on Kaggle, because the final submission and the graders' hidden-set run happen there (Section 6.2).

---

## 1. The lab in one page

| Part | What we build | Marks |
|---|---|---|
| **1. Multimodal RAG** | Answer questions about 3 economics PDFs (text, tables, figures) with JSON answers and citations, following strict rules: revisions, conflicts, an untrusted appendix, and user values for one chat only. Kaggle competition, ranked on a hidden question set. | 25 + 10 |
| **2. Diffusion fine-tuning** | Fine-tune SD 1.5 on Naruto BLIP captions with two LoRA runs, generate 120 images, measure KID, CLIP, LPIPS and a blinded human study, plus safety and a model card. | 27 |
| **3. Multi-agent system** | Planner → Vision → Prompt → Generation → Critique (at most 2 revisions), with a typed state, JSON traces, a Gradio app, 15 scenarios and 10 red-team cases. Domain-style requests use the Part 2 LoRA. | 28 |
| Demo | 15 minutes, both of us, run from the tagged code. **Each of us must be able to explain any part, including the other person's code.** | 10 |

**Overall architecture**

```
Part 1:  PDFs + kb_manifest.json -> chunks, figures, tables -> vector DB + BM25 -> answer rules -> JSON answer
Part 2:  Naruto dataset -> clean, split 90/10 -> LoRA run 1 and run 2 -> 20 prompts x 2 seeds -> metrics + human study
                                                       |
                                                       v  eval/p2_final_model.json (the only link between parts)
Part 3:  request + image -> Planner -> Vision -> Prompt -> Generation (img2img / pix2pix / Part 2 LoRA) -> Critique -> trace
```

**Rules that cost the most if we break them** (from the brief):
- A notebook without saved outputs earns at most 50%.
- A report number that does not match a notebook output gets 0 for that item.
- Hard-coded Part 1 answers or question-specific rules get 0 for Tasks 1.3 and 1.5.
- Fabricated results get 0 for the whole part, plus an SJSU integrity referral.
- Test questions and prompts must be **committed before tuning**, because the Git timestamp is checked.
- Everything must fit on one 16 GB GPU.
- Never commit PDFs, datasets, weights, images or tokens to GitHub.

---

## 2. Who does what

The brief says both of us work on all three parts. In Parts 2 and 3, the member who is not leading still builds and runs the full pipeline, at least as a baseline. Here is how we split it:

| | **Kavan** | **Muhammed** |
|---|---|---|
| **Part 1** | Complete system on his own: `p1_rag_kavan.ipynb` | Complete system on his own: `p1_rag_muhammed.ipynb` |
| **Part 2** | Own baseline: `p2_baseline_kavan.ipynb` (one LoRA, generate, evaluate) | **Lead.** Complete Part 2: `p2_finetune.ipynb` |
| **Part 3** | **Lead.** Complete Part 3: `p3_agents.ipynb` | Own baseline: `p3_baseline_muhammed.ipynb` (linear pipeline) |
| **Shared at the end** | Part 1 scoring (`p1_eval.ipynb`, already written), report assembly, tags | Kaggle Part 1 run, if his system is the one selected |
| **Both, personally** | Human ratings (Section 8), recorded calls, own rows in `CONTRIBUTIONS.md` and `AI_USAGE.md`, own report text | Same |

**At the final merge:**
- **Part 1:** we compare both complete systems on both golden sets and pick the stronger one, or combine them (Section 9).
- **Part 2:** Muhammed's `p2_finetune.ipynb` is the final system. Kavan's baseline stays in the repo as reproduction evidence.
- **Part 3:** Kavan's `p3_agents.ipynb` is the final system, rerun with Muhammed's final LoRA. Muhammed's baseline stays as reproduction evidence.

---

## 3. Where Kavan is right now (Nov 9)

**Already done (code only, nothing has run on a GPU yet; updated Nov 10):**
- [x] `src/p1_engine.py`: Part 1 ingestion, 36 numbered figures and tables, typed facts from tables, ChromaDB + BM25 hybrid retrieval, exact figure lookup, precedence rules, JSON answers, injection stripping.
- [x] `notebooks/p1_rag_kavan.ipynb`: generated from `src/`. It does the Kaggle run, validation, a tuning set, and golden-set prediction files.
- [x] `notebooks/p2_baseline_kavan.ipynb`: data cleaning and split, one LoRA (rank 4, 1500 steps), base vs LoRA generation, CLIP, LPIPS and KID.
- [x] `notebooks/p3_agents.ipynb` with `src/p3_agents.py`, `src/p3_models.py`, `src/p3_images.py` and `src/p3_app.py`: five agents, typed state, traces, critique loop, red-team scoring, Gradio app.
- [x] `eval/p1_test_prompts.csv` (Kavan's Part 1 tuning set), `eval/p3_scenarios.json` (15) and `eval/p3_red_team_cases.json` (10).
- [x] Shared scoring: `src/p1_score.py` and `notebooks/p1_eval.ipynb`.
- [x] Offline tests with fake models (`tests/`) and repo checks (`tools/`).
- [x] Nov 10: made all notebooks standalone (Colab, Kaggle or local), with a repo-shaped ZIP in the last cell. The Part 2 LoRA now reaches Part 3 through Hugging Face. Added `p3_results.ipynb` (human numbers, laptop).
- [x] Nov 10: fixed two Part 1 parser bugs that showed up with the pinned PyMuPDF 1.25.5: (1) a table's caption row was read as a year column, so Tables 1-2 to 1-4 gave no facts and every Table 1-1 row was labelled "World"; (2) kerned captions ("T", "able A-3") hid Table A-3. Now all 36 objects and 172 facts parse.

**Doing next, in order:**
1. [ ] Push this repo and share it with Muhammed (today).
2. [ ] Write `eval/p1_golden_kavan.csv` (25 questions, checked by hand against the PDFs) and commit it **before any more Part 1 tuning**. Due Nov 10.
3. [ ] Write `src/p1_conversation.py`, the real conversation memory, to replace the placeholder: follow-up rewriting, user values kept per chat, refusing permanent changes.
4. [ ] Run `p1_rag_kavan.ipynb` on the GPU, tune on `Questions.csv` and the tuning set only, and make the first Kaggle submissions.
5. [ ] Run `p2_baseline_kavan.ipynb` on the 4090. It trains the baseline LoRA and uploads it to `Kavan1804/data266-p2-baseline-kavan-lora` (needs `HF_TOKEN` as a Colab secret).
6. [ ] Run `p3_agents.ipynb`, which pulls that LoRA from Hugging Face, fix what breaks, and let it upload the test images to `Kavan1804/data266-p3-inputs`.
7. [ ] Write my report sections in my own words (`reports/sections/p1_kavan.tex`, `p3.tex`, Part 3 system card).
8. [ ] Freeze and push by Sat Nov 14, 11:59 PM.

**Muhammed has not started yet.** Everything for him is in Section 4 and in `PARTNER_LLM_PROMPT.md`.

---

## 4. Muhammed's tasks, in order

### Step 0: Setup (Nov 9–10)
- [ ] Accept the GitHub invite to **https://github.com/Kavan1804/data266_lab2** (private), then:
  ```bash
  git clone https://github.com/Kavan1804/data266_lab2.git
  cd data266_lab2
  git checkout -b muhammed
  git push -u origin muhammed
  ```
  Work **only on branch `muhammed`**. Never push to `main` or `kavan`.
- [ ] Download from Canvas: `DATA266_Lab2_Requirements.pdf` and the Part 1 files. Put the Part 1 files in `data/part1/`, which is gitignored. PDFs are never committed, so they are not in the repo.
- [ ] Join the Kaggle competition (`kaggle.com/t/e7463a4483964d5a9a41169a67416325`) and merge into **Pair_Programming_Team_24**. We share **5 submissions a day** as a team, so Muhammed uses at most 2 a day and Kavan at most 3.
- [ ] Make a Hugging Face account (or use your existing one) and a **write token**. Keep the token in the `HF_TOKEN` environment variable. **Never put it in a notebook or in Git.**
- [ ] Each notebook installs its own pinned packages, so there is nothing to set up for Colab. Keep the versions in `requirements.txt` and list anything extra in `requirements/muhammed.txt`.
- [ ] Put the Part 1 files somewhere the notebook looks: next to the notebook, in `/content`, or in `~/data266_lab2_data/` (a ZIP of them works too). On Kaggle, attach the competition data.

### Step 1: Commit the test sets before any tuning (by Tue Nov 10)
- [ ] `eval/p1_golden_muhammed.csv`: your own 25 golden questions (format in Section 6.1). **Do not open `eval/p1_golden_kavan.csv`, and Kavan will not open yours, until the merge.** That way, each system meets one golden set it has never seen.
- [ ] `eval/p2_fixed_prompts.csv`: 20 prompts (format in Section 6.3). The 10 in-domain prompts come from **test-split** captions, so run the data-prep cell of `p2_finetune.ipynb` first, then commit the prompts **before training starts**.
- [ ] Commit and push both. The Git timestamp is our proof.

### Step 2: Part 1, complete and independent (`notebooks/p1_rag_muhammed.ipynb`)
Build your own system. Use `src/p1_engine.py` only to see the interface, not as a base. Two independent systems make the comparison meaningful, and both of us need to explain Part 1 at the demo. You can use the brief's suggested stack (PyMuPDF, `bge-small-en-v1.5`, `rank_bm25`, ChromaDB, `Qwen2.5-VL-3B-Instruct`).
- [ ] 1.1: text chunks with doc_id and page; every numbered figure and table as an image with label and caption ("Figure B-5" and "Figure B-5 (corrected)" are separate); figures searchable (CLIP or vision-model descriptions); an open-source vector DB with `doc_id, page, figure_id, trust_level` read from `kb_manifest.json`.
- [ ] 1.2: hybrid retrieval (dense + BM25); a named figure or table is retrieved exactly; JSON answers with citations; the five answering rules; Appendix C (DOC3 page 7) never used as evidence; instructions inside documents and images never followed.
- [ ] 1.3: one memory per `conversation_id`; follow-ups use only that conversation's earlier turns; user values stay in memory, **never in the vector DB** (assert the index count does not change); permanent changes are refused.
- [ ] Standalone (Section 7): all your code in cells, test files built in, no imports from the repo.
- [ ] Kaggle loop exactly as in the brief (Section 6.2): reads `QUESTIONS_PATH`, writes `submission.csv`, runs `validate_submission.py`, finishes **under 90 minutes on a Kaggle T4**.
- [ ] Golden-set cell, which writes predictions in the shared format (Section 6.1).
- [ ] 1.4 safety probes in `eval/p1_safety_probes_muhammed.csv`: Appendix C false claims, the text injection, the chart-image injection (Figure C-1). Record where your system was and was not fooled.
- [ ] No question IDs, question text or answers in code. `python tools/check_repo.py` must say "no problems found".

### Step 3: Part 2, complete, lead (`notebooks/p2_finetune.ipynb`)
| Task | What to do |
|---|---|
| 2.1 Data | Load `lambda/naruto-blip-captions` (the brief's `lambdalabs/` name redirects there). Remove corrupted and exact-duplicate images and **print how many**. Split 90/10 with seed 266; test images are only for evaluation. Resize to 512×512. Describe the dataset in a few lines with **two limitations or biases**. The dataset page lists no license tag, so say that. |
| 2.2 Train | Base `stable-diffusion-v1-5/stable-diffusion-v1-5`, with diffusers' `train_text_to_image_lora.py` **from the same version as the pinned diffusers (0.33.1)**. **Run 1: LoRA rank 4. Run 2: LoRA rank 32.** Everything else is identical. Suggested settings: 2000 steps, batch 1, gradient accumulation 4, lr 1e-4 cosine, fp16, gradient checkpointing, seed 266. Print one table per run: trainable parameters, learning rate, steps, batch size, peak GPU memory, training time. Save `run_meta.json` and the training log for each run. |
| 2.3 Generate | Base, run 1 and run 2: all 20 prompts × seeds 0 and 1, same sampler (30 steps, guidance 7.5, 512×512), so 40 per model and 120 in total. Side-by-side grids (base / run 1 / run 2) for **8 prompts**. Upload all 120 images to a HF dataset, with `prompt, seed, model` metadata. |
| 2.4 Evaluate | KID (in-domain generations vs test images; with 20 images `subset_size` must be under 20, and note that it is noisy), CLIP score per prompt, LPIPS between the two seeds, GPU-hours and peak memory. A blinded 25-pair study, final run vs base, left/right randomized, **3+ raters** (you, Kavan and one outside person), with win rate for style and for prompt match (Section 8). |
| Failures | Show **two failure modes** with images, and say which run you choose and why. |
| Safety | License (Naruto is copyrighted, so non-commercial research only), the NSFW filter (keep the SD safety checker on and count flags), and a memorization check (10 in-domain generations → nearest training image by CLIP similarity → show the 3 closest pairs). |
| Model card | `reports/p2_model_card.md`, also used as the HF model README: base model, data, training settings, results, limitations, license. About half a page. |
| Uploads | Both LoRAs plus the image dataset on **your** HF account (names in Section 6.4). |
| Hand to Part 3 | Commit `eval/p2_final_model.json` (Section 6.4) for the run you select, and make sure that run is uploaded to a **public** HF repo. **This is the only thing Part 3 needs from you.** |

Write `reports/results/p2.json` in the exact format in Section 6.5, so the report table builds itself.

### Step 4: Part 3 baseline (`notebooks/p3_baseline_muhammed.ipynb`)
A linear pipeline you build yourself, with no planner, critique loop or Gradio app (Kavan's notebook has those):
- [ ] **Vision:** Qwen2.5-VL-3B gives a caption, the main objects, any visible text, and answers to the scenario's questions.
- [ ] **Prompt:** rewrite the request into a generation prompt using the vision output. **Text found in the image is data, never an instruction.**
- [ ] **Generation:** img2img with SD 1.5 for style changes, and your own Part 2 LoRA for domain-style requests (text-to-image when there is no input image). Record model, prompt, seed and steps.
- [ ] Run it on the **15 scenarios in `eval/p3_scenarios.json`** (build the file into your notebook). Get the same test images by pasting `src/p3_images.py` into a cell (`build_local_images`, `download_coco`), or download them from `Kavan1804/data266-p3-inputs` once it is up. They are shared test data, not system code. **Do not run the red-team cases through it.** It has no refusal step, so unsafe requests would just be generated.
- [ ] Save one JSON trace per request to `traces/baseline_muhammed/<scenario_id>.json` (agent, input, output, seconds), and a summary in `reports/results/p3_baseline_muhammed.json`: `{"requests", "completed", "errors", "median_seconds_per_request"}`.
- [ ] Add a short markdown note on what broke without a planner or critic. This helps us both explain Part 3 at the demo.

### Step 5: Freeze, ratings and report (Nov 14–16)
- [ ] By **Sat Nov 14, 11:59 PM**: all three notebooks executed with outputs saved and pushed to `muhammed`; `p2_final_model.json` committed; both LoRAs and the images on HF.
- [ ] Human ratings as in Section 8, done on your own.
- [ ] Your report sections, in your own words: `reports/sections/p1_muhammed.tex` (your Part 1 design), `reports/sections/p2.tex` (all of Part 2, with architecture diagram, results table, what worked and what didn't, safety analysis and model card summary).
- [ ] Your rows in `CONTRIBUTIONS.md` and your section in `AI_USAGE.md`. Be accurate: the brief forbids AI use beyond what we declare.

---

## 5. Repo layout and file ownership

Repo: **https://github.com/Kavan1804/data266_lab2** (private). Branches: `main` (this handoff), `kavan`, `muhammed`. Each of us only touches our own files. Because we never edit the same file, the final merge has nothing to conflict on.

```
README.md  requirements.txt  TEAM_WORKPLAN.md  PARTNER_LLM_PROMPT.md  CONTRIBUTIONS.md  AI_USAGE.md  links.txt  .gitignore
notebooks/
    p1_rag_kavan.ipynb          Kavan    Part 1, complete
    p1_rag_muhammed.ipynb       Muhammed Part 1, complete
    p1_eval.ipynb               shared   scores both Part 1 systems (no models, run on a laptop in the repo)
    p1_rag.ipynb                final    copy of the selected Part 1, run on Kaggle at tag p1-v1.0
    p2_finetune.ipynb           Muhammed Part 2, final
    p2_baseline_kavan.ipynb     Kavan    Part 2 baseline
    p3_agents.ipynb             Kavan    Part 3, final
    p3_results.ipynb            Kavan    Part 3 human numbers from saved runs + forms (no GPU, laptop)
    p3_baseline_muhammed.ipynb  Muhammed Part 3 baseline
src/        p1_engine.py p1_conversation*.py p3_*.py (Kavan, pasted into his notebooks by tools/build_notebooks.py)
            p1_score.py (shared, frozen: paste it into a cell) | src/muhammed/ (optional, Muhammed)
eval/       test sets, prompts, scenarios, selection file, human forms (eval/human/)
traces/     Part 3 JSON traces (traces/baseline_muhammed/ is Muhammed's)
reports/    report.tex, sections/, tables/, results/ (JSON written by notebooks), build_tables.py
requirements/  kavan.txt, muhammed.txt (merged into requirements.txt at the end)
tools/ tests/  builder and checks (Kavan)
data/ outputs/ DATA266_Lab2_Part1/   gitignored: raw data, weights, generated images, caches
```

| Owner | Files |
|---|---|
| **Kavan** (branch `kavan`) | `notebooks/p1_rag_kavan.ipynb`, `p2_baseline_kavan.ipynb`, `p3_agents.ipynb`, `p3_results.ipynb`; `src/p1_engine.py`, `src/p1_conversation*.py`, `src/p3_*.py`; `tools/`, `tests/`; `eval/p1_golden_kavan.csv`, `p1_test_prompts.csv`, `p1_safety_probes_kavan.csv`, `p3_scenarios.json`, `p3_red_team_cases.json`, `eval/human/p3_passfail_kavan.csv`, `eval/human/p1_groundedness_kavan.csv`, `eval/human/p2_ratings_kavan.csv`; `traces/` (except `baseline_muhammed/`); `reports/results/p1_kaggle_run_kavan.json`, `p1_predictions/kavan__*`, `p2_baseline_kavan.json`, `p3.json`; `reports/sections/p1_kavan.tex`, `p3.tex`; `requirements/kavan.txt` |
| **Muhammed** (branch `muhammed`) | `notebooks/p1_rag_muhammed.ipynb`, `p2_finetune.ipynb`, `p3_baseline_muhammed.ipynb`; `src/muhammed/`; `eval/p1_golden_muhammed.csv`, `p1_safety_probes_muhammed.csv`, `p2_fixed_prompts.csv`, `p2_final_model.json`, `eval/human/p2_pairs.csv`, `p2_pairs_key.csv`, `p2_ratings_muhammed.csv`, `p2_ratings_outside.csv`, `eval/human/p1_groundedness_muhammed.csv`, `eval/human/p3_passfail_muhammed.csv`; `traces/baseline_muhammed/`; `reports/results/p1_kaggle_run_muhammed.json`, `p1_predictions/muhammed__*`, `p2.json`, `p3_baseline_muhammed.json`; `reports/sections/p1_muhammed.tex`, `p2.tex`, `reports/p2_model_card.md`; `requirements/muhammed.txt` |
| **Frozen, shared** (nobody changes them after today) | `src/p1_score.py`, `notebooks/p1_eval.ipynb`, `tools/check_repo.py`, the formats in Section 6 |
| **Edited together at the merge** | `README.md`, `requirements.txt`, `CONTRIBUTIONS.md`, `AI_USAGE.md` (each edits their own section), `links.txt`, `.gitignore`, `eval/p1_selection.json`, `notebooks/p1_rag.ipynb`, `reports/report.tex`, `reports/sections/p1_eval.tex` |

---

## 6. Shared interfaces (fixed today, do not change)

### 6.1 Part 1: golden sets and prediction files
**Golden set** (`eval/p1_golden_<name>.csv`), one header row, 25 rows:

```
gold_id,conversation_id,turn,category,question,gold_answer,required_numbers,required_terms,gold_status,gold_sources,gold_figure_references,notes
```
- `gold_id`: `K01..K25` (Kavan) or `M01..M25` (Muhammed). `conversation_id`: e.g. `KC01` or `MC01`; multi-turn rows share one id and count `turn` 1, 2, 3.
- `category` and the brief's counts: `fact_table` 5, `figure` 6 (including near-identical ones), `two_source` 3, `multi_turn` 5 (in at least 2 conversations, with a user-supplied value), `conflict_revision` 3, `unanswerable_false_premise_injection` 3. `check_gold()` in `p1_score.py` checks this.
- `required_numbers`: the numbers that must appear, **separated by `;`** (e.g. `2.5;1.8`). A match is within ±1%.
- `required_terms`: for answers without numbers, words that must appear, separated by `;`. Leave it empty otherwise.
- `gold_status`: `answered | not_found | conflict | false_premise | session_override`.
- `gold_sources`: **every** acceptable source, comma separated, e.g. `DOC1:Table 1-3,DOC1:p8`. Use PDF page order for page numbers, not the printed numbers. Leave it empty for questions with no source (`not_found`).
- Nothing copied or paraphrased from `Questions.csv`. Check every answer in the PDFs yourself, including values read from charts.

**What each Part 1 notebook must provide:**
```python
new_conversation(conversation_id: str, mode: str = "hybrid") -> obj   # obj.answer(question) -> the brief's JSON dict
                                                                      # mode "dense" = baseline (dense retrieval only)
retrieve_sources(query: str, k: int, mode: str) -> list[str]          # e.g. ["DOC1:p3", "DOC2:p5:Figure A-6"]
# optional: obj.last_standalone = the follow-up rewritten as a full question (used for Recall@5)
```
**Golden-set cell.** Paste `src/p1_score.py` unchanged into a cell above it, and build your golden CSV into the notebook (Section 7), so it lands in `OUT_ROOT / "eval"`. Skip this on Kaggle to save time. It prints no scores on purpose:
```python
SYSTEM = "muhammed"   # or "kavan"
if not ON_KAGGLE:
    for path in sorted((OUT_ROOT / "eval").glob("p1_golden_*.csv")):
        gold = load_gold(path)
        if gold.empty:
            continue
        for mode in ("dense", "hybrid"):
            preds = run_golden(gold, new_conversation, retrieve_sources, mode)
            out = prediction_path(OUT_ROOT / "reports" / "results", SYSTEM, path.stem.replace("p1_golden_", ""), mode)
            out.parent.mkdir(parents=True, exist_ok=True)
            preds.to_csv(out, index=False)
```
This writes `reports/results/p1_predictions/<system>__<goldset>__<mode>.csv` (inside the ZIP, so it lands in the repo when you unzip) with columns `gold_id, conversation_id, turn, answer, status, citations, figure_references, top5_sources, seconds`. `p1_eval.ipynb` scores these. You can run `p1_eval.ipynb` yourself at any time to check your files load. Just don't tune on what it says.

### 6.2 Part 1: Kaggle notebook rules
- Standalone: Kaggle cannot import from our private repo, so every system cell lives in the notebook. Kavan's builder pastes `src/` into cells. Muhammed can just write code in cells.
- Finds the data by looking for `kb_manifest.json` next to the notebook, in `/content`, in `~/data266_lab2_data/`, in `data/part1/` or `DATA266_Lab2_Part1/`, or in `/kaggle/input/` (Kavan's data cell does exactly this, so copy it). Reads `QUESTIONS_PATH = os.environ.get("QUESTIONS_PATH", <data>/Questions.csv)` with `pd.read_csv(path, dtype=str, keep_default_na=False)`.
- Uses the brief's loop (one empty memory per conversation, turns in order). Writes `submission.csv` to the working folder (`/kaggle/working` on Kaggle) and runs `validate_submission.py` on it.
- Writes `reports/results/p1_kaggle_run_<name>.json` (under `OUT_ROOT`): `{"rows", "median_seconds_per_question", "total_minutes", "indexed_items"}`.
- Local models only, no API keys. If the competition notebook has no internet, attach models as Kaggle datasets.
- The brief's five statuses don't name one for "refuse a permanent change". Each of us picks one, explains it in our report section, and checks it against leaderboard feedback.

### 6.3 Part 2: fixed prompts
`eval/p2_fixed_prompts.csv`: `prompt_id,group,prompt,source`
- `prompt_id`: P01–P20. `group`: `in_domain` (10), `out_of_domain` (5), `compositional` (5). `source`: the test-image file name for in-domain prompts, otherwise `written`.

### 6.4 Part 2 → Part 3 (the only link between Muhammed's and Kavan's work)
`eval/p2_final_model.json` (Muhammed commits it; the template is `eval/p2_final_model.example.json`):
```json
{"source_notebook": "notebooks/p2_finetune.ipynb", "run_name": "run2",
 "hf_repo": "<muhammed-hf-username>/data266-p2-lora-run2", "local_path": null,
 "weight_name": "pytorch_lora_weights.safetensors", "scale": 1.0,
 "base_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
 "prompt_hint": "<style phrase, or empty>", "is_final": true}
```
- `tools/build_notebooks.py` builds this file into `p3_agents.ipynb` as `P2_LORA`, and `src/p3_models.find_lora()` uses it: a `local_path` that exists on the same machine wins, otherwise it downloads `hf_repo`. The HF repo must be **public**.
- If the file is missing, Part 3 uses Kavan's own baseline LoRA from `Kavan1804/data266-p2-baseline-kavan-lora` (uploaded by `p2_baseline_kavan.ipynb`). So Kavan can build and test all of Part 3 now, without waiting for anything.
- The `Load the models` cell in `p3_agents.ipynb` prints which LoRA was used, so the switch shows in the saved output.

**Hugging Face names**
| Repo | Owner | Type |
|---|---|---|
| `<muhammed-hf>/data266-p2-lora-run1` | Muhammed | model (LoRA rank 4) |
| `<muhammed-hf>/data266-p2-lora-run2` | Muhammed | model (LoRA rank 32) + model card |
| `<muhammed-hf>/data266-p2-generated-images` | Muhammed | dataset (120 images + metadata, and the 25 blinded pairs) |
| `Kavan1804/data266-p2-baseline-kavan-lora` | Kavan | model (optional) |
| `Kavan1804/data266-p3-inputs` | Kavan | dataset (Part 3 test images) |

### 6.5 Results files (every report number comes from these)
Notebooks write them, and `python reports/build_tables.py` turns them into LaTeX tables. A missing value prints `PENDING`.

- `reports/results/p1_eval.json`: written by `p1_eval.ipynb`. `baseline` and `final` hold `correct_of_25, recall_at_5, citation_accuracy, median_seconds_per_question`, and `final` adds `groundedness_mean, groundedness_within_one`.
- `reports/results/p2.json`: written by `p2_finetune.ipynb`:
  ```
  {"base": {...}, "run1": {...}, "run2": {...}, "selected_run": "run1|run2"}
  each: kid_mean, kid_std, clip_score, lpips_diversity, trainable_params, learning_rate, steps,
        batch_size, peak_gpu_gb, train_seconds, gpu_hours, win_rate_style, win_rate_prompt, rater_count
  (base has no training fields; win rates only on the selected run; use null for anything not computed)
  ```
- `reports/results/p3.json`: written by `p3_agents.ipynb` (human values `PENDING`), then rewritten by `p3_results.ipynb` once the forms are filled. `p3_runs.json` keeps the raw runs.
- `p1_kaggle_run_<name>.json`, `p2_baseline_kavan.json`, `p3_baseline_muhammed.json`: baselines and reproduction evidence.

---

## 7. Rules for every notebook
- Lives in `notebooks/`, runs with **Restart & Run All** in a fresh environment, and is saved **with outputs**.
- **Standalone:** no imports from the repo and no reading repo folders. Paste code into cells. Build small files (golden set, prompts, scenarios) into a cell that writes them out if they are missing (see the `EMBEDDED_FILES` cell in Kavan's notebooks). Download models and datasets from Hugging Face. The only thing you provide by hand is the Part 1 PDFs.
- **Output layout:** copy the setup cell from Kavan's notebooks. It defines `WORK` (big working files, never committed) and `OUT_ROOT`. Write every deliverable under `OUT_ROOT` using repo paths (`reports/results/...`, `traces/...`, `eval/human/...`). `OUT_ROOT` is the repo itself if the notebook happens to sit in one, otherwise `WORK/lab2_repo_files`.
- **Hugging Face token:** the `hf_token()` helper in that setup cell reads `HF_TOKEN` from a Colab secret (key icon), an environment variable or `huggingface-cli login`. Never type a token into a cell.
- **Cell 1** prints commit, Python, torch, GPU and seed (the brief's code), sets seed 266 everywhere, and caps GPU memory at 16 GB:
  ```python
  import subprocess, sys, random, numpy as np, torch
  SEED = 266; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(SEED)
      torch.cuda.set_per_process_memory_fraction(min(1.0, 16e9 / torch.cuda.get_device_properties(0).total_memory), 0)
  commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
  print({"commit": commit, "python": sys.version.split()[0], "torch": torch.__version__,
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "seed": SEED})
  ```
  (Part 2's required sampling seeds 0 and 1 stay as the brief says.)
- Installs its own pinned packages (`%pip install -q ...==...`) and never hides a failed install. Detects Kaggle, Colab or local. Finds files by relative path from the repo root. Checks required files exist. Uses the GPU when it is there.
- Loads one large model at a time where possible, frees memory between models (`del`, `gc.collect()`, `torch.cuda.empty_cache()`), and prints peak GPU memory.
- Training can be resumed: save `run_meta.json` and the log. On a rerun the notebook may load them, but it must **say so** and print the stored values. Typed-in numbers are never allowed.
- Readable markdown before each important section, in our normal student voice ("we used", "we observed"). Explain choices and failures honestly.
- **Last cell** builds a ZIP of that notebook's deliverables **in the repo's folder layout**: `notebooks/<name>.ipynb` (on Colab it fetches the notebook from the browser), results, CSV and JSON files, and traces. No raw data, caches, secrets or huge weights. It prints the exact path and contents and starts the download. Unzip it at the repo root and push. Also do File > Download .ipynb after it finishes, because a notebook cannot capture its own last cell. (Copy the last cell of any of Kavan's notebooks.)
- Cell 1 also prints `built_from` when there is no git (Colab, Kaggle). Kavan's builder fills it in. In your own notebooks, set `BUILT_FROM` to the commit you started from.
- Before pushing: `python tools/check_notebooks.py` and `python tools/check_repo.py`.

---

## 8. Human evaluation (people only, never an AI)
The notebooks read these CSVs and compute the numbers. A blank form prints `PENDING`. Each rater fills their own form **without looking at the other's**. Disagreements get settled in the integration call and written down in `CONTRIBUTIONS.md`.

| Form | Made by | Filled by | Columns |
|---|---|---|---|
| `eval/human/p1_groundedness_<name>.csv` | `p1_eval.ipynb` (after selection) | Kavan, Muhammed | `gold_id, rating` (1–5) |
| `eval/human/p2_ratings_<rater>.csv` | `p2_finetune.ipynb` | Kavan, Muhammed, outside rater | `pair_id, better_style (L/R), better_prompt (L/R)` |
| `eval/human/p3_passfail_<name>.csv` | already in the repo | Kavan, Muhammed | `scenario_id, mode, pass (0/1), critic_agree (0/1), note` |

After the forms are filled, Part 1 numbers come from `p1_eval.ipynb` and Part 3 numbers from `p3_results.ipynb`. Both run on a laptop in the repo with no GPU and no models. Part 2's win rates come from a small results cell or notebook that Muhammed writes the same way.

- **Part 2 blinding:** the notebook writes `eval/human/p2_pairs.csv` (pair_id, prompt, image file) and renders each pair as one image `pair_XX.png` (left | right), uploaded to the HF image dataset under `human_pairs/`. The left/right key goes to `eval/human/p2_pairs_key.csv`. **Nobody opens the key until every rating is in.**
- **Part 3:** `p3_agents.ipynb` shows every output image in its saved output, and writes `outputs/p3/review_sheet.csv`. Muhammed rates from the final rerun at the merge (Section 9, step 5), by opening that executed notebook.

---

## 9. Dates and the final merge

| When | What |
|---|---|
| **Mon Nov 9** | Kavan pushes the repo and invites Muhammed. **Kickoff call (record it):** walk through this plan and the existing code. |
| **Tue Nov 10** | Both golden sets and `p2_fixed_prompts.csv` committed and pushed (before tuning or training). |
| Nov 10–14 | Independent work. No handoffs are needed: Part 3 runs on Kavan's baseline LoRA, and Part 1 is scored from files. |
| **Sat Nov 14, 11:59 PM** | **Freeze.** Both branches pushed with executed notebooks; `p2_final_model.json` and the HF uploads done; Part 2 rating pairs on HF. |
| **Sun Nov 15** | **Integration call (record it).** Steps 1–6 below. Then human ratings. |
| **Mon Nov 16** | Ratings in → results cells rerun → report tables built → report finished. Kaggle run of `p1_rag.ipynb`. Rehearse the demo (record it). |
| **Tue Nov 17** | Tags, final Kaggle selection, Canvas zip, by 11:59 PM PT. |

**Merge procedure (Nov 15):**
1. `git checkout main && git pull`, then `git merge origin/muhammed`, then `git merge origin/kavan`. Ownership is split, so the only conflicts should be in the shared docs. Run `python tools/check_repo.py`.
2. **Part 1 cross-run:** now both golden sets are in `eval/`. Each of us builds both golden sets into our own Part 1 notebook (Kavan: `python tools/build_notebooks.py`), reruns it (Restart & Run All), unzips the ZIP into the repo, and pushes. That gives predictions for both sets.
3. **Part 1 scoring:** run `p1_eval.ipynb` and look at the cross table: 2 systems × 2 golden sets × dense/hybrid.
4. **Part 1 choice:** pick the system with the higher hybrid correct-out-of-25 on the golden set its author did not write. A tie goes to the Kaggle public score, then to speed. It must finish under 90 minutes. If we combine parts of both, the combined notebook is rerun and scored the same way before it counts. Write `eval/p1_selection.json` (`final_system`, `report_goldset` = the set the selected system's author did **not** write), copy the selected notebook to `notebooks/p1_rag.ipynb`, rerun `p1_eval.ipynb` so the groundedness forms appear, and rate.
5. **Part 3:** `eval/p2_final_model.json` is now on `main`. Kavan runs `python tools/build_notebooks.py` (this builds it in as `P2_LORA`), reruns `p3_agents.ipynb` on the 4090, and checks that the `Load the models` output shows Muhammed's LoRA. Unzip, push, then both fill the pass/fail forms from that run and run `p3_results.ipynb`.
6. **Requirements:** merge `requirements/kavan.txt` and `requirements/muhammed.txt` into `requirements.txt`, and pin torch to the exact version cell 1 printed.

**Verification before tagging (Nov 16–17):**
- [ ] Every notebook in `notebooks/` executed with outputs, first and last cells OK (`python tools/check_notebooks.py`)
- [ ] `notebooks/p1_rag.ipynb` run on Kaggle with the competition data, under 90 minutes, `validate_submission.py` passes, same rows and order as `Questions.csv`, submitted and **selected as final**
- [ ] `python tools/check_repo.py`: no question text or IDs, no secrets, no PDFs, weights or data
- [ ] `p2.json`, `p1_eval.json` and `p3.json` complete (no unexpected `PENDING`); `python reports/build_tables.py` run; every report number matches a notebook output
- [ ] Both LoRAs and the images on HF, model card on the selected run, Part 3 inputs on HF; `links.txt` filled in
- [ ] Report: IEEE format, own words, per part: architecture diagram, results table, what worked and what did not, safety analysis; Part 2 model card summary; Part 3 system card
- [ ] `CONTRIBUTIONS.md` (with recording links) and `AI_USAGE.md` written by each of us
- [ ] Tags: `git tag p1-v1.0 && git tag p2-v1.0 && git tag p3-v1.0 && git push origin --tags`, before the deadline
- [ ] Canvas: `Pair_Programming_Team_24.zip` = report PDF + `CONTRIBUTIONS.md` + `AI_USAGE.md` + `links.txt`
- [ ] Both of us can explain every part, including the other person's code

**Repo note for Kavan:** the brief wants the repo named `data266-lab2-team24`, private, with Read access for the instructor and ISAs (handles on Canvas). Rename `data266_lab2` on GitHub before tagging (GitHub redirects the old URL, so clones keep working), and add the graders.

---

## 10. Contribution entries (to copy into CONTRIBUTIONS.md at the end)
Tick only what really happened. Each of us edits only our own lines.

**Kavan Siddesh Kumar**
- [ ] Part 1: complete independent system (`p1_rag_kavan.ipynb`, `src/p1_engine.py`, `src/p1_conversation.py`), golden set `p1_golden_kavan.csv`, safety probes, shared scorer (`p1_score.py`, `p1_eval.ipynb`)
- [ ] Part 2: baseline run (`p2_baseline_kavan.ipynb`), blinded pair ratings
- [ ] Part 3: lead, complete system (`p3_agents.ipynb`, `src/p3_*.py`), scenarios and red-team cases, Gradio app, system card, pass/fail ratings
- [ ] Report: Part 1 (own system), Part 3, assembly. Repo setup, merge, tags, Canvas upload

**Muhammed Faiq Salman**
- [ ] Part 1: complete independent system (`p1_rag_muhammed.ipynb`), golden set `p1_golden_muhammed.csv`, safety probes
- [ ] Part 2: lead, complete system (`p2_finetune.ipynb`), two runs, uploads, human study and outside rater, model card
- [ ] Part 3: baseline (`p3_baseline_muhammed.ipynb`), pass/fail ratings
- [ ] Report: Part 1 (own system), Part 2. Kaggle run if his Part 1 is selected

**Meeting recordings** (date, topic, link): kickoff Nov 9/10, integration Nov 15, demo rehearsal Nov 16.
