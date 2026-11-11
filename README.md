# DATA 266 Lab 2, Team 24

Kavan Siddesh Kumar and Muhammed Faiq Salman (Pair_Programming_Team_24). The lab has three parts: multimodal RAG, diffusion fine-tuning, and a multi-agent image system.

> **Status (Nov 9):** Kavan's code is written and passes offline tests with fake models. **Nothing has run on a GPU yet**, and Muhammed's work starts now. `reports/results/` shows what has really run. The team plan is in [TEAM_WORKPLAN.md](TEAM_WORKPLAN.md).

## Layout
| Path | What it holds | Owner |
|---|---|---|
| `notebooks/p1_rag_kavan.ipynb` | Part 1, Kavan's complete system | Kavan |
| `notebooks/p1_rag_muhammed.ipynb` | Part 1, Muhammed's complete system | Muhammed |
| `notebooks/p1_eval.ipynb` | Scores both Part 1 systems on the golden sets (no models) | shared |
| `notebooks/p1_rag.ipynb` | The selected Part 1, run on Kaggle at tag `p1-v1.0` (created at the final merge) | final |
| `notebooks/p2_finetune.ipynb` | Part 2: both runs and the full evaluation | Muhammed (lead) |
| `notebooks/p2_baseline_kavan.ipynb` | Part 2 baseline | Kavan |
| `notebooks/p3_agents.ipynb` | Part 3: agents, evaluation and Gradio app | Kavan (lead) |
| `notebooks/p3_results.ipynb` | Part 3 human numbers from the saved runs + forms (laptop, no GPU) | Kavan |
| `notebooks/p3_baseline_muhammed.ipynb` | Part 3 baseline | Muhammed |
| `src/` | Code pasted into Kavan's notebooks, plus the shared scorer `p1_score.py` | |
| `eval/` | Golden sets, prompts, scenarios, red-team cases, human forms | |
| `traces/` | Part 3 JSON traces | generated |
| `reports/` | IEEE report, results JSON, table builder | |
| `tools/`, `tests/` | Notebook builder, repo checks, offline tests | Kavan |

**Every notebook is standalone.** Open just the `.ipynb` in Colab, Kaggle or Jupyter and Run all. Code and test files are built in, and models and datasets download by themselves. The only exception is the Part 1 PDFs: put them, or a ZIP of them, next to the notebook, in `/content`, or in `~/data266_lab2_data/`. The last cell downloads a ZIP in the repo's folder layout, so unzip it at the repo root and push.

Kavan's notebooks are generated: edit `src/` or `eval/`, then run `python tools/build_notebooks.py`. That clears notebook outputs, so rerun the notebook afterwards.

## Setup
Install a CUDA build of PyTorch, then:
```bash
pip install -r requirements.txt
```
Each notebook also installs its own pinned packages in its first code cells.

**Data.** Download the Part 1 files from Canvas and put them in `data/part1/` (or `DATA266_Lab2_Part1/`). Both are gitignored, as are all PDFs. Parts 2 and 3 download what they need from Hugging Face.

## GPU and memory
We run on Colab connected to an RTX 4090 (24 GB). Part 1 also has to run on Kaggle. The brief requires one 16 GB GPU, so cell 1 of every notebook caps PyTorch at 16 GB, and the notebooks print peak memory.

## How to run
- **Part 1, development:** we run this on Kaggle too. Make a notebook in the competition (GPU T4, internet on, competition data attached), upload `notebooks/p1_rag_<name>.ipynb`, and Save & Run All. It writes `submission.csv` and validates it first. After that, the tuning set and golden-set predictions run, and they are skipped when `QUESTIONS_PATH` is set. Download the ZIP from the last cell, unzip it at the repo root, then run `notebooks/p1_eval.ipynb` to score. The golden sets are built in when the notebook is generated, so rebuild it after editing `eval/p1_golden_*.csv`.
- **Part 1 on Kaggle (final submission):**
  1. Make a notebook in the competition with GPU on and the competition data attached.
  2. Upload `notebooks/p1_rag.ipynb` from tag `p1-v1.0`.
  3. Save and run all. Graders can swap questions with the `QUESTIONS_PATH` environment variable. It must finish within 90 minutes.
  4. Submit `submission.csv` from the Output tab and select it as final.

  If competition notebooks have no internet, attach `Qwen2.5-VL-3B-Instruct` and `bge-small-en-v1.5` as Kaggle datasets. The notebook finds them by folder name.
- **Part 2:** `notebooks/p2_finetune.ipynb` is the graded run. `notebooks/p2_baseline_kavan.ipynb` is the partner baseline.
- **Part 3:** `notebooks/p3_agents.ipynb`. Domain-style requests use the LoRA named in `eval/p2_final_model.json` (built in as `P2_LORA`), or Kavan's baseline LoRA from Hugging Face if that file is missing. After the human forms are filled, run `notebooks/p3_results.ipynb` in the repo. For the demo, set `LAB2_KEEP_GRADIO=1` before starting Jupyter so the app stays open.
- **Human forms:** fill in `eval/human/`, then rerun the results cells. Blank forms print `PENDING`.
- **Report:** `python reports/build_tables.py`, then build `reports/report.tex` on Overleaf (IEEEtran).

## Hugging Face
Weights and generated images go to Hugging Face, not GitHub. Add a write token named `HF_TOKEN` as a Colab secret (key icon), an environment variable, or with `huggingface-cli login`. Never put a token in a notebook. Kavan's account is `Kavan1804`, and Muhammed uses his own. The repo names are in [TEAM_WORKPLAN.md](TEAM_WORKPLAN.md) Section 6.4, and the final links go in `links.txt`.

## Models and APIs
We use no hosted APIs. All of these run locally:

| Use | Model |
|---|---|
| Text and vision (Kavan's Part 1, Part 3) | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Embeddings (Kavan's Part 1) | `BAAI/bge-small-en-v1.5` |
| Generation base | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| Editing | `timbrooks/instruct-pix2pix` |
| CLIP scores | `openai/clip-vit-base-patch32` |
| Dataset | `lambda/naruto-blip-captions` (the brief's `lambdalabs/` name now redirects there) |

Muhammed adds the models his Part 1 uses here at the final merge.

## Checks
```bash
python tests/test_offline.py            # logic checks with fake models, no GPU
python tests/dry_run_notebooks.py p1 p3 # runs Kavan's notebook cells with fake models, in a temp copy
python tools/check_notebooks.py         # syntax, first cell, last cell, saved outputs
python tools/check_repo.py              # question text, secrets, large files
```
These show the code runs. They say nothing about answer or image quality.

## Known limitations
- Chart values that are not printed on a chart are read approximately from the axis by the vision model.
- Kavan's fact parser only knows a fixed vocabulary of entities and variables. Other questions go to the model.
- The Part 3 safety checks use word patterns plus the model. A cleverly reworded request can get past the patterns.
- Naruto images are copyrighted, and the dataset page lists no license tag. Any model trained on them is for non-commercial research only.
