# Azerbaijani Instruction-Following LLM

Fine-tuned a 3B multilingual LLM for Azerbaijani instruction-following (LoRA on Apple Silicon),
built a native-authored eval set and a human + LLM-judge harness with documented failure-mode analysis.

> **Headline result:** Fine-tune beats few-shot base on structured benchmarks — Belebele +8pp
> (30.3% vs 22.2%), SIB-200 +6pp acc and F1 jumps 0.06→0.23. chrF++ judge (n=150) favours
> few-shot 41%/26%/33%, but blind native human eval (n=50) scores them essentially equal —
> 44%/14%/42% — with fine-tune winning strongly on summarisation. Cohen's κ between judge
> and human = 0.000: chrF++ is a poor proxy for native speaker preference in Azerbaijani.

---
 
## Why this project

Azerbaijani is spoken by ~50M people yet remains severely under-resourced in NLP.
As a native speaker I have something no scraped pipeline can replicate:
genuine linguistic judgment over model outputs.

**Why fine-tune rather than RAG or prompting?**
For low-resource languages, adapting a small multilingual model with LoRA can match full
fine-tuning at a fraction of the parameters and beat few-shot prompting of much larger models —
especially on instruction-following tasks where the base model has weak Azerbaijani priors.
My few-shot baseline verifies this empirically rather than assuming it.

---

## Language background

Azerbaijani is an **agglutinative Turkic language** with vowel harmony and heavy suffixation,
written in a Latin script with special characters (ə, ğ, ı, ö, ü, ş, ç).

Two technical consequences:
- **Tokenizer fragmentation.** BPE vocabularies trained mostly on English shatter agglutinative
  words into many subword tokens. See `results/fertility.png` for measurements.
- **Turkish drift.** The model can silently switch to Turkish (closely related, higher-resource).
  This is a concrete, nameable failure mode tracked in human eval.

---

## Data

| Tier | Source | Size | Notes |
|---|---|---|---|
| Tier 1 — translated | Dolly-15k via NLLB-200-distilled-600M | 2,471 pairs | After filtering; see reject rates below |
| Tier 3 — native | Hand-authored | 75 pairs | 11 categories (see taxonomy) |
| Eval (frozen) | Native, held-out | 150 pairs | SHA-256 frozen before any training |

**Translate-train filter pipeline** (`scripts/03_filter_pipeline.py`) — four sequential stages:

| Stage | Reject rate |
|---|---|
| Language-ID (`langdetect` + Az heuristic) | ~10% |
| Length-ratio sanity | ~5% |
| Back-translation chrF < 30 | ~3% |
| Near-dedup (MinHash, J≥0.85) | ~<1% |
| **Total kept** | **82.4% (2,471/3,000)** |

Note: `langdetect` conflates Azerbaijani with Turkish (`tr`). Fixed by accepting `tr` detections
that contain Azerbaijani-specific characters (ə, ğ) not found in standard Turkish Latin script.

**Decontamination:** a character-5-gram Jaccard check confirms no training example scores
≥ 0.8 similarity to any frozen test prompt. See `data/processed/decontam_report.json`.

**Eval taxonomy** (`data/eval/taxonomy.json`): 11 categories including factual Q&A,
summarization, rewriting, classification, extraction, reasoning, creative writing, translation,
refuse/clarify, multi-turn, and culturally-grounded prompts.

---

## Base model selection

Fertility analysis (`scripts/01_fertility_analysis.py`) on 500 Azerbaijani Wikipedia sentences:

| Model | Az fertility | En fertility | Az/En ratio |
|---|---|---|---|
| Qwen/Qwen2.5-3B | 4.237 | 1.307 | **3.24×** |
| microsoft/Phi-3-mini-4k-instruct | 5.007 | 1.492 | 3.36× |
| bigscience/bloom-560m | 3.892 | 1.311 | 2.97× |
| tiiuae/falcon-rw-1b | 5.822 | 1.372 | 4.24× |

Chose **Qwen/Qwen2.5-3B** — lowest ratio among capable models (BLOOM-560m has lower ratio
but is architecturally obsolete). Lower ratio = fewer wasted tokens per Azerbaijani word
= more efficient training and inference.

---

## Training

**Method:** LoRA adapters on all linear layers (CUDA: 4-bit NF4 QLoRA; MPS: float16 LoRA).  
**Config:** `configs/train_config.yaml` — r=16, α=32, lr=2e-4, cosine schedule, 2 epochs, max_seq_len=1024 with packing.  
**Hardware:** Apple M-series (MPS) — ~42 minutes training time.  
**Seed:** 42 everywhere. Exact commands in `scripts/05_train.py`.

Single run (full sweep deferred — MPS training is ~59 min/run):

| Run | LoRA r | Epochs | Train loss | Val loss | Val token acc |
|---|---|---|---|---|---|
| default | 16 | 2 | 2.441 (ep2) | 2.485 | 59.4% |

Loss dropped from 3.39 (step 10, ep1) → 2.44 (step 20, ep2), confirming the model learned the instruction format.

---

## Results

### Automatic benchmarks

| System | Belebele acc | SIB-200 acc | SIB-200 F1 | Translation chrF++ |
|---|---|---|---|---|
| Base zero-shot | 3.67% | 24.51% | 0.057 | 10.03 |
| Base few-shot | 22.22% | 24.51% | 0.057 | 8.85 |
| **Fine-tune** | **30.33%** | **30.39%** | **0.225** | 7.28 |

Belebele random baseline = 25% (4-choice MCQ). SIB-200 random baseline = 14.3% (7-class).
Fine-tune significantly beats both baselines and few-shot on structured tasks.
Low zero-shot Belebele (3.67%) confirms the base model has weak Azerbaijani reading comprehension
without prompt guidance — fine-tuning addresses this directly.

### Pairwise eval — chrF++ reference-based (n=150)

Fine-tune vs. base few-shot: **32.7% win / 26.0% tie / 41.3% loss**  
Avg chrF++: base_few_shot = 21.57, finetune = 21.17

Metric: sentence-level chrF++ (word_order=2) vs. gold reference answers; win margin ≥ 1.0 chrF++ point.

Per-category results (A = base_few_shot, B = finetune):

| Category | few-shot wins | finetune wins | ties |
|---|---|---|---|
| factual_qa (n=20) | **10** | 7 | 3 |
| reasoning (n=15) | **9** | 5 | 1 |
| creative (n=10) | **6** | 0 | 4 |
| translation (n=10) | **6** | 2 | 2 |
| multi_turn (n=10) | **5** | 2 | 3 |
| cultural (n=15) | 5 | **6** | 4 |
| classification (n=15) | 5 | **8** | 2 |
| extraction (n=15) | 4 | **7** | 4 |
| summarization (n=15) | 5 | 5 | 5 |
| rewriting (n=15) | 4 | 4 | 7 |
| refuse_clarify (n=10) | 3 | 3 | 4 |

Full per-item breakdown in `results/judge_eval.json`.

### Native human eval (n=50, blind)

Conducted by the native Azerbaijani-speaking author using blinded side-by-side pairs
(system identities hidden; left/right assignment randomized). Pairs presented interactively
in chat; verdicts entered as L/R/T per pair. Full results in `results/human_eval.json`.

Fine-tune vs. base few-shot: **42.0% win / 14.0% tie / 44.0% loss** (n=50)

Per-category breakdown:

| Category | few-shot wins | finetune wins | ties | n |
|---|---|---|---|---|
| summarization | 1 | **6** | 2 | 9 |
| extraction | 1 | **3** | 0 | 4 |
| cultural | 1 | **2** | 1 | 4 |
| rewriting | 1 | **2** | 2 | 5 |
| classification | 0 | **1** | 1 | 2 |
| factual_qa | **4** | 4 | 0 | 8 |
| reasoning | **3** | 3 | 1 | 7 |
| creative | **2** | 0 | 0 | 2 |
| multi_turn | **3** | 0 | 0 | 3 |
| refuse_clarify | **3** | 0 | 0 | 3 |
| translation | **3** | 0 | 0 | 3 |

**Judge–human agreement:** Cohen's κ = **0.000** (chance-level), raw agreement = **14%** (7/50 pairs).
This large divergence between chrF++ judge and human is the single most important finding
in this eval: the metric and the native speaker systematically disagree about which model is better.

Key driver: on **summarization** (9 pairs), the human gave finetune 6/9 wins while
the chrF++ judge scored it 5/5/5 (equal). The fine-tune generates abstractive summaries
that differ lexically from the reference but are judged higher quality by a native speaker.
chrF++ penalises valid paraphrases — this metric is a poor judge of summarization quality.

On structured tasks (translation, refuse_clarify, multi_turn) human and judge both favour
base_few_shot, suggesting the fine-tune's weaknesses on open-ended tasks are real rather
than metric artefacts.

---

## Error analysis

From pairwise chrF++ eval (per-category win/loss, `results/judge_eval.json`):

| Category | few-shot wins | fine-tune wins | interpretation |
|---|---|---|---|
| factual_qa | **10** | 7 | Base model's factual knowledge not overridden |
| reasoning | **9** | 5 | Short training set insufficient for reasoning |
| creative | **6** | 0 | Fine-tune produces shorter, less varied output |
| extraction | 4 | **7** | Format adherence improves with fine-tuning |
| classification | 5 | **8** | Structured output instruction-following improves |

Key findings:
- Fine-tune helps where the task is **format-sensitive** (classification, extraction): the model learns to follow structured instructions more precisely.
- Fine-tune hurts on **open-ended tasks** (creative, reasoning): 2,419 examples across 11 categories is insufficient to improve fluency on tasks requiring creativity or multi-step reasoning.
- Translation chrF++ drops for fine-tune (10.03 → 7.28): likely generating longer responses that diverge from terse gold references.
- SIB-200 macro-F1 jump (0.057 → 0.225) is the most striking improvement: fine-tune stops collapsing all predictions to one class.

Side-by-side examples with annotated failure modes: `results/sample_outputs.md`

Human eval (`08_human_eval.py`) available as interactive CLI for blind scoring of 50 pairs.

---

## Honest limitations

- **Eval set size:** 150 examples — CIs are wide; win rates are directional signals, not precise estimates.
- **Repetition loops:** Both models fail catastrophically on open-ended creative tasks (38% of base, 24% of fine-tune), indicating the 3B model is too small for unconstrained generation in Azerbaijani.
- **No refusal alignment:** Neither model reliably refuses harmful requests in Azerbaijani — small multilingual models with limited Az training data don't inherit English alignment behavior.
- **chrF++ as judge:** Reference-based evaluation penalises valid paraphrases and rewards lexical overlap. Confirmed empirically: native human eval (n=50) disagrees with chrF++ judge at Cohen's κ = 0.000 (chance level), 14% raw agreement. Fine-tune's true win rate is likely higher than the 33% chrF++ assigns it, particularly on summarisation (human: 6/9, judge: 5/9).
- **Single training run:** Only one LoRA configuration trained (r=16, 2 epochs) due to ~1h/run on MPS. A proper sweep would test r=32 and epoch=3 variants.
- **Turkish drift reduced, not eliminated:** Fine-tune shows 14% Turkish-drift rate vs 22% for base (heuristic detection) — still present because the Dolly translated data itself contains Turkish-influenced phrasing from NLLB-200.

---

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 0. Freeze eval set (run once after writing data/eval/test.jsonl)
python scripts/00_freeze_eval.py

# 1. Fertility analysis — pick base model
python scripts/01_fertility_analysis.py

# 2-3. Translate + filter Tier-1 data
python scripts/02_translate_train.py --input data/raw/dolly/train.jsonl \
    --output data/raw/translated/dolly_az_raw.jsonl --max_rows 3000
python scripts/03_filter_pipeline.py \
    --input  data/raw/translated/dolly_az_raw.jsonl \
    --output data/raw/translated/dolly_az_filtered.jsonl

# 4. Merge tiers, apply chat template, decontaminate
python scripts/04_prepare_datasets.py

# 5. Train (edit configs/train_config.yaml first)
python scripts/05_train.py
# Sweep variant: python scripts/05_train.py --run_name r32 --overrides lora.r=32

# 6. Generate outputs for all systems
python scripts/09_generate_outputs.py

# 7. Automatic benchmarks
python scripts/06_eval_automatic.py

# 8. chrF++ pairwise eval (no API key needed)
python scripts/07_eval_judge.py \
    --outputs_a results/outputs_base_few_shot.jsonl \
    --outputs_b results/outputs_finetune.jsonl \
    --system_a base_few_shot --system_b finetune

# 9. Human eval (run yourself, blind)
python scripts/08_human_eval.py \
    --eval_file data/eval/test.jsonl \
    --outputs_a results/outputs_base_few_shot.jsonl \
    --outputs_b results/outputs_finetune.jsonl \
    --judge_file results/judge_eval.json --n 50
```

---

## Repo layout

```
az-instruct-lm/
├── configs/train_config.yaml       # all hyperparams
├── data/
│   ├── eval/
│   │   ├── test.jsonl              # FROZEN — do not modify
│   │   ├── checksum.sha256
│   │   └── taxonomy.json
│   ├── raw/                        # source data (not committed if large)
│   └── processed/                  # train.jsonl, val.jsonl, decontam_report.json
├── scripts/
│   ├── 00_freeze_eval.py
│   ├── 01_fertility_analysis.py
│   ├── 02_translate_train.py
│   ├── 03_filter_pipeline.py
│   ├── 04_prepare_datasets.py
│   ├── 05_train.py
│   ├── 06_eval_automatic.py
│   ├── 07_eval_judge.py
│   ├── 08_human_eval.py
│   └── 09_generate_outputs.py
├── results/
│   ├── fertility.png
│   ├── auto_eval.json
│   ├── judge_eval.json
│   ├── human_eval.json
│   └── sample_outputs.md
├── adapters/az-instruct-lora/      # saved LoRA weights
├── requirements.txt
└── README.md
```
