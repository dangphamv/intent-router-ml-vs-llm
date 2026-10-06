# intent-router-ml-vs-llm

Intent classification for Vietnamese customer-support messages — **classical ML vs an LLM** — plus a **hybrid router**: a cheap model answers when it is confident, everything else goes to the LLM.

5 intents: `pricing` · `complaint` · `cancellation` · `tech_support` · `other`.

📖 [Detailed guide (HTML)](docs/guide.html) · 📓 [Notebook](notebooks/intent_router.ipynb) · ✍️ [Blog: When don't you need an LLM?](docs/blog-when-you-dont-need-an-llm.md) · 🏷️ [Labeling guide](data/LABELING_GUIDE.md)

## Results

300 test messages, latency measured per message, cost from actual tokens × list price.

| Method | Macro-F1 test (95% CI) | Macro-F1 CV 5-fold | p95 (ms) | $/100k messages |
|---|---|---|---|---|
| A. TF-IDF + logistic regression | 0.906 (0.87–0.94) | 0.886 ± 0.011 | 0.9 | ~0 |
| B. Embedding + logistic regression | 0.882 (0.84–0.92) | 0.902 ± 0.016 | 318 | 0.06 |
| C. Embedding + XGBoost | 0.896 (0.86–0.93) | 0.879 ± 0.020 | 318 | 0.06 |
| D. LLM zero-shot (`gpt-4o-mini`) | 0.943 (0.92–0.97) | — | 911 | 3.23 |

**Hybrid strategy** (threshold tuned on out-of-fold train+val: the cheapest threshold whose macro-F1 ≥ LLM-only):

| Hybrid | Threshold | % routed to LLM | Macro-F1 test | p50 (ms) | $/100k |
|---|---|---|---|---|---|
| A → D | 0.72 | 24.7% | 0.931 | 0.6 | 0.80 (−75%) |
| B → D | 0.81 | 26.3% | 0.945 | 274 | 0.91 (−72%) |

![Cost–quality trade-off](results/tradeoff.png)

Confusion matrices: [`results/confusion_matrices.png`](results/confusion_matrices.png) · full numbers: [`results/metrics.json`](results/metrics.json)

## Data

| | |
|---|---|
| Total | 2,000 messages (`data/intents.jsonl`), stratified 70/15/15 split |
| Hand-labeled | 300 seed messages (`data/seed_raw.tsv`) |
| Generated + labeled by LLM | 1,700 messages — generated with `gpt-4o-mini`, labeled independently by `gpt-4o` |
| LLM annotator vs human (300 seeds) | accuracy 99.0%, Cohen's κ 0.987 |
| Spot check of 100 LLM-labeled messages | 95% agreement (`data/audit_sample.csv`); corrected labels are written back to the dataset |
| Rare class | `cancellation` ~10% → balanced class weights + stratified split; ablation below |

The messages are Vietnamese by design and stay untranslated. The generation and annotation prompts use the Vietnamese guideline [`data/LABELING_GUIDE.vi.md`](data/LABELING_GUIDE.vi.md); [`data/LABELING_GUIDE.md`](data/LABELING_GUIDE.md) is its English translation.

> The seeds and the audit were written / reviewed with Claude Code following the guideline. Review `data/seed_raw.tsv` and `data/audit_sample.csv` before treating them as ground truth.

### Rare class: does class weighting help?

5-fold CV on train+val. In the "forced rare" scenario only 25% of `cancellation` messages are kept in each training fold; evaluation folds keep the natural distribution ([`results/rare_class_ablation.md`](results/rare_class_ablation.md)).

| Method | `cancellation` share in train | Recall none → balanced | Rare-class F1 none → balanced |
|---|---|---|---|
| A. TF-IDF + LR | 10.5% | 0.84 → 0.88 | 0.90 → 0.91 |
| A. TF-IDF + LR | 2.9% | 0.57 → **0.72** | 0.73 → **0.83** |
| B. Embedding + LR | 10.5% | 0.89 → 0.90 | 0.93 → 0.91 |
| B. Embedding + LR | 2.9% | 0.64 → **0.84** | 0.77 → **0.90** |
| C. Embedding + XGBoost | 2.9% | 0.47 → 0.54 | 0.63 → 0.70 |

At ~10%, class weighting barely changes anything. At ~3%, without weighting the model misses almost half of the customers who want to cancel, so `balanced` stays the default.

## Run

Requires [uv](https://docs.astral.sh/uv/). On macOS, XGBoost needs `brew install libomp`.

```bash
uv sync

# Notebook — reruns WITHOUT an API key (uses the cache in data/cache/)
uv run jupyter lab notebooks/intent_router.ipynb

# Rerun the whole pipeline (needs a key; cached calls are not repeated)
cp .env.example .env              # set OPENAI_API_KEY
uv run scripts/build_dataset.py   # generate → label → agreement → audit → split
uv run scripts/run_experiments.py # A–D, CV, latency, hybrid → results/
uv run scripts/rare_class_ablation.py  # class weighting vs rare class (no API calls)

# Router for project #3
uv run scripts/route.py "Gói Pro có bao nhiêu seat?" "ko muốn xài nữa, ngưng giúp mình"
```

```python
from intent_router.router import IntentRouter

router = IntentRouter.load()               # TF-IDF + LR (fit on train+val), threshold 0.72 → gpt-4o-mini
route = await router.route("app bị lỗi 500 khi lưu task")
route.label, route.confidence, route.handled_by
```

## Layout

```
data/
  LABELING_GUIDE.md      label definitions + tie-break rules (English)
  LABELING_GUIDE.vi.md   original Vietnamese guideline, used in the LLM prompts
  seed_raw.tsv           300 hand-labeled messages
  intents.jsonl          final dataset {text, label, split, source}
  dataset_report.json    agreement, class distribution, audit
  audit_sample.csv       100-message spot check
  cache/                 embeddings, LLM predictions, latency (rerun without a key)
src/intent_router/
  config.py              labels, models, price table
  llm.py                 zero-shot / annotator, structured output, cache
  embed.py               embeddings + latency probes, cache
  experiments.py         models A/B/C, CV, bootstrap CI, hybrid curve, threshold selection
  router.py              IntentRouter (reused in project #3)
scripts/                 build_dataset.py · run_experiments.py · rare_class_ablation.py · route.py
notebooks/               intent_router.ipynb
results/                 metrics.json, comparison.md, confusion_matrices.png, tradeoff.png, hybrid_*.csv, rare_class_ablation.*
docs/                    guide, blog
```

## Limitations

- 85% of labels come from an LLM, and the annotator (`gpt-4o`) is from the same family as model D, so results may favour D. On the 37 human-labeled test messages: D scores 0.973, A 0.905.
- Synthetic messages are "cleaner" than real logs; re-evaluate on production data.
