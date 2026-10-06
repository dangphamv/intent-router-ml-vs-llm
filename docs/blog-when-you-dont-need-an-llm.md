# When don't you need an LLM? Measuring it on intent classification

> TL;DR — On 2,000 Vietnamese customer support messages, a zero-shot LLM (`gpt-4o-mini`) reaches macro-F1 **0.943**; TF-IDF + logistic regression reaches **0.906** at **zero** cost and **under 1 ms** latency. Letting the cheap model answer when it is confident and routing only the **~25%** of hard messages to the LLM **cuts cost by 72–75%** with quality nearly on par with the LLM.

## The problem

A SaaS product receives support messages and needs to route them into 5 queues: `pricing` (price questions), `complaint` (complaints), `cancellation` (plan cancellation), `tech_support` (technical support), `other` (everything else). Real messages are messy: no diacritics ("cho minh huy goi voi" ("pls cancel my plan", no diacritics)), teencode ("gói pro bn tiền v" ("how much is the pro plan", teen slang)), Vietglish, and lots of messages that sit between two intents ("Dịch vụ tệ quá, hủy luôn cho tôi" ("The service is terrible, just cancel it for me")).

The question: do we need to call an LLM for **every** message?

## Data: 2,000 messages, 300 hand-labeled

- 300 seed messages hand-labeled according to the [labeling guide](../data/LABELING_GUIDE.md), which includes tie-breaking rules for ambiguous cases.
- 1,700 messages generated with `gpt-4o-mini` (10 writing styles, 11 personas, ~20% hard cases), labeled independently by `gpt-4o`.
- **Can the LLM labels be trusted?** On the 300 seed messages, the `gpt-4o` annotator agrees with the human **99.0%** of the time (Cohen's κ = 0.987). A review of 100 random LLM-labeled messages showed **95%** agreement; both exceed the 90% threshold. The 5 wrong labels were fixed in the dataset.
- `cancellation` is the rare class (209 messages, ~10%): stratified 70/15/15 split and balanced class weights.

## Results

The test set has 300 messages. Latency is measured one message at a time, like a real online router. Cost is computed from actual token counts times list prices.

| Approach | Test macro-F1 (95% CI) | 5-fold CV macro-F1 | p95 (ms) | $/100k msgs |
|---|---|---|---|---|
| A. TF-IDF + logistic regression | 0.906 (0.87–0.94) | 0.886 ± 0.011 | **0.7** | **~0** |
| B. Embedding + logistic regression | 0.882 (0.84–0.92) | **0.902** ± 0.016 | 318 | 0.06 |
| C. Embedding + XGBoost | 0.896 (0.86–0.93) | 0.879 ± 0.020 | 318 | 0.06 |
| D. LLM zero-shot (`gpt-4o-mini`) | **0.943** (0.92–0.97) | — | 911 | 3.23 |

Three things worth noting:

1. **The gap between the cheap models is smaller than the noise.** On test, A leads the cheap group; on 5-fold CV, B leads. The confidence intervals overlap, so don't pick a model based on a single split.
2. **XGBoost doesn't help on embeddings.** With 1,536-dimensional dense vectors and 1,400 training samples, a linear model is enough. Decision trees are a better fit for tabular features.
3. **TF-IDF with char n-grams is very strong on Vietnamese without diacritics.** "huy goi", "hủy gói" and "huỷ gói" (all "cancel plan") share character n-grams, while the embedding API has no clear advantage here.

The LLM wins by about 4 macro-F1 points, but it is **about 1,000x slower** than A and **about 50x more expensive** than B.

### Rare class: are class weights needed?

At the natural ~10% ratio, `class_weight="balanced"` changes almost nothing (`cancellation` F1 0.90 → 0.91 for A, 0.93 → 0.91 for B). To make the effect visible, I downsampled this class to ~2.9% in the training folds, keeping the evaluation folds unchanged:

| | `cancellation` recall, no weights | With weights |
|---|---|---|
| A. TF-IDF + LR | 0.57 | **0.72** |
| B. Embedding + LR | 0.64 | **0.84** |
| C. Embedding + XGBoost | 0.47 | 0.54 |

Without weights, the model **misses nearly half of the customers who want to cancel**. With weights, rare-class F1 rises by 7–13 points while precision stays nearly the same (≥ 0.97). XGBoost handles the rare class worst, one more reason to pick a linear model.

## Where do the models get confused?

| Confusion pair (true → predicted) | A | B | C | D |
|---|---|---|---|---|
| other → tech_support | 7 | 6 | 5 | 4 |
| tech_support → other | 5 | 5 | 3 | 1 |
| pricing → other | 2 | 5 | 3 | 2 |
| tech_support → complaint | 3 | 3 | 3 | 2 |
| cancellation → complaint | 2 | 3 | 0 | 1 |
| complaint → pricing | 1 | 2 | 0 | 3 |

- **`other` ↔ `tech_support`** tops the list for **every** approach, including the LLM. Is "Có tính năng nào đổi mã dự án không?" ("Is there a feature to change the project code?") a feature question (`other`) or a how-to question (`tech_support`)? This is a **label definition** problem, not a model problem. To improve it, fix the guideline first, then think about changing the model.
- **`tech_support` → `complaint`**: "Mỗi lần dùng là có lỗi, không thể nào yên tâm làm việc!" ("Every time I use it there's an error, I can't work in peace!"). This message both reports a bug and vents frustration; the boundary depends on tone.
- **`cancellation` → `complaint`**: "Hơi thất vọng, thôi hủy luôn cho đơn giản" ("A bit disappointed, let's just cancel to keep it simple"). The negative emotion overshadows the intent to cancel. This is the **most expensive** error in practice: a customer who wants to cancel gets pushed into the complaint queue.
- **The LLM has its own blind spots**: "Tại sao tháng này bị trừ tiền nhiều hơn tháng trước?" ("Why was I charged more this month than last month?") is predicted as `pricing` by the LLM because it sees a money topic, whereas by the rules, being charged incorrectly is `complaint`. The cheap model learns this rule from the data; the zero-shot LLM does not.

Of the 29 messages A gets wrong, the LLM gets 24 right. Conversely, of the 17 messages the LLM gets wrong, A gets 12 right. The two **fail in different places**, and that is exactly what makes the hybrid strategy effective.

## Hybrid strategy: cheap when confident, expensive when hard

```
p = cheap_model.predict_proba(text)
if p.max() > threshold:  return argmax(p)      # local, ~0.6 ms, $0
else:                    return llm(text)      # ~750 ms, $3.23 / 100k
```

**Choosing the threshold without looking at test.** The val set has only 300 messages, which is too noisy: in the first attempt, the threshold chosen on val gave F1 0.94 but dropped to 0.91 on test. So I used *out-of-fold* predictions from 5-fold CV on train+val (1,700 messages), swept the threshold from 0 to 1, and picked the **cheapest threshold whose macro-F1 is no lower than LLM-only** on the same data. The reasoning: the goal is to keep LLM quality at the lowest cost, not to maximize F1 at any price.

![Cost–quality trade-off](../results/tradeoff.png)

| Hybrid | Threshold | % routed to LLM | Test macro-F1 | p50 (ms) | $/100k | vs. LLM-only |
|---|---|---|---|---|---|---|
| A → D | 0.72 | 24.7% | 0.931 | **0.6** | **0.80** | −75% cost, −1.2 F1 points |
| A → D | 0.80 | 32.7% | 0.940 | 0.6 | 1.06 | −67% cost, −0.3 points |
| B → D | 0.81 | 26.3% | **0.945** | 274 | 0.91 | −72% cost, **equal F1** |
| D (LLM-only) | — | 100% | 0.943 | 737 | 3.23 | — |

The curve in the left chart has an interesting feature: over a range of thresholds, the hybrid is even **better** than LLM-only, because on the messages where the cheap model is very confident, it is more accurate than the LLM (e.g. the `complaint → pricing` case above).

## So when don't you need an LLM?

1. **When the intent has clear keyword signals** ("giá" (price), "hủy" (cancel), "lỗi 500" (error 500), "seat"): about 75% of traffic falls into this group, and a linear model on TF-IDF handles it in under 1 ms.
2. **When latency matters**: the p50 of the A → D router is 0.6 ms, because 3/4 of messages never touch the network.
3. **When volume is high**: at 10 million messages/month, LLM-only costs about $323, while A → D costs about $80. The absolute amount is still small with `gpt-4o-mini`, but the gap multiplies many times over with a larger model or longer prompts (few-shot, with context).
4. **When the errors come from the labels rather than the model**: the `other ↔ tech_support` pair trips up every model. Switching to an LLM doesn't fix an ambiguous definition.

**You still need an LLM** for the remaining ~25% of messages: long messages, multiple intents, emotion mixed with a request, or phrasing never seen in the training set.

## Limitations

- 85% of the labels come from an LLM (although checked at 99% / 95% agreement), and the `gpt-4o` annotator is from the same family as model D. This may **favor D**. On the subset of human-labeled messages in test (37 messages), D reaches 0.973 and A reaches 0.905. The trend holds, but the sample is too small to draw firm conclusions.
- Synthetic messages are "cleaner" than real ones. Before using this in production, re-evaluate on real logs.
- Embedding API and LLM latency were measured from one machine at one point in time. Absolute numbers will vary with network and load, but the relative differences between approaches are more reliable.

Code, data and a reproducible notebook (no API key needed): [README](../README.md).
