# intent-router-ml-vs-llm

Phân loại ý định tin nhắn khách hàng (tiếng Việt) — **ML cổ điển đấu với LLM** — và một **router kết hợp**: model rẻ trả lời khi tự tin, còn lại chuyển cho LLM.

5 ý định: `pricing` (hỏi giá) · `complaint` (khiếu nại) · `cancellation` (hủy gói) · `tech_support` (hỗ trợ kỹ thuật) · `other` (khác).

📓 [Notebook](notebooks/intent_router.ipynb) · ✍️ [Blog: Khi nào không cần LLM?](docs/blog-khi-nao-khong-can-llm.md) · 🏷️ [Hướng dẫn gán nhãn](data/LABELING_GUIDE.md)

## Kết quả

Test 300 tin, latency đo thật cho từng tin, chi phí từ token thực tế × giá niêm yết.

| Cách | Macro-F1 test (95% CI) | Macro-F1 CV 5-fold | p95 (ms) | $/100k tin |
|---|---|---|---|---|
| A. TF-IDF + logistic regression | 0,906 (0,87–0,94) | 0,886 ± 0,011 | 0,7 | ~0 |
| B. Embedding + logistic regression | 0,882 (0,84–0,92) | 0,902 ± 0,016 | 318 | 0,06 |
| C. Embedding + XGBoost | 0,896 (0,86–0,93) | 0,879 ± 0,020 | 318 | 0,06 |
| D. LLM zero-shot (`gpt-4o-mini`) | 0,943 (0,92–0,97) | — | 911 | 3,23 |

**Chiến lược kết hợp** (ngưỡng chọn trên out-of-fold train+val: rẻ nhất mà macro-F1 ≥ LLM-only):

| Hybrid | Ngưỡng | % chuyển LLM | Macro-F1 test | p50 (ms) | $/100k |
|---|---|---|---|---|---|
| A → D | 0,72 | 24,7% | 0,931 | 0,6 | 0,80 (−75%) |
| B → D | 0,81 | 26,3% | 0,945 | 274 | 0,91 (−72%) |

![Đánh đổi chi phí – chất lượng](results/tradeoff.png)

Confusion matrix: [`results/confusion_matrices.png`](results/confusion_matrices.png) · số liệu đầy đủ: [`results/metrics.json`](results/metrics.json)

## Dữ liệu

| | |
|---|---|
| Tổng | 2.000 tin (`data/intents.jsonl`), chia stratified 70/15/15 |
| Gán nhãn tay | 300 tin seed (`data/seed_raw.tsv`) |
| Sinh + gán nhãn bằng LLM | 1.700 tin — sinh bằng `gpt-4o-mini`, gán nhãn độc lập bằng `gpt-4o` |
| Annotator LLM vs người (300 seed) | accuracy 99,0%, Cohen's κ 0,987 |
| Kiểm tra mẫu 100 tin LLM-gán-nhãn | đồng thuận 95% (`data/audit_sample.csv`); nhãn sửa được ghi đè vào dataset |
| Lớp hiếm | `cancellation` ~10% → class weight cân bằng + stratified split |

> Seed và audit được soạn / review cùng Claude Code theo guideline. Hãy review lại `data/seed_raw.tsv` và `data/audit_sample.csv` trước khi dùng làm ground truth.

## Chạy

Yêu cầu [uv](https://docs.astral.sh/uv/). macOS cần `brew install libomp` cho XGBoost.

```bash
uv sync

# Notebook — chạy lại được KHÔNG cần API key (dùng cache trong data/cache/)
uv run jupyter lab notebooks/intent_router.ipynb

# Chạy lại toàn bộ pipeline (cần key; các call đã cache sẽ không gọi lại)
cp .env.example .env              # điền OPENAI_API_KEY
uv run scripts/build_dataset.py   # sinh → gán nhãn → đo đồng thuận → audit → split
uv run scripts/run_experiments.py # A–D, CV, latency, hybrid → results/

# Router cho dự án #3
uv run scripts/route.py "Gói Pro có bao nhiêu seat?" "ko muốn xài nữa, ngưng giúp mình"
```

```python
from intent_router.router import IntentRouter

router = IntentRouter.load()               # TF-IDF + LR, ngưỡng 0,72 → gpt-4o-mini
route = await router.route("app bị lỗi 500 khi lưu task")
route.label, route.confidence, route.handled_by
```

## Cấu trúc

```
data/
  LABELING_GUIDE.md      định nghĩa nhãn + quy tắc phân xử
  seed_raw.tsv           300 tin gán nhãn tay
  intents.jsonl          dataset cuối {text, label, split, source}
  dataset_report.json    đồng thuận, phân bố lớp, audit
  audit_sample.csv       100 tin kiểm tra mẫu
  cache/                 embedding, dự đoán LLM, latency (để chạy lại không cần key)
src/intent_router/
  config.py              nhãn, model, bảng giá
  llm.py                 zero-shot / annotator, structured output, cache
  embed.py               embedding + đo latency, cache
  experiments.py         model A/B/C, CV, bootstrap CI, hybrid curve, chọn ngưỡng
  router.py              IntentRouter (dùng lại cho dự án #3)
scripts/                 build_dataset.py · run_experiments.py · route.py
notebooks/               intent_router.ipynb
results/                 metrics.json, comparison.md, confusion_matrices.png, tradeoff.png, hybrid_*.csv
docs/                    blog
```

## Giới hạn

- 85% nhãn đến từ LLM, và annotator (`gpt-4o`) cùng họ với model D, nên kết quả có thể thiên vị D. Trên 37 tin test có nhãn người: D đạt 0,973, A đạt 0,905.
- Tin nhắn sinh tổng hợp "sạch" hơn log thật. Cần đánh giá lại trên dữ liệu production.
