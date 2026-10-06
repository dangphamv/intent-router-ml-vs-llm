| Method | Macro-F1 test (95% CI) | Macro-F1 CV 5-fold | p95 (ms) | $/100k messages |
|---|---|---|---|---|
| A. TF-IDF + logistic regression | 0.906 (0.87–0.94) | 0.886 ± 0.011 | 0.9 | 0.00 |
| B. Embedding + logistic regression | 0.882 (0.84–0.92) | 0.902 ± 0.016 | 317.8 | 0.06 |
| C. Embedding + XGBoost | 0.896 (0.86–0.93) | 0.879 ± 0.020 | 318.4 | 0.06 |
| D. LLM zero-shot | 0.943 (0.92–0.97) | — | 911.1 | 3.23 |

| Hybrid (threshold tuned on OOF train+val) | Threshold | % routed to LLM | Macro-F1 test | p95 (ms) | $/100k |
|---|---|---|---|---|---|
| A. → D (chosen) | 0.72 | 24.7% | 0.931 | 792 | 0.80 |
| A. → D (0.8) | 0.80 | 32.7% | 0.940 | 808 | 1.06 |
| B. → D (chosen) | 0.81 | 26.3% | 0.945 | 1062 | 0.91 |
| B. → D (0.8) | 0.80 | 26.0% | 0.945 | 1062 | 0.90 |
