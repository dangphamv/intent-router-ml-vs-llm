| Method | `cancellation` share in train | Class weight | Macro-F1 | cancellation precision | cancellation recall | cancellation F1 |
|---|---|---|---|---|---|---|
| A. TF-IDF + logistic regression | 2.9% | none | 0.836 | 1.000 | 0.572 | 0.725 ± 0.076 |
| A. TF-IDF + logistic regression | 2.9% | balanced | 0.863 | 0.986 | 0.724 | 0.832 ± 0.062 |
| A. TF-IDF + logistic regression | 10.5% | none | 0.878 | 0.968 | 0.837 | 0.896 ± 0.054 |
| A. TF-IDF + logistic regression | 10.5% | balanced | 0.886 | 0.959 | 0.876 | 0.914 ± 0.060 |
| B. Embedding + logistic regression | 2.9% | none | 0.861 | 0.992 | 0.640 | 0.772 ± 0.098 |
| B. Embedding + logistic regression | 2.9% | balanced | 0.897 | 0.976 | 0.842 | 0.901 ± 0.052 |
| B. Embedding + logistic regression | 10.5% | none | 0.903 | 0.971 | 0.887 | 0.925 ± 0.043 |
| B. Embedding + logistic regression | 10.5% | balanced | 0.902 | 0.932 | 0.898 | 0.912 ± 0.044 |
| C. Embedding + XGBoost | 2.9% | none | 0.818 | 0.990 | 0.466 | 0.629 ± 0.075 |
| C. Embedding + XGBoost | 2.9% | balanced | 0.839 | 0.990 | 0.544 | 0.700 ± 0.061 |
| C. Embedding + XGBoost | 10.5% | none | 0.875 | 0.954 | 0.786 | 0.859 ± 0.059 |
| C. Embedding + XGBoost | 10.5% | balanced | 0.879 | 0.952 | 0.814 | 0.873 ± 0.058 |
