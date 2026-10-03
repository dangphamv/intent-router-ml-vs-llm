import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from intent_router.config import DATA, LABELS, SEED

LABEL_ID = {l: i for i, l in enumerate(LABELS)}


def load_data() -> pd.DataFrame:
    return pd.read_json(DATA / "intents.jsonl", lines=True)


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)


def tfidf_lr():
    """Word 1-2 grams catch keywords ("hủy", "giá"); char 2-5 grams survive missing diacritics and typos."""
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
    ])
    return make_pipeline(features, LogisticRegression(C=10, class_weight="balanced", max_iter=5000))


def emb_lr():
    return LogisticRegression(C=10, class_weight="balanced", max_iter=5000)


class EmbXGB:
    """XGBoost over embeddings with balanced sample weights; exposes sklearn-style string labels."""

    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.08, subsample=0.8,
            colsample_bytree=0.4, tree_method="hist", random_state=SEED, n_jobs=-1,
        )
        self.classes_ = np.array(LABELS)

    def fit(self, X, y):
        y_id = np.array([LABEL_ID[l] for l in y])
        self.model.fit(X, y_id, sample_weight=compute_sample_weight("balanced", y_id))
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


FACTORIES = {"A": tfidf_lr, "B": emb_lr, "C": EmbXGB}
NAMES = {
    "A": "A. TF-IDF + logistic regression",
    "B": "B. Embedding + logistic regression",
    "C": "C. Embedding + XGBoost",
    "D": "D. LLM zero-shot",
}


def proba_in_label_order(model, X) -> np.ndarray:
    p = model.predict_proba(X)
    order = [list(model.classes_).index(l) for l in LABELS]
    return p[:, order]


def cross_validate(key: str, X, y, folds: int = 5) -> tuple[list[float], np.ndarray]:
    """Per-fold macro-F1 plus out-of-fold probabilities (used to tune the routing threshold)."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    y = np.asarray(y)
    scores, oof = [], np.zeros((len(y), len(LABELS)))
    for tr, te in skf.split(np.zeros(len(y)), y):
        m = FACTORIES[key]().fit(_take(X, tr), y[tr])
        oof[te] = proba_in_label_order(m, _take(X, te))
        scores.append(macro_f1(y[te], np.array(LABELS)[oof[te].argmax(1)]))
    return scores, oof


def _take(X, idx):
    return [X[i] for i in idx] if isinstance(X, list) else X[idx]


def per_message_ms(fn, inputs, repeats: int = 1) -> np.ndarray:
    """Wall-clock latency of predicting one message at a time, as an online router would."""
    out = []
    for x in inputs:
        start = time.perf_counter()
        for _ in range(repeats):
            fn(x)
        out.append((time.perf_counter() - start) * 1000 / repeats)
    return np.array(out)


def bootstrap_ci(y_true, y_pred, n: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = [
        macro_f1(y_true[idx], y_pred[idx])
        for idx in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n))
    ]
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def confused_pairs(y_true, y_pred, top: int = 5) -> list[tuple[str, str, int]]:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    pairs = [(LABELS[i], LABELS[j], int(cm[i, j])) for i in range(len(LABELS)) for j in range(len(LABELS)) if i != j]
    return sorted((p for p in pairs if p[2]), key=lambda p: -p[2])[:top]


def hybrid_curve(
    cheap_proba: np.ndarray,
    llm_pred: np.ndarray,
    y_true,
    cheap_cost: float,
    llm_cost: float,
    cheap_ms: np.ndarray,
    llm_ms: np.ndarray,
    thresholds=np.round(np.linspace(0, 1, 101), 2),
) -> pd.DataFrame:
    """Cheap model answers when max probability > t, otherwise the message goes to the LLM.

    Cost is per 100k messages; the cheap model always runs, the LLM only on routed messages.
    """
    conf = cheap_proba.max(1)
    cheap_pred = np.array(LABELS)[cheap_proba.argmax(1)]
    rows = []
    for t in thresholds:
        routed = conf <= t
        pred = np.where(routed, llm_pred, cheap_pred)
        ms = cheap_ms + np.where(routed, llm_ms, 0)
        rows.append({
            "threshold": float(t),
            "routed_pct": routed.mean() * 100,
            "macro_f1": macro_f1(y_true, pred),
            "cost_per_100k": cheap_cost + routed.mean() * llm_cost,
            "p95_ms": float(np.percentile(ms, 95)),
            "p50_ms": float(np.percentile(ms, 50)),
        })
    return pd.DataFrame(rows)


def pick_threshold(curve: pd.DataFrame, target_f1: float) -> float:
    """Cheapest threshold whose macro-F1 reaches `target_f1` (the LLM-only score on the same data)."""
    ok = curve[curve.macro_f1 >= target_f1]
    if ok.empty:
        ok = curve[curve.macro_f1 == curve.macro_f1.max()]
    return float(ok.sort_values(["cost_per_100k", "threshold"]).iloc[0].threshold)
