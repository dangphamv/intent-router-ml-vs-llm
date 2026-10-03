"""Run methods A–D and the hybrid router; write tables and figures to results/.

API calls (embeddings, LLM zero-shot, latency probes) are cached under data/cache/.
"""

import asyncio
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from intent_router import experiments as ex
from intent_router.config import CACHE, EMBED_MODEL, LABELS, LLM_MODEL, MODELS, RESULTS, cost_usd
from intent_router.embed import embed_texts, measure_latency
from intent_router.llm import ZERO_SHOT_PROMPT, classify_many

PER = 100_000


async def main():
    RESULTS.mkdir(exist_ok=True)
    df = ex.load_data()
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    trva = df[df.split != "test"].reset_index(drop=True)

    E = dict(zip(df.text, embed_texts(df.text.tolist())))
    X = {name: np.stack([E[t] for t in part.text]) for name, part in [("tr", tr), ("va", va), ("te", te), ("trva", trva)]}
    T = {name: part.text.tolist() for name, part in [("tr", tr), ("va", va), ("te", te), ("trva", trva)]}
    inputs = {"A": T, "B": X, "C": X}

    # LLM zero-shot on train+val (threshold tuning, out-of-fold) and test (reporting).
    llm = {
        s: await classify_many(T[s], LLM_MODEL, ZERO_SHOT_PROMPT, CACHE / f"zeroshot_{LLM_MODEL}.jsonl", concurrency=4)
        for s in ("trva", "te")
    }
    emb_lat = await measure_latency(T["te"], CACHE / f"embed_latency_{EMBED_MODEL}.jsonl")
    emb_ms = np.array([r["latency_ms"] for r in emb_lat])
    emb_cost = cost_usd(EMBED_MODEL, np.mean([r["input_tokens"] for r in emb_lat]) * PER)
    llm_ms = np.array([r["latency_ms"] for r in llm["te"]])
    llm_cost = cost_usd(
        LLM_MODEL,
        np.mean([r["input_tokens"] for r in llm["te"]]) * PER,
        np.mean([r["output_tokens"] for r in llm["te"]]) * PER,
    )

    results, preds, probas, oof, models, ms = {}, {}, {}, {}, {}, {}
    for key in ("A", "B", "C"):
        cv, oof[key] = ex.cross_validate(key, inputs[key]["trva"], trva.label.values)
        m = ex.FACTORIES[key]().fit(inputs[key]["tr"], tr.label.values)
        models[key] = m
        probas[key] = ex.proba_in_label_order(m, inputs[key]["te"])
        preds[key] = np.array(LABELS)[probas[key].argmax(1)]
        one = (lambda x, m=m: m.predict_proba([x])) if key == "A" else (lambda x, m=m: m.predict_proba(x[None]))
        local_ms = ex.per_message_ms(one, inputs[key]["te"], repeats=3)
        ms[key] = local_ms if key == "A" else local_ms + emb_ms
        results[key] = {"cv_macro_f1": cv, "cost_per_100k": 0.0 if key == "A" else emb_cost, "local_ms_p95": float(np.percentile(local_ms, 95))}
    preds["D"] = np.array([r["label"] for r in llm["te"]])
    ms["D"] = llm_ms
    results["D"] = {"cv_macro_f1": None, "cost_per_100k": llm_cost}

    human = (te.source == "human").values
    for key in "ABCD":
        lo, hi = ex.bootstrap_ci(te.label, preds[key])
        results[key].update({
            "name": ex.NAMES[key],
            "test_macro_f1": ex.macro_f1(te.label, preds[key]),
            "test_macro_f1_ci95": [lo, hi],
            "test_macro_f1_human_subset": ex.macro_f1(te.label[human], preds[key][human]),
            "p50_ms": float(np.percentile(ms[key], 50)),
            "p95_ms": float(np.percentile(ms[key], 95)),
            "per_class": classification_report(te.label, preds[key], labels=LABELS, output_dict=True, zero_division=0),
            "confused_pairs": ex.confused_pairs(te.label, preds[key]),
        })

    # Hybrid: tune the threshold on out-of-fold predictions over train+val (n≈1700), report on test.
    hybrid = {}
    llm_trva = np.array([r["label"] for r in llm["trva"]])
    llm_trva_f1 = ex.macro_f1(trva.label, llm_trva)
    for key in ("A", "B"):
        cheap_cost = results[key]["cost_per_100k"]
        # No latency probes on train+val; resample the test latency distributions (same population).
        rng = np.random.default_rng(0)
        cv_cheap_ms, cv_llm_ms = rng.choice(ms[key], len(trva)), rng.choice(llm_ms, len(trva))
        curve_cv = ex.hybrid_curve(oof[key], llm_trva, trva.label, cheap_cost, llm_cost, cv_cheap_ms, cv_llm_ms)
        curve_te = ex.hybrid_curve(probas[key], preds["D"], te.label, cheap_cost, llm_cost, ms[key], llm_ms)
        t = ex.pick_threshold(curve_cv, llm_trva_f1)
        curve_cv.to_csv(RESULTS / f"hybrid_{key}_cv.csv", index=False)
        curve_te.to_csv(RESULTS / f"hybrid_{key}_test.csv", index=False)
        hybrid[key] = {
            "chosen_threshold": t,
            "llm_only_cv_macro_f1": llm_trva_f1,
            "at_chosen": curve_te[curve_te.threshold == t].iloc[0].to_dict(),
            "at_0.8": curve_te[curve_te.threshold == 0.8].iloc[0].to_dict(),
            "cv_at_chosen": curve_cv[curve_cv.threshold == t].iloc[0].to_dict(),
        }

    # Error examples per confused pair, for the write-up.
    errors = {}
    for key in "ABCD":
        errors[key] = [
            {"true": a, "pred": b, "n": n, "examples": te.text[(te.label == a) & (preds[key] == b)].head(4).tolist()}
            for a, b, n in results[key]["confused_pairs"]
        ]

    (RESULTS / "metrics.json").write_text(json.dumps(
        {"llm_model": LLM_MODEL, "embed_model": EMBED_MODEL, "n_test": len(te), "n_test_human": int(human.sum()),
         "methods": results, "hybrid": hybrid, "errors": errors},
        ensure_ascii=False, indent=2, default=float,
    ))
    te.assign(**{f"pred_{k}": preds[k] for k in "ABCD"}).to_csv(RESULTS / "test_predictions.csv", index=False)
    write_tables(results, hybrid)
    plot_confusions(te.label, preds)
    plot_tradeoff(results, hybrid)

    MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": models["A"], "threshold": hybrid["A"]["chosen_threshold"], "labels": LABELS}, MODELS / "tfidf_lr.joblib")
    print((RESULTS / "comparison.md").read_text())


def write_tables(results, hybrid):
    lines = [
        "| Cách | Macro-F1 test (95% CI) | Macro-F1 CV 5-fold | p95 (ms) | $/100k tin nhắn |",
        "|---|---|---|---|---|",
    ]
    for k in "ABCD":
        r = results[k]
        cv = f"{np.mean(r['cv_macro_f1']):.3f} ± {np.std(r['cv_macro_f1']):.3f}" if r["cv_macro_f1"] else "—"
        lo, hi = r["test_macro_f1_ci95"]
        lines.append(f"| {r['name']} | {r['test_macro_f1']:.3f} ({lo:.2f}–{hi:.2f}) | {cv} | {r['p95_ms']:.1f} | {r['cost_per_100k']:.2f} |")
    lines += ["", "| Hybrid (ngưỡng chọn trên OOF train+val) | Ngưỡng | % chuyển LLM | Macro-F1 test | p95 (ms) | $/100k |", "|---|---|---|---|---|---|"]
    for k in ("A", "B"):
        for tag, row in (("chọn", hybrid[k]["at_chosen"]), ("0.8", hybrid[k]["at_0.8"])):
            lines.append(
                f"| {ex.NAMES[k][:2]} → D ({tag}) | {row['threshold']:.2f} | {row['routed_pct']:.1f}% | "
                f"{row['macro_f1']:.3f} | {row['p95_ms']:.0f} | {row['cost_per_100k']:.2f} |"
            )
    (RESULTS / "comparison.md").write_text("\n".join(lines) + "\n")


def plot_confusions(y_true, preds):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, k in zip(axes.flat, "ABCD"):
        cm = confusion_matrix(y_true, preds[k], labels=LABELS)
        ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30)
        ax.set_title(f"{ex.NAMES[k]}  (macro-F1 {ex.macro_f1(y_true, preds[k]):.3f})")
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrices.png", dpi=130)
    plt.close(fig)


def plot_tradeoff(results, hybrid):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ax = axes[0]
    colors = {"A": "#2a6fdb", "B": "#e07b00"}
    for k in ("A", "B"):
        c = ex.pd.read_csv(RESULTS / f"hybrid_{k}_test.csv")
        ax.plot(c.cost_per_100k, c.macro_f1, "-", color=colors[k], label=f"Hybrid {k} → D (quét ngưỡng)")
        ch = hybrid[k]["at_chosen"]
        ax.scatter(ch["cost_per_100k"], ch["macro_f1"], s=120, marker="*", color=colors[k], zorder=5,
                   label=f"{k}: ngưỡng chọn {ch['threshold']:.2f} ({ch['routed_pct']:.0f}% → LLM)")
    for k in "ABCD":
        r = results[k]
        ax.scatter(r["cost_per_100k"], r["test_macro_f1"], color="black", zorder=6)
        ax.annotate(k, (r["cost_per_100k"], r["test_macro_f1"]), textcoords="offset points", xytext=(6, -12))
    ax.set_xlabel("$ / 100k tin nhắn")
    ax.set_ylabel("Macro-F1 (test)")
    ax.set_title("Đánh đổi chi phí – chất lượng")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    for k in ("A", "B"):
        c = ex.pd.read_csv(RESULTS / f"hybrid_{k}_cv.csv")
        ax.plot(c.threshold, c.macro_f1, color=colors[k], label=f"{k}: macro-F1 (OOF)")
        ax2 = ax.twinx() if k == "A" else ax2
        ax2.plot(c.threshold, c.routed_pct, "--", color=colors[k], alpha=0.6, label=f"{k}: % chuyển LLM")
        ax.axvline(hybrid[k]["chosen_threshold"], color=colors[k], ls=":", alpha=0.8)
    ax.set_xlabel("Ngưỡng xác suất (cheap model trả lời khi max p > ngưỡng)")
    ax.set_ylabel("Macro-F1 (out-of-fold, train+val)")
    ax2.set_ylabel("% tin nhắn chuyển sang LLM")
    ax.set_title("Chọn ngưỡng trên out-of-fold train+val")
    ax.axhline(hybrid["A"]["llm_only_cv_macro_f1"], color="gray", ls="--", lw=1, label="LLM-only (OOF)")
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(RESULTS / "tradeoff.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    asyncio.run(main())
