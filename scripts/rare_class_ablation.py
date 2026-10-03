"""Does class weighting help the rare class (`cancellation`)? CV on train+val, no API calls (embeddings cached)."""

import json

import numpy as np
import pandas as pd

from intent_router import experiments as ex
from intent_router.config import RESULTS
from intent_router.embed import embed_texts

RARE = "cancellation"
KEEP = (1.0, 0.25)  # natural ~10% share, and an artificially rarer ~3% share


def main():
    df = ex.load_data()
    trva = df[df.split != "test"].reset_index(drop=True)
    X = {"A": trva.text.tolist(), "B": embed_texts(trva.text.tolist())}
    X["C"] = X["B"]
    rows = [r for k in "ABC" for keep in KEEP for r in ex.rare_class_ablation(k, X[k], trva.label.values, RARE, keep)]
    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS / "rare_class_ablation_folds.csv", index=False)

    summary = (
        raw.groupby(["method", "keep_frac", "balanced"])
        .agg(rare_share_train=("rare_share_train", "mean"), macro_f1=("macro_f1", "mean"),
             rare_precision=("rare_precision", "mean"), rare_recall=("rare_recall", "mean"),
             rare_f1=("rare_f1", "mean"), rare_f1_std=("rare_f1", "std"))
        .reset_index()
    )
    (RESULTS / "rare_class_ablation.json").write_text(json.dumps(summary.to_dict("records"), indent=2))

    lines = [
        f"| Cách | `{RARE}` trong train | Class weight | Macro-F1 | {RARE} precision | {RARE} recall | {RARE} F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in summary.itertuples():
        lines.append(
            f"| {ex.NAMES[r.method]} | {r.rare_share_train:.1%} | {'balanced' if r.balanced else 'không'} | "
            f"{r.macro_f1:.3f} | {r.rare_precision:.3f} | {r.rare_recall:.3f} | {r.rare_f1:.3f} ± {r.rare_f1_std:.3f} |"
        )
    (RESULTS / "rare_class_ablation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
