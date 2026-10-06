"""Build data/intents.jsonl: 300 hand-labeled seeds + LLM-generated messages labeled by an LLM annotator.

Steps: generate → annotate → measure annotator/human agreement → audit sample → stratified 70/15/15 split.
Every LLM call is cached under data/cache/, so reruns are free and deterministic.
"""

import asyncio
import csv
import json
import random
import re
import unicodedata

from openai import AsyncOpenAI
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm.asyncio import tqdm_asyncio

from intent_router.config import CACHE, DATA, LABEL_MODEL, LABELS, LLM_MODEL, SEED
from intent_router.llm import ANNOTATOR_PROMPT, annotate_many

TARGETS = {"pricing": 380, "complaint": 355, "cancellation": 145, "tech_support": 445, "other": 375}
BATCH = 20
OVERSAMPLE = 1.2

# Prompts below are intentionally in Vietnamese: they produced the cached dataset and ask the
# LLM to write Vietnamese messages. Writing styles, in order: polite with diacritics; teencode
# abbreviations without diacritics; no diacritics at all; Vietnamese-English mix; long-winded with
# 2-3 sentences of context; very short (2-6 words); annoyed with many exclamation marks; typed in a
# hurry with typos; fully English; formal like a work email.
STYLES = [
    "lịch sự, viết đầy đủ dấu",
    "viết tắt kiểu teencode, không dấu (vd: ko, dc, j, z, ak)",
    "hoàn toàn không dấu",
    "pha tiếng Anh kiểu Vietglish",
    "dài dòng, kể bối cảnh 2-3 câu rồi mới hỏi",
    "cực ngắn, 2-6 từ",
    "bực bội, nhiều dấu chấm than",
    "gõ vội, có lỗi chính tả",
    "tiếng Anh hoàn toàn",
    "trang trọng như email công việc",
]
# Personas: startup owner, IT manager, accountant, student, freelancer, agency PM, HR staff,
# developer, SME director, teacher, sysadmin.
PERSONAS = [
    "chủ startup", "trưởng phòng IT", "kế toán", "sinh viên", "freelancer", "PM ở agency",
    "nhân viên HR", "lập trình viên", "giám đốc doanh nghiệp vừa", "giáo viên", "admin hệ thống",
]


class Batch(BaseModel):
    messages: list[str]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def load_seed() -> list[dict]:
    rows = []
    for line in (DATA / "seed_raw.tsv").read_text().splitlines():
        label, text = line.split("\t", 1)
        rows.append({"text": text.strip(), "label": label, "source": "human"})
    return rows


async def generate(seed: list[dict]) -> list[dict]:
    path = CACHE / "generated_raw.jsonl"
    done = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    have = {(r["gen_label"], r["batch"]) for r in done}
    guide = (DATA / "LABELING_GUIDE.vi.md").read_text()
    rng = random.Random(SEED)
    jobs = []
    for label, n in TARGETS.items():
        examples = [r["text"] for r in seed if r["label"] == label]
        for b in range(int(n * OVERSAMPLE / BATCH) + 1):
            styles = rng.sample(STYLES, 4)
            persona = rng.choice(PERSONAS)
            shots = rng.sample(examples, 6)
            if (label, b) not in have:
                jobs.append((label, b, styles, persona, shots))

    c = AsyncOpenAI(max_retries=6)
    sem = asyncio.Semaphore(8)

    async def one(label, b, styles, persona, shots):
        # "Write {BATCH} DIFFERENT, natural customer messages for label X. Sender: persona. Spread
        # these styles evenly. Vary sub-topics, length and openings; don't repeat the examples. ~20%
        # hard cases that sound like another label but belong to X under the tie-break rules.
        # No numbering, no explanations. Reference examples: ..."
        prompt = (
            f"{guide}\n\nViết {BATCH} tin nhắn khách hàng KHÁC NHAU, tự nhiên như người thật gõ vào "
            f"khung chat, thuộc nhãn `{label}`.\n"
            f"- Người gửi: {persona}.\n"
            f"- Phân bổ đều các văn phong: {'; '.join(styles)}.\n"
            "- Đa dạng chủ đề con, độ dài và cách mở đầu; không lặp ý của ví dụ.\n"
            "- Khoảng 20% là ca khó: nghe gần một nhãn khác nhưng theo quy tắc phân xử vẫn "
            f"thuộc `{label}`.\n"
            "- Không đánh số, không giải thích.\n\nVí dụ tham khảo:\n"
            + "\n".join(f"- {s}" for s in shots)
        )
        async with sem:
            resp = await c.chat.completions.parse(
                model=LLM_MODEL,
                temperature=1.0,
                seed=SEED + b,
                messages=[{"role": "user", "content": prompt}],
                response_format=Batch,
            )
        rows = [
            {"gen_label": label, "batch": b, "text": m.strip()}
            for m in resp.choices[0].message.parsed.messages
            if m.strip()
        ]
        with path.open("a") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        done.extend(rows)

    if jobs:
        path.parent.mkdir(parents=True, exist_ok=True)
        await tqdm_asyncio.gather(*(one(*j) for j in jobs), desc="generate")

    seen = {normalize(r["text"]) for r in seed}
    out = []
    for label, n in TARGETS.items():
        pool = sorted((r for r in done if r["gen_label"] == label), key=lambda r: (r["batch"], r["text"]))
        rng.shuffle(pool)
        kept = []
        for r in pool:
            key = normalize(r["text"])
            if key and key not in seen and len(kept) < n:
                seen.add(key)
                kept.append({"text": r["text"], "gen_label": label, "source": "llm"})
        out.extend(kept)
    return out


def write_audit_sample(generated: list[dict], k: int = 100) -> None:
    """Random sample of LLM-labeled rows for human review. human_label is filled by hand."""
    path = DATA / "audit_sample.csv"
    if path.exists():
        return
    sample = random.Random(SEED).sample(generated, k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "llm_label", "human_label"])
        w.writeheader()
        for r in sample:
            w.writerow({"text": r["text"], "llm_label": r["label"], "human_label": ""})


def read_audit() -> dict | None:
    rows = list(csv.DictReader((DATA / "audit_sample.csv").open()))
    if not all(r["human_label"] for r in rows):
        return None
    y_h = [r["human_label"] for r in rows]
    y_l = [r["llm_label"] for r in rows]
    return {
        "n": len(rows),
        "agreement": accuracy_score(y_h, y_l),
        "disagreements": [r for r in rows if r["human_label"] != r["llm_label"]],
    }


async def main():
    seed = load_seed()
    generated = await generate(seed)
    rows = seed + generated

    ann = await annotate_many(
        [r["text"] for r in rows], LABEL_MODEL, ANNOTATOR_PROMPT, CACHE / f"annotator_{LABEL_MODEL}.jsonl"
    )
    for r, a in zip(rows, ann):
        r["llm_label"] = a["label"]
        if r["source"] == "llm":
            r["label"] = a["label"]

    y_h = [r["label"] for r in seed]
    y_l = [r["llm_label"] for r in seed]
    report = {
        "n_total": len(rows),
        "label_counts": {l: sum(r["label"] == l for r in rows) for l in LABELS},
        "annotator_model": LABEL_MODEL,
        "generator_model": LLM_MODEL,
        "seed_agreement": {
            "n": len(seed),
            "accuracy": accuracy_score(y_h, y_l),
            "cohen_kappa": cohen_kappa_score(y_h, y_l),
            "per_class": classification_report(y_h, y_l, labels=LABELS, output_dict=True, zero_division=0),
            "disagreements": [
                {"text": r["text"], "human": r["label"], "llm": r["llm_label"]}
                for r in seed
                if r["label"] != r["llm_label"]
            ],
        },
        "generation_intent_agreement": sum(r["gen_label"] == r["label"] for r in generated) / len(generated),
    }

    write_audit_sample(generated)
    report["audit"] = read_audit()
    if report["audit"]:
        fixes = {d["text"]: d["human_label"] for d in report["audit"]["disagreements"]}
        for r in generated:
            r["label"] = fixes.get(r["text"], r["label"])
        report["label_counts"] = {l: sum(r["label"] == l for r in rows) for l in LABELS}

    idx = list(range(len(rows)))
    labels = [r["label"] for r in rows]
    train, rest = train_test_split(idx, test_size=0.30, stratify=labels, random_state=SEED)
    val, test = train_test_split(
        rest, test_size=0.50, stratify=[labels[i] for i in rest], random_state=SEED
    )
    split = {**{i: "train" for i in train}, **{i: "val" for i in val}, **{i: "test" for i in test}}
    with (DATA / "intents.jsonl").open("w") as f:
        for i, r in enumerate(rows):
            out = {"text": r["text"], "label": r["label"], "split": split[i], "source": r["source"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    report["split_counts"] = {
        s: {l: sum(split[i] == s and labels[i] == l for i in idx) for l in LABELS}
        for s in ("train", "val", "test")
    }
    (DATA / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    sa = report["seed_agreement"]
    print(f"rows={len(rows)} counts={report['label_counts']}")
    print(f"annotator vs human (seed): acc={sa['accuracy']:.3f} kappa={sa['cohen_kappa']:.3f}")
    print(f"generation intent agreement: {report['generation_intent_agreement']:.3f}")
    print(f"audit: {report['audit'] and round(report['audit']['agreement'], 3)}")


if __name__ == "__main__":
    asyncio.run(main())
