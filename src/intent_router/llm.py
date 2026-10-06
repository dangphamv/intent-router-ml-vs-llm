import asyncio
import json
import time
from pathlib import Path
from typing import Literal

from openai import AsyncOpenAI, RateLimitError
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

from intent_router.config import DATA, LABELS

Label = Literal["pricing", "complaint", "cancellation", "tech_support", "other"]


class IntentPrediction(BaseModel):
    label: Label


ZERO_SHOT_PROMPT = f"""Classify the customer message sent to a SaaS project-management product's support channel.
Return exactly one label: {", ".join(LABELS)}.
- pricing: questions about prices, plans, quotas, discounts, payment methods
- complaint: dissatisfaction, billing errors, slow support, demands for refunds
- cancellation: cancel, downgrade, pause, stop renewal, delete/close account
- tech_support: how-to questions, bugs, login, integrations, API
- other: greetings, thanks, feature ideas, jobs, partnerships, off-topic"""

ANNOTATOR_PROMPT = (
    "You are a careful data annotator. Label the customer message with exactly one intent "
    "following this guideline (written in Vietnamese):\n\n"
    + (DATA / "LABELING_GUIDE.vi.md").read_text()
)


def client() -> AsyncOpenAI:
    return AsyncOpenAI(max_retries=6)


async def classify(c: AsyncOpenAI, text: str, model: str, system: str) -> dict:
    start = time.perf_counter()
    resp = await c.chat.completions.parse(
        model=model,
        temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
        response_format=IntentPrediction,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    parsed = resp.choices[0].message.parsed
    return {
        "text": text,
        "label": parsed.label if parsed else "other",
        "latency_ms": latency_ms,
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {r["text"]: r for r in rows}


async def classify_many(
    texts: list[str], model: str, system: str, cache_path: Path, concurrency: int = 8
) -> list[dict]:
    """Classify texts, reusing and appending to a JSONL cache keyed by text."""
    cache = load_cache(cache_path)
    todo = list(dict.fromkeys(t for t in texts if t not in cache))
    if todo:
        c = client()
        sem = asyncio.Semaphore(concurrency)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        async def one(t: str):
            async with sem:
                r = await classify(c, t, model, system)
            with cache_path.open("a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            cache[t] = r

        await tqdm_asyncio.gather(*(one(t) for t in todo), desc=cache_path.stem)
    return [cache[t] for t in texts]


class BatchItem(BaseModel):
    id: int
    label: Label


class BatchLabels(BaseModel):
    items: list[BatchItem]


async def annotate_many(
    texts: list[str], model: str, system: str, cache_path: Path, batch_size: int = 25, concurrency: int = 1
) -> list[dict]:
    """Label many messages per call (the annotator model is rate-limited); same cache format as classify_many."""
    cache = load_cache(cache_path)
    todo = list(dict.fromkeys(t for t in texts if t not in cache))
    if todo:
        c = client()
        sem = asyncio.Semaphore(concurrency)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        async def one(batch: list[str]):
            body = "\n".join(f"[{i}] {t}" for i, t in enumerate(batch))
            async with sem:
                for attempt in range(20):
                    try:
                        resp = await c.chat.completions.parse(
                            model=model,
                            temperature=0,
                            messages=[
                                {"role": "system", "content": system + "\n\nYou will receive numbered messages. Return one item per id."},
                                {"role": "user", "content": body},
                            ],
                            response_format=BatchLabels,
                        )
                        break
                    except RateLimitError:
                        await asyncio.sleep(15)
            got = {it.id: it.label for it in resp.choices[0].message.parsed.items}
            with cache_path.open("a") as f:
                for i, t in enumerate(batch):
                    if i in got:
                        r = {"text": t, "label": got[i]}
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        cache[t] = r

        batches = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]
        await tqdm_asyncio.gather(*(one(b) for b in batches), desc=cache_path.stem)
        missing = [t for t in todo if t not in cache]
        if missing:
            return await annotate_many(texts, model, system, cache_path, batch_size, concurrency)
    return [cache[t] for t in texts]
