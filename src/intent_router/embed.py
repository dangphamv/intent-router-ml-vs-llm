import asyncio
import json
import time
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI, OpenAI

from intent_router.config import CACHE, EMBED_MODEL


def _cache_path(model: str) -> Path:
    return CACHE / f"embeddings_{model}.npz"


def embed_texts(texts: list[str], model: str = EMBED_MODEL, batch_size: int = 256) -> np.ndarray:
    """Embed texts via the API, caching vectors on disk (float16) so reruns need no key."""
    path = _cache_path(model)
    cache: dict[str, np.ndarray] = {}
    if path.exists():
        z = np.load(path, allow_pickle=False)
        cache = dict(zip(z["texts"].tolist(), z["vectors"]))
    todo = list(dict.fromkeys(t for t in texts if t not in cache))
    if todo:
        c = OpenAI(max_retries=6)
        for i in range(0, len(todo), batch_size):
            batch = todo[i : i + batch_size]
            resp = c.embeddings.create(model=model, input=batch)
            for t, d in zip(batch, resp.data):
                cache[t] = np.asarray(d.embedding, dtype=np.float16)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(cache)
        np.savez_compressed(path, texts=np.array(keys), vectors=np.stack([cache[k] for k in keys]))
    return np.stack([cache[t] for t in texts]).astype(np.float32)


async def measure_latency(
    texts: list[str], cache_path: Path, model: str = EMBED_MODEL, concurrency: int = 4
) -> list[dict]:
    """One API call per message, as a router would make online. Cached as JSONL."""
    done = {}
    if cache_path.exists():
        done = {r["text"]: r for r in map(json.loads, cache_path.read_text().splitlines())}
    todo = [t for t in dict.fromkeys(texts) if t not in done]
    if todo:
        c = AsyncOpenAI(max_retries=6)
        sem = asyncio.Semaphore(concurrency)

        async def one(t: str):
            async with sem:
                start = time.perf_counter()
                resp = await c.embeddings.create(model=model, input=t)
                r = {
                    "text": t,
                    "latency_ms": (time.perf_counter() - start) * 1000,
                    "input_tokens": resp.usage.prompt_tokens,
                }
            with cache_path.open("a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            done[t] = r

        await asyncio.gather(*(one(t) for t in todo))
    return [done[t] for t in texts]
