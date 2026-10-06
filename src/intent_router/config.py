import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = DATA / "cache"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

load_dotenv(ROOT / ".env")

LABELS = ["pricing", "complaint", "cancellation", "tech_support", "other"]

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LABEL_MODEL = os.getenv("LABEL_MODEL", "gpt-4o")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
SEED = 42

# USD per 1M tokens (OpenAI list price, standard tier).
PRICES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


def cost_usd(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    p = PRICES[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1e6
