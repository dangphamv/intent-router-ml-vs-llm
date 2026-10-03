from dataclasses import dataclass
from pathlib import Path

import joblib
from openai import AsyncOpenAI

from intent_router.config import LLM_MODEL, MODELS
from intent_router.llm import ZERO_SHOT_PROMPT, classify


@dataclass
class Route:
    label: str
    confidence: float | None
    handled_by: str  # "local" or the LLM model name


class IntentRouter:
    """TF-IDF + LR answers when confident; anything at or below the threshold goes to the LLM."""

    def __init__(self, model, threshold: float, llm_model: str = LLM_MODEL):
        self.model = model
        self.threshold = threshold
        self.llm_model = llm_model
        self._client: AsyncOpenAI | None = None

    @classmethod
    def load(cls, path: Path = MODELS / "tfidf_lr.joblib", threshold: float | None = None) -> "IntentRouter":
        bundle = joblib.load(path)
        return cls(bundle["model"], bundle["threshold"] if threshold is None else threshold)

    def predict_local(self, text: str) -> tuple[str, float]:
        p = self.model.predict_proba([text])[0]
        i = p.argmax()
        return str(self.model.classes_[i]), float(p[i])

    async def route(self, text: str) -> Route:
        label, conf = self.predict_local(text)
        if conf > self.threshold:
            return Route(label, conf, "local")
        self._client = self._client or AsyncOpenAI(max_retries=3)
        r = await classify(self._client, text, self.llm_model, ZERO_SHOT_PROMPT)
        return Route(r["label"], conf, self.llm_model)
