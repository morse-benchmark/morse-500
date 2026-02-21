"""Reward function for GRPO training on temporal_reasoning.

This is intentionally simple: reward is 1.0 for an exact normalized match,
otherwise 0.0. Customize `normalize_answer` for partial credit or
category-specific scoring.
"""

from __future__ import annotations

import re
from typing import Iterable, List


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def normalize_answer(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    boxed = _BOXED_RE.findall(text)
    if boxed:
        text = boxed[-1]
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def score_answer(prediction: str, target: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(target) else 0.0


def reward_fn(samples: Iterable[dict], **_: object) -> List[float]:
    """Compute rewards for a batch of samples.

    Expected sample keys (choose one path and keep your verl config aligned):
      - prediction: `response` or `output_text`
      - reference: `answer` or `reference`
    """
    rewards: List[float] = []
    for sample in samples:
        prediction = sample.get("response") or sample.get("output_text") or ""
        target = sample.get("answer") or sample.get("reference") or ""
        rewards.append(score_answer(str(prediction), str(target)))
    return rewards
