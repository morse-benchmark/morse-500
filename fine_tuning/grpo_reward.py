"""Reward function for VERL GRPO training.

Implements VERL custom reward API:
compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
"""

from __future__ import annotations

import re
from typing import Any


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


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | list[str] | None,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> float:
    """Return scalar reward for one rollout sample."""
    del data_source, extra_info
    if ground_truth is None:
        return 0.0
    if isinstance(ground_truth, list):
        return max(score_answer(solution_str, str(target)) for target in ground_truth) if ground_truth else 0.0
    return score_answer(solution_str, str(ground_truth))
