from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_BUDGET_TOKENS = 500_000
DEFAULT_MODEL = "qwen3.6-35b-a3b"

MODEL_CATALOG = {
    "qwen3.8-27b": {"cost_weight": 1.0, "reasoning": True, "speed": "fast"},
    "qwen3.8-27b-noreason": {"cost_weight": 0.7, "reasoning": False, "speed": "fastest"},
    "qwen3.6-35b-a3b": {"cost_weight": 1.8, "reasoning": True, "speed": "balanced"},
    "qwen3.6-35b-a3b-noreason": {"cost_weight": 1.2, "reasoning": False, "speed": "balanced"},
}

PLAYGROUND_TASK_REWARD = 5
