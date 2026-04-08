from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineTimer:
    """Collect per-section wall times (ms) when enabled."""

    enabled: bool
    steps: dict[str, float] = field(default_factory=dict)

    def start(self) -> float:
        return time.perf_counter()

    def record(self, name: str, section_start: float) -> float:
        """Record milliseconds for the segment that ended at `section_start`. Returns new start time."""
        end = time.perf_counter()
        if self.enabled:
            self.steps[name] = (end - section_start) * 1000.0
        return end

    def log(self, operation: str, **context: str | int | float) -> None:
        if not self.enabled or not self.steps:
            return
        total_ms = sum(self.steps.values())
        step_parts = " ".join(
            f"{key}={value:.2f}ms" for key, value in sorted(self.steps.items())
        )
        ctx_parts = " ".join(
            f"{key}={value}" for key, value in sorted(context.items())
        )
        message = (
            f"pipeline_timing operation={operation} {step_parts} "
            f"total_ms={total_ms:.2f}ms"
        )
        if ctx_parts:
            message = f"{message} {ctx_parts}"
        logger.info(message)
