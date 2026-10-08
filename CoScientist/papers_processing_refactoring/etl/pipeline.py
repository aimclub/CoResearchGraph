from typing import Sequence

from .base import ETLStep
from .context import ETLContext


class ETLPipeline:
    def __init__(self, steps: Sequence[ETLStep]):
        self.steps = steps

    def run(self, ctx: ETLContext) -> ETLContext:
        for step in self.steps:
            step.execute(ctx)
        return ctx
