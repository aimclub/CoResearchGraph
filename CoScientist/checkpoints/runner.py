"""Tie the process-wide restore gate to the lifetime of actual ADK invocations."""

from contextlib import ExitStack, aclosing

from google.adk.runners import Runner

from CoScientist.checkpoints.plugin import CheckpointPlugin


class CheckpointRunner(Runner):
    """Keep restore blocked through execution and iterator cleanup, without a TTL."""

    async def run_async(self, **kwargs):
        with ExitStack() as scopes:
            for plugin in self.plugin_manager.plugins:
                if isinstance(plugin, CheckpointPlugin):
                    scopes.enter_context(plugin.track_run())
            # ADK can skip after_run_callback on errors. Close its iterator
            # before releasing the gate on success, failure, cancellation or
            # early stream closure; every concurrent invocation owns a scope.
            async with aclosing(super().run_async(**kwargs)) as events:
                async for event in events:
                    yield event
