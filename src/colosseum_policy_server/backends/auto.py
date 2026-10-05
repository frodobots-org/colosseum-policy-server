"""Dispatch registered model types without changing existing VLA adapters."""
from .droid import DroidBackend
from .inspect_agent import InspectAgentBackend


class AutoBackend:
    def __init__(self, options):
        self.options, self.active = dict(options), None

    async def start_session(self, model, request):
        if model.model_type == 'llm':
            self.active = InspectAgentBackend(self.options.get('llm', {}))
            await self.active.start_session(model, request)
        else:
            self.active = DroidBackend(self.options.get('vla', {}))

    async def infer(self, model, observation):
        if self.active is None:
            raise RuntimeError('Model session is not prepared')
        return await self.active.infer(model, observation)

    async def end_session(self):
        active, self.active = self.active, None
        if active is not None and hasattr(active, 'end_session'):
            await active.end_session()


def create_backend(options):
    return AutoBackend(options)
