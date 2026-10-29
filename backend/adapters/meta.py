from __future__ import annotations

from backend.adapters.base import AdapterDisabledError, CampaignPlan, ExecutionAdapter, ExecutionResult


class MetaAdapter(ExecutionAdapter):
    """Placeholder for publishing to Meta. Disabled and NOT implemented.

    Do not enable this with anyone else's app, token or ad account. Use your own
    test assets, or written permission from their owner, and create campaigns
    paused with a spend cap.
    """

    name = "meta"

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def execute(self, plan: CampaignPlan) -> ExecutionResult:
        if not self.enabled:
            raise AdapterDisabledError("The Meta adapter is disabled.")
        raise NotImplementedError("Meta publishing is not implemented yet.")
