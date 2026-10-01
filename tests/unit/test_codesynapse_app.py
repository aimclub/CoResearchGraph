from fastapi.testclient import TestClient

from CoScientist.integrations.codesynapse.app import create_app
from CoScientist.integrations.codesynapse.facade import CodesynapseFacade, StartRequest
from CoScientist.integrations.codesynapse.settings import CodesynapseIntegrationSettings
from CoScientist.integrations.codesynapse.store import InMemoryIntegrationStore


class _Executor:
    async def execute(self, request, hitl_handler):
        return "report"


def test_jwtless_app_keeps_capability_protected_control_plane():
    """Removing start-request JWT must not expose HITL/cancel/replay routes."""

    store = InMemoryIntegrationStore()
    app = create_app(
        CodesynapseIntegrationSettings(a2a_public_url="http://testserver"),
        facade=CodesynapseFacade(store=store, executor=_Executor()),
        store=store,
    )

    response = TestClient(app).post("/internal/runs/run-1/cancel")

    assert response.status_code == 401


def test_jwtless_app_installs_trace_delivery_for_capability_metadata():
    store = InMemoryIntegrationStore()
    facade = CodesynapseFacade(store=store, executor=_Executor())
    create_app(
        CodesynapseIntegrationSettings(a2a_public_url="http://testserver"),
        facade=facade,
        store=store,
    )

    dispatcher = facade._delivery_factory(
        StartRequest(
            external_run_id="external-1",
            tenant_id="root",
            project_id="project-1",
            research_request="Find a hypothesis",
            trace_callback_url="http://codesynapse.internal/events",
            trace_capability_token="trace-capability",
        )
    )

    assert dispatcher is not None
