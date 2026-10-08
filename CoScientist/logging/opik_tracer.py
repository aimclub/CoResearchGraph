import os

from CoScientist.config import get_settings

_tracers = {}

def get_multi_agent_tracer():
    settings = get_settings()

    # Read from WebSettings (runtime mutable) for enabled, standard settings for keys
    enabled = settings.web.opik_enabled
    api_key = settings.opik.api_key
    project_name = settings.opik.opik_project_name or "adk-coscientist"

    if not enabled:
        os.environ["OPIK_TRACK_DISABLE"] = "true"
        return None

    # Clear trace disable flag if tracking is active
    os.environ.pop("OPIK_TRACK_DISABLE", None)

    main_model = settings.llm.main_model
    coder_model = settings.llm.coder_model or settings.llm.main_model

    # Cache key based on settings to avoid recreating if settings didn't change
    cache_key = (api_key, project_name, main_model, coder_model)
    if cache_key in _tracers:
        return _tracers[cache_key]

    url_override = settings.opik.url_override

    if api_key:
        os.environ["OPIK_API_KEY"] = api_key
    else:
        os.environ.pop("OPIK_API_KEY", None)
    if url_override:
        os.environ["OPIK_URL_OVERRIDE"] = url_override
    else:
        os.environ.pop("OPIK_URL_OVERRIDE", None)
    os.environ["OPIK_PROJECT_NAME"] = project_name

    import opik

    # Don't let an opik misconfiguration (no key, no network) take down the app
    # on import — tracing is best-effort.
    try:
        opik.configure(
            api_key=api_key or None,
            url_override=url_override or None,
            project_name=project_name,
            use_local=False,
            install_mcp=False,
            #automatic_approvals=True,
        )
    except Exception as e:  # pragma: no cover - best-effort tracing setup
        print(f"[opik] configure failed, tracing may be disabled: {e!r}")

    if api_key:
        os.environ["OPIK_API_KEY"] = api_key
    if url_override:
        os.environ["OPIK_URL_OVERRIDE"] = url_override
    os.environ["OPIK_PROJECT_NAME"] = project_name

    from opik.integrations.adk import OpikTracer

    # Avoid dumping the full settings (which include API keys/passwords) into
    # trace metadata — only expose non-secret descriptors.
    _safe_metadata = {
        "main_model": settings.llm.main_model,
        "coder_model": settings.llm.coder_model or settings.llm.main_model,
    }

    tracer = OpikTracer(
        name="multi-agent-orchestrator",
        metadata=_safe_metadata,
        project_name=project_name,
    )
    _tracers[cache_key] = tracer
    return tracer


# Import-time initialization for CLI and module-level backwards compatibility
multi_agent_tracer = get_multi_agent_tracer()
