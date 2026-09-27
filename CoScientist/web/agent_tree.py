"""Session-scoped projection of the YAML agent system for the Web UI."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

from CoScientist.assembly.registry import REGISTRY
from CoScientist.assembly.schema import COMPOSITE_CLASSES, PIPELINE_ROOT_NAME, SystemConfig
from CoScientist.config import settings_scope
from CoScientist.config.settings import Settings


_MCP_SERVERS = {"websearch": "Tavily", "paper_analysis": "paper-analysis", "papers_search": "papers-search", "vault": "vault"}

_TITLES: dict[str, tuple[str, str]] = {
    "OrchestratorAgent": ("Координатор исследования", "Research coordinator"),
    "RootOrchestrator": ("Координатор синтеза", "Synthesis coordinator"),
    "PlannerAgent": ("Планировщик исследования", "Research planner"),
    "ContextInitAgent": ("Подготовка исследования", "Research framing"),
    "TZSpecAgent": ("Техническое задание", "Technical specification"),
    "TZQueryGenAgent": ("Поисковые запросы", "Search query preparation"),
    "ResultAggregatorAgent": ("Итоговый отчёт", "Result synthesis"),
    "NirReportAgent": ("Отчёт НИР", "Research report"),
    "HypothesesAgent": ("Генератор гипотез", "Hypothesis generator"),
    "ResearchAgent": ("Исследователь литературы", "Literature researcher"),
    "TaskExecutorAgent": ("Исполнитель задач", "Task executor"),
    "ExperimentModuleAgent": ("Экспериментальный модуль", "Experiment module"),
    "ExperimentPlannerAgent": ("Планировщик эксперимента", "Experiment planner"),
    "ExperimentExecutorAgent": ("Исполнитель эксперимента", "Experiment executor"),
    "ExperimentResultReviewAgent": ("Проверка результатов", "Result review"),
    "ExperimentAgent": ("ReAct: MCP-инструменты", "ReAct MCP tools"),
    "FedotAgent": ("FEDOT.MAS", "FEDOT.MAS"),
    "CoderAgent": ("Разработчик", "Coder"),
    "DatasetCollectorAgent": ("Сборщик данных", "Dataset collector"),
    "MedicalAgent": ("Медицинский эксперт", "Medical specialist"),
    "McpBuilderAgent": ("Создатель MCP-инструментов", "MCP tool builder"),
    "LiteratureOrchestrator": ("Анализ литературы", "Literature analysis"),
    "PaperRetriever": ("Аналитик публикаций", "Paper analyst"),
    "LiteratureSynthesisAgent": ("Обобщение литературы", "Literature synthesis"),
    "EvidenceVerifierAgent": ("Проверка источников", "Evidence verification"),
    "RouteSelectionAgent": ("Выбор маршрута синтеза", "Synthesis route selection"),
    "MolDesignAgent": ("Подбор молекулы", "Molecule selection"),
    "SynthRouteAgent": ("Маршруты синтеза", "Synthesis routes"),
    "EconomicsAgent": ("Расчёт стоимости", "Cost estimation"),
    "OptimizationAgent": ("Оптимизация условий синтеза", "Synthesis optimization"),
    "ReactorAgent": ("Эксперименты на реакторе", "Reactor experiments"),
    "ReportAgent": ("Итоговый отчёт по синтезу", "Synthesis report"),
}


# UI copy is deliberately separate from the English model-facing descriptions.
_PRESENTATION: dict[str, tuple[str, str]] = {
    "OrchestratorAgent": ("coord", "Распределяет задачи между агентами, учитывает их результаты и управляет ходом исследования. Может повторно обращаться к агентам, если нужны дополнительные данные или вычисления."),
    "RootOrchestrator": ("coord", "Управляет исследованием синтеза: от технического задания и анализа литературы до выбора молекул, оптимизации условий и отчёта."),
    "PlannerAgent": ("plan", "Составляет общий план исследования: определяет задачи, ожидаемые результаты и порядок работы."),
    "ContextInitAgent": ("context", "Уточняет цель, ограничения и исходные данные. Формирует контекст, на который опираются остальные агенты."),
    "TZSpecAgent": ("document", "Формирует и согласует техническое задание: требования к результату, ограничения и критерии проверки."),
    "TZQueryGenAgent": ("search", "Преобразует техническое задание в конкретные вопросы для поиска научной литературы."),
    "ResultAggregatorAgent": ("report", "Объединяет выводы, доказательства, таблицы и иллюстрации в итоговый отчёт. При запросе отчёта НИР обращается к соответствующему агенту."),
    "NirReportAgent": ("report", "Готовит отчёт о научно-исследовательской работе по ГОСТ и формирует документ для скачивания. Вызывается при включённой функции и согласии пользователя."),
    "HypothesesAgent": ("idea", "Предлагает проверяемые научные гипотезы, уточняет их по найденным данным и результатам экспериментов."),
    "ResearchAgent": ("search", "Ищет и анализирует научные публикации, извлекает факты и возвращает выводы со ссылками на источники."),
    "TaskExecutorAgent": ("execute", "Выбирает подходящего исполнителя для задачи и собирает результат вычислений или работы с данными."),
    "ExperimentModuleAgent": ("experiment", "Организует вычислительный эксперимент: подготовку инструментов, согласование плана, исполнение и проверку результатов. Может вызываться повторно для следующего эксперимента."),
    "ExperimentPlannerAgent": ("plan", "Составляет план вычислительного эксперимента, определяет необходимые действия и передаёт план на согласование."),
    "ExperimentExecutorAgent": ("execute", "Выполняет задачи согласованного плана через подходящих агентов. Учитывает результаты попыток и выбирает дальнейшие действия."),
    "ExperimentResultReviewAgent": ("review", "Обобщает результаты эксперимента, проверяет выполнение плана и передаёт результаты пользователю на согласование."),
    "ExperimentAgent": ("tools", "Выполняет вычисления и обрабатывает данные с помощью MCP-инструментов, выбранных для задачи."),
    "FedotAgent": ("ml", "Передаёт задачу системе FEDOT.MAS для автоматизированного анализа данных и машинного обучения, затем возвращает результаты."),
    "CoderAgent": ("code", "Пишет и запускает код для вычислений, обработки данных и построения графиков. При необходимости поручает сбор данных отдельному агенту."),
    "DatasetCollectorAgent": ("data", "Находит и подготавливает наборы данных, необходимые для вычислений и обучения моделей."),
    "MedicalAgent": ("medical", "Анализирует медицинские вопросы и научные источники, выполняет доступные профильные задачи обработки медицинских данных."),
    "McpBuilderAgent": ("tools", "Создаёт и подключает MCP-инструменты, когда для выполнения задачи не хватает существующих возможностей."),
    "LiteratureOrchestrator": ("search", "Распределяет вопросы технического задания между исследовательскими вызовами и собирает найденные материалы."),
    "PaperRetriever": ("search", "Находит релевантные публикации и извлекает из них сведения для исследования."),
    "LiteratureSynthesisAgent": ("document", "Обобщает найденные публикации, сравнивает подходы и формирует структурированный обзор литературы."),
    "EvidenceVerifierAgent": ("review", "Проверяет, подтверждаются ли выводы указанными источниками, и отмечает пробелы в доказательствах."),
    "RouteSelectionAgent": ("plan", "Сравнивает найденные маршруты синтеза и помогает выбрать маршрут для дальнейшей работы."),
    "MolDesignAgent": ("experiment", "Подбирает молекулы и их характеристики с учётом требований технического задания и выбранного маршрута синтеза."),
    "SynthRouteAgent": ("plan", "Разрабатывает и сравнивает возможные маршруты получения целевого соединения."),
    "EconomicsAgent": ("economics", "Оценивает стоимость выбранных вариантов синтеза по доступным данным о реагентах и условиях процесса."),
    "OptimizationAgent": ("experiment", "Организует подбор условий синтеза через внешнюю экспериментальную систему и собирает результаты оптимизации."),
    "ReactorAgent": ("experiment", "Передаёт задания внешней системе управления реактором и получает результаты экспериментов."),
    "ReportAgent": ("report", "Собирает результаты исследования синтеза в итоговый отчёт с обоснованиями и полученными данными."),
}


def _short(text: str, limit: int = 220) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    cut = value.rfind(". ", 0, limit)
    return value[: cut + 1 if cut >= 60 else limit].rstrip() + "…"


def _title(name: str, declared: str) -> dict[str, str]:
    known = _TITLES.get(name)
    if known:
        return {"ru": known[0], "en": known[1]}
    fallback = declared or name
    return {"ru": declared if any("а" <= char.lower() <= "я" for char in declared or "") else "Агент исследования", "en": fallback}


def agent_presentation(name: str, title: str = "", description: str = "") -> dict[str, Any]:
    """Shared UI copy for visible cards and the catalog of disabled agents."""
    icon, description_ru = _PRESENTATION.get(name, ("agent", "Описание на русском пока не добавлено."))
    return {
        "title": _title(name, title),
        "descriptionLocalized": {"ru": description_ru, "en": _short(description)},
        "icon": icon,
    }


def _full_graph(config: SystemConfig) -> tuple[dict[str, list[tuple[str, str]]], set[str]]:
    """Relations actually attached by the assembler, including its run wrapper."""
    graph: dict[str, list[tuple[str, str]]] = defaultdict(list)
    attached: set[str] = set()

    pre = [name for name in config.pipeline.pre if config.agent(name).is_enabled()]
    post = [name for name in config.pipeline.post if config.agent(name).is_enabled()]
    wrapper = bool(pre or post)
    entry = PIPELINE_ROOT_NAME if wrapper else config.root.name
    if wrapper:
        attached.add(PIPELINE_ROOT_NAME)
        graph[PIPELINE_ROOT_NAME].extend((name, "pipeline_pre") for name in pre)
        graph[PIPELINE_ROOT_NAME].append((config.root.name, "root"))
        graph[PIPELINE_ROOT_NAME].extend((name, "pipeline_post") for name in post)

    queue = [entry]
    while queue:
        name = queue.pop(0)
        if name in attached and name != PIPELINE_ROOT_NAME:
            continue
        attached.add(name)
        if name == PIPELINE_ROOT_NAME:
            queue.extend(child for child, _ in graph[name])
            continue
        agent = config.agent(name)
        relations: list[tuple[str, str]] = []
        relations.extend(
            (child, "delegate")
            for child in agent.subordinates
            if config.agent(child).is_enabled()
        )
        if agent.cls in COMPOSITE_CLASSES:
            # Composite assembly intentionally includes every declared child.
            if agent.is_enabled():
                relation = "sequence" if agent.cls == "sequential" else agent.cls
                relations.extend((child, relation) for child in agent.children)
        else:
            relations.extend(
                (child, "choice" if agent.cls == "custom:executor_switch" else "delegate")
                for child in agent.children
                if config.agent(child).is_enabled()
            )
        graph[name].extend(relations)
        queue.extend(child for child, _ in relations)
    return graph, attached


def project_agent_tree(
    run_settings: Settings,
    *,
    desired_revision: int,
    active_revision: int | None,
    running: bool,
) -> dict[str, Any]:
    """Build the public graph from the same mode-adjusted config as a run."""
    from CoScientist.agents import config_for_mode
    from CoScientist.assembly.schema import resolve_config_path

    with settings_scope(run_settings):
        config = config_for_mode()
        graph, attached = _full_graph(config)
        from CoScientist.web.agent_configuration_policy import (
            agent_control_policy,
            public_control_fields,
        )
        from CoScientist.web.agent_workflow import project_workflow
        controls = agent_control_policy(config)
        workflow = project_workflow(config, nir_enabled=run_settings.nir.enabled)

    virtual_root = "__mas_session__"
    full_root = PIPELINE_ROOT_NAME if PIPELINE_ROOT_NAME in attached else config.root.name
    visible = {
        name for name in attached
        if name not in (PIPELINE_ROOT_NAME,) and not config.agent(name).internal
    }

    # A virtual UI root is a container, not another system agent. It keeps the
    # collapsed graph connected when its real root is an internal workflow.
    collapsed: dict[tuple[str, str], dict[str, Any]] = {}

    def descend(source: str, target: str, relation: str, path: tuple[str, ...]) -> None:
        if target in path:
            return
        if target in visible:
            collapsed[(source, target)] = {
                "from": source,
                "to": target,
                "relation": relation,
            }
            return
        for child, child_relation in graph.get(target, []):
            descend(source, child, child_relation or relation, path + (target,))

    if full_root in visible:
        collapsed[(virtual_root, full_root)] = {
            "from": virtual_root, "to": full_root, "relation": "root",
        }
    else:
        for child, relation in graph.get(full_root, []):
            descend(virtual_root, child, relation, (full_root,))

    for parent in sorted(visible):
        for child, relation in graph.get(parent, []):
            descend(parent, child, relation, (parent,))

    nodes: list[dict[str, Any]] = [{
        "id": virtual_root,
        "name": "MAS",
        "title": {"ru": "Конфигуратор сессии", "en": "Session configurator"},
        "description": "",
        "descriptionLocalized": {
            "ru": "Управляет составом агентов выбранной сессии. Позволяет подключать и отключать дополнительных исполнителей. Изменения применяются со следующего запроса.",
            "en": "Controls the selected session's agent composition. Changes apply to the next request.",
        },
        "icon": "settings",
        "kind": "system",
        "stage": None,
        "toolKeys": [],
        "hasMcpTools": False,
        "selected": True,
        "availableInProfile": True,
        "effectiveEnabled": True,
        "canEnable": False,
        "canDisable": False,
        "controlReason": "Элемент интерфейса управления составом выбранной сессии.",
        "controlCode": "configuration",
        "requiredForPipeline": True,
    }]
    for name in config.build_order():
        if name not in visible:
            continue
        agent = config.agent(name)
        stage = (
            "pre" if name in config.pipeline.pre
            else "post" if name in config.pipeline.post
            else None
        )
        selected = not (name == "NirReportAgent" and not run_settings.nir.enabled)
        control = controls[name]
        nodes.append({
            "id": name,
            "name": name,
            **agent_presentation(name, agent.title, agent.description),
            "description": _short(agent.description),
            "kind": "agent",
            "class": agent.cls,
            "stage": stage,
            "toolKeys": list(agent.tools),
            "hasMcpTools": "dynamic_tools" in agent.tools or any(
                key in REGISTRY.tools and REGISTRY.tools[key].runtime_resolved
                for key in agent.tools
            ),
            "selected": selected,
            **public_control_fields(control),
        })

    return {
        "profile": resolve_config_path().stem,
        "startMode": run_settings.web.start_mode,
        "desiredRevision": desired_revision,
        "activeRevision": active_revision,
        "pending": active_revision is not None and active_revision != desired_revision,
        "running": running,
        "nodes": nodes,
        "edges": list(collapsed.values()),
        "workflow": workflow,
    }


def _selected_mcp(state: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    server_ids: set[str] = set()
    tool_names: set[str] = set()

    def add_server(server: Mapping[str, Any]) -> None:
        for key in ("server_id", "id", "name", "server_name"):
            if server.get(key):
                server_ids.add(str(server[key]))
        for tool in server.get("tools") or []:
            if isinstance(tool, Mapping):
                name = tool.get("name")
            else:
                name = tool
            if name:
                tool_names.add(str(name))

    for item in state.get("filtered_tools") or []:
        if isinstance(item, Mapping):
            add_server(item)
            if item.get("name"):
                tool_names.add(str(item["name"]))
    for item in state.get("deployed_mcps") or []:
        if isinstance(item, Mapping):
            add_server(item)
    envelope = state.get("experiment_active_envelope") or {}
    task = (envelope.get("task") or {}) if isinstance(envelope, Mapping) else {}
    for item in task.get("mcp_servers") or []:
        if isinstance(item, Mapping):
            add_server(item)
    return server_ids, tool_names


def agent_tools_payload(
    config: SystemConfig,
    agent_name: str,
    state: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Public tool list for a card; never returns MCP URLs or credentials."""
    from CoScientist.tools.mcp_catalog import load_presentation, resolve_tool_presentation
    from CoScientist.config import get_settings

    agent = config.agent(agent_name)
    presentation = load_presentation()
    tools: list[dict[str, Any]] = []
    seen: set[str] = set()

    dynamic = "dynamic_tools" in agent.tools
    selected_servers, selected_names = _selected_mcp(state)
    servers = {str(item.get("id")): item for item in catalog.get("servers") or []}

    if dynamic:
        for server_id, server in servers.items():
            aliases = {
                server_id,
                str(server.get("registry_id") or ""),
                str(server.get("name") or ""),
                *(str(value) for value in server.get("aliases") or []),
            }
            if aliases & selected_servers:
                selected_servers.add(server_id)

        for item in catalog.get("tools") or []:
            # Tool names are not globally unique; server selection is the
            # runtime boundary (DynamicMCPToolset loads the selected server).
            selected = str(item.get("server_id")) in selected_servers
            if selected:
                public = dict(item)
                public.update({"kind": "mcp", "selected": True})
                tools.append(public)
                seen.add(str(item.get("id")))

        # Product requirement: the FEDOT AutoML trainer is the first ReAct tool.
        fedot_id = "d6dac6fb09066080:train_ml"
        fedot = next(
            (dict(item) for item in catalog.get("tools") or [] if item.get("id") == fedot_id),
            {
                "id": fedot_id,
                "name": "train_ml",
                "display_name": {
                    "ru": "FEDOT AutoML — обучить модель",
                    "en": "FEDOT AutoML — train a model",
                },
                "summary": {
                    "ru": "Автоматически подбирает и обучает модель машинного обучения.",
                    "en": "Automatically selects and trains a machine-learning model.",
                },
                "status": "unavailable",
            },
        )
        fedot.update({
            "kind": "mcp",
            "selected": fedot_id in seen,
            "pinned": True,
        })
        tools = [fedot, *[item for item in tools if item.get("id") != fedot_id]]
        seen.add(fedot_id)

    for key in agent.tools:
        entry = REGISTRY.tools.get(key)
        if entry is None:
            continue
        remote = key in _MCP_SERVERS
        if remote:
            settings = get_settings()
            configured = {
                "websearch": bool(settings.services.tavily_api_key),
                "paper_analysis": bool(settings.mcp.paper_analysis_url),
                "papers_search": bool(settings.mcp.papers_search_url),
                "vault": bool(settings.mcp.vault_url),
            }
            if not configured[key]:
                continue
            alias = _MCP_SERVERS[key].casefold()
            server_ids = {
                sid for sid, server in servers.items()
                if alias in {str(value).casefold() for value in [server.get("name", ""), *(server.get("aliases") or [])]}
            }
            surface = [tool for tool in catalog.get("tools") or []
                       if str(tool.get("server_id")) in server_ids and tool.get("available_to_agents", True)]
            if surface:
                for tool in surface:
                    if str(tool["id"]) not in seen:
                        tools.append({**tool, "kind": "mcp", "selected": True})
                        seen.add(str(tool["id"]))
                continue
        for doc in entry.resolved_docs():
            if doc.name.startswith("<") or doc.name in seen:
                continue
            tools.append({
                "id": f"local:{agent_name}:{doc.name}",
                "name": doc.name,
                **resolve_tool_presentation(doc.name, doc.purpose, registry_key=key, presentation=presentation),
                "status": "saved" if remote else "available",
                "kind": "mcp" if remote else "local",
                "selected": True,
            })
            seen.add(doc.name)

    return {
        "agent": agent_name,
        "dynamic": dynamic,
        "selectionReady": bool(selected_servers or selected_names),
        "catalog": {
            "checkedAt": catalog.get("checked_at"),
            "stale": bool(catalog.get("stale")),
            "partial": bool(catalog.get("partial")),
        },
        "tools": tools,
    }


__all__ = ["agent_tools_payload", "project_agent_tree"]
