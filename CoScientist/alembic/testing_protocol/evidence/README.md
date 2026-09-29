# Материалы испытаний модуля разработки агентных прикладных инструментов (Alembic)

Дата испытаний: 2 сентября 2026 г.

| Файл | Что это |
|---|---|
| `build_synspace.log` | Полный журнал прямой сборки MCP-сервера по репозиторию https://github.com/whitead/synspace (5 стадий, docker commit, запуск сервера) |
| `build_mordred.log` | Полный журнал прямой сборки по репозиторию https://github.com/mordred-descriptor/mordred |
| `console_coscientist.log` | Сквозной прогон CoScientist: задача со статьёй → делегирование модулю → запуск сборки |
| `console_coscientist2.log` … `console_coscientist6.log` | Повторные сквозные прогоны (вариативность маршрутизации, транзиентные ошибки провайдера) |
| `agent_events*.log` | Те же прогоны, только строки событий агентов, без служебного вывода |
| `graph_mcpbuilder.txt` | Выжимка графа исполнения сессии: оркестратор → McpBuilderAgent → build_mcp_server |
| `mcp_synspace_check.txt` | Ответ поднятого MCP-сервера: список инструментов, их JSON Schema и результат реального вызова |
| `tool_gap_decision.txt` | Вывод проверки точки принятия решения о начале разработки инструмента |
| `check_mcp_server.py` | Скрипт: подключается к MCP-серверу, проверяет схемы инструментов, вызывает инструмент |
| `check_tool_gap_decision.py` | Скрипт: прогоняет точку принятия решения на двух состояниях сессии |
| `validate_artifacts.py` | Скрипт: проверка машиночитаемых артефактов прогонов по JSON Schema |
| `reuse_autogen.txt` | Сценарий «инструмент уже собран»: модуль нашёл готовый образ и поднял сервер за 1 секунду без пересборки |
| `mcp_autogen_check.txt` | Список инструментов поднятого сервера autogen и результат реального вызова инструмента |
| `check_reuse_existing_build.py` | Скрипт: вызывает те же функции, что и агент, и показывает путь повторного использования готовой сборки |
| `console_reuse.log`, `console_use_existing.log` | Чат-прогоны сценария «от статьи к уже собранному серверу» |
| `use_existing_server_run.txt` | Сценарий «работа по существующему собранному инструменту» целиком: условия запуска, промпт, ход выполнения, итоговый ответ |
| `console_use_existing12.log` | Полный журнал этого прогона |
| `check_reuse_reaches_agent.py` | Скрипт: настоящий путь «повторное использование сборки → список инструментов агента», без подмены состояния |
| `use_existing_synspace_run.txt` | Тот же сценарий на прикладной научной задаче: построение химического пространства вокруг аспирина инструментом уже собранного сервера synspace |
| `console_synspace_use4.log` | Полный журнал этого прогона |
| `md_to_docx.py` | Служебный скрипт сборки .docx из текста протокола |

Команды воспроизведения:

```
# прямая сборка MCP-сервера по ссылке на репозиторий
python CoScientist/alembic/start_chain.py https://github.com/whitead/synspace

# проверка поднятого сервера по протоколу MCP
python nirsii/evidence/check_mcp_server.py http://localhost:<порт>/mcp process_smiles_reos '{"smiles":"CN1C=NC2=C1C(=O)N(C(=O)N2C)C"}'

# доля структурно валидных машиночитаемых артефактов
python nirsii/evidence/validate_artifacts.py

# точка принятия решения о начале разработки инструмента
python nirsii/evidence/check_tool_gap_decision.py

# путь от повторного использования готовой сборки до списка инструментов агента
PYTHONPATH=. python nirsii/evidence/check_reuse_reaches_agent.py

# проверка конфигурации агентной системы
python -m CoScientist.assembly
```
