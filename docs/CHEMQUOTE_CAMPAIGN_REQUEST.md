# CampaignRequest ← chemquote: что реально можно получить от модуля экономики

Проверено на живом сервисе **chemquote 1.28.0**, `http://10.32.11.22:8000/mcp` (MCP, streamable HTTP),
2026-09-17. Все примеры значений — из реальных ответов сервиса.

> Полное описание инструментов — [MCP_Economic_model.md](../MCP_Economic_model.md) (актуализирован под 1.28.0).

---

## 1. `reagent_costs` — `{smiles: число}` руб/г  ✅ полностью

Главное, что модуль умеет.

**Инструмент:** `estimate_synthesis_cost` — один вызов на весь список реагентов.

```json
{"reagents": [{"smiles": "Cc1ccccc1", "qty": 100, "unit": "g"},
              {"smiles": "[Na+].[BH4-]", "qty": 10, "unit": "g"}],
 "similarity": "soft"}
```

**Откуда брать число:** `line_items[].chosen.unit_price` + `line_items[].chosen.pack_unit`.

**Пересчёт в руб/г:** `unit_price` дан за единицу фасовки (`kg` / `g` / `l` / `ml`), а не за грамм:

| `pack_unit` | руб/г |
|---|---|
| `g` | `unit_price` |
| `kg` | `unit_price / 1000` |
| `ml`, `l` | нужна плотность — сервис её не знает; запрашивать в `g`/`kg` |

**Реальные значения из прогона:**

| Вещество | SMILES | `unit_price` | `pack_unit` | руб/г |
|---|---|---|---|---|
| Толуол | `Cc1ccccc1` | 385.00 | kg | 0.385 |
| Дихлорметан | `ClCCl` | 93.60 | kg | 0.094 |
| Ацетилхлорид | `CC(=O)Cl` | 27.50 | g | 27.5 |
| Боргидрид натрия | `[Na+].[BH4-]` | 37.50 | g | 37.5 |

**Альтернатива для одной позиции:** `search_by_structure {"smiles": …, "mode": "exact"}` →
`unit_price`, `pack_unit`, `offer_count`. Но первым может прийти «навал» (см. ниже), поэтому
`estimate_synthesis_cost` надёжнее — он такие предложения обходит сам.

**Что отфильтровать / проверить:**

- `is_bulk: true` или пустой `pack_unit` / `pack_qty: "0"` — «цена по запросу», в расчёт не брать
  (для толуола `search_by_structure` первым выдал именно такое: 98.18 ₽ без фасовки).
- `basis_suspect: true` — автоматическая проверка сочла базис цены недостоверным.
- `match_level` ≠ `точный` (в режиме `soft`) — подобрано похожее вещество, читать `comment`.
- `price_currency` ≠ `RUB` — курсов в сервисе нет, в руб/г не переводится.
- Брать `unit_price` / `cost_per_unit` (себестоимость), **не** `cost_packs` (чек за целые упаковки).
- Цены — из прайс-листов, не из живых остатков.

---

## 2. `unavailable[]` — массив SMILES  ⚠️ частично (как прокси, с подтверждением оператора)

**Инструмент:** тот же `estimate_synthesis_cost` (или `rank_routes_by_cost`).

**Откуда брать:** `missing[]` с `reason: "no_match"` — вещества нет ни в одном прайс-листе;
дополнительно `unpriced_agents[]` из `rank_routes_by_cost` (агенты без заданного количества).

**Оговорки:**

- «Нет в прайсах» ≠ «не удалось закупить»: сервис не видит остатков и не оформляет заказ.
  В прогоне `no_match` получили **этанол** (`CCO`) и **бензальдегид** (`O=Cc1ccccc1`) — это дыра в
  покрытии базы, а не недоступность. Список годится только как черновик для оператора.
- `reason: "unit_mismatch"` (ДХМ запрошен в `ml`, прайс в `kg`) — ошибка единиц, **не** недоступность.
- `reason: "unresolved_name"` — имя не разрешилось в структуру, до поиска цены дело не дошло.

---

## 3. `product_smiles` — молярная масса для пересчёта выхода  ✅ (масса — да, сам SMILES — нет)

**Инструмент:** `resolve_chemicals {"names": ["O=Cc1ccccc1"]}` — принимает готовый SMILES или имя.

**Откуда брать:** `items[].molar_mass` (г/моль), а также `formula`, `inchikey`, `canonical_smiles`.

Пример: `O=Cc1ccccc1` → `molar_mass: "106.124"`, `formula: "C7H6O"`,
`inchikey: "HUMNYLRZRPPJDN-UHFFFAOYSA-N"`.

То же самое возвращает `rank_routes_by_cost` в `routes[].resolved_inputs[]` для всех участников
маршрута и в `routes[].target_smiles` для цели.

Сам SMILES продукта chemquote не определяет — он приходит из маршрута синтеза (стадия 4).

---

## 4. `reagents[]` — `{role, smiles}`  ⚠️ частично (SMILES/идентификация — да, роли — нет)

**Что даёт:**

- `resolve_chemicals` — канонический SMILES + InChIKey по имени (русскому, английскому, формуле)
  или по SMILES; `molar_mass` каждого вещества. InChIKey — надёжный ключ для сопоставления
  вещества между сервисами (надёжнее строки SMILES).
- `rank_routes_by_cost` → `routes[].steps_smiles` — собранные reaction SMILES, реально пошедшие
  в расчёт, и `resolved_inputs[]` — что во что превратилось и каким слоем.

Пример `resolve_chemicals`:

| Вход | `canonical_smiles` | `molar_mass` | `method` |
|---|---|---|---|
| бензальдегид | `O=Cc1ccccc1` | 106.124 | ru_iupac |
| ацетилхлорид | `CC(=O)Cl` | 78.498 | cas |
| AlCl3 | `[Cl][Al]([Cl])[Cl]` | 133.341 | pubchem |
| боргидрид натрия | — | — | **unresolved** |

**Чего не даёт:** ролей. В `rank_routes_by_cost` есть только `reactants` / `agents` / `products`;
катализатор и растворитель — оба «агенты», сервис их не различает. Роли задаёт стадия 4.

**Оговорки:** резолвер не всесилен (см. «боргидрид натрия» выше) — где есть SMILES, подавать SMILES;
строки, читаемые и как имя, и как SMILES (`CO`, `NO`, `Br`), сервис не угадывает и отказывает.

---

## 5. `base_stoichiometry` — только обратная проверка/пересчёт  ⚠️ не источник

Стехиометрия — **вход** chemquote (`equiv` у вещества или повтор компонента в reaction SMILES).

Что можно получить обратно из `rank_routes_by_cost` → `routes[].starting_materials[]`:
`moles`, `qty` (г), `molar_mass`, `source` (`stoichiometry` / `override`), `used_in_steps`.

Пример (цель 10 г, `CC(=O)Cl` + 2 экв. `CCO`, выход 0.8):

| SMILES | `moles` | `qty`, г | `molar_mass` |
|---|---|---|---|
| `CC(=O)Cl` | 0.141875 | 11.14 | 78.498 |
| `CCO` | 0.283749 | 13.07 | 46.069 |

Полезно, чтобы перевести мольные соотношения в граммы закупки на заданное количество продукта
и убедиться, что эквиваленты распарсились так, как задумано.

---

## Что chemquote **не** даёт

| Поле | Почему |
|---|---|
| `nominal_conditions` | `conditions` в шаге принимается строкой, но не интерпретируется и на расчёт не влияет |
| `objective` | не даёт; если метрика — стоимость продукта, `cost_per_unit` маршрута на `target_qty` даст базовую руб/г, но не `target_value` |
| `priorities[]` | — |
| `budget` | это точки/часы, не деньги; денежная сторона (`cost_packs`, реальный чек) в `CampaignRequest` не предусмотрена |
| `literature_hints` | это RAG/papers-стадия; chemquote лишь помогает перевести маршрут из статьи в SMILES (`{"text": …}`-шаги, `resolve_chemicals`) |

---

## Замечания по интеграции

- Адрес сервиса задаётся в `.env`: `MCP__MICROFLUIDIC_ECONOMIC_URL=http://10.32.11.22:8000/mcp`
  (читается как `settings.mcp.microfluidic_economic_url`; пока не задан — агент работает на заглушке
  `economics_mcp_stub`). Прежний `k3s-control-1.lab:30590` через VPN не резолвится.
- Описания инструментов для агента — `CoScientist/assembly/bindings.py` (ToolEntry `economics_mcp`),
  промпт — `microfluidics_economics` в `CoScientist/agents/prompts/templates.py`; оба под 1.28.0.
- Единицы: жидкости запрашивать в `g`/`kg`, иначе `unit_mismatch` против прайса в `kg`.
- Соли в reaction SMILES — в скобках: `([K+].[O-][Mn](=O)(=O)=O)`, иначе это два вещества.
- Где SMILES известен — подавать SMILES, а не имя; подстановку имён проверять по `resolved_inputs`.
