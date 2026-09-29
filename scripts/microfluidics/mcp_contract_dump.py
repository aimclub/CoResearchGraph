"""Record what an MCP server really exposes — tool names, input schemas, and a
sample answer — so the code and the prompts that talk to it are written against
the real contract instead of a guess.

Usage (reads the URLs and the key from .env; needs the network that reaches them):

    python scripts/microfluidics/mcp_contract_dump.py economics
    python scripts/microfluidics/mcp_contract_dump.py cfd

Writes ``tests/fixtures/<server>_mcp/``:

    tools.json               every tool: name, description, inputSchema
    <tool>.json              for the sample calls below: the raw CallToolResult
                             (content, structuredContent, isError)

The economics samples only read (price lookups, name resolution): nothing is changed
on the server. The CFD samples run a solver: safe on the emulator only.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from mcp import ClientSession  # noqa: E402
from mcp.client.streamable_http import streamablehttp_client  # noqa: E402

# Two-step SDS route in the named notation of rank_routes_by_cost.
SDS_ROUTE = {
    "route_id": "SDS-1",
    "target_name": "додецилсульфат натрия",
    "steps": [
        {
            "reactants": ["додеканол-1", "хлорсульфоновая кислота"],
            "products": ["додецилгидросульфат"],
            "conditions": "25 °C, 90 мин",
            "yield": 0.9,
        },
        {
            "reactants": ["@prev", "гидроксид натрия"],
            "products": ["додецилсульфат натрия"],
            "conditions": "30 °C, 30 мин",
            "yield": 0.95,
        },
    ],
}

# The same route by structure — what a priced ranking looks like.
SDS_ROUTE_SMILES = {
    "route_id": "SDS-2",
    "steps": [
        {
            "reactants": [{"smiles": "CCCCCCCCCCCCO"}, {"smiles": "OS(=O)(=O)Cl"}],
            "products": [{"smiles": "CCCCCCCCCCCCOS(=O)(=O)O"}],
            "conditions": "25 °C, 90 мин",
            "yield": 0.9,
        },
        {
            "reactants": ["@prev", {"smiles": "[Na+].[OH-]"}],
            "products": [{"smiles": "CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]"}],
            "conditions": "30 °C, 30 мин",
            "yield": 0.95,
        },
    ],
}

SERVERS: dict[str, dict[str, Any]] = {
    "economics": {
        "url_env": "MCP_MICROFLUIDIC_ECONOMIC",
        "key_env": None,
        "samples": [
            ("resolve_chemicals",
             {"names": ["додеканол-1", "хлорсульфоновая кислота", "гидроксид натрия"]}),
            ("search_reagents_by_name", {"query": "глицерин", "limit": 3}),
            ("get_price", {"name": "ацетон", "limit": 3}),
            ("estimate_synthesis_cost",
             {"reagents": [{"name": "глицерин", "qty": 100, "unit": "g"}]}),
            ("rank_routes_by_cost",
             {"routes": [SDS_ROUTE], "target_qty": 100, "target_unit": "g",
              "include_breakdown": True}),
            ("rank_routes_by_cost",
             {"routes": [SDS_ROUTE_SMILES], "target_qty": 100, "target_unit": "g",
              "include_breakdown": True},
             "rank_routes_by_cost_smiles"),
            # English names: do they resolve where the Russian ones did not?
            ("resolve_chemicals",
             {"names": ["1-dodecanol", "chlorosulfonic acid", "sodium dodecyl sulfate"]},
             "resolve_chemicals_en"),
            # An error on purpose: what isError looks like.
            ("resolve_chemicals", {"names": []}, "resolve_chemicals_error"),
        ],
    },
    "cfd": {
        "url_env": "MCP_MICROFLUIDIC_CFD_3_TOOLS",
        "key_env": "MICROFLUIDIC_CFD_3_TOOLS_KEY",
        # Safe on the emulator deployment; on a real solver the run below is a
        # long job — do not dump against it casually.
        "samples": [
            ("cfd_list_reactors", {}),
            ("cfd_run_reactor_experiment",
             {"reactor": "t_junction", "inlet_speed_m_per_s": 0.02,
              "concentration_a_mol_per_m3": 100.0, "concentration_b_mol_per_m3": 100.0,
              "rate_constant_m3_per_mol_s": 0.001, "wait_seconds": 120,
              "request_id": "contract-dump-run-1"}),
            ("cfd_get_experiment_result", {"request_id": "contract-dump-run-1"}),
            ("cfd_list_artifacts", {"request_id": "contract-dump-run-1", "limit": 5}),
            # Returned at once: what a pending run looks like.
            ("cfd_run_reactor_experiment",
             {"reactor": "t_junction", "inlet_speed_m_per_s": 0.03,
              "concentration_a_mol_per_m3": 100.0, "concentration_b_mol_per_m3": 100.0,
              "rate_constant_m3_per_mol_s": 0.001, "wait_seconds": 0,
              "request_id": "contract-dump-run-2"},
             "cfd_run_reactor_experiment_pending"),
            ("cfd_cancel_run", {"request_id": "contract-dump-run-2"}),
            # Errors on purpose: an unknown run, a single-inlet reactor for A + B.
            ("cfd_get_experiment_result", {"request_id": "no-such-run"},
             "cfd_get_experiment_result_unknown"),
            ("cfd_run_reactor_experiment",
             {"reactor": "vertical_pipe_500", "inlet_speed_m_per_s": 0.02,
              "concentration_a_mol_per_m3": 100.0, "concentration_b_mol_per_m3": 100.0,
              "rate_constant_m3_per_mol_s": 0.001, "wait_seconds": 60,
              "request_id": "contract-dump-run-3"},
             "cfd_run_reactor_experiment_single_inlet"),
        ],
    },
}


def _dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}")


async def dump(server: str, timeout: float) -> int:
    spec = SERVERS[server]
    url = os.getenv(spec["url_env"])
    if not url:
        print(f"{spec['url_env']} is not set in .env")
        return 2
    headers = {}
    if spec["key_env"]:
        key = os.getenv(spec["key_env"])
        if not key:
            print(f"{spec['key_env']} is not set in .env")
            return 2
        headers["X-API-Key"] = key

    out = ROOT / "tests" / "unit" / "microfluidics" / "fixtures" / f"{server}_mcp"
    out.mkdir(parents=True, exist_ok=True)

    async with streamablehttp_client(
        url, headers=headers, timeout=30, sse_read_timeout=timeout
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            print(f"{server}: {len(tools)} tools — {', '.join(t.name for t in tools)}")
            _dump(out / "tools.json", [
                {"name": t.name, "description": t.description,
                 "inputSchema": t.inputSchema,
                 "outputSchema": getattr(t, "outputSchema", None)}
                for t in tools
            ])

            names = {t.name for t in tools}
            for sample in spec["samples"]:
                tool, args = sample[0], sample[1]
                fixture = sample[2] if len(sample) > 2 else tool
                if tool not in names:
                    print(f"  SKIP {tool}: the server has no such tool")
                    continue
                print(f"  call {tool}({json.dumps(args, ensure_ascii=False)[:120]})")
                try:
                    result = await session.call_tool(tool, args)
                except Exception as exc:  # noqa: BLE001 — record, keep going
                    _dump(out / f"{fixture}.json",
                          {"exception": type(exc).__name__, "message": str(exc)})
                    continue
                _dump(out / f"{fixture}.json", result.model_dump(mode="json"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("server", choices=sorted(SERVERS))
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="read timeout per call, seconds")
    args = ap.parse_args()
    return asyncio.run(dump(args.server, args.timeout))


if __name__ == "__main__":
    raise SystemExit(main())
