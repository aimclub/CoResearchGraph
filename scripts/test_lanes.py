"""Test harness for isolated routing cases:
1. Research-only query
2. Hypotheses-only query
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Configure environment
os.environ["COSCIENTIST_CONFIG"] = str(REPO / "CoScientist" / "agents" / "experiments.yaml")
os.environ["COSCIENTIST_EXPERIMENT_HITL_AUTO_APPROVE"] = "1"
os.environ["COSCIENTIST_EXPERIMENT_AUDIT_STDOUT"] = "1"
os.environ["HYPOTHESES__MAX_ACTIVE"] = "2"

from dotenv import load_dotenv
load_dotenv(REPO / ".env")


async def run_case(case_name: str, query: str):
    print(f"\n{'='*70}\nSTART CASE: {case_name}\nQUERY: {query}\n{'='*70}\n", flush=True)
    from CoScientist.main import CoScientistManager

    session_id = f"test_{case_name}_{int(time.time())}"
    manager = CoScientistManager(session_id=session_id)
    await manager.initialize()
    
    start_t = time.time()
    try:
        result = await manager.run(query)
        elapsed = time.time() - start_t
        print(f"\n{'-'*70}\nCASE {case_name} COMPLETED in {elapsed:.1f}s\n{'-'*70}", flush=True)
        print("RESULT PREVIEW:", str(result)[:800], "\n", flush=True)
        return {"case": case_name, "success": True, "elapsed": elapsed, "result": str(result)}
    except Exception as exc:
        elapsed = time.time() - start_t
        print(f"\n{'-'*70}\nCASE {case_name} FAILED in {elapsed:.1f}s with error: {exc}\n{'-'*70}", flush=True)
        import traceback
        traceback.print_exc()
        return {"case": case_name, "success": False, "elapsed": elapsed, "error": str(exc)}
    finally:
        await manager.close()


async def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    results = {}
    if target in ("all", "research"):
        # Case 1: Research only
        res_q = (
            "Найди научные статьи и исследования про механизмы ингибирования KRAS G12C и основные структурные "
            "особенности связывания в кармане переключателя II (switch-II pocket). Эксперименты и вычисления не нужны."
        )
        results["research"] = await run_case("research_only", res_q)

    if target in ("all", "hypotheses"):
        # Case 2: Hypotheses only
        hyp_q = (
            "Сформулируй 2-3 научно обоснованные гипотезы для преодоления приобретенной резистентности к "
            "ингибиторам KRAS G12C (например, через вторичные мутации Y96D или активацию обходных путей RTK/SHP2). "
            "Эксперименты и вычисления не нужны."
        )
        results["hypotheses"] = await run_case("hypotheses_only", hyp_q)

    print("\n" + "="*70)
    print("SUMMARY RESULTS:")
    for k, v in results.items():
        print(f"  - {k}: success={v.get('success')} (elapsed={v.get('elapsed', 0):.1f}s)")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
