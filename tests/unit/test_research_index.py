"""Cross-run reuse: a finished research must be findable by the next run."""
import json

from CoScientist.graph.research.index import (
    ResearchIndex,
    format_priors,
    summarize,
)


def _graph(qid="Q1", question="Влияет ли кластеризация молекул на предсказание LD50?",
           hyp_status="refuted"):
    return {
        "research_id": f"research-{qid}",
        "root_id": qid,
        "nodes": [
            {"id": qid, "type": "ResearchQuestion", "status": "open",
             "attrs": {"formulation": question, "domain": "chemoinformatics"}},
            {"id": "H1", "type": "Hypothesis", "status": hyp_status,
             "attrs": {"formulation": "Кластеризация ухудшает предсказание LD50"}},
            {"id": "VM1", "type": "VerificationMethod", "status": "done",
             "attrs": {"procedure": "Сравнить кластерную и единую модель на held-out"}},
            {"id": "CL1", "type": "Conclusion", "status": "approved",
             "attrs": {"synthesis": "Выигрыш только при наличии якорных точек"}},
            {"id": "T1", "type": "Tool", "status": "available", "attrs": {"name": "RDKit"}},
        ],
        "edges": [{"type": "motivates", "from": qid, "to": "H1"}],
    }


def test_summarize_captures_verdicts_methods_and_tools():
    rec = summarize(_graph())
    assert rec["question"].startswith("Влияет ли кластеризация")
    assert rec["counts"] == {"hypotheses": 1, "confirmed": 0, "refuted": 1,
                             "evidence": 0, "conclusions": 1, "nodes": 5}
    assert rec["hypotheses"][0]["status"] == "refuted"
    assert "held-out" in rec["methods"][0]["procedure"]
    assert rec["tools"] == ["RDKit"]
    assert "кластеризация" in rec["tokens"]


def test_empty_or_rootless_graph_is_not_indexed():
    assert summarize({"nodes": [], "edges": []}) is None
    assert summarize({"nodes": [{"id": "X", "type": "Tool", "attrs": {}}]}) is None


def test_search_finds_a_prior_and_explains_why(tmp_path):
    idx = ResearchIndex(tmp_path / "research_index.json")
    idx.upsert(summarize(_graph()))
    idx.upsert(summarize(_graph(qid="Q2", question="Синтез аммиака при низком давлении")))

    hits = idx.search("кластеризация молекул и предсказание LD50")
    assert hits and hits[0]["question"].startswith("Влияет ли кластеризация")
    # explainability: the demo shows WHY a prior matched
    assert "кластеризация" in hits[0]["matched_tokens"]
    assert hits[0]["score"] > 0


def test_settled_research_outranks_an_open_one(tmp_path):
    idx = ResearchIndex(tmp_path / "i.json")
    idx.upsert(summarize(_graph(qid="Q1", hyp_status="formulated")))
    settled = summarize(_graph(qid="Q2", hyp_status="confirmed"))
    idx.upsert(settled)
    hits = idx.search("кластеризация молекул LD50", limit=2)
    assert hits[0]["research_id"] == settled["research_id"]


def test_exclude_id_keeps_the_current_research_out(tmp_path):
    idx = ResearchIndex(tmp_path / "i.json")
    rec = summarize(_graph())
    idx.upsert(rec)
    assert idx.search("кластеризация LD50", exclude_id=rec["research_id"]) == []


def test_index_survives_a_corrupt_file(tmp_path):
    p = tmp_path / "i.json"
    p.write_text("{not json", encoding="utf-8")
    idx = ResearchIndex(p)
    assert idx.all() == []
    assert idx.upsert(summarize(_graph())) is True
    assert len(ResearchIndex(p).all()) == 1


def test_digest_is_capped_and_mentions_verdicts(tmp_path):
    idx = ResearchIndex(tmp_path / "i.json")
    idx.upsert(summarize(_graph()))
    digest = format_priors(idx.search("кластеризация LD50"), budget=300)
    assert len(digest) <= 301
    assert "Прошлое исследование" in digest
