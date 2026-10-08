"""Medical tools: PubMed search, PICO extraction, study taxonomy classification, DICOM VLM analysis."""

import asyncio
import json
from typing import List, Optional, Dict, Any

import httpx
import litellm
from metapub import PubMedFetcher

from google.adk.tools import BaseTool, ToolContext
from google.adk.tools.base_toolset import BaseToolset
from google.adk.agents.readonly_context import ReadonlyContext

from CoScientist.config import get_settings
from CoScientist.logging.metrics import record_completion

settings = get_settings()

# ─── med_vlm service ─────────────────────────────────────────────────────────
_VLM_TASK_URL = settings.med_llm.task_url
_VLM_RESULT_URL = settings.med_llm.result_url
_VLM_AUTH = (settings.med_llm.login, settings.med_llm.password)
_VLM_POLL_INTERVAL_SEC = 10 
_VLM_MAX_POLLS = 60 


# ─── LLM helper ──────────────────────────────────────────────────────────────

async def _llm_json(prompt: str) -> dict:
    response = await litellm.acompletion(
        model=settings.llm.main_model,
        api_base=settings.llm.main_url,
        api_key=settings.llm.openai_api_key,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
        timeout=settings.llm.request_timeout,
    )
    record_completion(response, model=settings.llm.main_model, agent="MedicalAgent")
    return json.loads(response.choices[0].message.content)


# ─── Prompts ──────────────────────────────────────────────────────────────────

_PICO_PROMPT = """\
You are a medical AI assistant specializing in extracting PICO elements from research abstracts.

PICO definitions:
* population: The target patient population with characteristics
* intervention: The main intervention, treatment, or procedure
* comparison: The comparative intervention or control
* outcome: The measured clinical outcome

Extract PICO elements from the paper below.

Title: {title}
Abstract: {abstract}

Return a JSON object with keys: "population", "intervention", "comparison", "outcome".\
"""

_STUDY_TYPE_PROMPT = """\
You are a medical AI assistant specializing in research type analysis.
Classify the study as exactly one of: observational, experimental, literature review.

Title: {title}
Abstract: {abstract}

Return JSON: {{"category": "<observational|experimental|literature review>", "reasoning": "<brief explanation>"}}\
"""

_LIT_REVIEW_PROMPT = """\
You are a medical AI assistant. This paper is a literature review.
Classify it as exactly one of: historical, systematic, meta-analysis.

Title: {title}
Abstract: {abstract}

Return JSON: {{"category": "<historical|systematic|meta-analysis>", "reasoning": "<brief explanation>"}}\
"""

_OBS_TIME_PROMPT = """\
You are a medical AI assistant. This is an observational study.
Classify the temporal design as exactly one of: prospective, retrospective.

Title: {title}
Abstract: {abstract}

Return JSON: {{"category": "<prospective|retrospective>", "reasoning": "<brief explanation>"}}\
"""

_OBS_COHORT_PROMPT = """\
You are a medical AI assistant. This is an observational study.
Classify the study design as exactly one of: cross-sectional, cohort, case-control.

Title: {title}
Abstract: {abstract}

Return JSON: {{"category": "<cross-sectional|cohort|case-control>", "reasoning": "<brief explanation>"}}\
"""

_EXP_DISTR_PROMPT = """\
You are a medical AI assistant. This is an experimental study.
Classify the allocation method as exactly one of: randomized, non-randomized, propensity score matching.

Title: {title}
Abstract: {abstract}

Return JSON: {{"category": "<randomized|non-randomized|propensity score matching>", "reasoning": "<brief explanation>"}}\
"""

_EXP_ENV_PROMPT = """\
You are a medical AI assistant. This is an experimental study.
Classify the study environment as exactly one of: clinical, in vivo, in vitro, in silico.

Title: {title}
Abstract: {abstract}

Return JSON: {{"category": "<clinical|in vivo|in vitro|in silico>", "reasoning": "<brief explanation>"}}\
"""


# ─── Toolset ─────────────────────────────────────────────────────────────────

class MedToolset(BaseToolset):
    """Toolset for medical tasks: literature search, PICO analysis, taxonomy classification, DICOM VLM."""

    def __init__(self, prefix: str = "med_"):
        super().__init__()
        self.tool_name_prefix = prefix
    
    def get_tools(self, readonly_context: Optional[ReadonlyContext]) -> List[BaseTool]:
        return [
            self.search_pubmed,
            self.get_pico,
            self.get_study_taxonomy,
            self.analyze_medical_image,
        ]

    async def close(self) -> None:
        await asyncio.sleep(0)

    async def search_pubmed(
        self,
        keyword: str,
        num_results: int = 10,
    ) -> Dict[str, Any]:
        """
        Search PubMed for medical literature by keyword.

        Args:
            keyword: Search keyword or phrase. Automatically wrapped in quotes for exact matching.
            num_results: Maximum number of articles to return (default: 10).

        Returns:
            List of articles with title, authors, journal, year, and abstract.
        """
        if not (keyword.startswith('"') and keyword.endswith('"')):
            keyword = f'"{keyword}"'

        def _fetch() -> list:
            fetcher = PubMedFetcher()
            pmids = fetcher.pmids_for_query(keyword, retmax=num_results)
            articles = []
            for pmid in pmids:
                try:
                    article = fetcher.article_by_pmid(pmid)
                    articles.append({
                        "title": article.title,
                        "authors": article.authors,
                        "journal": article.journal,
                        "year": article.year,
                        "abstract": article.abstract,
                    })
                except Exception:
                    continue
            return articles

        try:
            articles = await asyncio.to_thread(_fetch)
            return {
                "status": "success",
                "keyword": keyword,
                "count": len(articles),
                "articles": articles,
            }
        except Exception as e:
            return {
                "status": "failure",
                "error": f"PubMed not reachable: {e}",
            }


    async def get_pico(
        self,
        title: str,
        abstract: str,
    ) -> Dict[str, Any]:
        """
        Extract PICO elements from a medical research paper.

        PICO framework: Population, Intervention, Comparison, Outcome.
        Useful for evidence-based medicine and systematic review workflows.

        Args:
            title: Title of the paper.
            abstract: Abstract of the paper.

        Returns:
            PICO breakdown with population, intervention, comparison, and outcome fields.
        """
        prompt = _PICO_PROMPT.format(title=title, abstract=abstract)
        try:
            result = await _llm_json(prompt)
            return {"status": "success", "pico": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def get_study_taxonomy(
        self,
        title: str,
        abstract: str,
    ) -> Dict[str, Any]:
        """
        Hierarchically classify the study type of a medical paper.

        Top-level types: observational | experimental | literature review.
        Observational subtypes — time: prospective/retrospective; design: cross-sectional/cohort/case-control.
        Experimental subtypes — allocation: randomized/non-randomized/propensity score matching;
                                environment: clinical/in vivo/in vitro/in silico.
        Literature review subtypes: historical | systematic | meta-analysis.

        Args:
            title: Title of the paper.
            abstract: Abstract of the paper.

        Returns:
            Taxonomy dict with hierarchical category labels and LLM reasoning for each level.
        """
        fmt = {"title": title, "abstract": abstract}
        taxonomy: Dict[str, Any] = {}

        try:
            type_result = await _llm_json(_STUDY_TYPE_PROMPT.format(**fmt))
            study_type = type_result.get("category", "").lower()
            taxonomy["type"] = {
                "category": study_type,
                "reasoning": type_result.get("reasoning", ""),
            }

            if study_type == "observational":
                time_result = await _llm_json(_OBS_TIME_PROMPT.format(**fmt))
                taxonomy["time"] = {
                    "category": time_result.get("category", ""),
                    "reasoning": time_result.get("reasoning", ""),
                }
                cohort_result = await _llm_json(_OBS_COHORT_PROMPT.format(**fmt))
                taxonomy["cohort"] = {
                    "category": cohort_result.get("category", ""),
                    "reasoning": cohort_result.get("reasoning", ""),
                }

            elif study_type == "experimental":
                distr_result = await _llm_json(_EXP_DISTR_PROMPT.format(**fmt))
                taxonomy["distribution"] = {
                    "category": distr_result.get("category", ""),
                    "reasoning": distr_result.get("reasoning", ""),
                }
                env_result = await _llm_json(_EXP_ENV_PROMPT.format(**fmt))
                taxonomy["environment"] = {
                    "category": env_result.get("category", ""),
                    "reasoning": env_result.get("reasoning", ""),
                }

            elif "literature" in study_type or study_type == "litreview":
                lit_result = await _llm_json(_LIT_REVIEW_PROMPT.format(**fmt))
                taxonomy["literature_review_type"] = {
                    "category": lit_result.get("category", ""),
                    "reasoning": lit_result.get("reasoning", ""),
                }

            return {"status": "success", "taxonomy": taxonomy}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def analyze_medical_image(
        self,
        artifact_id: str,
        question: str,
        tool_context: ToolContext = None,
    ) -> Dict[str, Any]:
        """
        Analyze a medical image (DICOM or standard format) using the med_vlm service.

        The image must be uploaded by the user first — the before_model_callback automatically
        saves each uploaded file as an artifact and injects its artifact_id into the conversation.
        Pass that artifact_id here.

        Submits the image to a multi-step VLM pipeline that performs:
        modality classification, visual inspection, clinical interpretation with ICD-10 codes,
        differential diagnosis, and ACR-guideline validation.

        Args:
            artifact_id: Artifact ID of the uploaded DICOM or image file
                         (shown as "artifact_id=..." in the conversation context).
            question: Clinical question or patient context (any language).

        Returns:
            Full medical analysis text from the VLM pipeline, including differential diagnosis.
        """
        if tool_context is None:
            return {"status": "error", "error": "No tool context available to load the artifact."}

        artifact = await tool_context.load_artifact(filename=artifact_id)
        if artifact is None:
            return {"status": "error", "error": f"Artifact '{artifact_id}' not found."}

        file_bytes = artifact.inline_data.data
        filename = artifact_id

        # filename = '/app/dicom.dcm'
        # with open(filename, 'rb') as f:
        #     file_bytes = f.read()  # ✅ actual bytes

        async with httpx.AsyncClient(auth=_VLM_AUTH, timeout=60.0) as client:
            submit_resp = await client.post(
                _VLM_TASK_URL,
                data={"text": question},
                files={"file": (filename, file_bytes)},
            )
            submit_resp.raise_for_status()
            task_id = submit_resp.json()["task_id"]

            for _ in range(_VLM_MAX_POLLS):
                await asyncio.sleep(_VLM_POLL_INTERVAL_SEC)
                poll_resp = await client.get(_VLM_RESULT_URL, params={"task_id": task_id})
                poll_resp.raise_for_status()
                result = poll_resp.json()
                if result.get("status") == "ok":
                    return {
                        "status": "success",
                        "task_id": task_id,
                        "analysis": result.get("text", ""),
                    }

        return {
            "status": "error",
            "task_id": task_id,
            "error": f"VLM service did not complete within {_VLM_MAX_POLLS * _VLM_POLL_INTERVAL_SEC}s.",
        }


med_toolset= MedToolset()
med_toolset_instance = med_toolset.get_tools(None)
