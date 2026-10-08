import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class RetrosynthesisServiceError(RuntimeError):
    pass


class RetrosynthesisServiceClient:
    def __init__(
        self,
        host: str,
        port: str,
        timeout: int = 60,
    ) -> None:
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
        self.timeout = timeout

    def _post(
        self,
        endpoint: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}{endpoint}"
        logger.info("Calling Retrosynthesis API: %s", url)

        try:
            response = requests.post(
                url,
                json=json,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise RetrosynthesisServiceError(
                f"Failed to call Retrosynthesis API at {url}: {e}"
            ) from e

        try:
            payload = response.json()
        except ValueError as e:
            raise RetrosynthesisServiceError(f"Invalid JSON response from {url}") from e

        if payload is None:
            raise RetrosynthesisServiceError(f"Empty JSON response from {url}")

        return payload

    def retrosynthesis_result(
        self,
        smiles: str,
        mode: str = "fast",
        max_routes: int = 5,
    ) -> Any:
        payload = self._post(
            "/api/v1/retrosynthesis/result",
            json={"smiles": smiles},
            params={"mode": mode},
        )
        if isinstance(payload, dict) and isinstance(payload.get("routes"), list):
            payload["routes"] = payload["routes"][: max(0, int(max_routes))]
        return payload

    def classify_reaction_smiles(
        self,
        smiles: list[str],
        num_results: int = 10,
    ) -> Any:
        return self._post(
            "/api/v1/reaction-classification/classify",
            json={"smiles": smiles, "num_results": num_results},
        )

    def forward_predict_products(
        self,
        smiles: list[str],
        backend: str = "wldn5",
        model_name: str = "pistachio",
        reagents: str = "",
        solvent: str = "",
    ) -> Any:
        return self._post(
            "/api/v1/forward/predict",
            json={
                "smiles": smiles,
                "backend": backend,
                "model_name": model_name,
                "reagents": reagents,
                "solvent": solvent,
            },
        )
