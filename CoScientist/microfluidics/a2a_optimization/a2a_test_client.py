"""Small dependency-free client used by the cross-platform A2A smoke tests."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_POLL_INTERVAL = 5.0


class A2ARequestError(RuntimeError):
    def __init__(self, message: str, response: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.response = response


@dataclass
class A2AClient:
    base_url: str
    timeout: float = 300.0
    # JSON-RPC endpoint relative to base_url ("" = base_url itself), the
    # method names of the server's protocol version, and extra HTTP headers
    # (e.g. ``A2A-Version`` for a v1.0 server).
    rpc_path: str = ""
    send_method: str = "message/send"
    get_method: str = "tasks/get"
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/") + "/"

    def agent_card(self) -> dict[str, Any]:
        return self._request(".well-known/agent-card.json")

    def send_message(
        self,
        *,
        text: str,
        experiment_id: str,
        context_id: str,
        message_id: str,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {
            "messageId": message_id,
            "role": "ROLE_USER",
            "contextId": context_id,
            "parts": [{"text": text}],
        }
        if task_id:
            message["taskId"] = task_id
        return self.rpc(
            method=self.send_method,
            params={
                "message": message,
                "metadata": {
                    "experiment_id": experiment_id,
                    "domain": "experiment-lab",
                },
            },
            request_id=f"message-{uuid.uuid4().hex}",
        )

    def get_task(self, task_id: str) -> dict[str, Any]:
        return self.rpc(
            method=self.get_method,
            params={"id": task_id},
            request_id=f"status-{uuid.uuid4().hex}",
        )

    def rpc(
        self, *, method: str, params: dict[str, Any], request_id: str
    ) -> dict[str, Any]:
        return self._request(
            self.rpc_path,
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            },
        )

    def _request(
        self, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        request = Request(
            self.base_url + path,
            data=payload,
            headers={**self.headers, **({"Content-Type": "application/json"} if payload else {})},
            method="POST" if payload else "GET",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            parsed = _parse_json(raw)
            raise A2ARequestError(
                f"HTTP {exc.code}: {raw}", parsed if isinstance(parsed, dict) else None
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise A2ARequestError(f"A2A endpoint is unavailable: {exc}") from exc
        parsed = _parse_json(raw)
        if not isinstance(parsed, dict):
            raise A2ARequestError(f"Expected a JSON object, received: {raw}")
        if parsed.get("error"):
            error = parsed["error"]
            raise A2ARequestError(
                f"JSON-RPC {error.get('code')}: {error.get('message')}", parsed
            )
        return parsed


def task_from_response(response: dict[str, Any]) -> dict[str, Any]:
    result = response.get("result") or {}
    task = result.get("task") if isinstance(result, dict) else None
    return task if isinstance(task, dict) else result


def task_phase(task: dict[str, Any]) -> str:
    parts = (((task.get("status") or {}).get("message") or {}).get("parts") or [])
    for part in parts:
        data = part.get("data") if isinstance(part, dict) else None
        if isinstance(data, dict) and data.get("phase"):
            return str(data["phase"])
    return ""


def task_state(task: dict[str, Any]) -> str:
    return str((task.get("status") or {}).get("state") or "")


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _parse_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None
