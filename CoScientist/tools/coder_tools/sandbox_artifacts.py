"""Bring a file out of the sandbox and put it where the rest of the system looks.

The sandbox agent uploads what it chooses to upload, and reports those in
``s3_uploads``. Everything else it produced — a checkpoint it forgot to publish,
a plot, an intermediate CSV, the whole ``/workspace`` — stays on a machine
nobody but the sandbox can read. ``list_sandbox_files`` could show that such a
file exists and there was no way to reach it.

The transfer itself is one hop: pull the bytes with the sandbox client, put them
in the same S3 bucket every other artifact lives in, and answer with the durable
``/api/artifact/<bucket>/<key>`` link. Nothing downstream needs teaching — the
report embeds that link, the chat renders it, the execution graph already reads
S3 references off a tool result into the card of the agent that produced them.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: Refuse rather than fill the disk: a sandbox directory arrives as a ZIP and
#: a training run can leave gigabytes behind. Raise it deliberately, per call.
DEFAULT_MAX_BYTES = 256 * 1024 * 1024


def _error(message: str, **extra: Any) -> Dict[str, Any]:
    return {"status": "error", "message": message, **extra}


def transfer_sandbox_artifact(
    remote_path: str,
    *,
    session_id: Optional[str] = None,
    sandbox_id: Optional[str] = None,
    tool_context: Any = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Dict[str, Any]:
    """Copy one sandbox path into durable storage and return how to open it.

    ``remote_path`` may name a file or a directory; the sandbox serves a
    directory as a ZIP, so the result is one object either way. ``sandbox_id``
    names a workspace other than the session's own; without it the transfer
    goes through whichever sandbox this session is bound to.
    """
    from CoScientist.graph.session_scope import session_key
    from CoScientist.reporting.s3_upload import upload_and_ref
    from CoScientist.tools.coder_tools import openhands_sandbox as sandbox
    from CoScientist.utils.report_links import artifact_link

    remote = str(remote_path or "").strip()
    if not remote:
        return _error("Укажите путь к файлу внутри песочницы.")

    user_id, sess = session_key(tool_context) if tool_context is not None else ("", "")
    scope = session_id or sess or "session"

    # A named temporary directory, not a named file: the sandbox decides the
    # extension (a directory comes back as .zip), and the object should keep it.
    with tempfile.TemporaryDirectory(prefix="sandbox-artifact-") as tmp:
        local = Path(tmp) / (Path(remote).name or "artifact")
        fetched = sandbox.download_sandbox_file(
            remote, str(local), session_id=session_id, sandbox_id=sandbox_id,
            tool_context=tool_context,
        )
        if fetched.get("status") != "ok":
            return _error(
                fetched.get("message") or f"Не удалось забрать {remote} из песочницы.",
                remote_path=remote,
            )

        size = int(fetched.get("size_bytes") or 0)
        if size > max_bytes:
            return _error(
                f"Файл слишком велик: {size} байт при пределе {max_bytes}. "
                "Заберите его частями или поднимите предел явно.",
                remote_path=remote, size_bytes=size,
            )

        ref = upload_and_ref(local, f"sandbox-artifacts/{scope}")
        if ref is None:
            return _error(
                "Файл получен из песочницы, но хранилище его не приняло — "
                "проверьте настройки S3. Ссылка нужна только чтобы файл "
                "пережил контейнер; чтобы просто сохранить его себе, есть "
                "«скачать».",
                remote_path=remote, size_bytes=size,
            )

    bucket, key = ref
    return {
        "status": "success",
        "remote_path": remote,
        "size_bytes": size,
        "bucket": bucket,
        "key": key,
        # The durable link, not a presigned one: a signature expires, the
        # object does not, and a report may be opened next week.
        "url": artifact_link(bucket, key),
        "sandbox_id": fetched.get("sandbox_id"),
    }


__all__ = ["transfer_sandbox_artifact", "DEFAULT_MAX_BYTES"]
