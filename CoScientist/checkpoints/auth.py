"""Only the trusted platform administrator may manage this instance's snapshots.

Bundles contain process-wide stores. This credential authorizes ALL runs on a
dedicated instance; end-user run/tenant authorization remains with Synapse.
"""

import logging
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)
_bearer = HTTPBearer(auto_error=False, scheme_name="CheckpointAdministrator")


async def require_checkpoint_admin(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    from CoScientist.config import get_settings

    token = get_settings().checkpoints.api_token
    if token is None:
        logger.warning(
            "[CHECKPOINT_AUTH] — checkpoint API disabled: no administrator credential"
        )
        raise HTTPException(
            status_code=503, detail="Checkpoint administration is not configured"
        )
    if credentials is None or not secrets.compare_digest(
        credentials.credentials.encode("utf-8"),
        token.get_secret_value().encode("utf-8"),
    ):
        logger.warning("[CHECKPOINT_AUTH] — rejected checkpoint administration request")
        raise HTTPException(
            status_code=401,
            detail="Checkpoint administrator credential required",
            headers={"WWW-Authenticate": "Bearer"},
        )
