"""A completion that never returns must fail, not hang forever.

The production symptom: the orchestrator delegated to HypothesesAgent, the
provider accepted the connection and then sent nothing, and the run sat silent
for fourteen minutes with an empty log. No exception was ever raised, so the
retry wrapper never fired. These tests pin the two halves of the fix — every
call carries a deadline, and the resulting Timeout is retried.
"""
import anyio
import pytest

from CoScientist.agents import common
from CoScientist.config import get_settings


@pytest.fixture
def anyio_backend():
    """The retry backoff sleeps on asyncio; the runtime does too."""
    return "asyncio"


def test_every_model_carries_a_deadline():
    expected = get_settings().llm.request_timeout
    assert expected > 0
    for llm in (common.make_llm(), common.make_coder_llm(), common.make_llm("openrouter/other")):
        assert llm._additional_args.get("timeout") == expected


def test_a_timeout_is_transient():
    """Whatever litellm raises on a deadline must be classified as retryable."""
    assert common._is_transient(TimeoutError("request timed out"))
    assert common._is_transient(RuntimeError("APITimeoutError: connection timeout"))
    assert not common._is_transient(ValueError("malformed tool schema"))


@pytest.mark.anyio
async def test_a_stalled_call_is_retried_then_succeeds(monkeypatch):
    attempts = []

    async def flaky(self, llm_request, stream=False):
        attempts.append(stream)
        if len(attempts) == 1:
            raise TimeoutError("request timed out")
        yield "answer"

    monkeypatch.setattr(common.LiteLlm, "generate_content_async", flaky)
    monkeypatch.setattr(common, "settings", get_settings())

    llm = common.make_llm()
    got = [r async for r in llm.generate_content_async(object())]

    assert got == ["answer"]
    assert len(attempts) == 2


@pytest.mark.anyio
async def test_a_partial_stream_is_never_retried(monkeypatch):
    """Retrying after output has been yielded would duplicate it."""
    attempts = []

    async def stalls_midway(self, llm_request, stream=False):
        attempts.append(stream)
        yield "first half"
        raise TimeoutError("request timed out")

    monkeypatch.setattr(common.LiteLlm, "generate_content_async", stalls_midway)

    llm = common.make_llm()
    with pytest.raises(TimeoutError):
        async for _ in llm.generate_content_async(object()):
            pass

    assert len(attempts) == 1


@pytest.mark.anyio
async def test_a_stream_that_opens_and_goes_quiet_is_not_waited_on_forever(monkeypatch):
    """The production hang: the socket stays open and no chunk ever arrives.

    The call has to end. Whether it ends by raising or, once the retries are
    spent, by the fail-soft path closing the agent's turn is the retry loop's
    business — what matters here is that the silence is bounded and retried
    instead of waited on.
    """
    attempts = []

    async def opens_then_silent(self, llm_request, stream=False):
        attempts.append(stream)
        await anyio.sleep(3600)
        yield "never reached"

    monkeypatch.setattr(common.LiteLlm, "generate_content_async", opens_then_silent)
    monkeypatch.setattr(common, "REQUEST_TIMEOUT", 0.05)

    llm = common.make_llm()
    got = []
    with anyio.fail_after(20):
        try:
            async for chunk in llm.generate_content_async(object()):
                got.append(chunk)
        except TimeoutError as err:
            assert "sent nothing" in str(err)

    # Either nothing, or the fail-soft path's one explanatory response — never
    # the model's own output, since none ever arrived.
    assert len(got) <= 1
    # Retried rather than hanging on the first attempt.
    assert len(attempts) > 1


@pytest.mark.anyio
async def test_a_stream_that_stalls_after_a_chunk_fails_instead_of_hanging(monkeypatch):
    """Mid-stream silence must surface as an error; retrying would duplicate."""
    attempts = []

    async def stalls_after_one(self, llm_request, stream=False):
        attempts.append(stream)
        yield "first chunk"
        await anyio.sleep(3600)

    monkeypatch.setattr(common.LiteLlm, "generate_content_async", stalls_after_one)
    monkeypatch.setattr(common, "REQUEST_TIMEOUT", 0.05)

    llm = common.make_llm()
    got = []
    with pytest.raises(TimeoutError, match="sent nothing"):
        async for chunk in llm.generate_content_async(object()):
            got.append(chunk)

    assert got == ["first chunk"]
    assert len(attempts) == 1
