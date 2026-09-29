"""read_result: values out of a tool result a server stored whole in S3.

Downloads are faked at ``_fetch``; the S3 service at ``s3_upload._get_service``.
"""
import asyncio
import json

from CoScientist.tools import read_result_tool as rr

# The shape of mordred's calculate_descriptors result, cut to four descriptors.
_MORDRED = {"values": [10.87, 194.08037556, -1.0293, 61.82],
            "names": ["ABC", "MW", "SLogP", "TopoPSA"], "n_descriptors": 4}


def _serve(monkeypatch, payload, seen=None):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def fetch(url):
        if seen is not None:
            seen.append(url)
        return body

    monkeypatch.setattr(rr, "_fetch", fetch)


def test_a_value_is_read_beside_its_name_in_parallel_lists(monkeypatch):
    """Three of mordred's 1613 descriptors took CoderAgent a hand-off, a shell
    download and a script, and its first curl broke on the unquoted link."""
    seen = []
    _serve(monkeypatch, _MORDRED, seen)

    out = asyncio.run(rr.read_result(
        "http://minio/r.json?X-Amz-Date=1&amp;X-Amz-Signature=abc",
        find=["MW", "SLogP", "TopoPSA", "nHBDon"]))

    assert out["found"]["MW"] == {"value": 194.08037556, "where": "values[1]", "matched": "names[1]"}
    assert (out["found"]["SLogP"]["value"], out["found"]["TopoPSA"]["value"]) == (-1.0293, 61.82)
    assert out["not_found"] == ["nHBDon"]
    assert out["structure"] == {"values": "list[4]", "names": "list[4]", "n_descriptors": "int"}
    assert seen == ["http://minio/r.json?X-Amz-Date=1&X-Amz-Signature=abc"]  # MCP servers escape &


def test_records_are_matched_by_field_and_long_fields_come_back_shortened(monkeypatch):
    _serve(monkeypatch, {"records": [{"smiles": "C", "MW": 16.04}, {"smiles": "CC", "MW": 30.07}],
                         "trace": list(range(100))})

    out = asyncio.run(rr.read_result("https://h/r.json", find=["MW"], keys=["trace", "absent"]))

    assert [(m["where"], m["value"]) for m in out["found"]["MW"]] == [
        ("records[0].MW", 16.04), ("records[1].MW", 30.07)]
    assert out["fields"] == {"trace": {"first": list(range(20)), "items": 100}}
    assert out["missing_keys"] == ["absent"]


def test_an_s3_key_is_read_through_a_fresh_link(monkeypatch):
    """A presigned link lives an hour; the s3_key does not expire."""
    seen = []
    _serve(monkeypatch, _MORDRED, seen)

    class _Service:
        def generate_presigned_url(self, key, expiration):
            return f"http://minio/bucket/{key}?ttl={expiration}"

    monkeypatch.setattr("CoScientist.reporting.s3_upload._get_service", lambda: _Service())
    key = "ephemeral/u/s/alembic_mordred/calculate_descriptors/1e284cd2/result/result.json"

    out = asyncio.run(rr.read_result(key, find=["MW"]))

    assert out["found"]["MW"]["value"] == 194.08037556
    assert seen == [f"http://minio/bucket/{key}?ttl={rr._LINK_TTL_SECONDS}"]


def test_a_file_that_cannot_be_read_comes_back_as_an_error(monkeypatch):
    _serve(monkeypatch, b"<Error><Code>AccessDenied</Code></Error>")
    assert "not JSON" in asyncio.run(rr.read_result("https://h/r.json"))["error"]

    def too_big(url):
        raise ValueError("the result is over 10 bytes (READ_RESULT_MAX_BYTES)")

    monkeypatch.setattr(rr, "_fetch", too_big)
    assert "READ_RESULT_MAX_BYTES" in asyncio.run(rr.read_result("https://h/r.json"))["error"]

    monkeypatch.setattr("CoScientist.reporting.s3_upload._get_service", lambda: None)
    assert "S3 is not configured" in asyncio.run(rr.read_result("a/key.json"))["error"]
