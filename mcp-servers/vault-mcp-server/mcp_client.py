import asyncio
import json
import os

import httpx
from fastmcp import Client

MCP_URL = os.environ.get("VAULT_MCP_URL", "http://localhost:7338/mcp")
USER_ID = "demo-user"
SESSION_ID = "demo-session-1"


async def call(client, tool, **args):
    result = await client.call_tool(tool, args)
    payload = json.loads(result.content[0].text)
    if "error" in payload:
        raise RuntimeError(f"{tool} failed: {payload['error']}")
    return payload


async def run_example():
    dummy_file = "dummy_test.txt"
    with open(dummy_file, "rb") as f:
        content = f.read()

    async with Client(MCP_URL) as client:
        # 1. Get an upload link (worker surface)
        print("[*] Requesting upload link...")
        up = await call(client, "get_upload_link",
                        user_id=USER_ID, session_id=SESSION_ID,
                        filename=dummy_file, feature="scratch")
        s3_key = up["s3_key"]
        print(f"[+] Upload key: {s3_key} (link expires in {up['expires_in']}s)")

        # 2. Upload with a plain HTTP PUT. No custom headers.
        print("[*] Uploading...")
        async with httpx.AsyncClient() as http:
            resp = await http.put(up["upload_url"], content=content)
            if resp.status_code != 200:
                print(f"[!] Upload failed: {resp.status_code}\n{resp.text}")
                return
        print("[+] Upload complete.")

        # 3. Set a description (framework surface)
        await call(client, "update_artifact_metadata", s3_key=s3_key,
                   description="A dummy test file for verifying the vault.")
        print("[+] Metadata updated.")

        # 4. Get a download link for the ephemeral object
        down = await call(client, "get_download_link", s3_key=s3_key)
        print(f"[+] Ephemeral link expires in {down['expires_in']}s (object remaining TTL).")

        # 5. Download
        async with httpx.AsyncClient() as http:
            resp = await http.get(down["presigned_url"])
            if resp.status_code != 200:
                print(f"[!] Download failed: {resp.status_code}")
                return
        print(f"[+] Success! Content: {resp.content.decode()}")
        with open("dummy_downloaded.txt", "wb") as f:
            f.write(resp.content)

        # 6. Promote to permanent (framework surface)
        promoted = await call(client, "promote_artifact", s3_key=s3_key)
        perm_key = promoted["s3_key"]
        print(f"[+] Promoted to: {perm_key}")

        # 7. Permanent objects use a plain URL with no expiry
        perm = await call(client, "get_download_link", s3_key=perm_key)
        assert perm["expires_in"] is None, "permanent links must not expire"
        async with httpx.AsyncClient() as http:
            resp = await http.get(perm["presigned_url"])
            if resp.status_code != 200:
                print(f"[!] Permanent download failed: {resp.status_code}")
                return
        print(f"[+] Permanent download OK: {resp.content.decode()}")

        # 8. Read through the resource handle
        resource = await client.read_resource(f"vault://{perm_key}")
        print(f"[+] Resource content: {resource[0].text}")

        # 9. Session manifest (derived query)
        manifest = await call(client, "get_session_manifest",
                              user_id=USER_ID, session_id=SESSION_ID)
        print(f"[+] Manifest holds {len(manifest['artifacts'])} artifacts.")

        # 10. Cleanup: dry-run first, then confirm
        dry = await call(client, "cleanup_session", user_id=USER_ID, session_id=SESSION_ID)
        print(f"[+] Dry-run cleanup would delete {dry['object_count']} objects.")
        done = await call(client, "cleanup_session",
                          user_id=USER_ID, session_id=SESSION_ID, confirm=True)
        print(f"[+] Cleanup deleted {done['deleted']} objects.")


if __name__ == "__main__":
    asyncio.run(run_example())
