"""
Upload predicted CSV data to OpenWebUI Knowledge Base as Markdown.

Flow:
1) Read CSV (default: output/anomaly_results.csv)
2) Convert to Markdown table
3) Upload Markdown content to OpenWebUI KB
"""

import argparse
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import requests


DEFAULT_BASE_URL = "http://localhost:3000"
DEFAULT_KB_ID = "445fe3fc-65d3-4705-9bf0-cf7770434438"
DEFAULT_CSV_PATH = "output/anomaly_results.csv"


def csv_to_markdown(df: pd.DataFrame, max_rows: int = 200) -> str:
    """Convert DataFrame to markdown without external tabulate dependency."""
    if df.empty:
        return "# Predicted Atmospheric Data\n\nNo rows available.\n"

    trimmed = df.head(max_rows).copy()
    trimmed = trimmed.fillna("")
    columns = [str(c) for c in trimmed.columns]

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows: List[str] = []
    for _, row in trimmed.iterrows():
        vals = []
        for c in columns:
            val = str(row[c]).replace("\n", " ").replace("|", "\\|")
            vals.append(val)
        rows.append("| " + " | ".join(vals) + " |")

    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    body = [
        "# Predicted Atmospheric Data",
        "",
        f"- Generated at: {generated_at}",
        f"- Total CSV rows: {len(df)}",
        f"- Rows included in this upload: {len(trimmed)}",
        "",
        header,
        separator,
        *rows,
        "",
    ]
    return "\n".join(body)


def build_headers(api_key: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key
    return headers


def create_authenticated_session(
    base_url: str,
    api_key: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    timeout: int = 30,
) -> Tuple[requests.Session, Dict[str, str], str]:
    """
    Build a session authenticated via API key or login cookie.

    Returns:
        (session, headers, auth_mode)
    """
    session = requests.Session()
    headers = build_headers(api_key)

    if api_key:
        return session, headers, "api_key"

    if not username or not password:
        return session, headers, "none"

    base = base_url.rstrip("/")
    signin_endpoints = [
        f"{base}/api/v1/auths/signin",
        f"{base}/api/auths/signin",
    ]
    payloads = [
        {"email": username, "password": password},
        {"username": username, "password": password},
    ]

    last_error = ""
    for endpoint in signin_endpoints:
        for payload in payloads:
            try:
                resp = session.post(endpoint, json=payload, timeout=timeout)
                if 200 <= resp.status_code < 300:
                    return session, headers, "session"
                last_error = f"{endpoint} -> HTTP {resp.status_code}: {resp.text[:200]}"
            except requests.RequestException as e:
                last_error = f"{endpoint} -> request failed: {e}"

    raise RuntimeError(f"OpenWebUI login failed. {last_error}")


def upload_file_to_openwebui(
    base_url: str,
    api_key: Optional[str],
    session: requests.Session,
    headers: Dict[str, str],
    auth_mode: str,
    file_path: Path,
    filename: str,
    timeout: int = 30,
) -> str:
    """
    Upload file to OpenWebUI and return `file_id`.

    OpenWebUI expects authenticated requests to `/api/v1/files/`.
    """
    if auth_mode == "none":
        raise RuntimeError(
            "OpenWebUI requires authentication to upload files to /api/v1/files/. "
            "Provide OPENWEBUI_API_KEY or (OPENWEBUI_USERNAME + OPENWEBUI_PASSWORD)."
        )

    base = base_url.rstrip("/")
    endpoints = [f"{base}/api/v1/files/", f"{base}/api/v1/files"]

    # For multipart upload, do not send application/json Content-Type.
    headers_upload = dict(headers)
    headers_upload.pop("Content-Type", None)

    last_error = ""
    for endpoint in endpoints:
        try:
            with file_path.open("rb") as fh:
                files = {"file": (filename, fh, "text/markdown")}
                resp = session.post(endpoint, headers=headers_upload, files=files, timeout=timeout)
            if not (200 <= resp.status_code < 300):
                last_error = f"{endpoint} -> HTTP {resp.status_code}: {resp.text[:400]}"
                continue

            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}

            # Common shapes: { "id": "..."} or { "file_id": "..."} or nested { "data": { ... } }
            file_id = (
                data.get("file_id")
                or data.get("id")
                or (data.get("data", {}) or {}).get("file_id")
                or (data.get("data", {}) or {}).get("id")
            )
            if file_id:
                return str(file_id)

            # If JSON parsing doesn't work, fall back to raw text (useful for debugging).
            last_error = f"{endpoint} -> upload succeeded but file_id not found in response: {resp.text[:400]}"
        except requests.RequestException as e:
            last_error = f"{endpoint} -> request failed: {e}"

    raise RuntimeError(f"OpenWebUI file upload failed. {last_error}")


def try_upload(
    base_url: str,
    kb_id: str,
    doc_name: str,
    markdown_text: str,
    api_key: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    timeout: int = 30,
) -> Tuple[bool, str]:
    knowledge_endpoint = f"{base_url.rstrip('/')}/api/v1/knowledge/{kb_id}/file/add"

    try:
        session, headers, auth_mode = create_authenticated_session(
            base_url=base_url,
            api_key=api_key,
            username=username,
            password=password,
            timeout=timeout,
        )
    except RuntimeError as e:
        return False, str(e)

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(markdown_text)
            tmp_path = Path(tmp.name)
    except Exception as e:
        return False, f"Failed to prepare markdown temp file: {e}"

    try:
        file_id = upload_file_to_openwebui(
            base_url=base_url,
            api_key=api_key,
            session=session,
            headers=headers,
            auth_mode=auth_mode,
            file_path=tmp_path,
            filename=doc_name,
            timeout=timeout,
        )

        knowledge_payload = {"file_id": file_id}
        resp = session.post(
            knowledge_endpoint,
            headers=headers,
            json=knowledge_payload,
            timeout=timeout,
        )

        if 200 <= resp.status_code < 300:
            return True, f"Uploaded markdown to KB via knowledge attachment (auth={auth_mode})"
        return (
            False,
            f"{knowledge_endpoint} -> HTTP {resp.status_code}: {resp.text[:500]}",
        )
    except requests.RequestException as e:
        return False, f"OpenWebUI request failed: {e}"
    except RuntimeError as e:
        return False, str(e)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload prediction CSV as Markdown to OpenWebUI Knowledge Base"
    )
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH, help="Path to prediction CSV file")
    parser.add_argument("--kb-id", default=DEFAULT_KB_ID, help="OpenWebUI Knowledge Base ID")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenWebUI base URL")
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENWEBUI_API_KEY", ""),
        help="OpenWebUI API key/token (or set OPENWEBUI_API_KEY env var)",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("OPENWEBUI_USERNAME", ""),
        help="OpenWebUI login username/email (optional, for session auth)",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("OPENWEBUI_PASSWORD", ""),
        help="OpenWebUI login password (optional, for session auth)",
    )
    parser.add_argument(
        "--doc-name",
        default="predicted_atmospheric_data.md",
        help="Document name inside the KB",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200,
        help="Max CSV rows to include in markdown table",
    )
    parser.add_argument(
        "--save-markdown",
        default="",
        help="Optional path to save generated markdown locally before upload",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"[ERROR] CSV file not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    markdown_text = csv_to_markdown(df, max_rows=args.max_rows)

    if args.save_markdown:
        md_path = Path(args.save_markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(markdown_text, encoding="utf-8")
        print(f"[INFO] Markdown file saved: {md_path}")

    ok, message = try_upload(
        base_url=args.base_url,
        kb_id=args.kb_id,
        doc_name=args.doc_name,
        markdown_text=markdown_text,
        api_key=args.api_key or None,
        username=args.username or None,
        password=args.password or None,
    )
    if not ok:
        print("[ERROR] Upload failed")
        print(f"[ERROR] {message}")
        print(
            "[HINT] Provide either OPENWEBUI_API_KEY or "
            "(OPENWEBUI_USERNAME + OPENWEBUI_PASSWORD), and verify the KB ID."
        )
        return 2

    print(f"[OK] {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
