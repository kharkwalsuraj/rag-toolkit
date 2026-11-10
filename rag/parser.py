# MinerU cloud client. A PDF is uploaded, polled until done, downloaded and unzipped into
# workdir/<doc_id>/, which also serves as the cache.

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import requests

from .exceptions import ConfigError, RAGError
from .schemas import ParsedDocument
from .utils import make_doc_id, slugify

log = logging.getLogger("rag.parser")

DEFAULT_BASE_URL = "https://mineru.net/api/v4"
MAX_FILE_BYTES = 200 * 1024 * 1024  # MinerU's per-file limit
MAX_ZIP_BYTES = 2 * 1024 ** 3
_MARKER = ".complete"



def load_mineru_output(path: Union[str, Path], doc_id: Optional[str] = None) -> ParsedDocument:
    path = Path(path)
    if path.is_file():
        content_file = path
    else:
        candidates = sorted(path.rglob("*content_list_v2.json"))
        if len(candidates) != 1:
            raise RAGError(
                f"Expected exactly one *content_list_v2.json under {path}, found {len(candidates)}"
            )
        content_file = candidates[0]

    try:
        content_list = json.loads(content_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RAGError(f"Cannot read {content_file}: {exc}") from exc

    root = content_file.parent
    markdown_files = sorted(root.glob("full.md")) or sorted(root.glob("*.md"))
    markdown = markdown_files[0] if markdown_files else None
    return ParsedDocument(
        doc_id=doc_id or slugify(root.name if path.is_file() else path.name),
        root=root,
        content_list=content_list,
        markdown_path=markdown,
    )


class MinerUClient:
    def __init__(
        self,
        token: Optional[str],
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "vlm",
        language: str = "en",
        ocr: bool = False,
        workdir: Union[str, Path] = ".rag_data/parsed",
        poll_interval: float = 5.0,
        timeout: float = 1800.0,
    ) -> None:
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.language = language
        self.ocr = ocr
        self.workdir = Path(workdir)
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._session = requests.Session()

    # public
    def parse(self, pdf_path: Union[str, Path], *, force: bool = False) -> ParsedDocument:
        pdf = Path(pdf_path)
        self._validate_pdf(pdf)
        doc_id = make_doc_id(pdf)
        out_dir = self.workdir / doc_id

        if (out_dir / _MARKER).exists() and not force:
            log.info(f"Using cached MinerU result for {pdf.name}")
            return load_mineru_output(out_dir, doc_id=doc_id)
        if not self.token:
            raise ConfigError("No MinerU token given. Set mineru_token in Settings (https://mineru.net/apiManage)")

        self.workdir.mkdir(parents=True, exist_ok=True)
        batch_id, upload_url = self._request_upload(pdf, doc_id)
        log.info(f"Uploading {pdf.name} ({pdf.stat().st_size / 1e6:.1f} MB)")
        self._upload(pdf, upload_url)
        entry = self._wait(batch_id, doc_id)

        with tempfile.TemporaryDirectory(dir=self.workdir) as tmp:
            zip_path = Path(tmp) / "result.zip"
            self._download(entry["full_zip_url"], zip_path)
            extracted = Path(tmp) / "extracted"
            _safe_extract(zip_path, extracted)
            load_mineru_output(extracted, doc_id=doc_id)  # validates before we publish it
            (extracted / "mineru_acquisition.json").write_text(
                json.dumps({"batch_id": batch_id, "source": pdf.name, "model": self.model}, indent=2)
            )
            (extracted / _MARKER).touch()
            if out_dir.exists():
                shutil.rmtree(out_dir)
            shutil.move(str(extracted), str(out_dir))

        log.info(f"MinerU parsing finished for {pdf.name}")
        return load_mineru_output(out_dir, doc_id=doc_id)

    # API calls
    @property
    def _auth(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _api(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        resp = self._http(method, f"{self.base_url}{path}", headers=self._auth, **kwargs)
        try:
            payload = resp.json()
        except ValueError as exc:
            raise RAGError(f"MinerU returned non-JSON response for {path}") from exc
        if payload.get("code") not in (0, "0"):
            raise RAGError(f"MinerU API error {payload.get('code')}: {payload.get('msg')}")
        return payload.get("data") or {}

    def _request_upload(self, pdf: Path, doc_id: str):
        data = self._api(
            "POST",
            "/file-urls/batch",
            json={
                "enable_formula": True,
                "enable_table": True,
                "language": self.language,
                "model_version": self.model,
                "files": [{"name": pdf.name, "is_ocr": self.ocr, "data_id": doc_id}],
            },
        )
        urls = data.get("file_urls") or []
        if not data.get("batch_id") or not urls:
            raise RAGError("MinerU did not return an upload URL")
        return data["batch_id"], urls[0]

    def _upload(self, pdf: Path, url: str, attempts: int = 3) -> None:
        last: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                with open(pdf, "rb") as fh:  # presigned URL: no auth header, no content-type
                    resp = self._session.put(url, data=fh, timeout=(30, 900))
                resp.raise_for_status()
                return
            except requests.RequestException as exc:
                last = exc
                time.sleep(min(2 ** attempt, 10))
        raise RAGError(f"Upload failed after {attempts} attempts: {last}") from last

    def _wait(self, batch_id: str, doc_id: str) -> Dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        last_msg = ""
        while True:
            data = self._api("GET", f"/extract-results/batch/{batch_id}")
            results = data.get("extract_result") or []
            entry = next((r for r in results if r.get("data_id") == doc_id), results[0] if results else None)
            if entry:
                state = entry.get("state")
                if state == "failed":
                    raise RAGError(f"MinerU failed to parse the file: {entry.get('err_msg')}")
                if state == "done" and entry.get("full_zip_url"):
                    return entry
                progress = entry.get("extract_progress") or {}
                msg = f"MinerU state: {state}"
                if progress.get("total_pages"):
                    msg += f" ({progress.get('extracted_pages', 0)}/{progress['total_pages']} pages)"
                if msg != last_msg:
                    log.info(msg)
                    last_msg = msg
            if time.monotonic() > deadline:
                raise RAGError(f"Timed out after {self.timeout:.0f}s waiting for MinerU")
            time.sleep(self.poll_interval)

    def _download(self, url: str, dest: Path, attempts: int = 3) -> None:
        last: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                with self._session.get(url, stream=True, timeout=(30, 300)) as resp:
                    resp.raise_for_status()
                    size = 0
                    with open(dest, "wb") as fh:
                        for block in resp.iter_content(1 << 20):
                            size += len(block)
                            if size > MAX_ZIP_BYTES:
                                raise RAGError("Result zip is unexpectedly large")
                            fh.write(block)
                return
            except requests.RequestException as exc:  # result zip can be briefly unavailable
                last = exc
                time.sleep(min(2 ** attempt * 2, 20))
        raise RAGError(f"Could not download MinerU result: {last}") from last

    def _http(self, method: str, url: str, attempts: int = 3, **kwargs: Any) -> requests.Response:
        last: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                resp = self._session.request(method, url, timeout=60, **kwargs)
                if resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                last = exc
                time.sleep(min(2 ** attempt, 10))
        raise RAGError(f"Request to MinerU failed after {attempts} attempts: {last}") from last

    @staticmethod
    def _validate_pdf(pdf: Path) -> None:
        if not pdf.is_file():
            raise RAGError(f"File not found: {pdf}")
        if pdf.suffix.lower() != ".pdf":
            raise RAGError(f"Only PDF files are supported: {pdf.name}")
        if pdf.stat().st_size > MAX_FILE_BYTES:
            raise RAGError(f"{pdf.name} exceeds MinerU's 200 MB limit; split it first")


# reject path traversal, symlinks and oversized archives
def _safe_extract(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    total = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if root != target and root not in target.parents:
                raise RAGError(f"Unsafe path in result zip: {info.filename}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise RAGError(f"Symlink in result zip: {info.filename}")
            total += info.file_size
            if total > MAX_ZIP_BYTES:
                raise RAGError("Result zip expands to an unexpected size")
        zf.extractall(dest)
