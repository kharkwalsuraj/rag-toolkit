"""MinerU cloud client (Precision API): upload a PDF, wait, download and unpack the result.

Flow: request upload URL -> PUT the file -> poll the batch -> download result zip -> extract.
Results are cached under ``workdir/<doc_id>/`` so the same PDF is never parsed twice.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import requests

from ..exceptions import ConfigError, ParsingError
from ..models import ParsedDocument
from ..utils import make_doc_id
from .loader import load_mineru_output

log = logging.getLogger("rag.parsing")

DEFAULT_BASE_URL = "https://mineru.net/api/v4"
MAX_FILE_BYTES = 200 * 1024 * 1024  # MinerU's per-file limit
MAX_ZIP_BYTES = 2 * 1024 ** 3
_MARKER = ".complete"


class MinerUClient:
    def __init__(
        self,
        token: Optional[str],
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "vlm",
        language: str = "en",
        ocr: bool = False,
        enable_table: bool = True,
        enable_formula: bool = True,
        workdir: Union[str, Path] = ".rag_data/parsed",
        poll_interval: float = 5.0,
        timeout: float = 1800.0,
        request_timeout: float = 60.0,
        on_progress: Optional[Callable[[str], None]] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.language = language
        self.ocr = ocr
        self.enable_table = enable_table
        self.enable_formula = enable_formula
        self.workdir = Path(workdir)
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.request_timeout = request_timeout
        self.on_progress = on_progress or (lambda msg: log.info("%s", msg))
        self._session = session or requests.Session()

    # public
    def parse(self, pdf_path: Union[str, Path], *, force: bool = False) -> ParsedDocument:
        """Parse a PDF (or return the cached result). Raises ``ParsingError`` on failure."""
        pdf = Path(pdf_path)
        self._validate_pdf(pdf)
        doc_id = make_doc_id(pdf)
        out_dir = self.workdir / doc_id

        if (out_dir / _MARKER).exists() and not force:
            self.on_progress(f"Using cached MinerU result for {pdf.name}")
            return load_mineru_output(out_dir, doc_id=doc_id)
        if not self.token:
            raise ConfigError("No MinerU token given. Pass mineru_token in Settings (or token= to MinerUClient). "
                              "Get one at https://mineru.net/apiManage")

        self.workdir.mkdir(parents=True, exist_ok=True)
        batch_id, upload_url = self._request_upload(pdf, doc_id)
        self.on_progress(f"Uploading {pdf.name} ({pdf.stat().st_size / 1e6:.1f} MB)")
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

        self.on_progress(f"MinerU parsing finished for {pdf.name}")
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
            raise ParsingError(f"MinerU returned non-JSON response for {path}") from exc
        if payload.get("code") not in (0, "0"):
            raise ParsingError(f"MinerU API error {payload.get('code')}: {payload.get('msg')}")
        return payload.get("data") or {}

    def _request_upload(self, pdf: Path, doc_id: str):
        data = self._api(
            "POST",
            "/file-urls/batch",
            json={
                "enable_formula": self.enable_formula,
                "enable_table": self.enable_table,
                "language": self.language,
                "model_version": self.model,
                "files": [{"name": pdf.name, "is_ocr": self.ocr, "data_id": doc_id}],
            },
        )
        urls = data.get("file_urls") or []
        if not data.get("batch_id") or not urls:
            raise ParsingError("MinerU did not return an upload URL")
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
        raise ParsingError(f"Upload failed after {attempts} attempts: {last}") from last

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
                    raise ParsingError(f"MinerU failed to parse the file: {entry.get('err_msg')}")
                if state == "done" and entry.get("full_zip_url"):
                    return entry
                progress = entry.get("extract_progress") or {}
                msg = f"MinerU state: {state}"
                if progress.get("total_pages"):
                    msg += f" ({progress.get('extracted_pages', 0)}/{progress['total_pages']} pages)"
                if msg != last_msg:
                    self.on_progress(msg)
                    last_msg = msg
            if time.monotonic() > deadline:
                raise ParsingError(f"Timed out after {self.timeout:.0f}s waiting for MinerU")
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
                                raise ParsingError("Result zip is unexpectedly large")
                            fh.write(block)
                return
            except requests.RequestException as exc:  # result zip can be briefly unavailable
                last = exc
                time.sleep(min(2 ** attempt * 2, 20))
        raise ParsingError(f"Could not download MinerU result: {last}") from last

    def _http(self, method: str, url: str, attempts: int = 3, **kwargs: Any) -> requests.Response:
        last: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                resp = self._session.request(method, url, timeout=self.request_timeout, **kwargs)
                if resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                last = exc
                time.sleep(min(2 ** attempt, 10))
        raise ParsingError(f"Request to MinerU failed after {attempts} attempts: {last}") from last

    @staticmethod
    def _validate_pdf(pdf: Path) -> None:
        if not pdf.is_file():
            raise ParsingError(f"File not found: {pdf}")
        if pdf.suffix.lower() != ".pdf":
            raise ParsingError(f"Only PDF files are supported: {pdf.name}")
        if pdf.stat().st_size > MAX_FILE_BYTES:
            raise ParsingError(f"{pdf.name} exceeds MinerU's 200 MB limit; split it first")


def _safe_extract(zip_path: Path, dest: Path) -> None:
    """Extract a zip while rejecting path traversal, symlinks and zip bombs."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    total = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if root != target and root not in target.parents:
                raise ParsingError(f"Unsafe path in result zip: {info.filename}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ParsingError(f"Symlink in result zip: {info.filename}")
            total += info.file_size
            if total > MAX_ZIP_BYTES:
                raise ParsingError("Result zip expands to an unexpected size")
        zf.extractall(dest)
