"""Private, content-addressed PDFs; client filenames never become filesystem paths."""
import hashlib
import os
import tempfile
from pathlib import Path
import pymupdf as fitz
from app.core.errors import ServiceError


class FileStorage:
    def __init__(self, root: Path, max_bytes: int, max_pages: int):
        self.root, self.max_bytes, self.max_pages = root.resolve(), max_bytes, max_pages

    def validate(self, content: bytes) -> int:
        if len(content) > self.max_bytes:
            raise ServiceError(413, "PDF exceeds upload limit")
        if not content.startswith(b"%PDF-"):
            raise ServiceError(422, "Expected a PDF")
        try:
            with fitz.open(stream=content, filetype="pdf") as pdf:
                if pdf.is_encrypted or not 0 < len(pdf) <= self.max_pages:
                    raise ServiceError(422, "Encrypted, empty or oversized page count")
                if any(p.rect.width > 1800 or p.rect.height > 1800 for p in pdf):
                    raise ServiceError(422, "PDF page dimensions exceed safe rendering limits")
                return len(pdf)
        except (RuntimeError, ValueError) as exc:
            raise ServiceError(422, "Unreadable PDF") from exc

    def save(self, content: bytes) -> tuple[str, str]:
        digest = hashlib.sha256(content).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True)
        key = digest + ".pdf"
        with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as out:
            path = Path(out.name)
            try:
                out.write(content)
                out.flush()
                os.fsync(out.fileno())
            except BaseException:
                path.unlink(missing_ok=True)
                raise
        try:
            os.replace(path, self.root / key)
        finally:
            path.unlink(missing_ok=True)
        return digest, key

    def path(self, key: str | None) -> Path:
        if not key or len(key) != 68 or key[64:] != ".pdf" or any(c not in "0123456789abcdef" for c in key[:64]):
            raise ServiceError(404, "Source PDF unavailable")
        path = self.root / key
        if not path.is_file():
            raise ServiceError(404, "Source PDF unavailable")
        return path

    def checked(self, key: str | None, sha256: str | None) -> bytes:
        content = self.path(key).read_bytes()
        if hashlib.sha256(content).hexdigest() != sha256:
            raise ServiceError(409, "Source PDF checksum changed")
        return content
