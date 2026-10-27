"""Text first, then Tesseract tha+eng. OCR failures stay visible per page."""
import subprocess
import tempfile
from pathlib import Path
import pymupdf as fitz


class TesseractExtractor:
    def extract(self, content: bytes) -> list[dict]:
        pages = []
        with fitz.open(stream=content, filetype="pdf") as pdf:
            for number, page in enumerate(pdf, 1):
                text, method, error = page.get_text().strip(), "text", None
                if not text:
                    method = "tesseract"
                    try:
                        with tempfile.TemporaryDirectory() as directory:
                            path = Path(directory) / "page.png"
                            page.get_pixmap(dpi=200).save(path)
                            result = subprocess.run(["tesseract", str(path), "stdout", "-l", "tha+eng"],
                                                    capture_output=True, timeout=90, check=True)
                            text = result.stdout.decode("utf-8").strip()
                        if not text:
                            error = "OCR returned no text"
                    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
                        error = f"OCR failed: {type(exc).__name__}"
                pages.append({"number": number, "text": text, "method": method, "error": error})
        return pages
