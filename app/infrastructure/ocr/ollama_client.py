"""Ollama Vision OCR adapter.

Replaces the legacy TyphoonOCRClient. Uses a local Ollama vision model
(e.g. qwen2.5-vl) via PyMuPDF for PDF rendering and the ``ollama`` Python
client for text extraction.

The output JSON schema is identical to what the old Typhoon adapter
produced, so ``build_index`` consumes it transparently.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
from typing import Callable

import fitz  # PyMuPDF
from ollama import Client as OllamaClient
from PIL import Image

logger = logging.getLogger(__name__)

# Qwen2.5-VL vision encoder patch size — image dimensions must be
# divisible by this value to avoid GGML_ASSERT tensor shape errors.
_PATCH_SIZE = 28

# Pages with longest edge above this threshold (in PDF points, 1pt = 1/72")
# are treated as A3 / engineering-drawing pages and skipped.
# A4 longest edge ≈ 842 pt; A3 ≈ 1190 pt.
_A3_THRESHOLD_PT = 1000

_DEFAULT_DPI = 200

_SYSTEM_PROMPT = (
    "You are an expert Thai electrical engineer. "
    "Extract all text and tables from this image and format it as Markdown. "
    "Rules: "
    "1. Keep all English engineering terms intact (e.g., 115kV, Transformer) do not translate them. "
    "2. Ensure Thai vowels and tones are accurate. "
    "3. Output ONLY valid JSON in this format: "
    '{ "filename": "string", "page": int, "content_markdown": "string" }.'
)

# Known problematic PDFs that cannot be reliably OCR-ed.
DEFAULT_SKIP_FILES: tuple[str, ...] = ()


class OllamaOCRClient:
    """OCR adapter backed by a local Ollama vision model."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5-vl",
        dpi: int = _DEFAULT_DPI,
    ) -> None:
        self._client = OllamaClient(host=host)
        self._model = model
        self._dpi = dpi

        # Verify connectivity
        try:
            self._client.list()
        except Exception as exc:
            raise ConnectionError(
                f"Cannot connect to Ollama at {host}. Is it running?"
            ) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_a4(page: fitz.Page) -> bool:
        rect = page.rect
        return max(rect.width, rect.height) <= _A3_THRESHOLD_PT

    @staticmethod
    def _render_page(page: fitz.Page, dpi: int) -> str:
        """Render a page to a base64-encoded PNG, correcting rotation and
        resizing to dimensions divisible by the vision-encoder patch size."""
        rotation = page.rotation
        if rotation != 0:
            page.set_rotation(0)

        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")

        if rotation != 0:
            page.set_rotation(rotation)

        img = Image.open(io.BytesIO(png_bytes))
        w, h = img.size
        new_w = (w // _PATCH_SIZE) * _PATCH_SIZE
        new_h = (h // _PATCH_SIZE) * _PATCH_SIZE
        if new_w != w or new_h != h:
            img = img.resize((new_w, new_h), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _ocr_page(self, image_b64: str, filename: str, page_num: int) -> dict:
        user_prompt = (
            f'Extract all content from this document page. '
            f'The source file is "{filename}", page {page_num}. '
            f'Return ONLY valid JSON with keys: "filename", "page", "content_markdown".'
        )
        response = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [image_b64],
                },
            ],
            format="json",
            options={"temperature": 0.1},
        )
        raw = response["message"]["content"]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"filename": filename, "page": page_num, "content_markdown": raw}
        data.setdefault("filename", filename)
        data.setdefault("page", page_num)
        data.setdefault("content_markdown", "")
        return data

    # ------------------------------------------------------------------
    # Public API (same contract as the old TyphoonOCRClient)
    # ------------------------------------------------------------------

    def extract_to_json(
        self,
        original_file_path: str,
        output_json_path: str,
        *,
        task_type: str = "default",
        markdown: bool = True,
    ) -> None:
        """Process a PDF and write the OCR result as a JSON file.

        The output JSON has the ``natural_text`` key that the ingestion
        pipeline expects — fully compatible with the old Typhoon output.
        """
        if not os.path.exists(original_file_path):
            raise FileNotFoundError(original_file_path)

        filename = os.path.basename(original_file_path)
        doc = fitz.open(original_file_path)
        total_pages = doc.page_count

        text_content = ""
        pages_processed = 0
        a3_skipped = 0

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc.load_page(page_idx)

            if not self._is_a4(page):
                longest = max(page.rect.width, page.rect.height)
                logger.info(
                    "  Page %d/%d skipped (A3/large: %.0fpt)", page_num, total_pages, longest
                )
                a3_skipped += 1
                continue

            rotation_info = f" (derotated from {page.rotation}°)" if page.rotation else ""
            logger.info("  Page %d/%d%s ...", page_num, total_pages, rotation_info)

            try:
                image_b64 = self._render_page(page, self._dpi)
                page_data = self._ocr_page(image_b64, filename, page_num)
                text_content += page_data.get("content_markdown", "") + "\n"
                pages_processed += 1
            except Exception as exc:
                logger.warning("  Page %d/%d failed: %s", page_num, total_pages, exc)
                continue

            if "จึงเรียน" in page_data.get("content_markdown", ""):
                logger.info("  Closing phrase detected on page %d; stopping.", page_num)
                break

        doc.close()

        if a3_skipped:
            logger.info("  %d A3/large page(s) skipped", a3_skipped)

        final_output = {
            "natural_text": text_content.strip(),
            "markdown": markdown,
            "task_type": task_type,
            "total_pages": total_pages,
            "pages_processed": pages_processed,
            "original_file_path": original_file_path,
            "output_json_path": output_json_path,
            "model_used": self._model,
        }
        os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)
