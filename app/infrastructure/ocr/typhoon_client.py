from __future__ import annotations

import json
import logging
import os
from typing import Callable

from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from pypdf import PdfReader
from typhoon_ocr.ocr_utils import get_anchor_text, render_pdf_to_base64png

logger = logging.getLogger(__name__)

TYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1"
TYPHOON_MODEL = "typhoon-ocr-preview"

_PROMPTS_SYS: dict[str, Callable[[str], str]] = {
    "default": lambda base_text: (
        "Below is an image of a document page along with its dimensions. "
        "Simply return the markdown representation of this document, presenting tables in markdown format as they naturally appear.\n"
        "If the document contains images, use a placeholder like dummy.png for each image.\n"
        "Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
    ),
    "structure": lambda base_text: (
        "Below is an image of a document page, along with its dimensions and possibly some raw textual content previously extracted from it. "
        "Note that the text extraction may be incomplete or partially missing. Carefully consider both the layout and any available text to reconstruct the document accurately.\n"
        "Your task is to return the markdown representation of this document, presenting tables in HTML format as they naturally appear.\n"
        "If the document contains images or figures, analyze them and include the tag <figure>IMAGE_ANALYSIS</figure> in the appropriate location.\n"
        "Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
    ),
}


class TyphoonOCRClient:
    def __init__(self, api_key: str | None) -> None:
        if not api_key:
            raise RuntimeError("TYHOON_API_KEY is required for OCR ingestion.")
        self._client = OpenAI(base_url=TYPHOON_BASE_URL, api_key=api_key)

    def _get_total_pages(self, filename: str) -> int:
        with open(filename, "rb") as f:
            return len(PdfReader(f).pages)

    def extract_to_json(
        self,
        original_file_path: str,
        output_json_path: str,
        task_type: str = "default",
        markdown: bool = True,
    ) -> None:
        if not os.path.exists(original_file_path):
            raise FileNotFoundError(original_file_path)

        total_pages = self._get_total_pages(original_file_path)
        text_content = ""
        prompt_fn = _PROMPTS_SYS.get(task_type, _PROMPTS_SYS["default"])

        for page_num in range(total_pages):
            page = page_num + 1
            image_b64 = render_pdf_to_base64png(
                original_file_path, page, target_longest_image_dim=1800
            )
            anchor = get_anchor_text(
                original_file_path, page, pdf_engine="pdfreport", target_length=8000
            )
            messages = [
                ChatCompletionUserMessageParam(
                    role="user",
                    content=[
                        {"type": "text", "text": prompt_fn(anchor)},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                )
            ]
            response = self._client.chat.completions.create(
                model=TYPHOON_MODEL,
                messages=messages,
                max_tokens=16384,
                temperature=0.1,
                top_p=0.6,
                extra_body={"repetition_penalty": 1.2},
            )
            text_output = response.choices[0].message.content
            if not text_output:
                logger.warning("No OCR output for page %d of %s", page, original_file_path)
                continue

            text_content += text_output + "\n"
            if "จึงเรียน" in text_output:
                logger.info("Last-page marker detected on page %d; stopping.", page)
                break

        final_output = {
            "natural_text": text_content.strip(),
            "markdown": markdown,
            "task_type": task_type,
            "total_pages": total_pages,
            "original_file_path": original_file_path,
            "output_json_path": output_json_path,
        }
        os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)
