"""
Offline OCR Ingestion Script — น้องกอฟ (PEA Substation AI Assistant)
===================================================================

Standalone script that processes PDF documents into structured JSON files
using a local Ollama vision model. This decouples document ingestion from
the online FastAPI serving layer.

Usage:
    python scripts/run_local_ocr.py                          # Process all PDFs
    python scripts/run_local_ocr.py --model llama3-typhoon-vision  # Use a specific model
    python scripts/run_local_ocr.py --force                   # Re-process already-converted files
    python scripts/run_local_ocr.py --pdf-dir data/standards --json-dir data/standards_json

Requirements:
    - Ollama running locally (default: http://localhost:11434)
    - A vision-capable model pulled (e.g. `ollama pull qwen2.5-vl`)
    - PyMuPDF (`pip install PyMuPDF`)
    - ollama Python client (`pip install ollama`)
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF is required. Install it with: pip install PyMuPDF")
    sys.exit(1)

try:
    from ollama import Client as OllamaClient
except ImportError:
    print("ERROR: ollama Python client is required. Install it with: pip install ollama")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "qwen2.5-vl"
DEFAULT_PDF_DIR = "data/sdd-data"
DEFAULT_JSON_DIR = "data/sdd-data_json"
DEFAULT_DPI = 200  # Resolution for PDF→image rendering
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Pages with longest edge > this threshold (in points) are considered A3 or
# larger engineering drawings and will be skipped. A4 portrait longest edge
# is ~842pt; A3 portrait is ~1190pt. The 1000pt cutoff sits safely between.
A3_LONGEST_EDGE_THRESHOLD = 1000  # points (1pt = 1/72 inch)

SYSTEM_PROMPT = (
    "You are an expert Thai electrical engineer. "
    "Extract all text and tables from this image and format it as Markdown. "
    "Rules: "
    "1. Keep all English engineering terms intact (e.g., 115kV, Transformer) do not translate them. "
    "2. Ensure Thai vowels and tones are accurate. "
    "3. Output ONLY valid JSON in this format: "
    '{ "filename": "string", "page": int, "content_markdown": "string" }.'
)

# Known problematic files that should be skipped
SKIP_FILES = [
    "สถานีไฟฟ้าลำลูกกา 3 (คพจ.2).pdf",
    "สฟ.ปากท่อ 2 (อนุมัติ + แบบ).pdf",
    "อนุมัติ อุบล 5.pdf",
    "สำเนาอนุมัติแบบและค่าใช้จ่ายสฟ.กันทรวิชัย.pdf",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_a4_page(page: fitz.Page) -> bool:
    """
    Return True if the page is A4 (or smaller). A3 / engineering-drawing
    pages whose longest edge exceeds the threshold are rejected.

    Uses the *MediaBox* dimensions (in points, 1pt = 1/72"):
      - A4:  595 × 842 pt  → longest edge ≈ 842
      - A3: 842 × 1190 pt  → longest edge ≈ 1190
    """
    rect = page.rect  # page dimensions *before* rotation is applied
    longest_edge = max(rect.width, rect.height)
    return longest_edge <= A3_LONGEST_EDGE_THRESHOLD


def render_page_to_base64(
    page: fitz.Page, dpi: int = DEFAULT_DPI
) -> str:
    """
    Render a fitz.Page to a base64-encoded PNG string.

    Auto-corrects page rotation so the vision model always receives an
    upright image regardless of the PDF's native /Rotate metadata.
    """
    # --- Auto-correct rotation -------------------------------------------
    # page.rotation returns the effective rotation (0, 90, 180, 270).
    # Setting it to 0 before rendering ensures the pixmap is upright.
    rotation = page.rotation
    if rotation != 0:
        page.set_rotation(0)

    # Render at the requested DPI (default 72 → scale factor = dpi/72)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")

    # Restore original rotation so we don't mutate the document permanently
    if rotation != 0:
        page.set_rotation(rotation)

    return base64.b64encode(png_bytes).decode("utf-8")


def call_ollama_vision(
    client: OllamaClient,
    image_b64: str,
    filename: str,
    page_num: int,
    model: str,
) -> dict:
    """
    Send a base64 image to the local Ollama vision model and return the
    parsed JSON response.
    """
    user_prompt = (
        f'Extract all content from this document page. '
        f'The source file is "{filename}", page {page_num}. '
        f'Return ONLY valid JSON with keys: "filename", "page", "content_markdown".'
    )

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_prompt,
                "images": [image_b64],
            },
        ],
        format="json",
        options={"temperature": 0.1},
    )

    raw_text = response["message"]["content"]

    # Try to parse the JSON response
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        # If the model returned invalid JSON, wrap it as best-effort
        data = {
            "filename": filename,
            "page": page_num,
            "content_markdown": raw_text,
        }

    # Ensure required keys are present
    data.setdefault("filename", filename)
    data.setdefault("page", page_num)
    data.setdefault("content_markdown", "")

    return data


def process_pdf(
    client: OllamaClient,
    pdf_path: str,
    json_dir: str,
    model: str,
    force: bool = False,
    dpi: int = DEFAULT_DPI,
) -> bool:
    """
    Process a single PDF file: render each page, run OCR via Ollama,
    and save the combined result as a JSON file.

    - A3 (or larger) pages are automatically skipped.
    - Rotated pages are de-rotated before rendering.

    Returns True if the file was processed, False if skipped.
    """
    filename = os.path.basename(pdf_path)
    stem = os.path.splitext(filename)[0]
    output_path = os.path.join(json_dir, f"{stem}_ocr.json")

    # Skip if output already exists (unless --force)
    if os.path.exists(output_path) and not force:
        print(f"  ⏭️  Skipping (already exists): {filename}")
        return False

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    print(f"  📄 Processing: {filename} ({total_pages} page(s))")

    pages_data = []
    a3_skipped = 0
    combined_markdown = ""

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page = doc.load_page(page_idx)

        # --- Filter: skip A3 / large engineering-drawing pages -----------
        if not is_a4_page(page):
            longest = max(page.rect.width, page.rect.height)
            print(
                f"      Page {page_num}/{total_pages} ... "
                f"⏭️  skipped (A3/large: longest edge {longest:.0f}pt)"
            )
            a3_skipped += 1
            continue

        # --- Rotation info (for logging) ---------------------------------
        rotation_info = ""
        if page.rotation != 0:
            rotation_info = f", derotated from {page.rotation}°"

        print(f"      Page {page_num}/{total_pages}{rotation_info} ... ", end="", flush=True)

        t0 = time.time()

        # Render page to base64 image (auto-corrects rotation)
        image_b64 = render_page_to_base64(page, dpi=dpi)

        # Call Ollama
        page_data = call_ollama_vision(client, image_b64, filename, page_num, model)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")

        pages_data.append(page_data)
        combined_markdown += page_data.get("content_markdown", "") + "\n"

        # Early stop if "จึงเรียน" (formal closing phrase) is detected
        if "จึงเรียน" in page_data.get("content_markdown", ""):
            print(f"      ⏹️  Detected closing phrase 'จึงเรียน' — stopping early.")
            break

    doc.close()

    if a3_skipped:
        print(f"      ℹ️  {a3_skipped} A3/large page(s) skipped")

    # Build the final output JSON (compatible with existing pipeline)
    final_output = {
        "natural_text": combined_markdown.strip(),
        "markdown": True,
        "total_pages": total_pages,
        "pages_processed": len(pages_data),
        "original_file_path": pdf_path,
        "output_json_path": output_path,
        "model_used": model,
        "pages": pages_data,
    }

    # Write to disk
    os.makedirs(json_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"  ✅ Saved: {output_path}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Offline OCR ingestion using local Ollama vision model",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama vision model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--pdf-dir",
        default=DEFAULT_PDF_DIR,
        help=f"Directory containing source PDFs (default: {DEFAULT_PDF_DIR})",
    )
    parser.add_argument(
        "--json-dir",
        default=DEFAULT_JSON_DIR,
        help=f"Directory for output JSON files (default: {DEFAULT_JSON_DIR})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"DPI for PDF page rendering (default: {DEFAULT_DPI})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process files even if output JSON already exists",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_OLLAMA_HOST,
        help=f"Ollama server URL (default: {DEFAULT_OLLAMA_HOST})",
    )
    args = parser.parse_args()

    pdf_dir = args.pdf_dir
    json_dir = args.json_dir

    if not os.path.isdir(pdf_dir):
        print(f"ERROR: PDF directory not found: {pdf_dir}")
        sys.exit(1)

    # Create Ollama client with explicit host
    client = OllamaClient(host=args.host)

    # Verify Ollama is reachable
    try:
        client.list()
    except Exception as e:
        print(f"ERROR: Cannot connect to Ollama at {args.host}. Is it running? ({e})")
        sys.exit(1)

    os.makedirs(json_dir, exist_ok=True)

    # Collect PDF files
    pdf_files = []
    for root, _dirs, files in os.walk(pdf_dir):
        for fname in sorted(files):
            if fname.lower().endswith(".pdf"):
                if fname in SKIP_FILES:
                    print(f"  ⚠️  Skipping known problematic file: {fname}")
                    continue
                pdf_files.append(os.path.join(root, fname))

    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        sys.exit(0)

    print(f"\n🔍 Found {len(pdf_files)} PDF(s) in {pdf_dir}")
    print(f"📂 Output directory: {json_dir}")
    print(f"🤖 Model: {args.model}")
    print(f"🌐 Host: {args.host}")
    print(f"📐 DPI: {args.dpi}")
    print("-" * 60)

    processed = 0
    skipped = 0
    failed = 0

    for pdf_path in pdf_files:
        try:
            was_processed = process_pdf(
                client, pdf_path, json_dir, args.model, force=args.force, dpi=args.dpi
            )
            if was_processed:
                processed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ❌ FAILED: {pdf_path} — {e}")
            failed += 1

    print("-" * 60)
    print(f"✅ Processed: {processed}  ⏭️ Skipped: {skipped}  ❌ Failed: {failed}")
    print("Done.")


if __name__ == "__main__":
    main()
