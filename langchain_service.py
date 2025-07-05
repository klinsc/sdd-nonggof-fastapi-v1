import os
from typing import Callable

from dotenv import load_dotenv

load_dotenv()
from openai import OpenAI
from typhoon_ocr.ocr_utils import get_anchor_text, render_pdf_to_base64png

# Define the system prompts for OCR tasks
PROMPTS_SYS = {
    "default": lambda base_text: (
        f"Below is an image of a document page along with its dimensions. "
        f"Simply return the markdown representation of this document, presenting tables in markdown format as they naturally appear.\n"
        f"If the document contains images, use a placeholder like dummy.png for each image.\n"
        f"Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
    ),
    "structure": lambda base_text: (
        f"Below is an image of a document page, along with its dimensions and possibly some raw textual content previously extracted from it. "
        f"Note that the text extraction may be incomplete or partially missing. Carefully consider both the layout and any available text to reconstruct the document accurately.\n"
        f"Your task is to return the markdown representation of this document, presenting tables in HTML format as they naturally appear.\n"
        f"If the document contains images or figures, analyze them and include the tag <figure>IMAGE_ANALYSIS</figure> in the appropriate location.\n"
        f"Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
    ),
}


def get_prompt(prompt_name: str) -> Callable[[str], str]:
    """
    Fetches the system prompt based on the provided PROMPT_NAME.

    :param prompt_name: The identifier for the desired prompt.
    :return: The system prompt as a string.
    """
    return PROMPTS_SYS.get(prompt_name, lambda x: "Invalid PROMPT_NAME provided.")


def get_total_pages(filename: str) -> int:
    """
    Retrieves the total number of pages in a PDF file.

    :param filename: The path to the PDF file.
    :return: The total number of pages in the PDF.
    """
    from pypdf import PdfReader

    with open(filename, "rb") as f:
        reader = PdfReader(f)
        return len(reader.pages)


def extract_text_and_image_from_pdf(
    original_file_path, output_json_path, markdown=True
):
    if not os.path.exists(original_file_path):
        raise FileNotFoundError(f"The file {original_file_path} does not exist.")

    # Use uitils to get the total number of pages in the PDF
    total_pages = get_total_pages(original_file_path)

    # Set the task type to "default" for now.
    task_type = "default"

    # Create the text_content variable to store the extracted text.
    text_content = ""

    # Iterate through each page in the PDF
    for page_num in range(total_pages):
        page = page_num + 1  # Page numbers are 1-based in the prompt

        # Render the first page to base64 PNG and then load it into a PIL image.
        image_base64 = render_pdf_to_base64png(
            original_file_path, page, target_longest_image_dim=1800
        )

        # Extract anchor text from the PDF (first page)
        anchor_text = get_anchor_text(
            original_file_path, page, pdf_engine="pdfreport", target_length=8000
        )

        # Retrieve and fill in the prompt template with the anchor_text
        prompt_template_fn = get_prompt(task_type)
        PROMPT = prompt_template_fn(anchor_text)

        from openai.types.chat import ChatCompletionUserMessageParam

        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content=[
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            )
        ]

        # send messages to openai compatible api
        openai = OpenAI(
            base_url="https://api.opentyphoon.ai/v1",
            api_key=os.getenv("TYHOON_API_KEY"),
        )

        response = openai.chat.completions.create(
            model="typhoon-ocr-preview",
            messages=messages,
            max_tokens=16384,
            temperature=0.1,
            top_p=0.6,
            extra_body={
                "repetition_penalty": 1.2,
            },
        )
        text_output = response.choices[0].message.content
        if not text_output:
            print(f"Warning: No text output for page {page}. Skipping this page.")
            continue

        print(text_output)

        # Append the text output to the text_content variable
        text_content += text_output + "\n"

        # If the text output contains word "จึงเรียน", that means the page is the last page.
        if "จึงเรียน" in text_output:
            print("Last page detected, stopping further processing.")
            break

    # Prepare the final output in JSON format
    final_output = {
        "natural_text": text_content.strip(),
        "markdown": markdown,
        "task_type": task_type,
        "total_pages": total_pages,
        "original_file_path": original_file_path,
        "output_json_path": output_json_path,
    }

    # Save the final output to the specified JSON file
    with open(output_json_path, "w", encoding="utf-8") as f:
        import json

        json.dump(final_output, f, ensure_ascii=False, indent=4)


# ! from langchain_community.document_loaders import PyPDFLoader # This import might become unused or replaced
import json
import os

from langchain_core.documents import Document  # Added for creating Document objects

pdf_directory = "data/sdd-data"
# Create a directory to store OCR'd JSON files (containing Markdown)
ocr_output_directory = "data/sdd-data_json"  # Changed directory name for clarity
os.makedirs(ocr_output_directory, exist_ok=True)

problematic_files = [
    "สถานีไฟฟ้าลำลูกกา 3 (คพจ.2).pdf",
    "สฟ.ปากท่อ 2 (อนุมัติ + แบบ).pdf",
    "อนุมัติ อุบล 5.pdf",
    "สำเนาอนุมัติแบบและค่าใช้จ่ายสฟ.กันทรวิชัย.pdf",
]
all_docs_loaded = []  # To accumulate all documents from all files

for root, dirs, files in os.walk(pdf_directory):
    for file in files:
        # Check if the file is in the problematic files list
        if file in problematic_files:
            print(f"Skipping problematic file: {file}")
            continue

        if file.endswith(".pdf"):  # Still processing original PDFs
            original_file_path = os.path.join(root, file)
            # Define the path for the OCR'd JSON file
            ocr_file_name = (
                os.path.splitext(file)[0] + "_ocr.json"
            )  # Appending _ocr.json to the filename
            ocr_file_path = os.path.join(ocr_output_directory, ocr_file_name)

            file_to_load = None
            current_file_docs = []  # To store docs from the current file

            # Step 1: Perform OCR on the PDF and save output as JSON (containing Markdown)
            print(f"Attempting to OCR (to JSON Markdown): {original_file_path}")
            try:
                # Replace the following comment with the actual typhoon-ocr call.
                # This call should now generate a JSON file at ocr_file_path
                # containing the Markdown representation.
                # Example (ensure you have the correct function and parameters):
                # typhoon_ocr_module_or_function.ocr_to_json_markdown(original_file_path, output_json_path=ocr_file_path)

                # Check if the ocr_file_path already exists
                if os.path.exists(ocr_file_path):
                    print(
                        f"Placeholder: OCR output JSON file already exists: {ocr_file_path}. Skipping OCR step."
                    )
                else:
                    # >>> START OF TYPHOON-OCR INTEGRATION POINT (Outputting JSON) <<<
                    # Example:
                    extract_text_and_image_from_pdf(
                        original_file_path,
                        output_json_path=ocr_file_path,
                        markdown=True,
                    )  # Hypothetical
                    # >>> END OF TYPHOON-OCR INTEGRATION POINT <<<

                # This is a placeholder. Ensure your OCR tool creates the JSON file.
                if not os.path.exists(ocr_file_path):
                    # For testing, you might manually create a sample JSON file here
                    # or copy a pre-made one.
                    # Example of creating a dummy JSON for testing:
                    # sample_json_content = {"natural_text": f"# Sample Markdown for {file}\n\nThis is test content."}
                    # with open(ocr_file_path, 'w', encoding='utf-8') as f_json:
                    #     json.dump(sample_json_content, f_json)
                    # print(f"Placeholder: Created dummy JSON {ocr_file_path} as OCR step is not implemented.")
                    raise FileNotFoundError(
                        f"OCR output JSON file not found: {ocr_file_path}. Ensure typhoon-ocr creates this file."
                    )

                print(
                    f"Successfully OCR'd (or placeholder for OCR to JSON): {original_file_path} to {ocr_file_path}"
                )
                file_to_load = ocr_file_path
            except Exception as ocr_error:
                print(f"Failed to OCR (to JSON) {original_file_path}: {ocr_error}")
                problematic_files.append(original_file_path)
                continue  # Skip to the next file if OCR fails

            # Step 2: Load Markdown from the OCR'd JSON file
            if file_to_load:
                print(f"Attempting to load Markdown from JSON file: {file_to_load}")
                try:
                    with open(file_to_load, "r", encoding="utf-8") as f:
                        json_data = json.load(f)

                    # IMPORTANT: Adjust "natural_text" to the actual key in your JSON
                    # that holds the Markdown string.
                    natural_text = json_data.get("natural_text")

                    if natural_text is None:
                        raise ValueError(
                            f"Key 'natural_text' not found in JSON file: {file_to_load}. Check JSON structure."
                        )

                    # Create a LangChain Document object from the Markdown content.
                    # Metadata can be added as needed.
                    doc = Document(
                        page_content=natural_text,
                        metadata={
                            "source": original_file_path,
                            "ocr_json_source": file_to_load,
                        },
                    )
                    current_file_docs = [doc]  # Assuming one document per JSON file
                    all_docs_loaded.extend(current_file_docs)  # Accumulate docs

                    print(
                        f"Successfully loaded Markdown from: {file_to_load}, created {len(current_file_docs)} document(s)."
                    )
                    # You can preview content if needed:
                    # print(f"Content preview: {current_file_docs[0].page_content[:200]}...")

                except json.JSONDecodeError as je:
                    print(
                        f"Failed to parse JSON from {file_to_load} (original: {original_file_path}): {je}"
                    )
                    problematic_files.append(original_file_path)
                except ValueError as ve:  # Catches the "natural_text" key error
                    print(
                        f"Error processing JSON content from {file_to_load} (original: {original_file_path}): {ve}"
                    )
                    problematic_files.append(original_file_path)
                except Exception as e:
                    print(
                        f"Failed to load/process Markdown from JSON {file_to_load} (original: {original_file_path}): {e}"
                    )
                    problematic_files.append(original_file_path)
            else:
                print(
                    f"Skipping loading for {original_file_path} as OCR (to JSON) step did not produce a file to load."
                )

# After the loop, 'all_docs_loaded' will contain all Document objects.
# You might want to assign it to 'docs' if subsequent cells expect 'docs'.
docs = all_docs_loaded
print(f"\nTotal documents loaded from all JSON files: {len(docs)}")

if problematic_files:
    print(
        "\nProblematic files found (either failed OCR to JSON or failed loading/processing JSON):"
    )
    for f in problematic_files:
        print(f)
else:
    print("\nNo problematic files found with OCR (to JSON) and JSON loading.")

# The variable 'docs' will now be a list of Document objects,
# where each Document's page_content is the Markdown string from a JSON file.
# Subsequent cells like 'print(docs[0])' should work if 'docs' is not empty.

# print(docs[0])
# print(len(docs[0].page_content))  # จะเห็นว่ายังไม่คลีน ต้องคลีนก่อนใช้งานจริง เพราะ RAG ไม่คิดเยอะ

"""## Cleaning"""


def clean_document(doc: Document) -> Document:
    """Clean up the page_content of a Document object."""
    cleaned = doc.page_content.replace("\\n", "\n")
    # Add more cleaning rules as needed
    # doc.page_content = doc.page_content.replace("\n\n", "\n")
    # doc.page_content = doc.page_content.replace(" ", "")
    # doc.page_content = doc.page_content.replace("|", "")
    # doc.page_content = doc.page_content.replace("-", "")
    # doc.page_content = doc.page_content.replace("#", "")

    return Document(page_content=cleaned, metadata=doc.metadata)


docs = [clean_document(doc) for doc in docs]

"""## Document Chunk and Overlab"""

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=3000, chunk_overlap=1000
)
texts = text_splitter.split_documents(docs)
len(texts)

# print(len(texts[0].page_content))
# print(texts[0].page_content)

from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer


class CustomHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name, **kwargs):
        self.model = SentenceTransformer(model_name, **kwargs)

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.model.encode([text], convert_to_tensor=True)
        return embedding.tolist()[0]


embeddings = CustomHuggingFaceEmbeddings(model_name="BAAI/bge-m3")

embeddings.model.to("cuda")  # Move the model to GPU if available


from langchain_community.vectorstores import Chroma

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=texts,
    collection_name="rag-chroma",
    embedding=embeddings,
)


import torch
from huggingface_hub import login

# Check if CUDA is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
torch.set_float32_matmul_precision("high")

# Check if MPS (Metal Performance Shaders) is available for Apple Silicon Macs
if torch.backends.mps.is_available():
    device = "mps"
print(device)

# Set environment variable to ensure CUDA errors are raised immediately
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if device != "cuda":
    raise NotImplementedError

hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig

model_id = "scb10x/typhoon2.1-gemma3-4b"

tokenizer = AutoTokenizer.from_pretrained(model_id)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_quant_type="nf4",  # or "fp4"
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print(model)


# Example usage of the model with a prompt
messages = [
    {
        "role": "system",
        "content": "You are a male AI assistant named Typhoon created by SCB 10X to be helpful, harmless, and honest. Typhoon is happy to help with analysis, question answering, math, coding, creative writing, teaching, role-play, general discussion, and all sorts of other tasks. Typhoon responds directly to all human messages without unnecessary affirmations or filler phrases like “Certainly!”, “Of course!”, “Absolutely!”, “Great!”, “Sure!”, etc. Specifically, Typhoon avoids starting responses with the word “Certainly” in any way. Typhoon follows this information in all languages, and always responds to the user in the language they use or request. Typhoon is now being connected with a human. Write in fluid, conversational prose, Show genuine interest in understanding requests, Express appropriate emotions and empathy. Also showing information in term that is easy to understand and visualized.",
    },
    {"role": "user", "content": "ขอสูตรไก่ย่าง"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is False.
).to(model.device)

outputs = model.generate(
    input_ids,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
)
response = outputs[0][input_ids.shape[-1] :]
print(tokenizer.decode(response, skip_special_tokens=True))
