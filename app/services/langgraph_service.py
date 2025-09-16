import json
import os
import sqlite3
from datetime import datetime
from enum import Enum
from typing import Callable

import torch
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document  # Added for creating Document objects
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

# from transformers.utils.quantization_config import BitsAndBytesConfig
from typhoon_ocr.ocr_utils import get_anchor_text, render_pdf_to_base64png

# ! from langchain_community.document_loaders import PyPDFLoader # This import might become unused or replaced

load_dotenv()


class DatasetName(str, Enum):
    STANDARDS = "standards"
    SDD_DATA = "sdd-data"


# The dataset name for the documents
dataset_name: DatasetName = DatasetName.SDD_DATA

# ---- SQLite setup for saving user queries ----
DB_DIR = os.path.join("storage")
DB_PATH = os.path.join(DB_DIR, "app.sqlite3")
os.makedirs(DB_DIR, exist_ok=True)


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                session_id TEXT,
                content TEXT NOT NULL,
                metadata TEXT
            )
            """
        )
        conn.commit()


def _coerce_message_content_to_text(content) -> str:
    # Handles LangChain message.content being str or list (for multimodal)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    parts.append("[image]")
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p])
    return str(content)


def save_user_query(
    content: str, session_id: str | None = None, metadata: dict | None = None
):
    meta_str = None
    if metadata is not None:
        try:
            meta_str = json.dumps(metadata, ensure_ascii=False)
        except Exception:
            meta_str = str(metadata)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO user_queries (created_at, session_id, content, metadata) VALUES (?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), session_id, content, meta_str),
        )
        conn.commit()


# Initialize DB on import
init_db()
# ---- end SQLite setup ----


def get_doc():
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
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            },
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

    return docs


docs = get_doc()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=3000, chunk_overlap=1000
)
texts = text_splitter.split_documents(docs)
len(texts)

# print(len(texts[0].page_content))
# print(texts[0].page_content)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
)


def embed_text(texts: list[Document]) -> Chroma | None:
    try:

        # Add to vectorDB
        vectorstore = Chroma.from_documents(
            documents=texts,
            collection_name=dataset_name.value,
            embedding=embeddings,
            persist_directory="storage/chroma_data",  # Directory to store Chroma data
        )

        return vectorstore

    except Exception as e:
        print(f"Error during embedding or vector store creation: {e}")
        return None


import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

from langchain_core.tools import tool


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""

    # Check if the vector store already exists
    if os.path.exists("storage/chroma_data"):

        # Load the existing vector store
        vector_store = Chroma(
            collection_name=dataset_name.value,
            embedding_function=embeddings,
            persist_directory="storage/chroma_data",
        )
    else:
        vector_store = embed_text(texts)

    if vector_store is None:
        raise ValueError("Vector store could not be created or loaded.")

    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

graph_builder = StateGraph(MessagesState)


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    # Persist the latest human query (if any)
    try:
        latest_human = next(
            (m for m in reversed(state["messages"]) if m.type == "human"), None
        )
        if latest_human is not None:
            content_text = _coerce_message_content_to_text(
                getattr(latest_human, "content", "")
            )
            # session_id is optional; pass it via state when invoking the graph if you have one
            session_id = state.get("session_id") if isinstance(state, dict) else None
            save_user_query(content_text, session_id=session_id, metadata=None)
    except Exception as e:
        # Non-fatal: log and continue
        print(f"Warning: failed to save user query to SQLite: {e}")

    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use ten sentences maximum."
        "Respond **only in Thai language**."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}


def build_graph():
    from langgraph.graph import END
    from langgraph.prebuilt import tools_condition

    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    return graph_builder.compile()
