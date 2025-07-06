# import json
# import os
# from enum import Enum
# from typing import Callable, List, Literal

# import torch
# from langchain_core.documents import Document  # Added for creating Document objects
# from llama_index.core import (
#     Settings,
#     SimpleDirectoryReader,
#     StorageContext,
#     VectorStoreIndex,
#     load_index_from_storage,
# )
# from llama_index.core.chat_engine.types import BaseChatEngine, ChatMode
# from llama_index.core.query_engine import BaseQueryEngine
# from llama_index.core.schema import Document as LlamaDocument
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.llama_cpp import LlamaCPP
# from openai import OpenAI
# from typhoon_ocr.ocr_utils import get_anchor_text, render_pdf_to_base64png

# from app.config import VarSettings


# class DatasetName(str, Enum):
#     STANDARDS = "standards"
#     SDD_DATA = "sdd-data"


# # The dataset name for the documents
# dataset_name: DatasetName = DatasetName.SDD_DATA


# def get_doc():
#     # Define the system prompts for OCR tasks
#     PROMPTS_SYS = {
#         "default": lambda base_text: (
#             f"Below is an image of a document page along with its dimensions. "
#             f"Simply return the markdown representation of this document, presenting tables in markdown format as they naturally appear.\n"
#             f"If the document contains images, use a placeholder like dummy.png for each image.\n"
#             f"Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
#             f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
#         ),
#         "structure": lambda base_text: (
#             f"Below is an image of a document page, along with its dimensions and possibly some raw textual content previously extracted from it. "
#             f"Note that the text extraction may be incomplete or partially missing. Carefully consider both the layout and any available text to reconstruct the document accurately.\n"
#             f"Your task is to return the markdown representation of this document, presenting tables in HTML format as they naturally appear.\n"
#             f"If the document contains images or figures, analyze them and include the tag <figure>IMAGE_ANALYSIS</figure> in the appropriate location.\n"
#             f"Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
#             f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
#         ),
#     }

#     def get_prompt(prompt_name: str) -> Callable[[str], str]:
#         """
#         Fetches the system prompt based on the provided PROMPT_NAME.

#         :param prompt_name: The identifier for the desired prompt.
#         :return: The system prompt as a string.
#         """
#         return PROMPTS_SYS.get(prompt_name, lambda x: "Invalid PROMPT_NAME provided.")

#     def get_total_pages(filename: str) -> int:
#         """
#         Retrieves the total number of pages in a PDF file.

#         :param filename: The path to the PDF file.
#         :return: The total number of pages in the PDF.
#         """
#         from pypdf import PdfReader

#         with open(filename, "rb") as f:
#             reader = PdfReader(f)
#             return len(reader.pages)

#     def get_response(PROMPT: str, image_base64: str):
#         openai = OpenAI(
#             base_url="https://api.opentyphoon.ai/v1",
#             api_key=VarSettings.TYHOON_API_KEY,
#         )

#         # Ensure messages are typed as List[ChatCompletionMessageParam]
#         from openai.types.chat import ChatCompletionUserMessageParam

#         typed_messages: list[ChatCompletionUserMessageParam] = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": PROMPT},
#                     {
#                         "type": "image_url",
#                         "image_url": {"url": f"data:image/png;base64,{image_base64}"},
#                     },
#                 ],
#             }
#         ]

#         return openai.chat.completions.create(
#             model="typhoon-ocr-preview",
#             # messages=messages,
#             messages=typed_messages,
#             max_tokens=16384,
#             temperature=0.1,
#             top_p=0.6,
#             extra_body={
#                 "repetition_penalty": 1.2,
#             },
#         )

#     def extract_text_and_image_from_pdf(
#         original_file_path, output_json_path, markdown=True
#     ):
#         if not os.path.exists(original_file_path):
#             raise FileNotFoundError(f"The file {original_file_path} does not exist.")

#         # Use uitils to get the total number of pages in the PDF
#         total_pages = get_total_pages(original_file_path)

#         # Set the task type to "default" for now.
#         task_type = "default"

#         # Create the text_content variable to store the extracted text.
#         text_content = ""

#         # Iterate through each page in the PDF
#         for page_num in range(total_pages):
#             # Log the current page being processed
#             print(f"Processing page {page_num + 1} of {total_pages}...")

#             page = page_num + 1  # Page numbers are 1-based in the prompt

#             # Render the first page to base64 PNG and then load it into a PIL image.
#             image_base64 = render_pdf_to_base64png(
#                 original_file_path, page, target_longest_image_dim=1800
#             )

#             # Extract anchor text from the PDF (first page)
#             anchor_text = get_anchor_text(
#                 original_file_path, page, pdf_engine="pdfreport", target_length=8000
#             )

#             # Retrieve and fill in the prompt template with the anchor_text
#             prompt_template_fn = get_prompt(task_type)
#             PROMPT = prompt_template_fn(anchor_text)

#             # Get the response from the OpenAI API
#             response = get_response(PROMPT, image_base64)

#             # Get the text output from the response
#             text_output = response.choices[0].message.content
#             print(text_output)

#             # Append the text output to the text_content variable
#             if text_output != None:
#                 text_content += text_output + "\n"

#             # If the text output contains word "จึงเรียน", that means the page is the last page.
#             if text_output != None and "จึงเรียน" in text_output:
#                 print("Last page detected, stopping further processing.")
#                 break

#         # Prepare the final output in JSON format
#         final_output = {
#             "natural_text": text_content.strip(),
#             "markdown": markdown,
#             "task_type": task_type,
#             "total_pages": total_pages,
#             "original_file_path": original_file_path,
#             "output_json_path": output_json_path,
#         }

#         # Save the final output to the specified JSON file
#         with open(output_json_path, "w", encoding="utf-8") as f:
#             import json

#             json.dump(final_output, f, ensure_ascii=False, indent=4)

#     pdf_directory = f"data/{dataset_name.value}"  # Directory containing the PDF files

#     ocr_output_directory = f"data/{dataset_name.value}_json"  # Create a directory to store OCR'd JSON files (containing Markdown)
#     os.makedirs(ocr_output_directory, exist_ok=True)

#     problematic_files = [
#         "สถานีไฟฟ้าลำลูกกา 3 (คพจ.2).pdf",
#         "สฟ.ปากท่อ 2 (อนุมัติ + แบบ).pdf",
#         "อนุมัติ อุบล 5.pdf",
#         "สำเนาอนุมัติแบบและค่าใช้จ่ายสฟ.กันทรวิชัย.pdf",
#     ]
#     all_docs_loaded: List[Document] = []  # To accumulate all documents from all files

#     for root, dirs, files in os.walk(pdf_directory):
#         for file in files:
#             # Check if the file is in the problematic files list
#             if file in problematic_files:
#                 print(f"Skipping problematic file: {file}")
#                 continue

#             if file.endswith(".pdf"):  # Still processing original PDFs
#                 original_file_path = os.path.join(root, file)
#                 # Define the path for the OCR'd JSON file
#                 ocr_file_name = (
#                     os.path.splitext(file)[0] + "_ocr.json"
#                 )  # Appending _ocr.json to the filename
#                 ocr_file_path = os.path.join(ocr_output_directory, ocr_file_name)

#                 file_to_load = None
#                 current_file_docs = []  # To store docs from the current file

#                 # Step 1: Perform OCR on the PDF and save output as JSON (containing Markdown)
#                 print(f"Attempting to OCR (to JSON Markdown): {original_file_path}")
#                 try:
#                     # Replace the following comment with the actual typhoon-ocr call.
#                     # This call should now generate a JSON file at ocr_file_path
#                     # containing the Markdown representation.
#                     # Example (ensure you have the correct function and parameters):
#                     # typhoon_ocr_module_or_function.ocr_to_json_markdown(original_file_path, output_json_path=ocr_file_path)

#                     # Check if the ocr_file_path already exists
#                     if os.path.exists(ocr_file_path):
#                         print(
#                             f"Placeholder: OCR output JSON file already exists: {ocr_file_path}. Skipping OCR step."
#                         )
#                     else:
#                         # >>> START OF TYPHOON-OCR INTEGRATION POINT (Outputting JSON) <<<
#                         # Example:
#                         extract_text_and_image_from_pdf(
#                             original_file_path,
#                             output_json_path=ocr_file_path,
#                             markdown=True,
#                         )  # Hypothetical
#                         # >>> END OF TYPHOON-OCR INTEGRATION POINT <<<

#                     # This is a placeholder. Ensure your OCR tool creates the JSON file.
#                     if not os.path.exists(ocr_file_path):
#                         # For testing, you might manually create a sample JSON file here
#                         # or copy a pre-made one.
#                         # Example of creating a dummy JSON for testing:
#                         # sample_json_content = {"natural_text": f"# Sample Markdown for {file}\n\nThis is test content."}
#                         # with open(ocr_file_path, 'w', encoding='utf-8') as f_json:
#                         #     json.dump(sample_json_content, f_json)
#                         # print(f"Placeholder: Created dummy JSON {ocr_file_path} as OCR step is not implemented.")
#                         raise FileNotFoundError(
#                             f"OCR output JSON file not found: {ocr_file_path}. Ensure typhoon-ocr creates this file."
#                         )

#                     print(
#                         f"Successfully OCR'd (or placeholder for OCR to JSON): {original_file_path} to {ocr_file_path}"
#                     )
#                     file_to_load = ocr_file_path
#                 except Exception as ocr_error:
#                     print(f"Failed to OCR (to JSON) {original_file_path}: {ocr_error}")
#                     problematic_files.append(original_file_path)
#                     continue  # Skip to the next file if OCR fails

#                 # Step 2: Load Markdown from the OCR'd JSON file
#                 if file_to_load:
#                     print(f"Attempting to load Markdown from JSON file: {file_to_load}")
#                     try:
#                         with open(file_to_load, "r", encoding="utf-8") as f:
#                             json_data = json.load(f)

#                         # IMPORTANT: Adjust "natural_text" to the actual key in your JSON
#                         # that holds the Markdown string.
#                         natural_text = json_data.get("natural_text")

#                         if natural_text is None:
#                             raise ValueError(
#                                 f"Key 'natural_text' not found in JSON file: {file_to_load}. Check JSON structure."
#                             )

#                         # Create a LangChain Document object from the Markdown content.
#                         # Metadata can be added as needed.
#                         doc = Document(
#                             page_content=natural_text,
#                             metadata={
#                                 "source": original_file_path,
#                                 "ocr_json_source": file_to_load,
#                             },
#                         )
#                         current_file_docs = [doc]  # Assuming one document per JSON file
#                         all_docs_loaded.extend(current_file_docs)  # Accumulate docs

#                         print(
#                             f"Successfully loaded Markdown from: {file_to_load}, created {len(current_file_docs)} document(s)."
#                         )
#                         # You can preview content if needed:
#                         # print(f"Content preview: {current_file_docs[0].page_content[:200]}...")

#                     except json.JSONDecodeError as je:
#                         print(
#                             f"Failed to parse JSON from {file_to_load} (original: {original_file_path}): {je}"
#                         )
#                         problematic_files.append(original_file_path)
#                     except ValueError as ve:  # Catches the "natural_text" key error
#                         print(
#                             f"Error processing JSON content from {file_to_load} (original: {original_file_path}): {ve}"
#                         )
#                         problematic_files.append(original_file_path)
#                     except Exception as e:
#                         print(
#                             f"Failed to load/process Markdown from JSON {file_to_load} (original: {original_file_path}): {e}"
#                         )
#                         problematic_files.append(original_file_path)
#                 else:
#                     print(
#                         f"Skipping loading for {original_file_path} as OCR (to JSON) step did not produce a file to load."
#                     )

#     # After the loop, 'all_docs_loaded' will contain all Document objects.
#     # You might want to assign it to 'docs' if subsequent cells expect 'docs'.
#     docs = all_docs_loaded
#     print(f"\nTotal documents loaded from all JSON files: {len(docs)}")

#     if problematic_files:
#         print(
#             "\nProblematic files found (either failed OCR to JSON or failed loading/processing JSON):"
#         )
#         for f in problematic_files:
#             print(f)
#     else:
#         print("\nNo problematic files found with OCR (to JSON) and JSON loading.")

#     return docs


# def get_texted_doc():
#     pdf_directory = f"data/{dataset_name.value}"

#     return SimpleDirectoryReader(pdf_directory).load_data()


# def get_query_engine(
#     docs: List[Document], engine_type: Literal["as_chat_engine", "as_query_engine"]
# ):
#     def messages_to_prompt(messages):
#         prompt = ""
#         for message in messages:
#             if message.role == "system":
#                 prompt += f"<|start_header_id|>system<|end_header_id|>{message.content}<|eot_id|>\n"
#             elif message.role == "user":
#                 prompt += f"<|start_header_id|>user<|end_header_id|>{message.content}<|eot_id|>\n"
#             elif message.role == "assistant":
#                 prompt += f"<|start_header_id|>assistant<|end_header_id|>{message.content}<|eot_id|>\n"

#         # add final assistant prompt
#         prompt = prompt + "<|start_header_id|>assistant<|end_header_id|><|eot_id|>\n"
#         return prompt

#     def completion_to_prompt(completion):
#         return f"<|start_header_id|>system<|end_header_id|><|eot_id|>\n<|start_header_id|>user<|end_header_id|>{completion}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"

#     llm = LlamaCPP(
#         model_url="https://huggingface.co/scb10x/typhoon2.1-gemma3-4b-gguf/resolve/main/typhoon2.1-gemma3-4b-q4_k_m.gguf",
#         # ? It takes too long to inference using 12B model, around ~2.8min used with 5060ti 16GB
#         # ? model_url="https://huggingface.co/scb10x/typhoon2.1-gemma3-12b-gguf/resolve/main/typhoon2.1-gemma3-12b-q4_k_m.gguf",
#         temperature=0.1,
#         context_window=8192,
#         max_new_tokens=1024,  # อาจจะตอบเพิ่มออกมาอีก 1024 bytes
#         generate_kwargs={},
#         model_kwargs={
#             "repetition-penalty": 1.4,
#             "no_repeat_ngram_size": 4,
#             # "response_format": { "type": "json_object" },
#             "n_gpu_layers": -1,
#         },
#         # model_kwargs={"n_gpu_layers": 33},
#         messages_to_prompt=messages_to_prompt,
#         completion_to_prompt=completion_to_prompt,
#         verbose=True,
#     )

#     embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
#     Settings.chunk_size = 3840
#     Settings.chunk_overlap = 256
#     Settings.llm = llm
#     Settings.embed_model = embed_model

#     langchain_documents = docs
#     llama_docs = [
#         LlamaDocument(text=doc.page_content, metadata=doc.metadata)
#         for doc in langchain_documents
#     ]

#     persist_dir = "./storage"
#     index = None

#     if os.path.exists(persist_dir) and os.listdir(persist_dir):
#         # rebuild storage context
#         storage_context = StorageContext.from_defaults(persist_dir="storage")
#         # load index
#         index = load_index_from_storage(storage_context, index_id=dataset_name.value)
#     else:
#         # Create new index and persist it
#         index = VectorStoreIndex.from_documents(llama_docs, show_progress=True)
#         index.set_index_id(dataset_name.value)
#         index.storage_context.persist("./storage")

#     if engine_type == "as_chat_engine":
#         return index.as_chat_engine(
#             similarity_top_k=1,
#             chat_mode=ChatMode.CONDENSE_QUESTION,
#             # verbose=True,
#             streaming=True,
#             # text_qa_template=prompt_template,
#         )

#     return index.as_query_engine(
#         similarity_top_k=1,
#         chat_mode=ChatMode.CONDENSE_QUESTION,
#         # verbose=True,
#         streaming=True,
#         # text_qa_template=prompt_template,
#     )


# def build_llm():
#     print("Building LLM...")

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     if device == "cuda":
#         print("Using GPU for LLM processing.")
#     else:
#         print("Using CPU for LLM processing.")

#     docs = get_doc()
#     if not docs:
#         print("No documents found. Exiting LLM build process.")
#         return

#     engine = get_query_engine(docs, engine_type="as_query_engine")
#     if not isinstance(engine, BaseQueryEngine) and not isinstance(
#         engine, BaseChatEngine
#     ):
#         print("Failed to create a valid query engine. Exiting LLM build process.")
#         return

#     print("LLM build process completed successfully.")
#     return engine


# query_engine = build_llm()
