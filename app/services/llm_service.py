import torch
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def load_storage_context():
    return StorageContext.from_defaults(persist_dir="./storage")


def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|start_header_id|>system<|end_header_id|>{message.content}<|eot_id|>\n"
        elif message.role == "user":
            prompt += (
                f"<|start_header_id|>user<|end_header_id|>{message.content}<|eot_id|>\n"
            )
        elif message.role == "assistant":
            prompt += f"<|start_header_id|>assistant<|end_header_id|>{message.content}<|eot_id|>\n"
    prompt += "<|start_header_id|>assistant<|end_header_id|><|eot_id|>\n"
    return prompt


def completion_to_prompt(completion):
    return (
        "<|start_header_id|>system<|end_header_id|><|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>{completion}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def create_embedding_model():
    return HuggingFaceEmbedding(model_name="BAAI/bge-m3")


def create_llm_model():
    model_url = (
        "https://huggingface.co/scb10x/typhoon2.1-gemma3-4b-gguf/resolve/main/"
        "typhoon2.1-gemma3-4b-q4_k_m.gguf"
    )

    return LlamaCPP(
        model_url=model_url,
        temperature=0.1,
        context_window=8192,
        max_new_tokens=1024,
        generate_kwargs={},
        model_kwargs={
            "repetition-penalty": 1.4,
            "no_repeat_ngram_size": 4,
            "n_gpu_layers": -1,
        },
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )


def load_index(storage_context=None):
    if storage_context is None:
        raise ValueError("Storage context must be provided")
    return load_index_from_storage(storage_context=storage_context)


def create_query_engine(index):
    if index is None:
        raise ValueError("Index must be provided")

    from llama_index.core import Settings

    # Set global settings before using
    Settings.chunk_size = 3840
    Settings.chunk_overlap = 256
    Settings.llm = create_llm_model()
    Settings.embed_model = create_embedding_model()

    return index.as_query_engine(
        similarity_top_k=3,
        response_mode="tree_summarize",
        verbose=True,
    )
