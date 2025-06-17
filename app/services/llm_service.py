import torch
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def load_storage_context():
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    return storage_context


def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|start_header_id|>system<|end_header_id|>{message.content}<|eot_id|>\n"
            # prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{message.content}<|eot_id|>\n"
        elif message.role == "user":
            prompt += (
                f"<|start_header_id|>user<|end_header_id|>{message.content}<|eot_id|>\n"
            )
        elif message.role == "assistant":
            prompt += f"<|start_header_id|>assistant<|end_header_id|>{message.content}<|eot_id|>\n"

    prompt = prompt + "<|start_header_id|>assistant<|end_header_id|><|eot_id|>\n"
    return prompt


def completion_to_prompt(completion):
    return f"<|start_header_id|>system<|end_header_id|><|eot_id|>\n<|start_header_id|>user<|end_header_id|>{completion}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"


def create_embedding_model():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    return embed_model


def create_llm_model():
    model_url = "https://huggingface.co/scb10x/typhoon2.1-gemma3-4b-gguf/resolve/main/typhoon2.1-gemma3-4b-q4_k_m.gguf"
    llm = LlamaCPP(
        model_url=model_url,
        model_path=None,
        temperature=0.1,
        context_window=8192,
        max_new_tokens=1024,  # อาจจะตอบเพิ่มออกมาอีก 1024 bytes
        generate_kwargs={},
        model_kwargs={
            "repetition-penalty": 1.4,
            "no_repeat_ngram_size": 4,
            # "response_format": { "type": "json_object" },
            "n_gpu_layers": -1,
        },
        # model_kwargs={"n_gpu_layers": 33},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )

    return llm


def load_index(storage_context=None):
    if storage_context is None:
        raise ValueError("Storage context must be provided")

    loaded_index = load_index_from_storage(storage_context=storage_context)

    return loaded_index


def create_query_engine(index):
    if index is None:
        raise ValueError("Index must be provided")

    from llama_index.core import Settings, VectorStoreIndex

    Settings.chunk_size = 3840
    Settings.chunk_overlap = 256

    llm = create_llm_model()
    embed_model = create_embedding_model()

    Settings.llm = llm
    Settings.embed_model = embed_model
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        response_mode="tree_summarize",
        verbose=True,
    )

    return query_engine
