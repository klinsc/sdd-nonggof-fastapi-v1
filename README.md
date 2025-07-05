conda create -n llama-gpu python=3.11 -y
conda activate llama-gpu
conda install -c conda-forge cmake ninja git -y
pip install numpy
conda install -c nvidia cuda-toolkit -y
conda install -c nvidia cudnn -y

git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python
set LLAMA_CUBLAS=1
pip install . --verbose --force-reinstall --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu129

pip install -U "triton-windows<3.4"

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install "fastapi[standard]" uvicorn python-dotenv openai typhoon-ocr pypdf langchain-core langchain sentence-transformers torch huggingface_hub transformers bitsandbytes accelerate langchain-community chromadb