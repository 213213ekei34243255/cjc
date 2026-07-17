FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Build llama.cpp
RUN git clone https://github.com/ggml-org/llama.cpp.git && \
    cd llama.cpp && \
    cmake -B build && \
    cmake --build build -j$(nproc)

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x start.sh

EXPOSE 10000

CMD ["./start.sh"]
