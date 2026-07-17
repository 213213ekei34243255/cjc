FROM python:3.11-slim

WORKDIR /app

# Install build tools
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

# Copy project
COPY . .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x start.sh

EXPOSE 10000

CMD ["./start.sh"]
