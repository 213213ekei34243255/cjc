FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Clone and build llama.cpp (low memory)
RUN git clone --depth=1 https://github.com/ggml-org/llama.cpp.git && \
    cd llama.cpp && \
    cmake -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j1

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Make startup script executable
RUN chmod +x start.sh

EXPOSE 10000

CMD ["./start.sh"]
