FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create missing compatibility symlinks
RUN cd /app/llama-bin && \
    ln -sf libllama-common.so.0.0.10056 libllama-common.so.0 && \
    ln -sf libllama.so.0.0.10056 libllama.so.0 && \
    ln -sf libggml.so.0.16.0 libggml.so.0 && \
    ln -sf libggml-base.so.0.16.0 libggml-base.so.0

ENV LD_LIBRARY_PATH=/app/llama-bin

RUN chmod +x start.sh
RUN chmod +x /app/llama-bin/*

EXPOSE 10000

CMD ["./start.sh"]
