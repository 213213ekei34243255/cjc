FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Make binaries executable
RUN chmod +x start.sh && \
    chmod +x /app/llama-bin/*

# Create shared library symlinks if they don't already exist
RUN cd /app/llama-bin && \
    [ -e libllama-common.so.0 ] || ln -s libllama-common.so.0.0.10056 libllama-common.so.0 && \
    [ -e libllama.so.0 ] || ln -s libllama.so.0.0.10056 libllama.so.0 && \
    [ -e libggml.so.0 ] || ln -s libggml.so.0.16.0 libggml.so.0 && \
    [ -e libggml-base.so.0 ] || ln -s libggml-base.so.0.16.0 libggml-base.so.0

# Tell Linux where the shared libraries are
ENV LD_LIBRARY_PATH=/app/llama-bin

EXPOSE 10000

CMD ["./start.sh"]
