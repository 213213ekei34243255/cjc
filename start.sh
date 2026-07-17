#!/bin/bash
set -e

echo "Starting llama-server..."

/app/llama.cpp/build/bin/llama-server \
    -m /var/data/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    --host 127.0.0.1 \
    --port 8080 &

echo "Waiting for model..."
sleep 15

echo "Starting Flask..."

exec gunicorn app:app \
    -k gthread \
    --threads 4 \
    -w 1 \
    --timeout 120 \
    --bind 0.0.0.0:${PORT}
