#!/bin/bash
set -e
export LD_LIBRARY_PATH=/app/llama-bin:$LD_LIBRARY_PATH

echo "========================================"
echo "Starting llama-server..."
echo "========================================"

/app/llama-bin/llama-server \
    -m /var/data/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    --host 127.0.0.1 \
    --port 8080 \
    > /tmp/llama.log 2>&1 &

LLAMA_PID=$!

echo "Waiting for llama-server to load the model..."

for i in {1..120}; do
    if curl -s http://127.0.0.1:8080/health >/dev/null 2>&1; then
        echo "✓ llama-server is ready."
        break
    fi

    if ! kill -0 $LLAMA_PID 2>/dev/null; then
        echo "❌ llama-server exited unexpectedly."
        echo "----- llama-server log -----"
        cat /tmp/llama.log
        exit 1
    fi

    sleep 2
done

if ! curl -s http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "❌ Timed out waiting for llama-server."
    echo "----- llama-server log -----"
    cat /tmp/llama.log
    exit 1
fi

echo "========================================"
echo "llama-server is ready."
echo "Starting Gunicorn..."
echo "========================================"

exec gunicorn app:app \
    --bind 0.0.0.0:${PORT} \
    --workers 1 \
    --threads 4 \
    --worker-class gthread \
    --timeout 120
