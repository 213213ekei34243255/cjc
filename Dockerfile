FROM python:3.11-slim

WORKDIR /app

# Only install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (including llama-bin/)
COPY . .

# Make executables runnable
RUN chmod +x start.sh
RUN chmod +x llama-bin/llama-server

EXPOSE 10000

CMD ["./start.sh"]
