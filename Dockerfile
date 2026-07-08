FROM python:3.11-slim

LABEL maintainer="InferMesh" \
      description="Distributed LLM Inference Orchestrator" \
      version="1.0.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir hatchling && \
    pip install --no-cache-dir \
        fastapi uvicorn[standard] httpx \
        pydantic pydantic-settings PyYAML python-dotenv \
        anyio aiofiles \
        prometheus-client \
        structlog rich \
        numpy pandas scipy \
        grpcio grpcio-tools protobuf \
        redis \
        sortedcontainers tenacity \
        typer click tabulate \
        matplotlib seaborn

# Copy application code
COPY app/ ./app/
COPY benchmarks/ ./benchmarks/
COPY proto/ ./proto/
COPY configs/ ./configs/

# Compile gRPC stubs
RUN python -m grpc_tools.protoc \
    -I proto \
    --python_out=app/api/ \
    --grpc_python_out=app/api/ \
    --pyi_out=app/api/ \
    proto/inference.proto 2>/dev/null || true

# Create results directory
RUN mkdir -p results logs

# Non-root user for security
RUN useradd -m -u 1000 infermesh && chown -R infermesh:infermesh /app
USER infermesh

EXPOSE 8000 50051

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "asyncio"]
