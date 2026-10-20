FROM python:3.12-slim

WORKDIR /app

# System dependencies for unstructured document parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    libreoffice-core \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY core/ core/
COPY mcp_server/ mcp_server/
COPY scripts/ scripts/
COPY agents/ agents/
COPY plugins/ plugins/
COPY clients/example/ clients/example/

# Create directory for user clients
RUN mkdir -p clients

EXPOSE 8080

# Default: run MCP server (stdio mode)
CMD ["python", "scripts/mcp_server.py"]
