FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed by chromadb or others)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your full backend code
COPY . .

# Ensure mount point exists for chroma_db
RUN mkdir -p /app/chroma_db

# Expose port for uvicorn
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
