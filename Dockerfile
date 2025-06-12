FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code
COPY . .

# Create data directory
RUN mkdir -p data

# Note: The following files need to be manually added to the data directory:
# - GoogleNews-vectors-negative300.bin
# - qa_formatted.parquet

CMD ["python", "main.py"] 