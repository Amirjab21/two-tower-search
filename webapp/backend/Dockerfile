FROM python:3.9-slim

WORKDIR /app

RUN pip3 install --no-cache-dir --default-timeout=100 torch --index-url https://download.pytorch.org/whl/cpu && \
    python3 -c "import torch; print('PyTorch version:', torch.__version__)"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Development command with reload
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 