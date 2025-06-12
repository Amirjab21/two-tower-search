# Two Tower Model Docker Setup

This repository contains a Docker setup for running the Two Tower Model training script.

## Prerequisites

1. Docker installed on your system
2. The following data files:
   - `GoogleNews-vectors-negative300.bin`
   - `qa_formatted.parquet`

## Setup Instructions

1. Place the required data files in the `data/` directory:
```bash
data/
├── GoogleNews-vectors-negative300.bin
└── qa_formatted.parquet
```

2. Build the Docker image:
```bash
docker build -t two-tower-model .
```

3. Run the container:
```bash
docker run --gpus all -v $(pwd)/data:/app/data two-tower-model
```

Note: If you don't have a GPU, remove the `--gpus all` flag:
```bash
docker run -v $(pwd)/data:/app/data two-tower-model
```

## Data Files

- `GoogleNews-vectors-negative300.bin`: Pre-trained Word2Vec embeddings from Google News
- `qa_formatted.parquet`: Question-Answer dataset in Parquet format

## Environment Variables

You can customize the training by setting the following environment variables when running the container:

```bash
docker run -v $(pwd)/data:/app/data \
  -e LEARNING_RATE=0.01 \
  -e NUM_EPOCHS=3 \
  -e BATCH_SIZE=512 \
  two-tower-model
``` 