# Two Tower Search

This project contains the code used to train and serve a two‑tower retrieval model.  
The model embeds questions and answers separately and uses cosine similarity to match
user queries with relevant answers.  The repository also provides a small web
application for interactive search.

## Repository structure

```
.
├── build_document_embeddings.py  # create FAISS index of answer embeddings
├── main.py                       # training script (CPU)
├── main_gpu.py                   # training script with GPU support
├── models.py                     # query tower, answer tower and dataset classes
├── webapp/                       # backend (FastAPI) and frontend (React)
├── checkpoints/                  # saved model weights
├── requirements.txt              # Python dependencies
└── ...
```

## Model architecture

Both the query and answer towers are implemented as GRU encoders.  Each tower
receives a sequence of Word2Vec embeddings and projects the final hidden state
through a small feed‑forward layer to create fixed‑length representations.
During training we maximise the similarity between matching query/answer pairs
while pushing apart mismatched pairs using a margin‑based loss.

## Training

1. Download the following data files and place them under `data/`:
   - `GoogleNews-vectors-negative300.bin` &ndash; pre‑trained Word2Vec weights
   - `qa_formatted.parquet` &ndash; training set of query/answer pairs

2. Install the dependencies (or build the Docker image provided):
   ```bash
   docker build -t two-tower-model .
   ```

3. Start training:
   ```bash
   docker run --gpus all -v $(pwd)/data:/app/data two-tower-model
   ```
   If you do not have a GPU remove the `--gpus all` flag.

The scripts `main.py` and `main_gpu.py` can also be run directly if the
requirements are installed locally.  Training parameters such as learning rate
and batch size can be modified via environment variables when using Docker.

## Building document embeddings

After training, run `build_document_embeddings.py` to compute answer embeddings
and store them in a FAISS index.  These embeddings are used for fast retrieval
during inference.

## Web application

The `webapp` directory contains a FastAPI backend and a React frontend.  Use
`docker-compose up --build` from within `webapp/` to start the development
server.  The application exposes a simple interface for querying the trained
model and inspecting the retrieved answers.
