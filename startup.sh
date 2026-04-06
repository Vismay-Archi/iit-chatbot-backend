#!/bin/bash
echo "Downloading sentence transformer model..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"
echo "Model ready. Starting uvicorn..."
uvicorn app:app --host 0.0.0.0 --port 8000
 
