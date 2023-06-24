#!/bin/bash

# --ncs means the snapshot feature is disabled.

echo "server is starting..."
/home/venv/bin/torchserve --foreground --model-store model_store --models my_model=all-MiniLM-L6-v2.mar --ncs