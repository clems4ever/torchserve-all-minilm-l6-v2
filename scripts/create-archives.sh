#!/bin/bash

rm -rf model_store
mkdir -p model_store

torch-model-archiver \
--model-name all-MiniLM-L6-v2 \
--version 1.0 \
--model-file embedder_model_files/pytorch_model.bin \
--handler handlers/embedding/handler.py  \
--extra-files "embedder_model_files/" \
--export-path model_store

torch-model-archiver \
--model-name ms-marco-TinyBERT-L-2-v2 \
--version 1.0 \
--model-file cross_encoder_model_files/pytorch_model.bin \
--handler handlers/cross_encoder/handler.py  \
--extra-files "cross_encoder_model_files/" \
--export-path model_store

echo "Archive created!"