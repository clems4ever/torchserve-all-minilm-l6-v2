# TorchServe Embeddings

This repository contains everything needed to deploy a production-ready service for
computing sentence embeddings for similarity computation using the model
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

This model is used for computing sentence similarities. It can be used in combination with a vector database
like pinecone, milvus, weaviate or qdrant.

This repository has been created because there was no example in the torchserve repository for deploying
a huggingface model for sentence similarity, there are only resources for sequence classification, generation, question answering,
and token classification as you can check [here](https://github.com/pytorch/serve/tree/master/examples/Huggingface_Transformers).

## Get Started

There are two ways to test the server, by running it as a process or as a docker container.

### Deploy with docker

First, make sure you have docker installed.

```make
make build-docker
docker run -p 8080:8080 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it torchserve-all-minilm-l6-v2
```

Then, go to [Usage](#usage) to check how to use the service.

### Deploy as a process

First, make sure you have Python 3 and Java 11+ installed.

```bash
make serve
```

Then, go to [Usage](#usage) to check how to use the service.

### Usage

This should start a server localy that you can query with a curl like the following:

```bash
curl --location 'http://127.0.0.1:8080/predictions/my_model' \
--header 'Content-Type: application/json' \
--data '{
    "input": ["hello", "hi"]
}'
```

You should get an output similar to

```bash
[
  [
    0.016306497156620026,
    0.10300007462501526,
    -0.17589513957500458,
    -0.010497087612748146,
    -0.06088363379240036,
    0.00311004975810647,
    0.07295241206884384,
    ...
  ]
]
```

## Aknowledgments

Many thanks to [Stane Aurelius](https://supertype.ai/author/saurelius/) who wrote
[a great post](https://supertype.ai/notes/serving-pytorch-w-torchserve/) about the details on how to deploy a model with
tochserve. I highly recommend reading it.

Many thanks to the community publishing models on [HuggingFace](https://huggingface.co/) and particularly to the team who
have produced the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model and have shared it.

## License

The code in this repository is licensed under the MIT license.
