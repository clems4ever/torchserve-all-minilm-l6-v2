# Running all-MiniLM-L6-v2 with TorchServe 

This repository contains everything needed to deploy a production-ready service for
computing sentence similarity embeddings using the model
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and TorchServe.
Those embeddings can then be used in combination with a vector database like Pinecone, Milvus,
Weaviate or Qdrant.

This repository has been created because there was no simple example in the torchserve repository
for deploying a huggingface model for sentence similarity. The closest I could find was resources
for sequence classification, generation, question answering, and token classification as you can
check [here](https://github.com/pytorch/serve/tree/master/examples/Huggingface_Transformers).


## Get Started

There are two ways to test the server, by running it as a process or as a docker container.

### Deploy with docker

First, make sure you have docker installed.

```make
docker run -p 8080:8080 -it ghcr.io/clems4ever/torchserve-all-minilm-l6-v2:latest
```

Then, go to [Usage](#usage) to check how to use the service.

Note that the command can be further optimized for production by using settings documented in the 
[TorchServe documentation](https://github.com/pytorch/serve/blob/master/docker/README.md#running-torchserve-in-a-production-docker-environment)

Also note that by using this docker image and even if you have a GPU available, the inferences will be done on CPU with
the basic capabilities, i.e., without leveraging any CPU extension. If you want to leverage your GPU or some CPU
capabilities, you need to adapt the Dockerfile according to the documentation of
[TorchServe](https://github.com/pytorch/serve/blob/master/docker/README.md).

### Deploy as a process

First, make sure you have Python 3 and Java 11+ installed.

```bash
make serve
```

Then, go to [Usage](#usage) to check how to use the service. If you have a GPU available, it will use it for faster
inferences.

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
    0.019096793606877327,
    0.03446517512202263,
    0.09162796288728714,
    0.0701652243733406,
    -0.029946573078632355,
    ...
  ],
  [
    -0.06470940262079239,
    -0.03830110654234886,
    0.013061972334980965,
    -0.0003482792235445231,
    ...
  ]
]
```

## Aknowledgments

Many thanks to:

- [Stane Aurelius](https://supertype.ai/author/saurelius/) who wrote
[a great post](https://supertype.ai/notes/serving-pytorch-w-torchserve/) about the details on how to deploy a model with
tochserve. I highly recommend reading it.

- The community publishing models on [HuggingFace](https://huggingface.co/) and particularly to the team who
have produced the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model and have shared it.

- The Microsoft Research team who produced the paper about [MiniLM](https://arxiv.org/abs/2002.10957).

- The [HuggingFace](https://huggingface.co/) team who hosts the models and make them easily available to everyone.

## License

The code in this repository is licensed under the MIT license.
