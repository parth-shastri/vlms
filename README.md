# Vision Language Models (vlms)

A code base for working and learning about Vision-Language Models (VLMs) using PyTorch and Hugging Face Transformers.

## Features

- Vision-language model implementations and utilities
- Built on PyTorch with CUDA support
- Integration with Hugging Face Transformers and Datasets libraries
- Support for modern vision-language architectures

## Installation

This project uses Poetry for dependency management. To install:

```sh
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone <repository-url>
cd vlms

# Install dependencies
poetry install
```

## Requirements

Python 3.12+
CUDA-capable GPU (CUDA 12.1)
PyTorch 2.5.1+
transformers 4.47.1+
datasets 3.2.0+
tokenizers 0.21.0+

## Project Structure

```sh
paligemma/
├── __init__.py
├── cache_utils.py
├── inference_utils.py
├── inference.py
├── model_utils.py
├── modelling_gemma.py
├── modelling_paligemma.py
├── modelling_siglip.py
└── processing_paligemma.py
```



