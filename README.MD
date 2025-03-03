# CLIP-Based Image Finder

A forward-thinking, easy-to-use tool that uses [OpenAI's CLIP model](https://github.com/openai/CLIP) and [FAISS](https://github.com/facebookresearch/faiss) to help you:
1. **Embed** images in a folder,
2. Build a **FAISS** index for similarity search, and
3. Use a **Streamlit** app to search images either by **text query** or by **image upload**.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Building the Database](#1-building-the-database)
  - [2. Running the Streamlit App](#2-running-the-streamlit-app)
- [Docker Usage](#docker-usage)
- [Folder Structure](#folder-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Fully Automated**: Provide the folder path; the system scans for images, extracts CLIP embeddings, and builds an index if not already present.
- **Multi-Modal**: Supports both text-based and image-based querying. 
- **Fast Similarity Search**: Uses **FAISS** for real-time nearest-neighbor queries on CPU or GPU.
- **User-Friendly Interface**: Built with **Streamlit**, so you can run everything locally in your browser.

## Prerequisites

- Python 3.8+ (recommended)
- [PyTorch](https://pytorch.org/) (CPU or GPU build, depending on your hardware)
- CLIP (installed automatically through `pip install git+https://github.com/openai/CLIP.git`)
- FAISS ([faiss-cpu](https://pypi.org/project/faiss-cpu/) for CPU, [faiss-gpu](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) for GPU)
- [Streamlit](https://streamlit.io/)

Make sure you have a compatible C++ compiler if you’re installing faiss-cpu from source. On most systems, using `pip` will suffice.

## Installation

1. **Clone this repository**:

   ```bash
   git clone https://github.com/username/image-finder.git
   cd image-finder
