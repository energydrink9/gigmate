![example workflow](https://github.com/energydrink9/gigmate/actions/workflows/python-app.yml/badge.svg)

# GigMate

GigMate is an AI-driven music completion and accompaniment system that leverages deep learning to generate missing musical segments or provide real-time musical collaboration. It is designed to assist musicians in practice, composition, and live performance by generating high-quality audio based on existing tracks.

## 🚀 Project Overview

GigMate is a reimplementation and adaptation of **MusicGen by Meta**, using a similar approach based on **interleaving multiple codebooks** for audio representation. However, it is a much simpler version, built from scratch as a personal challenge by **Michele Lugano** while studying machine learning and transformer architectures. The project is an ambitious experiment, possibly a stretch beyond reach, but an exciting mission to tackle.

### Key Features:
- **Music Continuation**: Predicts and generates musical sequences based on a given track.
- **Live Accompaniment (Future Goal)**: Aims to provide real-time responses to live input, making it a virtual bandmate.
- **Multi-Instrument Support**: Handles multiple instruments in a single composition, ensuring synchronization.
- **Custom AI Models**: Utilizes a Transformer-based architecture with Cached Multihead Attention for efficient sequence generation.
- **Streaming and Offline Processing**: Designed to support both pre-recorded and live-streamed audio in the future.
- **Latency Optimization**: Implements caching and optimized inference techniques to minimize response time.

## 🔬 Research & Innovation

GigMate builds upon several research areas in deep learning and music generation, integrating state-of-the-art methodologies:

- **Transformer-based Sequence Modeling**: Inspired by works like OpenAI's Jukebox and Google's Music Transformer, GigMate employs a Transformer model to generate structured musical sequences.
- **KV Caching for Efficient Attention**: Implements a custom Cached Multihead Attention mechanism to reduce computational overhead and enable real-time inference.
- **Audio Tokenization & Embedding**: Uses EnCodec for audio tokenization, ensuring compact and meaningful representations of musical sequences.
- **Variational Sampling & Temperature Control**: Provides flexible sampling strategies for creative control over musical outputs.

## 🏗️ Project Status

**⚠️ Current Limitations:**
- The model is **not yet fully functional**. While it can generate some audio, training has not yet reached a good enough solution for music continuation.
- **Performance is still insufficient** for live music accompaniment, though that remains a key long-term goal.
- The project is in its early stages and **would benefit from contributors** to improve training, inference speed, and quality of generated music.

Despite these challenges, the intention is to **explore and push the boundaries of AI-generated music** rather than produce a perfect working system immediately.

## 🏗️ Project Structure

The codebase is structured into several modules:

```
./src/gigmate/
│── api/            # API endpoints for inference and serving
│── dataset/        # Dataset handling and preprocessing
│── domain/         # Core logic for prediction and completion
│── model/          # Transformer-based model architecture
│── scripts/        # Utility scripts for audio synchronization and latency measurement
│── training/       # Training pipeline and optimization techniques
│── utils/          # Utility functions for audio processing, device management, and more
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- PyTorch
- Torchaudio
- FastAPI
- EnCodec

### Installation

Clone the repository and install dependencies:

```sh
git clone https://github.com/energydrink9/gigmate.git
cd gigmate
pip install poetry
poetry install
poetry install -E dataset
```

#### Environment Variables

If you want to run training or dataset generation (for inference it's not needed), make sure the following environment variables are set before running the application (replace the <> tokens with the actual secrets):

```sh
export CLEARML_API_ACCESS_KEY=<YOUR_CLEARML_API_ACCESS_KEY_HERE>
export CLEARML_API_SECRET_KEY=<YOUR_CLEARML_API_SECRET_KEY_HERE>
```

Also this one in case you are running on Apple Silicon:
```sh
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Running the API
To start the API server for audio completion:

```sh
poetry run python -m gigmate.api.complete_audio # To start the API
```


### Inference
```sh
poetry run python -m gigmate.play # To start the client
```

### Dataset generation
Please refer to the following repository for dataset generation information: https://github.com/energydrink9/stem_continuation_dataset_generator


## 📖 Usage

### Completing an Audio File
You can use the API to generate missing musical segments by sending an audio file:

```python
import requests

url = "http://localhost:8000/predict"
files = {"request": open("input_audio.ogg", "rb")}
response = requests.post(url, files=files)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### Streaming Audio Completion
GigMate supports WebSocket-based streaming for real-time accompaniment (in future iterations).

## 🏋️‍♂️ Training Your Own Model

In order to start a training run, make sure to have a dataset available and then install the [latest nightly version](https://pytorch.org/get-started/locally/) of PyTorch. On Mac you can use the following command:
```
pip3 install --pre torch torchvision torchaudio torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
```

After that, run the following command to start training:

```
python -m gigmate.train
```

## 📜 References
GigMate is inspired by and builds upon several key papers and frameworks:
- **MusicGen (Meta, 2023)** - High-Fidelity Music Generation with Transformers
- **Music Transformer**: Huang et al. (2018) - Generating Music with Long-Term Structure
- **Jukebox**: OpenAI (2020) - A Generative Model for Music
- **EnCodec**: Défossez et al. (2022) - High-Fidelity Neural Audio Codec
- **Efficient Transformers**: Vaswani et al. (2017) - Attention Is All You Need

## 📌 Future Improvements
- Improve music continuation quality through better training techniques
- Support for additional musical styles and genres
- Enhanced real-time streaming performance
- Integration with MIDI for structured music generation
- Mobile and desktop applications for enhanced accessibility

## 🤝 Contributing
GigMate is an ambitious but early-stage project, and contributions are highly welcome! If you'd like to improve GigMate, please follow these steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a Pull Request

## 📜 License
GigMate is licensed under the MIT License. See `LICENSE` for more details.

## 📬 Contact
For questions or collaboration inquiries, feel free to reach out:
- GitHub: [energydrink9](https://github.com/energydrink9)
- Email: michele.lugano9@gmail.com
