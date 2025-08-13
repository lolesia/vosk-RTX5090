# Run Vosk on RTX 5090: Docker Compose with SM_90 + CUDA 12.9 support

Docker Compose for building and running the Vosk speech recognition service with support for the RTX 5090 GPU (SM_90 architecture, CUDA 12.9).
The project includes Kaldi / Vosk / Vosk-server rebuilt from source with fixed dependencies and makefiles.
JIT compilation support (compute tag) has been implemented, allowing Vosk to run on modern GPUs without official support.
It also includes a fixed asr_server_gpu.py file, which can optionally be passed to the container.
