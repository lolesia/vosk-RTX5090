FROM nvidia/cuda:12.9.0-devel-ubuntu22.04 

ARG KALDI_MKL
ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Etc/UTC

RUN  apt-get update && \
     apt-get install -y --no-install-recommends \
        wget \
        bzip2 \
        unzip \
        xz-utils \
        g++ \
        make \
        cmake \
        git \
        python3 \
        python3-dev \
        python3-pip \
        zlib1g-dev \
        automake \
        autoconf \
        libtool \
        pkg-config \
        ca-certificates \
        gfortran \
        libgfortran5 \
        libgfortran-9-dev \
    && rm -rf /var/lib/apt/lists/*


RUN \
    git clone -b vosk --single-branch https://github.com/alphacep/kaldi /opt/kaldi \
    && cd /opt/kaldi/tools \
    && sed -i 's:status=0:exit 0:g' extras/check_dependencies.sh \
    && sed -i 's:--enable-ngram-fsts:--enable-ngram-fsts --disable-bin:g' Makefile \
    # Убираем лишний патч OpenFst, он не нужен для nvToolsExt
    # && sed -i 's/LIBS="-ldl"/LIBS="-ldl -L\/usr\/local\/cuda\/lib64 -lnvToolsExt"/' Makefile \
    && make -j $(nproc) openfst cub \
    && if [ "x$KALDI_MKL" != "x1" ] ; then \
          extras/install_openblas_clapack.sh; \
        else \
          extras/install_mkl.sh; \
        fi \
    \
    && cd /opt/kaldi/src \
    && if [ "x$KALDI_MKL" != "x1" ] ; then \
          ./configure --mathlib=OPENBLAS_CLAPACK --shared --use-cuda=yes --cudatk-dir=/usr/local/cuda --cuda-arch="-gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90"; \
        else \
          ./configure --mathlib=MKL --shared --use-cuda=yes --cudatk-dir=/usr/local/cuda --cuda-arch="-gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90"; \
        fi \
    && sed -i 's:-msse -msse2:-msse -msse2:g' kaldi.mk \
    && sed -i 's: -O1 : -O3 :g' kaldi.mk \
    && export CPATH=/usr/local/cuda/include/nvtx3:$CPATH \
    && sed -i 's/#include <nvToolsExt.h>/\/\/#include <nvToolsExt.h>/' cudamatrix/cu-common.h \
    && sed -i 's/-lnvToolsExt//g' kaldi.mk \
    && sed -i 's/-L\\\\\/usr\\\\\/local\\\\\/cuda\\\\\/lib64 -lnvToolsExt//g' kaldi.mk \
    && sed -i '1s/^/#include <cfloat>\n/' cudadecoder/cuda-decoder-kernels.cu \
    && make clean depend -j$(nproc) && make -j $(nproc) online2 lm rnnlm cudafeat cudadecoder \
    \
    && pip3 install --upgrade websockets cffi \
    \
    && git clone https://github.com/alphacep/vosk-api /opt/vosk-api \
    && cd /opt/vosk-api/src \
    && sed -i 's/-lnvToolsExt//g' Makefile \
    && HAVE_CUDA=1 HAVE_MKL=$KALDI_MKL KALDI_ROOT=/opt/kaldi KALDI_CUDA_ARCH=90 make -j $(nproc) \
    && cd /opt/vosk-api/python \
    && python3 ./setup.py install \
    \
    && git clone https://github.com/alphacep/vosk-server /opt/vosk-server \
    \
    && rm -rf /opt/vosk-api/src/*.o \
    && rm -rf /opt/kaldi \
    && rm -rf /root/.cache \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/vosk-server
COPY asr_server_gpu.py .

# Открываем порт для WebSocket-сервера.
EXPOSE 2700
# Команда, которая будет выполняться при запуске контейнера.
CMD [ "python3", "./asr_server_gpu.py", "/opt/vosk-server/model"]
