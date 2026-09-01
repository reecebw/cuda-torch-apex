# syntax=docker/dockerfile:1
#
# CUDA + Python + torch(cu128) + NVIDIA apex, as a runtime image.
#
# apex is compiled in the `devel` stage against the exact torch below and
# emitted as a wheel; the `runtime` stage installs that wheel on top of a CUDA
# *runtime* base, so the ~10 GB CUDA toolkit and the apex build tree never
# reach the published image.

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

# ── devel: compile apex, emit a wheel ───────────────────────────────────

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS devel

ARG PYTHON=python3.12
ARG TORCH_VERSION=2.11.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
# nvcc needs ~2 GB per job; keep this under (RAM_GB / 2) and the core count of
# whatever builds this. 16 suits a 32-vCPU builder, 4 a GitHub-hosted runner.
ARG MAX_JOBS=4
ARG APEX_REF=master
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates \
        ${PYTHON} ${PYTHON}-dev ${PYTHON}-venv python3-pip libaio-dev \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/foundry-env \
    PATH=/opt/foundry-env/bin:/usr/local/cuda/bin:$PATH
# Without ninja, torch's BuildExtension falls back to serial distutils.
RUN ${PYTHON} -m venv ${VIRTUAL_ENV} \
    && pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir ninja

RUN pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch==${TORCH_VERSION}+cu128

RUN git clone https://github.com/NVIDIA/apex.git /opt/apex \
    && cd /opt/apex && git checkout ${APEX_REF} \
    && git rev-parse HEAD > /opt/apex-commit.txt

RUN cd /opt/apex && \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS=${MAX_JOBS} \
    pip wheel -v --no-cache-dir --no-build-isolation --no-deps \
        --config-settings "--build-option=--cpp_ext" \
        --config-settings "--build-option=--cuda_ext" \
        -w /wheels .

# ── runtime: torch + the apex wheel, no toolkit ─────────────────────────

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG PYTHON=python3.12
ARG TORCH_VERSION=2.11.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ENV DEBIAN_FRONTEND=noninteractive

# build-essential/${PYTHON}-dev/libaio-dev stay: consumers of this image still
# `pip install` packages that compile C extensions against this venv. The CUDA
# toolkit does not stay — that is the ~10 GB this image exists to drop.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget curl ca-certificates gnupg \
        ${PYTHON} ${PYTHON}-dev ${PYTHON}-venv python3-pip libaio-dev \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/foundry-env \
    PATH=/opt/foundry-env/bin:/usr/local/cuda/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
RUN ${PYTHON} -m venv ${VIRTUAL_ENV} \
    && pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir ninja

RUN pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch==${TORCH_VERSION}+cu128

COPY --from=devel /wheels /tmp/wheels
COPY --from=devel /opt/apex-commit.txt /opt/apex-commit.txt
RUN pip install --no-cache-dir /tmp/wheels/apex-*.whl && rm -rf /tmp/wheels

# Constraint file for consumers: keeps a downstream `pip install` from
# resolving a different torch and breaking apex's ABI binding.
RUN echo "torch==${TORCH_VERSION}" > /opt/torch-constraints.txt

# Loading apex's compiled extensions needs no GPU, so this ABI check runs here.
RUN python -c "import torch, apex, amp_C, fused_layer_norm_cuda; print(torch.__version__)"

LABEL org.opencontainers.image.source=https://github.com/reecebw/cuda-torch-apex \
      org.opencontainers.image.description="CUDA 12.8 + Python 3.12 + torch cu128 + apex (prebuilt)" \
      torch=${TORCH_VERSION} \
      cuda=${CUDA_VERSION}

CMD ["/bin/bash"]
