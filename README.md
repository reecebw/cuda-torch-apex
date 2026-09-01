# cuda-torch-apex

A prebuilt, public base image carrying the slow, rarely-changing half of a
GPU training/inference stack:

| Component | Version |
|---|---|
| CUDA runtime | 12.8.1 (`nvidia/cuda:12.8.1-base-ubuntu24.04`) |
| Python | 3.12, in a venv at `/opt/foundry-env` |
| torch | `2.11.0+cu128` (from `https://download.pytorch.org/whl/cu128`) |
| NVIDIA apex | compiled `--cpp_ext --cuda_ext` for `TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"` |

Published to `ghcr.io/reecebw/cuda-torch-apex`, public, so unauthenticated
pulls work with no rate limit.

Also in the image:

- `/opt/torch-constraints.txt` — `torch==2.11.0`, to pass to downstream
  installs as `pip install -c /opt/torch-constraints.txt ...`.
- `/opt/apex-commit.txt` — the apex commit the wheel was built from.
- `ninja`, so a downstream `BuildExtension` does not fall back to serial
  distutils.
- `build-essential`, `python3.12-dev`, `libaio-dev` — kept because consumers
  still `pip install` packages that compile C extensions against this venv.

Not in the image: the CUDA **toolkit** (`nvcc`, ~10 GB installed) and apex's
build tree. Both exist only in the `devel` build stage.

The runtime stage sits on the CUDA **`base`** flavour (cudart plus the NVIDIA
container-runtime env vars, ~0.1 GB compressed) rather than `runtime`
(~2.2 GB): torch's cu128 wheels ship their own cuBLAS/cuDNN/cuFFT/NCCL, so the
`runtime` flavour is a second copy of libraries nothing loads. Build with
`--build-arg CUDA_RUNTIME_FLAVOR=runtime` if a consumer ever needs the system
copies.

## apex is ABI-bound to torch

apex's compiled extensions link against a specific libtorch. Installing a
different torch into this image silently breaks them at import time. The tag
therefore names the pin it is valid for (`cu128-torch2.11.0`), and downstream
installs must carry `-c /opt/torch-constraints.txt` so a transitive dependency
cannot pull torch out from under apex.

A new torch means a new build and a new tag, never a re-push of an existing one.

## Consume it by digest

Tags can be moved; digests cannot. Pin the digest so a rebuild of the base
cannot change what a downstream content-addressed image resolves to:

```dockerfile
FROM ghcr.io/reecebw/cuda-torch-apex@sha256:<digest>
RUN . /opt/foundry-env/bin/activate && \
    pip install -c /opt/torch-constraints.txt -e '/opt/yourpackage[all]'
```

Current build of `cu128-torch2.11.0` (2026-09-01):

```
ghcr.io/reecebw/cuda-torch-apex@sha256:3aee00adfba311c712696490f3e4765bd611d2ace5cc59d12df93b162c3574d4
```

Re-check the digest for the tag with:

```bash
docker buildx imagetools inspect ghcr.io/reecebw/cuda-torch-apex:cu128-torch2.11.0
```

## Rebuild

Actions → **build-and-push** → *Run workflow*. It also runs on any push to
`main` that touches the `Dockerfile`. Inputs: `tag` (the ABI pin) and
`max_jobs` (apex compile parallelism — nvcc needs ~2 GB per job, so keep it
under both the core count and RAM_GB/2 of the builder).

Locally, on a machine with more cores:

```bash
docker build --build-arg MAX_JOBS=16 \
  -t ghcr.io/reecebw/cuda-torch-apex:cu128-torch2.11.0 .
```

apex takes ~50 min at `MAX_JOBS=16` on 32 vCPUs and several hours on a
GitHub-hosted runner's 4. It is a once-per-ABI cost.

Build args: `CUDA_VERSION`, `UBUNTU_VERSION`, `PYTHON`, `TORCH_VERSION`,
`TORCH_INDEX_URL`, `TORCH_CUDA_ARCH_LIST`, `MAX_JOBS`, `APEX_REF`.
