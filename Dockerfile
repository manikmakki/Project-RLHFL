FROM ubuntu:26.04

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev curl build-essential gnupg \
    && rm -rf /var/lib/apt/lists/*

# Intel GPU user-space runtime — Ubuntu 26.04 ships Level Zero 1.28.2 and
# compute-runtime 26.05.37020.3 (first release with Battlemage/Xe2 support)
# in its main/universe repositories.  No external PPA needed.
# Package names changed in 26.04: intel-level-zero-gpu → libze-intel-gpu1.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libze1 \
        libze-intel-gpu1 \
        intel-opencl-icd \
        libigdgmm12 \
    && rm -rf /var/lib/apt/lists/*

# Make 'python' and 'pip' resolve to python3/pip3
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install PyTorch with native Intel XPU support.
# The XPU index at download.pytorch.org/whl/xpu contains BOTH +xpu and +cu128 wheels
# for the same base version.  Without an explicit local-version pin pip non-deterministically
# picks +cu128 (CUDA build, XPU backend inert), so we pin the +xpu suffix directly.
# torch-2.12.0+xpu-cp314-cp314-linux_x86_64.whl is the current stable XPU build for Python 3.14.
RUN pip install --no-cache-dir --break-system-packages \
    "torch==2.12.0+xpu" \
    --index-url https://download.pytorch.org/whl/xpu

# BitsAndBytes multi-backend wheel — XPU support ships since 0.45.0.
# The standard PyPI wheel is CUDA-only; the XPU-enabled build is on the torch index.
RUN pip install --no-cache-dir --break-system-packages \
    "bitsandbytes>=0.45.5" \
    --extra-index-url https://download.pytorch.org/whl/xpu

# Triton for Intel XPU — Unsloth uses it for chunked-CE, RoPE, and layernorm kernels.
RUN pip install --no-cache-dir --break-system-packages \
    triton \
    --extra-index-url https://download.pytorch.org/whl/xpu

# Unsloth — must come after torch+bnb+triton.
# --no-deps skips its torch pin (requires <2.11) which conflicts with our XPU wheel.
RUN pip install --no-cache-dir --break-system-packages \
    "unsloth>=2026.5.4" \
    --no-deps

# Patch unsloth_zoo to recognise Intel XPU as a valid accelerator.
# upstream tracking: unslothai/unsloth PR #6106 (opened 2026-06-08, not yet merged).
# get_device_type() currently raises NotImplementedError for torch.xpu devices; this
# inserts the XPU check immediately before that raise so the import succeeds.
# The patch is idempotent: if the string is already absent (PR merged) it no-ops.
RUN python3 -c "\
import pathlib; \
needle = 'raise NotImplementedError(\"Unsloth cannot find any torch accelerator? You need a GPU.\")'; \
patch = 'if hasattr(torch, \"xpu\") and torch.xpu.is_available():\n        return \"xpu\"\n    ' + needle; \
hits = [p for p in pathlib.Path('/usr/local/lib').rglob('unsloth_zoo/device_type.py') if needle in p.read_text()]; \
[p.write_text(p.read_text().replace(needle, patch)) or print('Patched', p) for p in hits] or print('Patch not needed (already applied or PR merged)')"

# Install remaining dependencies (torch already satisfies the >=2.6.0 floor)
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy project source
COPY services/ ./
COPY scripts/ ./scripts/
COPY start.sh ./

RUN chmod +x start.sh

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["./start.sh"]
