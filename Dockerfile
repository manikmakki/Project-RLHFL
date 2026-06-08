FROM intel/intel-extension-for-pytorch:2.8.10-xpu-pip-base

# intel/intel-extension-for-pytorch:*-xpu-pip-base provides:
#   - Ubuntu base with Intel GPU compute runtime (Level Zero + OpenCL)
#   - PyTorch built against Intel XPU
#   - intel_extension_for_pytorch
# The kernel xe/i915 driver is supplied by the host; the container only needs
# the user-space runtime, which this image already contains.

WORKDIR /app

# Install project dependencies.
# torch and intel_extension_for_pytorch are already installed in the base image;
# pip will skip them if the version pin in requirements.txt is satisfied, or
# upgrade in-place if a newer pin is specified — keep the pin loose (>=) to avoid
# forcing a CPU-wheel downgrade that would break XPU support.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY services/ ./
COPY scripts/ ./scripts/
COPY start.sh ./

RUN chmod +x start.sh

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["./start.sh"]
