# Optional provider-neutral CUDA container. Docker build was not run on a GPU.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir -c requirements/pilot-constraints.txt -e '.[gpu,tracking,kaggle]'
ENV RSNA_STATE_DIR=/state MPLBACKEND=Agg HF_HUB_DISABLE_TELEMETRY=1
CMD ["rsna-knee", "doctor"]

