FROM comfyhidream-comfyui

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    HIDREAM_HOST=0.0.0.0 \
    HIDREAM_PORT=7861 \
    HIDREAM_MODEL_TYPE=dev \
    HIDREAM_MODEL_PATH=/models/HiDream-O1-Image-Dev

WORKDIR /workspace/HiDream-O1-Image
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-docker.txt

COPY . /workspace/HiDream-O1-Image
RUN chmod +x /workspace/HiDream-O1-Image/deploy/vast/entrypoint.sh

EXPOSE 7861
CMD ["/workspace/HiDream-O1-Image/deploy/vast/entrypoint.sh"]
