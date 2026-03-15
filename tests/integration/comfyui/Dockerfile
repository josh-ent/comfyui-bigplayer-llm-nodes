FROM python:3.12-slim

ARG COMFYUI_REF=v0.17.2

ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN git clone --branch ${COMFYUI_REF} --depth 1 https://github.com/Comfy-Org/ComfyUI.git

WORKDIR /opt/ComfyUI

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt \
    && python -m pip install -r manager_requirements.txt

EXPOSE 8188

CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--front-end-version", "Comfy-Org/ComfyUI_frontend@latest", "--enable-manager", "--cpu", "--disable-auto-launch", "--disable-api-nodes", "--base-directory", "/opt/bigplayer/runtime"]
