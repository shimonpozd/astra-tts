FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ASTRA_CONFIG_ROOT="/app"

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        libasound2 \
        libportaudio2 \
        portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

RUN chmod +x docker-entrypoint.sh

ENV PYTHONPATH="/app${PYTHONPATH:+:${PYTHONPATH}}"

EXPOSE 7010

ENTRYPOINT ["./docker-entrypoint.sh"]
