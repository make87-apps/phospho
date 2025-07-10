FROM ghcr.io/make87/debian:bookworm AS base-image

ARG VIRTUAL_ENV=/make87/venv

# Install Python, Git, and required build tools
RUN apt-get update \
    && apt-get install --no-install-suggests --no-install-recommends -y \
        build-essential \
        python3 \
        python3-venv \
        curl \
        ca-certificates \
        git \
    && python3 -m venv ${VIRTUAL_ENV} \
    && curl -LsSf https://astral.sh/uv/install.sh | bash -s -- \
    && mv /root/.local/bin/uv ${VIRTUAL_ENV}/bin/uv \
    && mv /root/.local/bin/uvx ${VIRTUAL_ENV}/bin/uvx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
WORKDIR /app

COPY . .

ARG EXTRAS=""

RUN set -eux; \
    if [ -f ./pip.conf ]; then \
        echo "Found pip.conf, creating PIP_CONFIG_FILE env"; \
        export PIP_CONFIG_FILE="$(pwd)/pip.conf"; \
    fi; \
    if [ -n "${EXTRAS}" ]; then \
        uv pip install ".[${EXTRAS}]"; \
    else \
        uv pip install .; \
    fi

FROM ghcr.io/make87/python3-debian12:latest

ARG VIRTUAL_ENV=/make87/venv
COPY --from=base-image ${VIRTUAL_ENV} ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

ENTRYPOINT ["python3", "-m", "app.main"]
