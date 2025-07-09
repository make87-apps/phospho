FROM ghcr.io/make87/debian:bookworm AS base-image

RUN apt-get update \
    && apt-get install --no-install-suggests --no-install-recommends -y \
        build-essential \
        python3-pip \
        python3 \
        libpython3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl gnupg && \
    curl -fsSL https://raw.githubusercontent.com/phospho-app/phosphobot/main/install.sh | sed 's/sudo //g' | bash && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


ENTRYPOINT ["/usr/local/bin/phosphobot", "run", "--port", "80"]
