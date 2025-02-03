FROM python:3.11-slim

# Install Node.js 18+
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    make \
    curl \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy and set working directory
COPY . /app
WORKDIR /app

ENV PATH="/root/.cargo/bin/:$PATH"

# Download the latest installer
ADD https://astral.sh/uv/0.4.17/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Install just
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin

# Install Python dependencies using uv
RUN uv sync
RUN uv pip install -r pyproject.toml --no-build-isolation
RUN pip install pre-commit
# Setup git hooks
RUN git config --global --add safe.directory /app && \
    pre-commit install

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create volume for caches
VOLUME [ "/app/.pytest_cache", "/app/.ruff_cache" ]

# Set just as the entrypoint with a default command
ENTRYPOINT ["/usr/local/bin/just"]
CMD ["--list"]
