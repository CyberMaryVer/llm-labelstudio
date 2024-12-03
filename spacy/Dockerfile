FROM python:3.10-slim

ENV PYTHONUNBUFFERED=True \
    PORT=9090 \
    WORKERS=2 \
    THREADS=4

# Install OS-level Dependencies
RUN set -e && apt-get update && apt-get upgrade -y && \
    apt-get install -y git build-essential python3-dev libffi-dev curl gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

CMD exec gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app
