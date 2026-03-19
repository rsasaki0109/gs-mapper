FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies (including COLMAP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    colmap \
    && rm -rf /var/lib/apt/lists/*

# Install package
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY configs/ configs/

RUN pip install --no-cache-dir -e ".[app]"

# Copy remaining files
COPY app.py .
COPY scripts/ scripts/
COPY docs/gallery/ docs/gallery/
COPY docs/training_metrics.json docs/

EXPOSE 8501 8080

# Default: run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
