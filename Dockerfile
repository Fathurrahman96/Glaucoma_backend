FROM python:3.10-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1

# Install sistem library (untuk opencv & pillow & lainnya)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Upgrade pip dan tools build Python
RUN pip install --upgrade pip setuptools wheel

# Install semua paket Python
RUN pip install -r requirements.txt --no-cache-dir --verbose

# Salin sisa kode
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
