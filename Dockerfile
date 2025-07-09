# Gunakan image Python resmi
FROM python:3.10-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Nonaktifkan cache pip
ENV PIP_NO_CACHE_DIR=1

# Install dependensi sistem (wajib untuk opencv + pillow)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Salin requirements.txt
COPY requirements.txt .

# Install semua dependensi Python (termasuk tensorflow-cpu)
RUN pip install --upgrade pip && pip install -r requirements.txt

# Salin seluruh kode proyek
COPY . .

# Buka port Flask
EXPOSE 5000

# Jalankan aplikasi Flask
CMD ["python", "app.py"]
