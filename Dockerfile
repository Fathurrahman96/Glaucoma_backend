FROM python:3.10
FROM python:3.10

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1

# Install dependencies sistem (wajib untuk opencv & pillow)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Salin file requirements
# Install dependencies sistem (wajib untuk opencv & pillow)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Salin file requirements
COPY requirements.txt .

# Upgrade pip + tools dan install Python package
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
# Upgrade pip + tools dan install Python package
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Salin semua kode ke container
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
