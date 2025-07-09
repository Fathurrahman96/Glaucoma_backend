# Ganti base image awalmu (misal: FROM python:3.9-slim-buster)
# dengan image TensorFlow resmi. Ini paling penting.
FROM tensorflow/tensorflow:2.13.0-cpu-python3.10

# Tetapkan direktori kerja di dalam container
WORKDIR /app

# Nonaktifkan cache pip untuk mengurangi ukuran image
ENV PIP_NO_CACHE_DIR=1

# Salin hanya requirements.txt terlebih dahulu untuk memanfaatkan Docker cache layer
# Pastikan requirements.txt ada di root folder proyekmu, sejajar dengan Dockerfile
COPY requirements.txt .

# Konfigurasi Virtual Environment
# Variabel VIRTUAL_ENV membantu pip dan python tahu di mana venv berada
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Buat virtual environment
# Ini akan membuat folder /opt/venv di dalam container
RUN python -m venv $VIRTUAL_ENV

# Instal dependencies dari requirements.txt
# Gunakan /bin/bash -c untuk menjalankan beberapa perintah dalam satu RUN layer
# Pastikan pip berhasil menginstal semua paket
RUN /bin/bash -c "source ${VIRTUAL_ENV}/bin/activate && pip install -r requirements.txt"

# Salin sisa kode aplikasi dari host ke container
# Pastikan file Python utama (misal: app.py) ada di root folder proyekmu
COPY . .

# Paparkan port yang akan digunakan aplikasi Flask-mu
EXPOSE 5000 # Ganti 5000 jika aplikasi Flask-mu berjalan di port lain

# Perintah untuk menjalankan aplikasi saat container dimulai
# Ganti 'app.py' dengan nama file Python utama yang menjalankan aplikasimu
CMD ["python", "app.py"]
