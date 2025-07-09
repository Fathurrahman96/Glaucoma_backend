FROM tensorflow/tensorflow:2.13.0-cpu-python3.10

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1

COPY requirements.txt .

# Tidak perlu virtualenv, karena Docker container sendiri sudah environment terisolasi
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
