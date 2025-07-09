FROM tensorflow/tensorflow:2.13.0

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
