FROM python:3.10-slim

# Рабочая директория
WORKDIR /code

# Зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Копирование файлов
COPY ozon-services .

# По умолчанию запуск train
# CMD ["python", "train.py"]
