# Dockerfile для контейнеризации программы распознавания глаголицы
# Используется Python 3.9 на основе slim-образа для минимизации размера

FROM python:3.11-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл с зависимостями и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем скрипт и обученную модель
COPY ocr/. ./ocr/
COPY rabbit_mq/. ./rabbit_mq/
COPY glagolitic_model_full.pth .
COPY __main__.py .

# Экспозируем порт, на котором работает приложение
EXPOSE 5174

# Точка входа: запуск скрипта распознавания
ENTRYPOINT ["python", "__main__.py"]