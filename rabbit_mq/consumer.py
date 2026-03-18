import pika
import json
import logging
from typing import Optional

from ocr.recognition import start_recognition

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RabbitMQConsumer:
    def __init__(self):
        self.host = "localhost"
        self.port = 5672
        self.username = "admin"
        self.password = "admin123"
        self.virtual_host = "/"
        self.queue_name = "Document.Queue"
        
        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[pika.adapters.blocking_connection.BlockingChannel] = None
        
    def connect(self):
        """Установка соединения с RabbitMQ"""
        try:
            # Создаем credentials
            credentials = pika.PlainCredentials(self.username, self.password)
            
            # Параметры соединения
            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                virtual_host=self.virtual_host,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )
            
            # Устанавливаем соединение
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # Объявляем очередь (на случай, если она еще не создана)
            self.channel.queue_declare(queue=self.queue_name, durable=True)
            
            logger.info(f"Успешно подключились к RabbitMQ. Очередь: {self.queue_name}")
            
        except Exception as e:
            logger.error(f"Ошибка подключения к RabbitMQ: {e}")
            raise
    
    def process_message(self, ch, method, properties, body):
        """Обработка полученного сообщения"""
        try:
            # Пытаемся распарсить JSON
            try:
                message = json.loads(body)
                logger.info(f"Получено сообщение (JSON): {message}")
            except json.JSONDecodeError:
                # Если не JSON, обрабатываем как строку
                message = body.decode('utf-8')
                logger.info(f"Получено сообщение (строка): {message}")
            
            # Здесь можно добавить свою логику обработки сообщения
            # Например, сохранение в БД, вызов API и т.д.
            start_recognition()
            
            # Подтверждаем обработку сообщения
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info(f"Сообщение обработано и подтверждено")
            
        except Exception as e:
            logger.error(f"Ошибка при обработке сообщения: {e}")
            # Отклоняем сообщение и не возвращаем в очередь
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    
    def consume(self):
        """Запуск consumer'а"""
        try:
            if not self.connection or self.connection.is_closed:
                self.connect()
            
            # Настраиваем получение только одного сообщения за раз
            self.channel.basic_qos(prefetch_count=1)
            
            # Подписываемся на очередь
            self.channel.basic_consume(
                queue=self.queue_name,
                on_message_callback=self.process_message
            )
            
            logger.info(f"Ожидание сообщений в очереди {self.queue_name}. Для выхода нажмите CTRL+C")
            
            # Запускаем цикл получения сообщений
            self.channel.start_consuming()
            
        except KeyboardInterrupt:
            logger.info("Получен сигнал остановки")
            self.stop()
        except Exception as e:
            logger.error(f"Ошибка в процессе потребления: {e}")
            self.stop()
    
    def stop(self):
        """Остановка consumer'а и закрытие соединения"""
        try:
            if self.channel and self.channel.is_open:
                self.channel.stop_consuming()
            
            if self.connection and self.connection.is_open:
                self.connection.close()
                logger.info("Соединение с RabbitMQ закрыто")
        except Exception as e:
            logger.error(f"Ошибка при закрытии соединения: {e}")