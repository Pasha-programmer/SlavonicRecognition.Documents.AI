import logging

from rabbit_mq.consumer import RabbitMQConsumer

logger = logging.getLogger(__name__)

def main():
    """Основная функция"""
    consumer = RabbitMQConsumer()
    
    try:
        consumer.connect()
        consumer.consume()
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        consumer.stop()

if __name__ == "__main__":
    main()