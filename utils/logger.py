import logging


def setup_logger():
    logging.basicConfig(
        filename='chatbot_log.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )

    logging.info("Logger initialized")
