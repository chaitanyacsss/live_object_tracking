import datetime
import logging
import os

loggers = {}


def myLogger(name):
    global loggers
    if not os.path.exists("logs"):
        os.makedirs("logs")

    if loggers.get(name):
        return loggers.get(name)
    else:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        now = datetime.datetime.now()
        handler = logging.FileHandler(
            'logs/common_logger_'
            + now.strftime("%Y-%m-%d")
            + '.log')
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler())
        loggers[name] = logger

        return logger
