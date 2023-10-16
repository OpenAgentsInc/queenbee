import logging
import queue
from logging.handlers import QueueHandler, QueueListener


def init_log():
    que = queue.Queue()
    queue_handler = QueueHandler(que)
    handler = logging.StreamHandler()
    listener = QueueListener(que, handler)
    root = logging.getLogger()
    root.addHandler(queue_handler)
    try:
        # replace root handler with queue
        h = root.handlers.pop()
        f = h.formatter
        handler.setFormatter(f)
    except IndexError:
        # no root handler to pop
        pass
    listener.start()
