import logging
import logging.config
import logging.handlers

from multiprocessing import Queue, Process


class MultiProcessLogger(Process):
    def __init__(self, conf_file_path, queue: Queue):
        Process.__init__(self)
        logging.config.fileConfig(conf_file_path)
        self.queue = queue

    @staticmethod
    def logger(name: str = __name__) -> logging.Logger:
        return logging.getLogger(name)

    def add_logger(self):
        qh = logging.handlers.QueueHandler(self.queue)
        logger = logging.getLogger()
        logger.addHandler(qh)

    def run(self):
        while True:
            try:
                record = self.queue.get()
                if record is None:  # We send this as a sentinel to tell the listener to quit.
                    break
                logger = logging.getLogger(record.name)
                logger.handle(record)  # No level or filter logic applied - just do it!
            except Exception:
                import sys
                import traceback
                print('Whoops! Problem:', file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

    def stop(self):
        self.queue.put_nowait(None)
