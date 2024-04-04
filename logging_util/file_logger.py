import logging
import sys

class Logger:
    def __init__(self, logger_name=None, add_stream_handler = True, std_err = False, level = 'debug'):
        self.logger_name = logger_name if logger_name else __name__
        self.add_stream_handler = add_stream_handler
        self.std_err = std_err
        self.level = self.LEVELS.get(level, logging.INFO)
        self.logger = None
    
    LEVELS = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR, 'critical': logging.CRITICAL}
    
    @staticmethod
    def remove_handlers(logger):
        while logger.handlers:
            handler = logger.handlers[0]
            handler.close()
            logger.removeHandler(handler)
        return logger
    
    def set_logger(self, dt_fmt_basic = True):
        logging.basicConfig(
            format="%(asctime)s :: [%(levelname)s] :: %(message)s", 
            datefmt='%d %B, %Y %I:%M:%S %p %z',
            level=logging.ERROR,
            stream = sys.stderr,
            force=True)
        logging.getLogger().removeHandler(logging.getLogger().handlers[0])
        self.logger = logging.getLogger(self.logger_name) ## IMP: __name__ is important for scope of logger
        self.logger.setLevel(logging.DEBUG)
        if dt_fmt_basic:
            formatter = logging.Formatter("%(asctime)s :: [%(levelname)s] :: %(message)s")
        else:
            formatter = logging.Formatter("%(asctime)s :: [%(levelname)s] :: %(message)s", datefmt='%d %B, %Y %I:%M:%S %p %z')
        if self.add_stream_handler:
            if not self.std_err:
                # self.logger = self.remove_handlers(self.logger)
                stream_handler = logging.StreamHandler(stream = sys.stdout)
            else:
                stream_handler = logging.StreamHandler(stream = sys.stderr)
            stream_handler.setFormatter(formatter)
            stream_handler.setLevel(self.level)
            self.logger.addHandler(stream_handler)
    
    def get_logger(self, dt_fmt_basic = True):
        if not self.logger:
            self.set_logger(dt_fmt_basic)
        return self.logger
    
    def add_file_handler(self, filepath, level = logging.INFO):
        file_handler = logging.FileHandler(filepath)
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s', datefmt='%d %B, %Y %I:%M:%S %p %z')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        return self.get_logger()
    
    def __str__(self):
        to_print("logger not initialized")
        if self.logger:
            to_print = ""
            to_print = "".join([to_print,f"Logger is set. Name :: {self.logger.name} with the following handlers :"])
            for num, i in enumerate(self.logger.handlers, start = 1):
                h_name, h_op, h_level = i.__str__().replace("<","").replace(">","").split(" ")
                to_print = "".join([to_print, "\n", f"{num} :: {h_name} :: {h_level}"])
        return to_print
    
def file_logger(level=logging.INFO, want_stream=True, filepath = "/Users/aadilzikre/Documents/Personal/tmp.log"):
    # initialize a logger
    logger = logging.getLogger(__name__)
    # set level to the loggger (this will act as a global level)
    logger.setLevel(logging.DEBUG)
    # initialize a file handler for properly maintaining a log file
    # set level and format for the handler
    file_handler = logging.FileHandler(filepath)
    file_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s', datefmt='%d %B, %Y %I:%M:%S %p %z')
    file_handler.setFormatter(formatter)
    # add the handler to the logger and you are all set for logging
    logger.addHandler(file_handler)
    # add stram handler
    if want_stream:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        formatter_sh = logging.Formatter(
            '%(asctime)s :: %(message)s', datefmt='%d %B, %Y %I:%M:%S %p %z')
        stream_handler.setFormatter(formatter_sh)
        logger.addHandler(stream_handler)
    return logger