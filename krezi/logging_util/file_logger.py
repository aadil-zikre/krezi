import logging
import sys

class Logger:
    def __init__(self, logger_name=None, add_stream_handler = True, std_err = False, level = 'debug', **kwargs):
        self.logger_name = logger_name if logger_name else __name__
        self.add_stream_handler = add_stream_handler
        self.std_err = std_err
        self.level = self.LEVELS.get(level, logging.INFO)
        self.logger = None
        self.dt_fmt_basic = kwargs.get('dt_fmt_basic', False)
    
    LEVELS = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR, 'critical': logging.CRITICAL}
    
    @staticmethod
    def remove_handlers(logger):
        while logger.handlers:
            handler = logger.handlers[0]
            handler.close()
            logger.removeHandler(handler)
        return logger
    
    def set_logger(self, dt_fmt_basic=None):
        logging.basicConfig(
            format="%(asctime)s :: [%(levelname)s] :: %(message)s", 
            datefmt='%d %B, %Y %I:%M:%S %p %z',
            level=logging.ERROR,
            stream = sys.stderr,
            force=True)
        logging.getLogger().removeHandler(logging.getLogger().handlers[0])
        self.logger = logging.getLogger(self.logger_name) ## IMP: __name__ is important for scope of logger
        self.logger.setLevel(logging.DEBUG)
        dt_fmt_basic = self.dt_fmt_basic if dt_fmt_basic is None else dt_fmt_basic
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
    
    def add_file_handler(self, filepath, level = logging.INFO, dt_fmt_basic=None):
        file_handler = logging.FileHandler(filepath)
        file_handler.setLevel(level)
        dt_fmt_basic = self.dt_fmt_basic if dt_fmt_basic is None else dt_fmt_basic
        if dt_fmt_basic:
            formatter = logging.Formatter(
                '%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s')
        else:
            formatter = logging.Formatter(
                '%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s', datefmt='%d %B, %Y %I:%M:%S %p %z')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        return self.get_logger()
    
    def add_log_prefix(self, prefix):
        self.prefix = prefix
        for handler in self.logger.handlers:
            if self.prefix:
                formatter = logging.Formatter(handler.formatter._fmt.replace("%(message)s", "") + f"{prefix} :: %(message)s", datefmt=handler.formatter.datefmt)
                handler.setFormatter(formatter)
        return self.get_logger()
    
    def remove_log_prefix(self):
        for handler in self.logger.handlers:
            if self.prefix:
                formatter = logging.Formatter(handler.formatter._fmt.replace(f"{self.prefix} :: %(message)s", "%(message)s"), datefmt=handler.formatter.datefmt)
                handler.setFormatter(formatter)
        self.prefix = None
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