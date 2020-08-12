import logging
import rootpath

from keras.callbacks import Callback


class Logger(logging.getLoggerClass()):
    """ Initialze log with output to given path """

    def __init__(self, path="{}/res/log/output.log".format(rootpath.detect()), level=logging.DEBUG):
        # Create handlers
        super().__init__(__name__)
        self.setLevel(level)

        stream_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(path)
        stream_handler.setLevel(level)
        file_handler.setLevel(level)

        format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(format)
        file_handler.setFormatter(format)

        # Add handlers to the log
        self.addHandler(stream_handler)
        self.addHandler(file_handler)

    def log(self, msg, level=logging.INFO):
        """ Logs message with log with given level """
        super().log(level, msg)


class LoggingCallback(Callback):
    """
        Callback that logs Keras verbose messages at end of epoch.
    """

    def __init__(self, log_fcn):
        Callback.__init__(self)
        self.log = log_fcn

    def on_epoch_end(self, epoch, logs={}):
        msg = "Epoch: %i, %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        self.log(msg)
