import logging
import logging.handlers
import os
import sys
import datetime

def get_logger(log_name):
    logger = logging.getLogger()

    logger.setLevel(logging.INFO)

    smpl_fmn = logging.Formatter(fmt='%(message)s')

    hdlr = logging.StreamHandler(sys.stdout)
    hdlr.setLevel(logging.INFO)
    hdlr.setFormatter(smpl_fmn)

    logger.addHandler(hdlr)

    app_home = os.path.abspath(os.path.join(os.path.dirname(__file__), "..") )

    log_dir = os.path.join(app_home, "log")

    now = "{0:%Y%m%d%H%M%S}".format(datetime.datetime.now())

    file_hdlr = logging.handlers.RotatingFileHandler(
        filename=os.path.join(app_home, log_dir, log_name + "_" + now + ".log"),
        backupCount = 10, encoding='utf-8'
    )
    file_hdlr.setLevel(logging.INFO)
    file_hdlr.setFormatter(smpl_fmn)

    logger.addHandler(file_hdlr)

    return logger

