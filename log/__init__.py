import os
import logging.config
from conf.settings import *

_log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../conf/logging.conf')
logging.config.fileConfig(_log_file_path, disable_existing_loggers=False)

if FLASK_ENV == 'prod':
    log = logging.getLogger('errorLogger')
else:
    log = logging.getLogger('errorLogger')
