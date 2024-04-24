import datetime
import math
import time
import os
import builtins
from pathlib import Path

import logging
import sys

l:logging.Logger = None
DEBUG = 0

def init_logger(logger: logging.Logger = None, level: int = logging.INFO):
    global l
    if logger is None:
        l = logging.getLogger()
        l.setLevel(level)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                                      '%m-%d-%Y %H:%M:%S')

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(formatter)

        file_handler = logging.FileHandler('logs.log')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        l.addHandler(file_handler)
        l.addHandler(stdout_handler)
    else:
        l = logger
    return l



def print(*args,debug=None, logger=None, **kwargs):
    global l, DEBUG

    if debug is None:
        debug = DEBUG

    if l is None and logger is None:
        match debug:
            case bool():
                debug = debug
            case int():
                debug = True if debug >= logging.DEBUG else False
    else:
        if logger is None:
            logger = l
        else:
            match debug:
                case bool():
                    debug = logging.DEBUG if debug else logging.INFO
                case int():
                    debug = debug
            logger = init_logger(logger)
        logger.log(debug,*args, **kwargs)
    if debug:
        builtins.print(*args, **kwargs)


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r", start_time = time.time()):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    eta = ((time.time() - start_time) * (total - iteration) / iteration) if iteration > 0 else math.inf
    # express eta as the time it will take to complete the loop in days, hours, minutes, seconds and date/time of completion
    if eta != math.inf:
        eta = time.strftime("%d days, %H hours, %M minutes, %S seconds", time.gmtime(eta))
    else:
        eta = "Unknown"
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix} (eta: {eta})', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print("")

def get_date_time():
    return datetime.datetime.now().strftime("%Y%m%d-%H")

def get_project_dir():
    """ get project root dir """
    return Path(__file__).parent.parent.parent

def get_build_dir():
    return os.path.join(get_project_dir(), "build")

def get_src_dir():
    return os.path.join(get_project_dir(), "src")

def get_data_dir():
    return os.path.join(get_project_dir(), "data")

def get_output_dir():
    return os.path.join(get_project_dir(), "out")