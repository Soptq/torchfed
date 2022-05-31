from loguru import logger
from tqdm import tqdm
import datetime


existed_logger_name = []


def make_filter(name):
    def _filter(record):
        return record["extra"].get("name") == name
    return _filter


def get_logger(name, level="INFO"):
    if name in existed_logger_name:
        return logger.bind(name=name)
    logger.add(
        lambda m: tqdm.write(m, end=""),
        colorize=True,
        backtrace=True,
        diagnose=True,
        level=level,
        filter=make_filter(name)
    )
    time_info = datetime.datetime.now().replace(microsecond=0).isoformat().replace(":", "-")
    logger.add(
        "logs/{time_info}+{name}.log".format(time_info=time_info, name=name),
        backtrace=True,
        diagnose=True,
        level=level,
        filter=make_filter(name)
    )
    existed_logger_name.append(name)
    return logger.bind(name=name)
