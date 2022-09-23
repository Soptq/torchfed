from loguru import logger
from tqdm import tqdm


existed_logger_name = []


def make_filter(name):
    def _filter(record):
        return record["extra"].get("name") == name
    return _filter


def get_logger(exp_id, name, level="INFO"):
    if len(existed_logger_name) == 0:
        logger.remove()
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
    logger.add(
        "logs/{exp_id}/{name}.log".format(exp_id=exp_id, name=name),
        backtrace=True,
        diagnose=True,
        level=level,
        filter=make_filter(name)
    )
    existed_logger_name.append(name)
    return logger.bind(name=name)
