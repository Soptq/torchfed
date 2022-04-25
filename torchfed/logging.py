from loguru import logger
from tqdm import tqdm


def get_logger(name, level="INFO"):
    """
    Get a logger with the given name and level.
    """
    logger.remove()
    logger.add(
        lambda m: tqdm.write(m, end=""),
        colorize=True,
        backtrace=True,
        diagnose=True,
        level=level,
    )
    logger.add(
        "logs/{name}.log".format(name=name),
        backtrace=True,
        diagnose=True,
        level=level,
    )
    return logger
