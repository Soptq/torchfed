from loguru import logger
from tqdm import tqdm


def get_logger(sub_dir, name, level="INFO"):
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
        "logs/{sub_dir}/{name}.log".format(sub_dir=sub_dir, name=name),
        backtrace=True,
        diagnose=True,
        level=level,
    )
    return logger
