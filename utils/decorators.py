# Copyright (C) 2022 Florian Wolf
#
# This program is licensed under the Creative Commons Attribution 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
import logging
from time import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()


def log_time(func):
    def wraps(*args, **kwargs):
        try:
            start = time()
            return_value = func(*args, **kwargs)
            logger.info("{} took {}s".format(func.__name__, time() - start))
            return return_value
        except Exception as exc:
            logger.exception("func {} raised exception {}".format(func.__name__, exc))
            raise exc

    return wraps
