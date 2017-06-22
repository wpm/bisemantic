"""
Machine learning model to detect relationships between pairs of natural language text.
"""

import logging

__version__ = "1.0.0"

# Column labels in DataFrame input.
text_1 = "text1"
text_2 = "text2"
label = "label"

logger = logging.getLogger(__name__)


# noinspection PyShadowingBuiltins
def configure_logger(level, format):
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(format))
    logger.addHandler(h)
