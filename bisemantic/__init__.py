"""
Machine learning models to detect equivalence between pairs of natural language text.
"""

__version__ = "1.0.0"

# Column labels in DataFrame input.
text_1 = "text1"
text_2 = "text2"
label = "label"

# noinspection PyUnresolvedReferences
from bisemantic.main import load_data, TextualEquivalenceModel
