import sys
from io import StringIO
from unittest import TestCase

from bisemantic.main import main


def main_function_output(args):
    sys.argv = ["bisemantic"] + args
    sys.stdout = s = StringIO()
    try:
        main()
    except SystemExit:
        pass
    sys.stdout = sys.__stdout__
    return s.getvalue()


class TestCommandLine(TestCase):
    def test_version(self):
        actual = main_function_output(["--version"])
        self.assertEqual("""bisemantic 1.0.0\n""", actual)
